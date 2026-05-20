"""WS /ws/tts — Streaming TTS with chunked diffusion + pipelined first chunk.

Each WebSocket message is a JSON object::

    {
      "type":     "tts.request",
      "text":     "<your text>",
      "language": "en",              // optional
      "voice":    "ajay",           // optional — any profile known to the server (incl. hot-added)
      "epochs":   16,                // optional — sets BOTH first- and rest-chunk steps when the
                                     // specific keys below are omitted (alias: inference_steps)
      "epochs_fc":   4,             // optional — first-chunk num_step only (aliases: first_chunk_epochs)
      "epochs_rest": 12,            // optional — mid + last chunk num_step (aliases: rest_chunk_epochs)
      "digit_words_lang": "hi",     // optional — legacy; prefer digit_pronunciation
      "digit_words_hint": "hinglish",  // optional — English digits in Indic/SEA text
      "digit_pronunciation": "ta"   // optional — ISO / alias for how digits are spoken
    }

``language`` is optional (defaults to the server's ``OMNIVOICE_LANGUAGE``
setting).  ``voice`` is optional (defaults to ``OMNIVOICE_DEFAULT_VOICE``)
and must match a profile known to the server (startup-loaded or hot-added
via ``POST /api/voices``).

Latency strategy (see top-level docs)
─────────────────────────────────────
1. **Aggressive first-chunk text split** (~25 chars).
2. **Diffusion steps for the first chunk** (``FIRST_CHUNK_STEPS``) unless the
   client sends ``epochs_fc`` / ``first_chunk_epochs``, or ``epochs`` /
   ``inference_steps`` when no per-stage override is given.
3. **Rest-chunk steps** (``MID_CHUNK_CFG`` / ``LAST_CHUNK_CFG``) unless the client
   sends ``epochs_rest`` / ``rest_chunk_epochs``, or ``epochs`` when that key
   is not overridden for rest chunks.
4. **Bypass the batch timeout** for chunk 0 via ``submit_immediate``.
5. **Pipelined generation**: while chunk N streams, chunk N+1 is generating.
6. **Crossfade overlap** hides any audible mismatch between the fast first
   chunk and the higher-quality chunks that follow.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..audio_utils import Crossfader, b64_encode, np_to_wav_bytes
from ..config import (
    CROSSFADE_MS,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEED,
    EPOCHS_MAX,
    EPOCHS_MIN,
    FIRST_CHUNK_CFG,
    LAST_CHUNK_CFG,
    MID_CHUNK_CFG,
    SPEED_MAX,
    SPEED_MIN,
    cfg_with_epochs,
)
from ..lang_utils import resolve_language
from ..text_utils import split_first_chunk_early, split_to_chunks


def _coerce_text(raw) -> str:
    """Robustly convert any ``msg["text"]`` payload into a single string.

    Accepts plain strings, lists/tuples (joined with spaces), or coerces other
    types via ``str(...)``.  Returns the trimmed string, or ``""`` on None.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (list, tuple)):
        return " ".join(str(part).strip() for part in raw if part is not None).strip()
    return str(raw).strip()


def _coerce_speed(raw) -> tuple[Optional[float], Optional[str]]:
    """Validate a user-supplied speed value.

    Returns ``(speed, error_message)``.  ``speed`` is ``None`` when the user
    didn't provide a value (server falls back to DEFAULT_SPEED).
    """
    if raw is None:
        return None, None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("", "default", "none", "auto"):
            return None, None
        try:
            raw = float(s)
        except ValueError:
            return None, f"speed must be a number, got {raw!r}"
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None, f"speed must be a number, got {raw!r}"
    if not (SPEED_MIN <= v <= SPEED_MAX):
        return None, f"speed {v} is out of range [{SPEED_MIN}, {SPEED_MAX}]"
    return v, None


def _coerce_epochs(raw) -> tuple[Optional[int], Optional[str]]:
    """Validate client ``epochs`` / ``inference_steps`` (maps to ``num_step``)."""
    if raw is None:
        return None, None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("", "default", "none", "auto"):
            return None, None
        try:
            raw = int(s, 10)
        except ValueError:
            return None, f"epochs must be an integer, got {raw!r}"
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None, f"epochs must be an integer, got {raw!r}"
    if not (EPOCHS_MIN <= v <= EPOCHS_MAX):
        return None, f"epochs {v} is out of range [{EPOCHS_MIN}, {EPOCHS_MAX}]"
    return v, None


def _epochs_field_provided(raw) -> bool:
    """True when the client sent a value other than 'use server default' sentinels."""
    if raw is None:
        return False
    if isinstance(raw, str) and raw.strip().lower() in ("", "default", "none", "auto"):
        return False
    return True


def _resolve_fc_rest_epochs(msg: dict) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Derive first-chunk and rest-chunk ``num_step`` overrides from the WS payload.

    Precedence:
      * ``epochs_fc`` / ``first_chunk_epochs`` applies to chunk 0 only.
      * ``epochs_rest`` / ``rest_chunk_epochs`` / ``mid_chunk_epochs`` applies to chunks ≥1.
      * ``epochs`` / ``inference_steps`` fills whichever of the above was **not** explicitly set.
    """
    raw_fc = (
        msg.get("epochs_fc")
        or msg.get("first_chunk_epochs")
        or msg.get("firstChunkEpochs")
    )
    raw_rest = (
        msg.get("epochs_rest")
        or msg.get("rest_chunk_epochs")
        or msg.get("restChunkEpochs")
        or msg.get("mid_chunk_epochs")
    )
    raw_legacy = msg.get("epochs")
    if raw_legacy is None:
        raw_legacy = msg.get("inference_steps", msg.get("inferenceSteps"))

    fc_opt: Optional[int]
    rest_opt: Optional[int]

    if _epochs_field_provided(raw_fc):
        fc_opt, err = _coerce_epochs(raw_fc)
        if err:
            return None, None, err
    else:
        fc_opt = None

    if _epochs_field_provided(raw_rest):
        rest_opt, err = _coerce_epochs(raw_rest)
        if err:
            return None, None, err
    else:
        rest_opt = None

    leg_opt: Optional[int] = None
    if _epochs_field_provided(raw_legacy):
        leg_opt, err = _coerce_epochs(raw_legacy)
        if err:
            return None, None, err

    if fc_opt is None:
        fc_opt = leg_opt
    if rest_opt is None:
        rest_opt = leg_opt

    return fc_opt, rest_opt, None


logger = logging.getLogger("omnivoice.streaming")

router = APIRouter()

# Binary WebSocket frame header format (8 bytes, little-endian):
#   chunk_index  : uint16  (0-65535)
#   flags        : uint16  (bit 0 = is_last, bit 1 = is_first)
#   sample_count : uint32  (number of PCM16 samples in this frame)
_BIN_HEADER = struct.Struct("<HHI")
_FLAG_FIRST = 0x02
_FLAG_LAST  = 0x01


def _np_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 audio to raw PCM16 little-endian bytes (no WAV header)."""
    pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    return pcm16.tobytes()


def _coerce_opt_str(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


@router.websocket("/ws/tts")
async def ws_tts(websocket: WebSocket):
    await websocket.accept()
    logger.info("WS connected: %s", websocket.client)
    batcher = websocket.app.state.batcher
    sample_rate: int = getattr(websocket.app.state, "sample_rate", 24_000)
    default_voice: str = getattr(websocket.app.state, "default_voice", "")

    try:
        while True:
            # ──────────────────────────────────────────────────────────
            # 1. Receive request
            # ──────────────────────────────────────────────────────────
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break

            mt = msg.get("type", "")
            if mt == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if mt not in ("tts.request", ""):
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown type '{mt}'. Send tts.request.",
                })
                continue

            text = _coerce_text(msg.get("text"))
            if not text:
                await websocket.send_json({"type": "error", "message": "Empty text."})
                continue

            available_voices: dict = getattr(websocket.app.state, "voice_profiles", {}) or {}

            # Language: client sends ISO code, name, or 'auto' (None = auto).
            raw_lang = msg.get("language")
            if raw_lang is None or (isinstance(raw_lang, str) and not raw_lang.strip()):
                raw_lang = DEFAULT_LANGUAGE
            language = resolve_language(raw_lang)

            raw_voice = msg.get("voice")
            voice = (str(raw_voice).strip() if raw_voice else "") or default_voice
            if available_voices and voice not in available_voices:
                await websocket.send_json({
                    "type": "error",
                    "message": (
                        f"Voice '{voice}' not loaded. Available: "
                        f"{sorted(available_voices.keys())}"
                    ),
                })
                continue

            # Model type: client can send it but it's informational —
            # the worker uses whatever was loaded at startup.
            model_type: str = getattr(websocket.app.state, "model_type", "triton")

            speed, speed_err = _coerce_speed(msg.get("speed"))
            if speed_err:
                await websocket.send_json({"type": "error", "message": speed_err})
                continue
            if speed is None:
                speed = DEFAULT_SPEED

            digit_words_lang = _coerce_opt_str(
                msg.get("digit_words_lang") or msg.get("digitWordsLang")
            )
            digit_words_hint = _coerce_opt_str(
                msg.get("digit_words_hint") or msg.get("digitWordsHint")
            )
            digit_pronunciation = _coerce_opt_str(
                msg.get("digit_pronunciation") or msg.get("digitPronunciation")
            )

            epochs_fc, epochs_rest, epochs_err = _resolve_fc_rest_epochs(msg)
            if epochs_err:
                await websocket.send_json({"type": "error", "message": epochs_err})
                continue

            # Binary audio transport (opt 2.2): client opts in per-request.
            # Binary mode sends raw PCM16 as WebSocket binary frames with an
            # 8-byte header, eliminating base64 + WAV overhead (~60% smaller).
            binary_audio: bool = bool(msg.get("binary_audio", False))

            fc_cfg = cfg_with_epochs(FIRST_CHUNK_CFG, epochs_fc)

            logger.info(
                "WS TTS request  voice=%s  lang=%s  speed=%s  model=%s  epochs_fc=%s  epochs_rest=%s  text=%.80s",
                voice, language or "auto",
                f"{speed:.2f}" if speed is not None else "default",
                model_type,
                epochs_fc if epochs_fc is not None else "default",
                epochs_rest if epochs_rest is not None else "default",
                text,
            )

            # ──────────────────────────────────────────────────────────
            # 2. Split: aggressive short first chunk + remainder
            # ──────────────────────────────────────────────────────────
            first_text, remainder = split_first_chunk_early(text)
            rest_chunks = split_to_chunks(remainder) if remainder else []
            all_chunks  = [first_text] + rest_chunks
            n           = len(all_chunks)
            logger.info(
                "Split %d chunk(s)  first=%d chars  rest=%d chunks",
                n, len(first_text), len(rest_chunks),
            )

            t_req = time.perf_counter()
            first_latency_ms: Optional[float] = None
            total_samples = 0
            cumulative_audio_ms = 0.0
            prev_gen_end: Optional[float] = None
            xfader = Crossfader(sample_rate, CROSSFADE_MS)

            # ──────────────────────────────────────────────────────────
            # 3. Generate first chunk via the PRIORITY first-chunk queue.
            #    Rest-chunks are NOT pre-submitted yet — doing so would
            #    put them in the executor queue ahead of other streams'
            #    first chunks and cause 5-10x FCL inflation under
            #    concurrent load.  We submit them right after sending
            #    the first-chunk response.
            # ──────────────────────────────────────────────────────────
            prefetch_task: Optional[asyncio.Task] = None

            t_first_await_start = time.perf_counter()
            try:
                first_audio_raw, first_gen_ms = await batcher.submit_first_chunk_raw(
                    first_text, fc_cfg,
                    language=language, voice=voice, speed=speed,
                    digit_words_lang=digit_words_lang,
                    digit_words_hint=digit_words_hint,
                    digit_pronunciation=digit_pronunciation,
                )
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue
            gen_end = time.perf_counter()

            # Raw numpy path — no WAV decode needed
            first_audio = xfader.process(first_audio_raw, is_first=True, is_last=(n == 1))
            total_samples += len(first_audio)
            first_latency_ms = (gen_end - t_req) * 1000.0

            chunk_audio_ms = len(first_audio) / sample_rate * 1000.0
            cumulative_audio_ms += chunk_audio_ms
            chunk_wall_ms = (gen_end - t_first_await_start) * 1000.0
            since_request_ms = (gen_end - t_req) * 1000.0
            prev_gen_end = gen_end

            logger.info(
                "FIRST CHUNK  latency=%.1fms  gen_gpu=%.1fms  wait=%.1fms  "
                "audio=%.0fms  text='%s'",
                first_latency_ms, first_gen_ms,
                chunk_wall_ms - first_gen_ms, chunk_audio_ms, first_text,
            )

            # Encode to WAV only once, right before sending
            if binary_audio:
                # Binary mode: send raw PCM16 + separate JSON metadata
                pcm_bytes = _np_to_pcm16_bytes(first_audio)
                flags = _FLAG_FIRST | (_FLAG_LAST if n == 1 else 0)
                header = _BIN_HEADER.pack(0, flags, len(first_audio))
                await websocket.send_bytes(header + pcm_bytes)
                await websocket.send_json({
                    "type":                   "response.audio.meta",
                    "chunk_index":            0,
                    "chunk_text":             first_text,
                    "chunk_audio_ms":         round(chunk_audio_ms, 1),
                    "chunk_gen_ms":           round(first_gen_ms, 1),
                    "chunk_wall_ms":          round(chunk_wall_ms, 1),
                    "since_request_ms":       round(since_request_ms, 1),
                    "cumulative_audio_ms":    round(cumulative_audio_ms, 1),
                    "sample_rate":            sample_rate,
                    "language":               language or "auto",
                    "voice":                  voice,
                    "speed":                  speed,
                    "model_type":             model_type,
                    "first_chunk_latency_ms": round(first_latency_ms, 1),
                    "epochs":                 int(fc_cfg["num_step"]),
                })
            else:
                wav_out = np_to_wav_bytes(first_audio, sample_rate)
                # First chunk includes full static metadata; subsequent chunks omit it (3.4)
                await websocket.send_json({
                    "type":                   "response.audio.delta",
                    "delta":                  b64_encode(wav_out),
                    "encoding":               "wav/pcm16",
                    "sample_rate":            sample_rate,
                    "chunk_index":            0,
                    "chunk_text":             first_text,
                    "chunk_audio_ms":         round(chunk_audio_ms, 1),
                    "chunk_audio_sec":        round(chunk_audio_ms / 1000.0, 3),
                    "chunk_gen_ms":           round(first_gen_ms, 1),
                    "chunk_wall_ms":          round(chunk_wall_ms, 1),
                    "since_prev_chunk_ms":    None,
                    "since_request_ms":       round(since_request_ms, 1),
                    "cumulative_audio_ms":    round(cumulative_audio_ms, 1),
                    "cumulative_audio_sec":   round(cumulative_audio_ms / 1000.0, 3),
                    "language":               language or "auto",
                    "voice":                  voice,
                    "speed":                  speed,
                    "model_type":             model_type,
                    "first_chunk_latency_ms": round(first_latency_ms, 1),
                    "epochs":                 int(fc_cfg["num_step"]),
                })

            # NOW that the first chunk is sent, pre-submit chunk 1.
            # This is intentionally deferred: submitting before the
            # first-chunk response would put rest-chunks in the executor
            # queue ahead of other streams' first chunks.
            if n > 1:
                base_1 = LAST_CHUNK_CFG if n == 2 else MID_CHUNK_CFG
                cfg_1 = cfg_with_epochs(base_1, epochs_rest)
                prefetch_task = asyncio.create_task(
                    batcher.submit_raw(
                        all_chunks[1], cfg_1,
                        language=language, voice=voice, speed=speed,
                        digit_words_lang=digit_words_lang,
                        digit_words_hint=digit_words_hint,
                        digit_pronunciation=digit_pronunciation,
                    )
                )

            # ──────────────────────────────────────────────────────────
            # 4. Remaining chunks — pipelined
            # ──────────────────────────────────────────────────────────
            for i in range(1, n):
                is_last = (i == n - 1)

                t_await_start = time.perf_counter()
                if prefetch_task is not None:
                    try:
                        audio_raw = await prefetch_task
                    except Exception as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        break
                else:
                    base_i = LAST_CHUNK_CFG if is_last else MID_CHUNK_CFG
                    cfg_i = cfg_with_epochs(base_i, epochs_rest)
                    audio_raw = await batcher.submit_raw(
                        all_chunks[i], cfg_i,
                        language=language, voice=voice, speed=speed,
                        digit_words_lang=digit_words_lang,
                        digit_words_hint=digit_words_hint,
                        digit_pronunciation=digit_pronunciation,
                    )
                gen_end = time.perf_counter()

                prefetch_task = None
                if i + 1 < n:
                    base_next = (
                        LAST_CHUNK_CFG if (i + 1 == n - 1) else MID_CHUNK_CFG
                    )
                    cfg_next = cfg_with_epochs(base_next, epochs_rest)
                    prefetch_task = asyncio.create_task(
                        batcher.submit_raw(
                            all_chunks[i + 1], cfg_next,
                            language=language, voice=voice, speed=speed,
                            digit_words_lang=digit_words_lang,
                            digit_words_hint=digit_words_hint,
                            digit_pronunciation=digit_pronunciation,
                        )
                    )

                # Raw numpy path — no WAV decode needed
                audio = xfader.process(audio_raw, is_first=False, is_last=is_last)
                total_samples += len(audio)

                chunk_wall_ms       = (gen_end - t_await_start) * 1000.0
                since_prev_chunk_ms = (gen_end - prev_gen_end) * 1000.0 if prev_gen_end else 0.0
                since_request_ms    = (gen_end - t_req) * 1000.0
                chunk_audio_ms      = len(audio) / sample_rate * 1000.0
                cumulative_audio_ms += chunk_audio_ms
                prev_gen_end = gen_end

                rest_epoch_cfg = cfg_with_epochs(
                    LAST_CHUNK_CFG if is_last else MID_CHUNK_CFG,
                    epochs_rest,
                )
                epochs_this = int(rest_epoch_cfg["num_step"])

                logger.info(
                    "CHUNK[%d]  wall=%.1fms  since_prev=%.1fms  since_req=%.1fms  "
                    "audio=%.0fms",
                    i, chunk_wall_ms, since_prev_chunk_ms, since_request_ms,
                    chunk_audio_ms,
                )

                # Encode to WAV only once, right before sending
                if binary_audio:
                    pcm_bytes = _np_to_pcm16_bytes(audio)
                    flags = (_FLAG_FIRST if i == 0 else 0) | (_FLAG_LAST if is_last else 0)
                    header = _BIN_HEADER.pack(i, flags, len(audio))
                    await websocket.send_bytes(header + pcm_bytes)
                    await websocket.send_json({
                        "type":                 "response.audio.meta",
                        "chunk_index":          i,
                        "chunk_text":           all_chunks[i],
                        "chunk_audio_ms":       round(chunk_audio_ms, 1),
                        "chunk_gen_ms":         round(chunk_wall_ms, 1),
                        "chunk_wall_ms":        round(chunk_wall_ms, 1),
                        "since_prev_chunk_ms":  round(since_prev_chunk_ms, 1),
                        "since_request_ms":     round(since_request_ms, 1),
                        "cumulative_audio_ms":  round(cumulative_audio_ms, 1),
                        "epochs":               epochs_this,
                    })
                else:
                    wav_out = np_to_wav_bytes(audio, sample_rate)
                    # Slim payload — omit static metadata already sent in chunk 0 (3.4)
                    await websocket.send_json({
                        "type":                 "response.audio.delta",
                        "delta":                b64_encode(wav_out),
                        "encoding":             "wav/pcm16",
                        "sample_rate":          sample_rate,
                        "chunk_index":          i,
                        "chunk_text":           all_chunks[i],
                        "chunk_audio_ms":       round(chunk_audio_ms, 1),
                        "chunk_audio_sec":      round(chunk_audio_ms / 1000.0, 3),
                        "chunk_gen_ms":         round(chunk_wall_ms, 1),
                        "chunk_wall_ms":        round(chunk_wall_ms, 1),
                        "since_prev_chunk_ms":  round(since_prev_chunk_ms, 1),
                        "since_request_ms":     round(since_request_ms, 1),
                        "cumulative_audio_ms":  round(cumulative_audio_ms, 1),
                        "cumulative_audio_sec": round(cumulative_audio_ms / 1000.0, 3),
                        "epochs":               epochs_this,
                    })

            tail = xfader.flush()
            if tail is not None:
                total_samples += len(tail)

            # ──────────────────────────────────────────────────────────
            # 5. Done
            # ──────────────────────────────────────────────────────────
            total_audio_ms = round(total_samples / sample_rate * 1000)
            total_wall_ms  = round((time.perf_counter() - t_req) * 1000, 1)

            epochs_first = int(fc_cfg["num_step"])
            if n <= 1:
                epochs_rest_done = epochs_first
            else:
                epochs_rest_done = int(
                    cfg_with_epochs(MID_CHUNK_CFG, epochs_rest)["num_step"]
                )

            await websocket.send_json({
                "type":                   "response.audio.done",
                "total_chunks":           n,
                "total_audio_ms":         total_audio_ms,
                "total_gen_ms":           total_wall_ms,
                "first_chunk_latency_ms": round(first_latency_ms or 0, 1),
                "language":               language or "auto",
                "voice":                  voice,
                "speed":                  speed,
                "model_type":             model_type,
                "epochs_first_chunk":     epochs_first,
                "epochs_rest_chunk":      epochs_rest_done,
            })
            logger.info(
                "Done.  chunks=%d  audio=%dms  wall=%.0fms  first_chunk=%.0fms  "
                "voice=%s  lang=%s  speed=%s  epochs_fc=%d  epochs_rest=%d",
                n, total_audio_ms, total_wall_ms, first_latency_ms or 0,
                voice, language or "auto",
                f"{speed:.2f}" if speed is not None else "default",
                epochs_first, epochs_rest_done,
            )

    except WebSocketDisconnect:
        logger.info("WS disconnected: %s", websocket.client)
    except Exception as exc:
        logger.exception("WS error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        logger.info("WS closed: %s", websocket.client)

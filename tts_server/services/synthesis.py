from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import HTTPException

from ..audio_utils import b64_encode, wav_bytes_to_np
from ..config import DEFAULT_LANGUAGE, DEFAULT_SPEED, LAST_CHUNK_CFG, MID_CHUNK_CFG, cfg_with_epochs
from ..lang_utils import resolve_language
from ..schemas import BatchTTSItem, BatchTTSRequest, BatchTTSResponse


class DefaultSynthesisService:
    async def synth_batch(self, req: BatchTTSRequest, app_state: Any) -> BatchTTSResponse:
        if not req.texts:
            raise HTTPException(400, "'texts' list cannot be empty.")
        batcher = getattr(app_state, "batcher", None)
        if batcher is None:
            raise HTTPException(503, "Server not ready.")

        available_voices: dict = getattr(app_state, "voice_profiles", {})
        default_voice: str = getattr(app_state, "default_voice", "")
        voice = (req.voice or default_voice).strip() or default_voice
        if available_voices and voice not in available_voices:
            raise HTTPException(400, f"Voice '{voice}' is not loaded. Available voices: {sorted(available_voices.keys())}")

        cfg_base = LAST_CHUNK_CFG if req.use_high_quality else MID_CHUNK_CFG
        cfg = cfg_with_epochs(cfg_base, req.epochs)
        language = resolve_language(req.language or DEFAULT_LANGUAGE)
        speed = req.speed if req.speed is not None else DEFAULT_SPEED
        batches_before = batcher.total_batches
        t0 = time.perf_counter()

        tasks = [
            asyncio.create_task(
                batcher.submit(
                    text,
                    cfg,
                    language=language,
                    voice=voice,
                    speed=speed,
                    digit_words_lang=req.digit_words_lang,
                    digit_words_hint=req.digit_words_hint,
                    digit_pronunciation=req.digit_pronunciation,
                )
            )
            for text in req.texts
        ]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)
        total_ms = (time.perf_counter() - t0) * 1000.0

        items: list[BatchTTSItem] = []
        for i, result in enumerate(results_raw):
            if isinstance(result, Exception):
                raise HTTPException(500, f"Generation failed for item {i}: {result}")
            audio, sr = wav_bytes_to_np(result)
            items.append(BatchTTSItem(id=i, audio_base64=b64_encode(result), audio_ms=round(len(audio) / sr * 1000), sample_rate=sr))

        return BatchTTSResponse(
            results=items,
            total_gen_ms=round(total_ms, 1),
            batch_size=len(req.texts),
            server_batches_formed=batcher.total_batches - batches_before,
            language=language or "auto",
            voice=voice,
            speed=speed,
            epochs=int(cfg["num_step"]),
        )


"""HTTP API for voice profiles — list and hot-add without server restart."""

from __future__ import annotations

import asyncio
import logging
import re
import json
from datetime import datetime, timezone
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..config import MAX_WORKERS, MODEL_ID, VOICE_DIR
from ..voice_init import build_voice_init_spec
from ..voice_profiles import (
    VoiceEmbedding,
    VoiceProfile,
    list_profiles,
    load_profile_by_name,
)
from ..worker import worker_add_voice

logger = logging.getLogger("omnivoice.voices")

router = APIRouter()

_ALLOWED_AUDIO_EXT = frozenset({".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a"})


def _slug_voice_name(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise HTTPException(400, "Voice name is empty or invalid after normalisation.")
    if len(s) > 64:
        s = s[:64]
    return s


def _safe_audio_suffix(filename: str) -> str:
    suf = Path(filename or "").suffix.lower()
    if suf not in _ALLOWED_AUDIO_EXT:
        raise HTTPException(
            400,
            f"Unsupported audio extension {suf!r}. Allowed: "
            f"{', '.join(sorted(_ALLOWED_AUDIO_EXT))}",
        )
    return suf


async def _broadcast_worker_add_voice(
    executor: ProcessPoolExecutor,
    name: str,
    spec: dict,
    model_id: str,
) -> list[int]:
    """Run ``worker_add_voice`` until every pool process has reported a unique PID."""
    loop = asyncio.get_running_loop()
    max_w = int(getattr(executor, "_max_workers", None) or MAX_WORKERS)
    seen: set[int] = set()
    max_attempts = max(max_w * 12, 24)
    for _ in range(max_attempts):
        if len(seen) >= max_w:
            break
        pid = int(await loop.run_in_executor(
            executor, worker_add_voice, name, spec, model_id,
        ))
        seen.add(pid)
    if len(seen) < max_w:
        logger.warning(
            "Hot-voice '%s': only %d distinct worker PID(s) after %d executor "
            "submissions (expected %d). Voice may be missing on some workers.",
            name, len(seen), max_attempts, max_w,
        )
    return sorted(seen)


@router.get("/api/voices")
async def list_voice_profiles(request: Request):
    """Return profile names discoverable under ``voice_reference/`` plus server state."""
    names_disk = list_profiles()
    loaded = getattr(request.app.state, "voice_profiles", {}) or {}
    default_voice = getattr(request.app.state, "default_voice", "")
    sample_rate = int(getattr(request.app.state, "sample_rate", 24_000))

    details: list[dict] = []
    for n in names_disk:
        try:
            p = load_profile_by_name(n)
        except Exception as exc:
            details.append({"name": n, "error": str(exc)})
            continue
        try:
            ref_audio = str(p.resolve_ref_audio())
        except Exception as exc:
            ref_audio = str(exc)
        details.append({
            "name":             p.name,
            "language":       p.language,
            "ref_audio":      ref_audio,
            "ref_audio_file": p.ref_audio,
            "embedding_path": str(p.resolve_embedding_path()),
            "cached":         p.has_cached_embedding(),
            "embedding_meta": p.embedding_meta,
            "loaded_in_server": n in loaded,
            "is_default":     n == default_voice,
        })

    return {
        "names":           names_disk,
        "default_voice":   default_voice,
        "sample_rate":     sample_rate,
        "loaded_names":    sorted(loaded.keys()),
        "profiles":        details,
    }


@router.post("/api/voices")
async def create_voice_profile(
    request: Request,
    name: str = Form(..., description="Persona id, e.g. mani (stored as mani_ref.json)"),
    ref_text: str = Form(..., description="Transcript aligned with the reference clip"),
    language: str = Form(""),
    file: UploadFile = File(..., description="Reference audio (wav/mp3/flac/…)"),
):
    """Upload reference audio + metadata, write ``<name>_ref.json``, hot-load all workers."""
    executor: ProcessPoolExecutor | None = getattr(request.app.state, "executor", None)
    profiles: dict | None = getattr(request.app.state, "voice_profiles", None)
    if executor is None or profiles is None:
        raise HTTPException(503, "Server not ready (workers not initialised).")

    slug = _slug_voice_name(name)
    if not ref_text or not str(ref_text).strip():
        raise HTTPException(400, "ref_text is required.")

    suf = _safe_audio_suffix(file.filename or "")
    json_path = VOICE_DIR / f"{slug}_ref.json"
    audio_rel = f"{slug}_ref{suf}"
    audio_path = VOICE_DIR / audio_rel

    VOICE_DIR.mkdir(parents=True, exist_ok=True)

    body = await file.read()
    if not body:
        raise HTTPException(400, "Empty audio upload.")
    max_bytes = 25 * 1024 * 1024
    if len(body) > max_bytes:
        raise HTTPException(400, f"Audio file too large (max {max_bytes // (1024*1024)} MiB).")

    audio_path.write_bytes(body)

    lang_clean = (language or "").strip()
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "name":            slug,
        "ref_text":        ref_text.strip(),
        "ref_audio":       audio_rel,
        "language":        lang_clean,
        "embedding_path":  None,
        "embedding_meta":  {},
        "created_at":      created_at,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Drop stale cache so a re-upload of the same slug cannot reuse old tokens.
    stale_npz = VOICE_DIR / f"{slug}_embedding.npz"
    try:
        if stale_npz.exists():
            stale_npz.unlink()
    except OSError as exc:
        logger.warning("Could not remove stale embedding %s: %s", stale_npz, exc)

    try:
        profile = VoiceProfile.from_json(json_path)
        profile.resolve_ref_audio()
    except Exception as exc:
        try:
            audio_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except OSError:
            pass
        try:
            json_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except OSError:
            pass
        raise HTTPException(400, f"Invalid profile or audio: {exc}") from exc

    sr = int(getattr(request.app.state, "sample_rate", 24_000))
    try:
        emb = profile.load_cached_embedding()
        sr = int(emb.sampling_rate)
    except Exception:
        pass

    spec, used_cache = build_voice_init_spec(profile, sr)
    try:
        worker_pids = await _broadcast_worker_add_voice(
            executor, profile.name, spec, MODEL_ID,
        )
    except Exception as exc:
        logger.exception("Hot-load failed for voice '%s'", slug)
        raise HTTPException(500, f"Worker hot-load failed: {exc}") from exc

    # Refresh profile from disk (worker may have written embedding .npz)
    profile = load_profile_by_name(slug)
    emb_path = profile.resolve_embedding_path()
    if emb_path.exists():
        try:
            embedding = VoiceEmbedding.from_npz(emb_path)
            profile.update_embedding_metadata(embedding)
            profile = VoiceProfile.from_json(json_path)
        except Exception as exc:
            logger.warning("[%s] could not sync embedding metadata to JSON: %s", slug, exc)

    profiles[profile.name] = profile

    return {
        "ok":               True,
        "name":             profile.name,
        "json_path":        str(json_path),
        "ref_audio":        audio_rel,
        "used_npz_cache":   used_cache,
        "worker_pids":      worker_pids,
        "embedding_path":   str(profile.resolve_embedding_path()),
        "has_embedding":    profile.has_cached_embedding(),
        "embedding_meta":   profile.embedding_meta,
    }

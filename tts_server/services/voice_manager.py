"""Voice management helpers used by the `voices` router.

This centralises common operations so the router stays focused on HTTP
handling and the logic can be reused or unit-tested more easily.
"""
from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from fastapi import HTTPException

from ..config import MAX_WORKERS, MODEL_ID, VOICE_DIR
from ..voice_init import build_voice_init_spec
from ..voice_profiles import (
    VoiceEmbedding,
    VoiceProfile,
    find_profile_json,
    list_profiles,
    load_profile_by_name,
)
from ..worker import (
    worker_add_voice,
    worker_remove_voice,
    worker_set_default_voice,
)

logger = logging.getLogger("omnivoice.voice_manager")

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
            f"Unsupported audio extension {suf!r}. Allowed: {', '.join(sorted(_ALLOWED_AUDIO_EXT))}",
        )
    return suf


async def _broadcast_to_workers(
    executor: ProcessPoolExecutor,
    fn,
    *args,
    label: str = "task",
) -> list[int]:
    """Run picklable *fn* on the pool until each worker PID has been seen.

    Returns a sorted list of observed PIDs.
    """
    loop = asyncio.get_running_loop()
    max_w = int(getattr(executor, "_max_workers", None) or MAX_WORKERS)
    seen: set[int] = set()
    max_attempts = max(max_w * 12, 24)
    for _ in range(max_attempts):
        if len(seen) >= max_w:
            break
        pid = int(await loop.run_in_executor(executor, fn, *args))
        seen.add(pid)
    if len(seen) < max_w:
        logger.warning(
            "Broadcast %s: only %d distinct worker PID(s) after %d submissions (expected %d).",
            label, len(seen), max_attempts, max_w,
        )
    return sorted(seen)


def _try_slug(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _load_profile_fuzzy(raw: str) -> VoiceProfile:
    raw = (raw or "").strip()
    if not raw:
        raise HTTPException(400, "Empty voice name.")
    for key in (raw, _try_slug(raw)):
        if not key:
            continue
        try:
            return load_profile_by_name(key)
        except FileNotFoundError:
            continue
    raise HTTPException(404, f"Voice {raw!r} not found. Available: {list_profiles()}")


def _unlink_profile_files(profile: VoiceProfile) -> None:
    try:
        if profile.has_cached_embedding():
            profile.resolve_embedding_path().unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("unlink npz: %s", exc)
    try:
        profile.resolve_ref_audio().unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("unlink ref audio: %s", exc)
    try:
        profile.json_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("unlink json: %s", exc)

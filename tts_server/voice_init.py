"""Shared helpers to build worker voice-init specs from :class:`VoiceProfile`.

Used at server startup (``app.lifespan``) and for hot-adding profiles at runtime
without restarting workers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from omnivoice.utils.audio import load_audio

from .config import REF_PROMPT_MAX_SEC
from .voice_profiles import VoiceEmbedding, VoiceProfile

logger = logging.getLogger("omnivoice.voice_init")


def trim_ref_audio(
    path: Path, sr: int, max_sec: float, ref_text: Optional[str],
) -> tuple[np.ndarray, Optional[str]]:
    """Load + trim reference audio to ``max_sec`` and proportionally shorten transcript."""
    wav   = load_audio(str(path), sr)
    total = wav.shape[-1]
    max_s = max(1, int(max_sec * sr))
    if total <= max_s:
        return wav, ref_text

    trimmed = wav[..., :max_s]
    frac    = trimmed.shape[-1] / total
    short_text = ref_text
    if ref_text and frac < 1.0:
        n   = max(32, int(len(ref_text) * frac))
        cut = ref_text[:n]
        sp  = cut.rfind(" ")
        if sp > n // 2:
            cut = cut[:sp]
        short_text = cut.strip()

    logger.warning(
        "Reference audio trimmed %.1fs → %.1fs (max=%.1fs).",
        total / sr, trimmed.shape[-1] / sr, max_sec,
    )
    return trimmed, short_text


def serialise_embedding(embedding: VoiceEmbedding) -> dict:
    """Return a dict that crosses the spawn boundary cleanly."""
    return {
        "ref_audio_tokens": embedding.ref_audio_tokens.astype(np.int64),
        "ref_text":         embedding.ref_text,
        "ref_rms":          float(embedding.ref_rms),
        "sampling_rate":    int(embedding.sampling_rate),
        "model_id":         embedding.model_id,
        "num_codebooks":    int(embedding.num_codebooks),
        "num_tokens":       int(embedding.num_tokens),
    }


def build_voice_init_spec(
    profile: VoiceProfile,
    sr: int,
) -> tuple[dict, bool]:
    """Prepare the spawn-safe init spec for one voice profile.

    Returns ``(spec, used_cache)`` where ``used_cache=True`` means the
    pre-built numpy embedding was used (fast path).
    """
    spec: dict = {
        "cached_embedding": None,
        "raw_ref_bytes":    None,
        "raw_ref_sr":       None,
        "raw_ref_text":     None,
        "cache_save_path":  None,
        "ref_audio_path":   str(profile.resolve_ref_audio()),
        "full_ref_text":    profile.ref_text,
    }

    if profile.has_cached_embedding():
        try:
            embedding = profile.load_cached_embedding()
            spec["cached_embedding"] = serialise_embedding(embedding)
            logger.info(
                "  [%s] cached embedding %s  tokens=(%d, %d)",
                profile.name,
                profile.resolve_embedding_path(),
                embedding.num_codebooks, embedding.num_tokens,
            )
            return spec, True
        except Exception as exc:
            logger.warning(
                "  [%s] could not load cached embedding (%s) — rebuilding.",
                profile.name, exc,
            )

    ref_path = profile.resolve_ref_audio()
    ref_clip, ref_text_short = trim_ref_audio(
        ref_path, sr, REF_PROMPT_MAX_SEC, profile.ref_text,
    )
    spec["raw_ref_bytes"]   = ref_clip.flatten().astype(np.float32).tobytes()
    spec["raw_ref_sr"]      = sr
    spec["raw_ref_text"]    = ref_text_short
    spec["cache_save_path"] = str(profile.resolve_embedding_path())
    logger.info(
        "  [%s] cold-path ref clip %.2fs  (%s)",
        profile.name, ref_clip.shape[-1] / sr, ref_path,
    )
    return spec, False

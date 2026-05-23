"""HTTP API for voice profiles — list and hot-add without server restart."""

from __future__ import annotations

import asyncio
import logging
import re
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

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

logger = logging.getLogger("omnivoice.voices")

router = APIRouter()

from ..services.voice_manager import (
    _slug_voice_name,
    _safe_audio_suffix,
    _broadcast_to_workers,
    _try_slug,
    _load_profile_fuzzy,
    _unlink_profile_files,
)


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
        worker_pids = await _broadcast_to_workers(
            executor, worker_add_voice, profile.name, spec, MODEL_ID,
            label=f"add_voice:{slug}",
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


def _unlink_profile_files(profile: VoiceProfile) -> None:
    """Remove npz, reference audio, and JSON (best effort)."""
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


@router.get("/api/voices/{name}")
async def get_one_voice(name: str) -> dict:
    """Return one profile for edit forms."""
    p = _load_profile_fuzzy(name)
    try:
        ref_abs = str(p.resolve_ref_audio())
    except Exception as exc:
        ref_abs = str(exc)
    return {
        "name":             p.name,
        "ref_text":         p.ref_text,
        "language":         p.language,
        "ref_audio":        ref_abs,
        "ref_audio_file":   p.ref_audio,
        "embedding_path":   str(p.resolve_embedding_path()),
        "cached":           p.has_cached_embedding(),
        "embedding_meta":   p.embedding_meta,
        "created_at":       p.created_at,
        "json_path":        str(p.json_path),
    }


@router.delete("/api/voices/{name}")
async def delete_voice(name: str, request: Request) -> dict:
    """Delete JSON + reference audio + npz and unload from all workers."""
    executor: ProcessPoolExecutor | None = getattr(request.app.state, "executor", None)
    profiles: dict | None = getattr(request.app.state, "voice_profiles", None)
    if executor is None or profiles is None:
        raise HTTPException(503, "Server not ready.")

    profile = _load_profile_fuzzy(name)
    key = profile.name
    disk = list_profiles()
    if key not in disk:
        raise HTTPException(404, f"Voice {key!r} not on disk.")
    if len(disk) <= 1:
        raise HTTPException(400, "Cannot delete the only voice profile in voice_reference/.")

    was_default = getattr(request.app.state, "default_voice", "") == key
    loaded = key in profiles

    _unlink_profile_files(profile)
    profiles.pop(key, None)

    if loaded:
        try:
            await _broadcast_to_workers(
                executor, worker_remove_voice, key, label=f"remove:{key}",
            )
        except Exception as exc:
            logger.exception("worker_remove_voice failed for %s", key)
            raise HTTPException(500, str(exc)) from exc

    if was_default and profiles:
        new_def = sorted(profiles.keys())[0]
        request.app.state.default_voice = new_def
        request.app.state.voice_profile = profiles[new_def]
        try:
            await _broadcast_to_workers(
                executor, worker_set_default_voice, new_def,
                label=f"default:{new_def}",
            )
        except Exception as exc:
            logger.warning("worker_set_default_voice: %s", exc)

    return {
        "ok":             True,
        "deleted":        key,
        "default_voice":  getattr(request.app.state, "default_voice", ""),
    }


@router.put("/api/voices/{name}")
async def update_voice(
    request: Request,
    name: str,
    new_name: Optional[str] = Form(default=None),
    ref_text: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    """Update ref_text / language, replace audio, and/or rename (moves files on disk)."""
    executor: ProcessPoolExecutor | None = getattr(request.app.state, "executor", None)
    profiles: dict | None = getattr(request.app.state, "voice_profiles", None)
    if executor is None or profiles is None:
        raise HTTPException(503, "Server not ready.")

    old = _load_profile_fuzzy(name)
    old_key = old.name
    old_json = old.json_path

    raw_new = (new_name or "").strip()
    new_slug = _slug_voice_name(raw_new) if raw_new else old_key
    if new_slug != old_key and find_profile_json(new_slug) is not None:
        raise HTTPException(409, f"Profile {new_slug!r} already exists.")

    has_file = bool(file and getattr(file, "filename", None) and str(file.filename).strip())
    upload_bytes: bytes | None = None
    if has_file:
        upload_bytes = await file.read()
        if not upload_bytes:
            raise HTTPException(400, "Empty audio upload.")
        max_bytes = 25 * 1024 * 1024
        if len(upload_bytes) > max_bytes:
            raise HTTPException(400, f"Audio too large (max {max_bytes // (1024*1024)} MiB).")

    final_ref = (ref_text.strip() if ref_text is not None else old.ref_text)
    if not final_ref:
        raise HTTPException(400, "ref_text cannot be empty.")

    final_lang = str(language).strip() if language is not None else (old.language or "")
    ref_changed = ref_text is not None and ref_text.strip() != old.ref_text
    lang_changed = language is not None and final_lang != (old.language or "")

    if raw_new and new_slug == old_key and not has_file and ref_text is None and language is None and not ref_changed:
        raise HTTPException(400, "new_name matches the current profile; nothing to update.")

    if not raw_new and not has_file and ref_text is None and language is None:
        raise HTTPException(400, "No changes (pass new_name, ref_text, language, and/or file).")

    if new_slug == old_key and not has_file and not ref_changed and not lang_changed:
        if ref_text is not None or language is not None:
            raise HTTPException(400, "No changes.")

    lang_only = (
        new_slug == old_key
        and not has_file
        and not ref_changed
        and lang_changed
    )

    if lang_only:
        old.language = final_lang
        old.save_json()
        profiles[old_key] = VoiceProfile.from_json(old_json)
        return {"ok": True, "name": old_key, "metadata_only": True, "worker_pids": []}

    renamed = new_slug != old_key
    wipe_npz = has_file or ref_changed

    old_audio = old.resolve_ref_audio()
    ext = _safe_audio_suffix(str(file.filename)) if has_file else (old_audio.suffix.lower() or ".mp3")
    new_json = VOICE_DIR / f"{new_slug}_ref.json"
    new_rel = f"{new_slug}_ref{ext}"
    new_audio = VOICE_DIR / new_rel
    old_npz = old.resolve_embedding_path() if old.has_cached_embedding() else None

    if wipe_npz and old_npz and old_npz.exists():
        try:
            old_npz.unlink()
        except OSError as exc:
            logger.warning("npz unlink: %s", exc)

    if has_file:
        assert upload_bytes is not None
        new_audio.write_bytes(upload_bytes)
        if old_audio.resolve() != new_audio.resolve():
            try:
                old_audio.unlink(missing_ok=True)
            except OSError:
                pass
    elif renamed:
        try:
            shutil.move(str(old_audio), str(new_audio))
        except OSError as exc:
            raise HTTPException(500, f"Could not move reference audio: {exc}") from exc
    else:
        new_audio = old_audio
        new_rel = old.ref_audio

    if renamed and old_json.resolve() != new_json.resolve():
        if old_npz and old_npz.exists() and not wipe_npz:
            dest_npz = VOICE_DIR / f"{new_slug}_embedding.npz"
            try:
                shutil.move(str(old_npz), str(dest_npz))
            except OSError as exc:
                logger.warning("npz move: %s", exc)

    emb_name: str | None = None
    emb_meta: dict = {}
    if wipe_npz:
        emb_name, emb_meta = None, {}
    elif renamed:
        emb_name = f"{new_slug}_embedding.npz"
        emb_meta = dict(old.embedding_meta or {})
    else:
        emb_name = old.embedding_path
        emb_meta = dict(old.embedding_meta or {})

    payload = {
        "name":            new_slug,
        "ref_text":        final_ref,
        "ref_audio":       new_rel,
        "language":        final_lang,
        "embedding_path":  emb_name,
        "embedding_meta":  emb_meta,
        "created_at":      old.created_at or datetime.now(timezone.utc).isoformat(),
    }
    new_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if renamed and old_json.resolve() != new_json.resolve() and old_json.exists():
        try:
            old_json.unlink()
        except OSError:
            pass

    try:
        profile = VoiceProfile.from_json(new_json)
        profile.resolve_ref_audio()
    except Exception as exc:
        raise HTTPException(400, f"Invalid profile after update: {exc}") from exc

    sr = int(getattr(request.app.state, "sample_rate", 24_000))
    try:
        emb = profile.load_cached_embedding()
        sr = int(emb.sampling_rate)
    except Exception:
        pass

    spec, _used = build_voice_init_spec(profile, sr)

    if old_key in profiles:
        try:
            await _broadcast_to_workers(
                executor, worker_remove_voice, old_key, label=f"remove:{old_key}",
            )
        except Exception as exc:
            logger.warning("worker_remove_voice(%s): %s", old_key, exc)
    profiles.pop(old_key, None)

    try:
        pids = await _broadcast_to_workers(
            executor, worker_add_voice, profile.name, spec, MODEL_ID,
            label=f"update:{profile.name}",
        )
    except Exception as exc:
        logger.exception("Hot-reload failed for %s", profile.name)
        raise HTTPException(500, f"Worker reload failed: {exc}") from exc

    profile = load_profile_by_name(new_slug)
    ep = profile.resolve_embedding_path()
    if ep.exists():
        try:
            profile.update_embedding_metadata(VoiceEmbedding.from_npz(ep))
            profile = VoiceProfile.from_json(new_json)
        except Exception as exc:
            logger.warning("[%s] embedding metadata: %s", new_slug, exc)

    profiles[profile.name] = profile

    if getattr(request.app.state, "default_voice", "") == old_key and renamed:
        request.app.state.default_voice = profile.name
        request.app.state.voice_profile = profile
        try:
            await _broadcast_to_workers(
                executor, worker_set_default_voice, profile.name,
                label=f"default:{profile.name}",
            )
        except Exception as exc:
            logger.warning("worker_set_default_voice: %s", exc)
    elif getattr(request.app.state, "default_voice", "") == profile.name:
        request.app.state.voice_profile = profile

    return {
        "ok":            True,
        "name":          profile.name,
        "renamed_from":  old_key if renamed else None,
        "json_path":     str(new_json),
        "worker_pids":   pids,
        "has_embedding": profile.has_cached_embedding(),
        "embedding_meta": profile.embedding_meta,
    }

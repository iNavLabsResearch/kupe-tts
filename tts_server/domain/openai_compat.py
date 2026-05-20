"""OpenAI Speech API compatibility — voice/model resolution helpers."""

from __future__ import annotations

from typing import Union

from ..config import MODEL_ID

# OpenAI TTS preset voice names (https://platform.openai.com/docs/guides/text-to-speech)
OPENAI_PRESET_VOICES: tuple[str, ...] = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)

# Models accepted by POST /v1/audio/speech (only one weights bundle is loaded).
ACCEPTED_SPEECH_MODELS: frozenset[str] = frozenset(
    {
        MODEL_ID,
        "omnivoice",
        "tts-1",
        "tts-1-hd",
        "gpt-4o-mini-tts",
        "gpt-4o-mini-tts-2025-12-15",
    }
)

SUPPORTED_RESPONSE_FORMATS: frozenset[str] = frozenset(
    {"mp3", "opus", "aac", "flac", "wav", "pcm"}
)


def extract_voice_name(voice: Union[str, object]) -> str:
    """Normalise ``voice`` field — plain string or ``{"id": "..."}`` object."""
    if isinstance(voice, str):
        return voice.strip()
    voice_id = getattr(voice, "id", None)
    if voice_id is not None:
        return str(voice_id).strip()
    if isinstance(voice, dict):
        raw = voice.get("id")
        if raw is not None:
            return str(raw).strip()
    raise ValueError("voice must be a string or an object with an 'id' field")


def resolve_openai_voice(
    voice: str,
    available_profiles: dict[str, object],
    default_voice: str,
) -> str:
    """Map an OpenAI ``voice`` value to a loaded OmniVoice profile name."""
    if not voice:
        return default_voice

    lowered = voice.lower()
    for name in available_profiles:
        if name.lower() == lowered:
            return name

    if lowered in OPENAI_PRESET_VOICES:
        profiles = sorted(available_profiles.keys())
        if not profiles:
            return default_voice
        idx = OPENAI_PRESET_VOICES.index(lowered)
        return profiles[idx % len(profiles)]

    raise ValueError(
        f"Voice '{voice}' is not loaded. "
        f"Use a profile name ({sorted(available_profiles.keys())}) "
        f"or an OpenAI preset ({list(OPENAI_PRESET_VOICES)})."
    )


def validate_speech_model(model: str) -> None:
    """Reject unknown model ids when strict OpenAI aliases are expected."""
    normalised = model.strip().lower()
    if normalised not in {m.lower() for m in ACCEPTED_SPEECH_MODELS}:
        raise ValueError(
            f"Model '{model}' is not supported. "
            f"Accepted: {sorted(ACCEPTED_SPEECH_MODELS)}"
        )

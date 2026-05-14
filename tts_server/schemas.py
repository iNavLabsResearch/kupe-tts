"""Pydantic models for HTTP request / response validation."""

from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, BaseModel, Field

from .config import EPOCHS_MAX, EPOCHS_MIN


class BatchTTSRequest(BaseModel):
    texts:            list[str]                 = Field(..., min_length=1)
    language:         Optional[str]             = Field(
        default=None,
        description=(
            "Language for the synthesised speech.  Accepts:\n"
            "  • 'auto' / 'none' / '' → let the model auto-detect (DEFAULT)\n"
            "  • ISO-639-3 code → 'en', 'hi', 'gu', 'pa', 'bn', 'ta', 'te', "
            "'mr', 'kn', 'ml', 'zh', 'ja', 'ko', 'ar', …\n"
            "  • Canonical English name → 'English', 'Hindi', 'Gujarati', "
            "'Panjabi', 'Chinese', 'Japanese', 'Korean', …\n"
            "If omitted, the server's OMNIVOICE_LANGUAGE default is used."
        ),
    )
    voice:            Optional[str]             = Field(
        default=None,
        description=(
            "Voice profile name to use (e.g. 'ajay', 'soham'). May be any "
            "profile loaded at startup or hot-added via POST /api/voices. "
            "If omitted, the server's default voice is used."
        ),
    )
    speed:            Optional[float]           = Field(
        default=None,
        ge=0.25,
        le=3.0,
        description=(
            "Speaking-speed multiplier.  1.0 = normal pace, <1.0 slower, "
            ">1.0 faster.  Allowed range: 0.25 to 3.0.  If omitted, the "
            "server's OMNIVOICE_DEFAULT_SPEED is used (None → model default)."
        ),
    )
    use_high_quality: bool = False
    epochs: Optional[int] = Field(
        default=None,
        ge=EPOCHS_MIN,
        le=EPOCHS_MAX,
        description=(
            "Diffusion / iterative decoding steps (OmniVoice ``num_step``). "
            "Higher values tend to improve quality at the cost of latency. "
            "Omitted → server defaults (see ``first_chunk_steps`` / "
            "``rest_chunk_steps`` on ``/health``).  JSON alias: "
            "``inference_steps``."
        ),
        validation_alias=AliasChoices("epochs", "inference_steps"),
    )
    digit_words_lang: Optional[str] = Field(
        default=None,
        description=(
            "Legacy alias for ``digit_pronunciation`` when value is one of the "
            "cardinal locales ``en``/``hi``/``gu``/``kn`` or another recognised "
            "code. Prefer ``digit_pronunciation`` for new clients."
        ),
    )
    digit_words_hint: Optional[str] = Field(
        default=None,
        description=(
            "When ``digit_pronunciation`` / ``digit_words_lang`` omitted, "
            "``hinglish`` / ``en`` / ``english_digits`` uses **English** digit "
            "words inside Indic / SEA scripts — typical modern Hinglish."
        ),
    )
    digit_pronunciation: Optional[str] = Field(
        default=None,
        description=(
            "How digits should be **spoken** (ISO code or alias), e.g. ``ta``, "
            "``bn``, ``hi``, ``en``, ``ar``. Overrides script auto-detect. "
            "See server ``digit_to_words.supported_digit_pronunciations()``."
        ),
        validation_alias=AliasChoices("digit_pronunciation", "digitPronunciation"),
    )


class BatchTTSItem(BaseModel):
    id:           int
    audio_base64: str
    audio_ms:     int
    sample_rate:  int


class BatchTTSResponse(BaseModel):
    results:               list[BatchTTSItem]
    total_gen_ms:          float
    batch_size:            int
    server_batches_formed: int
    language:              str
    voice:                 str
    speed:                 Optional[float] = None
    epochs:                int = Field(
        description="``num_step`` applied to every item in this response.",
    )

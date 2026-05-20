"""Centralised configuration — every tunable knob lives here.

Values come from environment variables with sensible defaults.
Import this module from anywhere in the package.

Quantization
────────────
``OMNIVOICE_WEIGHT_DTYPE`` selects how the model weights are loaded.
The value is normalised to one of these canonical codes:

  ``fp32``   : full-precision (largest, slowest, most accurate)
                aliases: ``float32``, ``f32``, ``32``
  ``fp16``   : default for CUDA — half-precision floating point
                aliases: ``float16``, ``f16``, ``half``, ``16``
  ``bf16``   : bfloat16 — same memory as fp16 with wider dynamic range
                aliases: ``bfloat16``, ``bf16``, ``brain16``
  ``int8``   : 8-bit weights via ``bitsandbytes`` (lower VRAM, slight quality loss)
                aliases: ``i8``, ``8bit``, ``8``
  ``int4``   : 4-bit NF4 weights via ``bitsandbytes`` (lowest VRAM, larger quality loss)
                aliases: ``i4``, ``4bit``, ``4``, ``nf4``

Lookup is case-insensitive.  CPU is always loaded as fp32 regardless of this
setting.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# HTTP bind / reverse proxy (nginx)
# ---------------------------------------------------------------------------
# Uvicorn listen address.  Default 127.0.0.1 — expose only via nginx on :80/:443.
# Set HOST=0.0.0.0 for direct LAN access without a reverse proxy.
BIND_HOST: str = os.getenv("HOST", os.getenv("OMNIVOICE_BIND_HOST", "127.0.0.1")).strip()
BIND_PORT: int = int(os.getenv("PORT", "8000"))

# Trust X-Forwarded-* from nginx (set OMNIVOICE_TRUST_PROXY=0 to disable).
TRUST_PROXY_HEADERS: bool = os.getenv("OMNIVOICE_TRUST_PROXY", "1") == "1"
# Comma-separated IPs/CIDRs nginx may connect from; ``*`` trusts any (typical on same host).
FORWARDED_ALLOW_IPS: str = os.getenv("OMNIVOICE_FORWARDED_ALLOW_IPS", "127.0.0.1,::1")

# ---------------------------------------------------------------------------
# Model / device
# ---------------------------------------------------------------------------
MODEL_ID:    str        = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE:      str | None = os.getenv("OMNIVOICE_DEVICE", None)  # None → auto

# Inference backend: "triton" (fast, default) or "standard" (original PyTorch).
# "triton" requires `pip install omnivoice-triton`.
_RAW_MODEL_TYPE = os.getenv("OMNIVOICE_MODEL_TYPE", "triton").strip().lower()
if _RAW_MODEL_TYPE not in ("triton", "standard"):
    raise ValueError(
        f"OMNIVOICE_MODEL_TYPE='{_RAW_MODEL_TYPE}' is invalid. "
        f"Accepted values: 'triton', 'standard'."
    )
MODEL_TYPE: str = _RAW_MODEL_TYPE

# Weight dtype / quantization — alias map → canonical code (see docstring).
_DTYPE_ALIASES: dict[str, str] = {
    # fp32
    "fp32": "fp32", "float32": "fp32", "f32": "fp32", "32": "fp32",
    # fp16
    "fp16": "fp16", "float16": "fp16", "f16": "fp16",
    "half": "fp16", "16": "fp16",
    # bf16
    "bf16": "bf16", "bfloat16": "bf16", "brain16": "bf16",
    # int8
    "int8": "int8", "i8": "int8", "8bit": "int8", "8": "int8",
    # int4
    "int4": "int4", "i4": "int4", "4bit": "int4", "4": "int4", "nf4": "int4",
}
_RAW_WEIGHT_DTYPE = os.getenv("OMNIVOICE_WEIGHT_DTYPE", "fp16").strip().lower()
if _RAW_WEIGHT_DTYPE not in _DTYPE_ALIASES:
    raise ValueError(
        f"OMNIVOICE_WEIGHT_DTYPE='{_RAW_WEIGHT_DTYPE}' is invalid. "
        f"Accepted values (case-insensitive): {sorted(_DTYPE_ALIASES)}"
    )
WEIGHT_DTYPE: str = _DTYPE_ALIASES[_RAW_WEIGHT_DTYPE]

# ---------------------------------------------------------------------------
# Voice profiles
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR    = PROJECT_ROOT / "voice_reference"

# Default voice profile name — used when a request doesn't specify one.
DEFAULT_VOICE: str = os.getenv("OMNIVOICE_DEFAULT_VOICE", "ajay")

# Comma-separated list of profiles to preload at server startup.
#   - "all" / "auto" / unset (default)  → auto-discover EVERY voice JSON in
#     ``voice_reference/`` and load them all.  Drop a new ``<name>_ref.json``
#     file and the next restart picks it up automatically.
#   - "ajay,soham,shanti"               → load only the named profiles.
#
# Setting the var explicitly is for power users who want to constrain which
# voices the workers preload; the default just loads everything in the folder.
def _parse_voice_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]

_RAW_VOICE_PROFILES: str = os.getenv("OMNIVOICE_VOICE_PROFILES", "all").strip()
VOICE_PROFILES_AUTO: bool = _RAW_VOICE_PROFILES.lower() in ("", "all", "auto", "*")
VOICE_PROFILES: list[str] = (
    [] if VOICE_PROFILES_AUTO else _parse_voice_list(_RAW_VOICE_PROFILES)
)
# When an explicit list is given, ensure the default voice is included.
if (not VOICE_PROFILES_AUTO) and DEFAULT_VOICE not in VOICE_PROFILES:
    VOICE_PROFILES.insert(0, DEFAULT_VOICE)

# Maximum reference clip duration (seconds).  Longer clips are trimmed before
# the audio tokenizer runs.
REF_PROMPT_MAX_SEC: float = float(os.getenv("REF_PROMPT_MAX_SEC", "12"))

# ---------------------------------------------------------------------------
# Default language
#   Accepts:
#     - "auto" / "none" / ""   → language-agnostic mode (server passes None
#                                to OmniVoice, which lets the model figure out
#                                the language from the script + reference voice)
#     - ISO-639-3 code         → "en", "hi", "gu", "pa", "bn", "ta", "te",
#                                "mr", "kn", "ml", "zh", "ja", "ko", "ar", …
#     - English name           → "English", "Hindi", "Gujarati", "Panjabi",
#                                "Chinese", "Japanese", "Korean", …
#
#   See ``omnivoice.utils.lang_map.LANG_NAME_TO_ID`` (~700 supported).
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE: str = os.getenv("OMNIVOICE_LANGUAGE", "auto").strip()

# ---------------------------------------------------------------------------
# Speaking-speed parameter
#   - 1.0  → normal pace (model default)
#   - <1.0 → slower (e.g. 0.8 = 20% slower)
#   - >1.0 → faster (e.g. 1.5 = 50% faster)
# Hard range: 0.25 ≤ speed ≤ 3.0 (matches vLLM-Omni docs).  Set to 0 / "" /
# "default" to pass ``None`` through and let OmniVoice estimate naturally.
# ---------------------------------------------------------------------------
SPEED_MIN: float = 0.25
SPEED_MAX: float = 3.0


def _parse_speed_env(raw: str) -> float | None:
    raw = raw.strip().lower()
    if raw in ("", "0", "default", "none", "auto"):
        return None
    try:
        v = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"OMNIVOICE_DEFAULT_SPEED='{raw}' is not a number."
        ) from exc
    if not (SPEED_MIN <= v <= SPEED_MAX):
        raise ValueError(
            f"OMNIVOICE_DEFAULT_SPEED={v} is out of range "
            f"[{SPEED_MIN}, {SPEED_MAX}]."
        )
    return v


DEFAULT_SPEED: float | None = _parse_speed_env(
    os.getenv("OMNIVOICE_DEFAULT_SPEED", "default")
)

# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------
CHUNK_CHARS:       int = int(os.getenv("CHUNK_CHARS", "60"))
FIRST_CHUNK_CHARS: int = int(os.getenv("OMNIVOICE_FIRST_CHUNK_CHARS", "25"))

# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------
MAX_WORKERS:      int   = max(1, int(os.getenv("OMNIVOICE_MAX_WORKERS",           "1")))
MAX_BATCH_SIZE:   int   = max(1, int(os.getenv("OMNIVOICE_MAX_BATCH_SIZE",        "8")))
BATCH_TIMEOUT_MS: float = max(0.0, float(os.getenv("OMNIVOICE_BATCH_TIMEOUT_MS", "50")))
MAX_CONCURRENT:   int   = max(1, int(os.getenv("OMNIVOICE_MAX_CONCURRENT",        "16")))

# First-chunk priority batching — collection window for batching concurrent
# first-chunk requests together.  Short enough to not hurt single-stream FCL,
# long enough to catch burst arrivals from concurrent WebSocket connections.
FC_BATCH_TIMEOUT_MS: float = max(0.0, float(
    os.getenv("OMNIVOICE_FC_BATCH_TIMEOUT_MS", "30")
))

# Maximum items in a single REST-chunk GPU dispatch.  Keeping this at 1 means
# the scheduler checks for priority FC items between EVERY rest-chunk call,
# minimising worst-case FC wait.  Increase for better throughput at the cost
# of higher maximum FCL under concurrent load.
MAX_REST_BATCH: int = max(1, int(os.getenv("OMNIVOICE_MAX_REST_BATCH", "1")))

# ---------------------------------------------------------------------------
# Attention / performance
# ---------------------------------------------------------------------------
ATTN_IMPL:         str  = os.getenv("OMNIVOICE_ATTN_IMPL",   "sdpa")
USE_SAGE_ATTN:     bool = os.getenv("OMNIVOICE_SAGE_ATTN",   "1") == "1"
USE_TF32:          bool = os.getenv("OMNIVOICE_TF32",        "1") == "1"
USE_CUDNN_BENCH:   bool = os.getenv("OMNIVOICE_CUDNN_BENCH", "1") == "1"
USE_TORCH_COMPILE: bool = os.getenv("OMNIVOICE_COMPILE",     "1") == "1"
SORT_BATCH:        bool = os.getenv("OMNIVOICE_SORT_BATCH",  "1") == "1"
TORCH_COMPILE_MODE: str = os.getenv("OMNIVOICE_COMPILE_MODE", "reduce-overhead")

# Synchronise CUDA before timing measurements.  Adds ~0.5-1 ms of stall but
# gives precise GPU timing.  Disable in production for lower latency.
SYNC_TIMING: bool = os.getenv("OMNIVOICE_SYNC_TIMING", "0") == "1"

# Executor backend: "thread" (single-GPU, shared memory, lower overhead) or
# "process" (multi-GPU safe, uses mp.spawn).  Default "thread" for single-GPU.
_RAW_EXECUTOR_TYPE = os.getenv("OMNIVOICE_EXECUTOR", "thread").strip().lower()
if _RAW_EXECUTOR_TYPE not in ("thread", "process"):
    raise ValueError(
        f"OMNIVOICE_EXECUTOR='{_RAW_EXECUTOR_TYPE}' is invalid. "
        f"Accepted values: 'thread', 'process'."
    )
EXECUTOR_TYPE: str = _RAW_EXECUTOR_TYPE

# ---------------------------------------------------------------------------
# Streaming / crossfade
# ---------------------------------------------------------------------------
CROSSFADE_MS: int = max(0, int(os.getenv("OMNIVOICE_CROSSFADE_MS", "80")))

# ---------------------------------------------------------------------------
# First-chunk latency optimisation
# ---------------------------------------------------------------------------
FIRST_CHUNK_STEPS:    int   = max(1, int(os.getenv("OMNIVOICE_FIRST_CHUNK_STEPS", "4")))
FIRST_CHUNK_GUIDANCE: float = float(os.getenv("OMNIVOICE_FIRST_CHUNK_GUIDANCE", "1.0"))

# Rest-chunk (mid + last) diffusion steps.  Lower → faster GPU calls → lower
# max FC latency when the GPU is busy with a rest-chunk.  Default 16 matches
# a solid quality / speed trade-off; tune down (e.g. 8) for busier GPUs.
REST_CHUNK_STEPS: int = max(1, int(os.getenv("OMNIVOICE_REST_CHUNK_STEPS", "16")))

# Adaptive early exit: skip remaining diffusion steps when the fraction of
# still-masked tokens drops below this value for ALL items in the batch.
# 0.0 = disabled (always run all steps).  0.02 = exit when ≤2% tokens remain
# masked, typically saving 1-3 forward passes with minimal quality impact.
EARLY_EXIT_THRESHOLD: float = max(0.0, float(
    os.getenv("OMNIVOICE_EARLY_EXIT", "0.0")
))

# ---------------------------------------------------------------------------
# Generation config dicts (plain dicts for safe pickling across processes)
# ---------------------------------------------------------------------------
FIRST_CHUNK_CFG: dict = dict(
    num_step=FIRST_CHUNK_STEPS,
    guidance_scale=FIRST_CHUNK_GUIDANCE,
    t_shift=0.1,
    denoise=True,
    postprocess_output=False,
    layer_penalty_factor=5.0,
    position_temperature=5.0,
    class_temperature=0.0,
    early_exit_threshold=EARLY_EXIT_THRESHOLD,
)

MID_CHUNK_CFG: dict = dict(
    num_step=REST_CHUNK_STEPS,
    guidance_scale=1.5,
    t_shift=0.1,
    denoise=True,
    postprocess_output=False,
    layer_penalty_factor=5.0,
    position_temperature=5.0,
    class_temperature=0.0,
    early_exit_threshold=EARLY_EXIT_THRESHOLD,
)

LAST_CHUNK_CFG: dict = dict(
    num_step=REST_CHUNK_STEPS,
    guidance_scale=2.0,
    t_shift=0.1,
    denoise=True,
    postprocess_output=True,
    layer_penalty_factor=5.0,
    position_temperature=5.0,
    class_temperature=0.0,
    early_exit_threshold=EARLY_EXIT_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Client-overridable diffusion depth (``num_step`` in generation config)
# ---------------------------------------------------------------------------
EPOCHS_MIN: int = 1
EPOCHS_MAX: int = 128


def cfg_with_epochs(base: dict, epochs: int | None) -> dict:
    """Shallow copy of ``base`` with optional ``num_step`` (clamped).

    API / WebSocket field name is ``epochs``; it maps to OmniVoice
    ``num_step`` (iterative decoding / diffusion depth).
    """
    cfg = dict(base)
    if epochs is not None:
        cfg["num_step"] = max(EPOCHS_MIN, min(EPOCHS_MAX, int(epochs)))
    return cfg

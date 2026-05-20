from __future__ import annotations

from typing import Optional

from ..config import EPOCHS_MAX, EPOCHS_MIN, SPEED_MAX, SPEED_MIN


def coerce_text(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (list, tuple)):
        return " ".join(str(part).strip() for part in raw if part is not None).strip()
    return str(raw).strip()


def coerce_opt_str(raw) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def coerce_speed(raw) -> tuple[Optional[float], Optional[str]]:
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
    if raw is None:
        return False
    if isinstance(raw, str) and raw.strip().lower() in ("", "default", "none", "auto"):
        return False
    return True


def resolve_fc_rest_epochs(msg: dict) -> tuple[Optional[int], Optional[int], Optional[str]]:
    raw_fc = msg.get("epochs_fc") or msg.get("first_chunk_epochs") or msg.get("firstChunkEpochs")
    raw_rest = (
        msg.get("epochs_rest")
        or msg.get("rest_chunk_epochs")
        or msg.get("restChunkEpochs")
        or msg.get("mid_chunk_epochs")
    )
    raw_legacy = msg.get("epochs")
    if raw_legacy is None:
        raw_legacy = msg.get("inference_steps", msg.get("inferenceSteps"))

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

    leg_opt = None
    if _epochs_field_provided(raw_legacy):
        leg_opt, err = _coerce_epochs(raw_legacy)
        if err:
            return None, None, err

    if fc_opt is None:
        fc_opt = leg_opt
    if rest_opt is None:
        rest_opt = leg_opt
    return fc_opt, rest_opt, None


from __future__ import annotations

import logging
from typing import Any

import torch

from omnivoice.models.omnivoice import OmniVoice

log = logging.getLogger("omnivoice.worker")


def build_load_kwargs(device: str, weight_dtype: str, attn_impl: str) -> dict:
    load_kw: dict[str, Any] = {"device_map": device}
    if device == "cpu":
        load_kw["dtype"] = torch.float32
        if weight_dtype != "fp32":
            log.warning("device=cpu does not support %s; falling back to fp32.", weight_dtype)
    elif weight_dtype == "fp32":
        load_kw["dtype"] = torch.float32
    elif weight_dtype == "fp16":
        load_kw["dtype"] = torch.float16
    elif weight_dtype == "bf16":
        load_kw["dtype"] = torch.bfloat16
    elif weight_dtype in ("int8", "int4"):
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                f"OMNIVOICE_WEIGHT_DTYPE={weight_dtype!r} requires `transformers` and `bitsandbytes`."
            ) from exc
        if weight_dtype == "int8":
            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
        else:
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
            )
        load_kw["dtype"] = torch.float16
    else:
        load_kw["dtype"] = torch.float16
    if attn_impl and attn_impl not in ("auto", ""):
        load_kw["attn_implementation"] = attn_impl
    return load_kw


def load_model(model_id: str, load_kw: dict) -> OmniVoice:
    try:
        return OmniVoice.from_pretrained(model_id, **load_kw)
    except TypeError:
        load_kw.pop("attn_implementation", None)
        log.warning("attn_implementation kwarg not accepted; retrying without it.")
        return OmniVoice.from_pretrained(model_id, **load_kw)


def attach_triton_hybrid(runner: Any, model: OmniVoice, *, enable_sage: bool) -> None:
    from omnivoice_triton.models.faster_runner import _CUDAGraphForward
    from omnivoice_triton.models.patching import apply_sage_attention, apply_triton_kernels, find_patchable_model

    runner._model = model
    patch_range = getattr(runner, "patch_range", (0, 24))
    patchable = find_patchable_model(model)
    apply_triton_kernels(patchable, patch_range=patch_range)
    if enable_sage:
        apply_sage_attention(patchable, patch_range=patch_range)
    graph_forward = _CUDAGraphForward(model)
    model.forward = graph_forward  # type: ignore[method-assign]
    runner._graph_forward = graph_forward
    log.info("Triton hybrid runner attached to loaded OmniVoice model.")


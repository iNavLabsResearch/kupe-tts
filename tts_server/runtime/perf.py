from __future__ import annotations

import logging

import torch

from ..config import USE_CUDNN_BENCH, USE_TF32

log = logging.getLogger("omnivoice.worker")


def patch_sage_attention(enable: bool) -> None:
    if not enable:
        log.info("SageAttention disabled.")
        return
    try:
        from sageattention import sageattn  # type: ignore[import]
    except ImportError:
        log.warning("SageAttention requested but not installed — using SDPA.")
        return

    import torch.nn.functional as F

    _orig_sdpa = F.scaled_dot_product_attention

    def _smart_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if attn_mask is None and dropout_p == 0.0 and query.dim() == 4 and query.dtype in (torch.float16, torch.bfloat16) and query.is_cuda:
            try:
                return sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)
            except Exception:
                pass
        return _orig_sdpa(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa
        )

    F.scaled_dot_product_attention = _smart_sdpa  # type: ignore[assignment]
    torch.nn.functional.scaled_dot_product_attention = _smart_sdpa
    log.info("SageAttention: smart SDPA wrapper installed.")


def apply_torch_perf_flags() -> None:
    if not torch.cuda.is_available():
        return
    if USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if USE_CUDNN_BENCH:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


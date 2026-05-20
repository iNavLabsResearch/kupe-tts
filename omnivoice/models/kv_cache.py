"""Prefix KV cache helpers for OmniVoice iterative diffusion.

During masked diffusion only the *target* audio tokens change each step; the
conditional prefix (style + text + reference audio) is fixed.  Logits are read
only from target positions, so we can:

1. Run the LLM once on the prefix with ``use_cache=True`` and store
   ``past_key_values``.
2. Each diffusion step re-embed the suffix and run the LLM with that past KV
   (suffix queries attend to prefix + suffix keys).

The unconditional CFG branch has no fixed prefix in the packed layout, so it
still uses a full forward each step.

Requires a Hugging Face causal decoder (``use_cache`` + ``past_key_values``).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch

logger = logging.getLogger(__name__)


def llm_supports_kv_cache(llm: torch.nn.Module) -> bool:
    """Return True if the backbone looks like a HF decoder with KV cache."""
    cfg = getattr(llm, "config", None)
    if cfg is None:
        return False
    if getattr(cfg, "is_decoder", False):
        return True
    # Qwen/Llama-style configs
    model_type = getattr(cfg, "model_type", "") or ""
    return model_type in {
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "gemma",
        "gemma2",
        "phi",
        "phi3",
        "gpt2",
    }


def _bidirectional_sdpa_mask(
    bsz: int, q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Additive SDPA mask (zeros) = full bidirectional attention, matching diffusion."""
    return torch.zeros(bsz, 1, q_len, kv_len, device=device, dtype=dtype)


def llm_forward_prefix_cache(
    llm: torch.nn.Module,
    prefix_embeds: torch.Tensor,
) -> Any:
    """One forward on the fixed prefix; returns ``past_key_values``."""
    bsz, prefix_len, _ = prefix_embeds.shape
    attn = _bidirectional_sdpa_mask(
        bsz, prefix_len, prefix_len, prefix_embeds.device, prefix_embeds.dtype,
    )
    kwargs: dict[str, Any] = dict(
        inputs_embeds=prefix_embeds,
        attention_mask=attn,
        use_cache=True,
        return_dict=True,
    )

    out = llm(**kwargs)
    past = getattr(out, "past_key_values", None)
    if past is None:
        raise RuntimeError("LLM did not return past_key_values with use_cache=True.")
    return past


def llm_forward_suffix_with_past(
    llm: torch.nn.Module,
    suffix_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    suffix_len: int,
    prefix_len: int,
) -> torch.Tensor:
    """Forward suffix embeddings attending to cached prefix + current suffix."""
    device = suffix_embeds.device
    dtype = suffix_embeds.dtype
    bsz = suffix_embeds.size(0)
    total_kv = prefix_len + suffix_len
    attn = _bidirectional_sdpa_mask(bsz, suffix_len, total_kv, device, dtype)

    cache_pos = torch.arange(
        prefix_len, prefix_len + suffix_len, device=device, dtype=torch.long,
    )
    position_ids = cache_pos.unsqueeze(0).expand(bsz, -1)

    kwargs: dict[str, Any] = dict(
        inputs_embeds=suffix_embeds,
        past_key_values=past_key_values,
        attention_mask=attn,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )

    out = llm(**kwargs)
    return out[0]

"""Prefix KV cache helpers for OmniVoice iterative diffusion.

During masked diffusion only the *target* audio tokens change each step; the
conditional prefix (style + text + reference audio) is fixed.  Logits are read
only from target positions, so we can:

1. Run the LLM once on the prefix with ``use_cache=True`` and store
   ``past_key_values``.
2. Each diffusion step clone that prefix cache (HF mutates cache in-place even
   when ``use_cache=False``), re-embed the suffix, and forward.

The unconditional CFG branch has no fixed prefix in the packed layout, so it
still uses a full forward each step.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def llm_supports_kv_cache(llm: torch.nn.Module) -> bool:
    """Return True if the backbone looks like a HF decoder with KV cache."""
    cfg = getattr(llm, "config", None)
    if cfg is None:
        return False
    if getattr(cfg, "is_decoder", False):
        return True
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


def cache_seq_length(past_key_values: Any) -> int:
    """Return the number of tokens stored in *past_key_values*."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        layer0 = past_key_values[0]
        if isinstance(layer0, tuple) and len(layer0) > 0:
            return int(layer0[0].shape[-2])
    raise TypeError(
        f"Unsupported past_key_values type: {type(past_key_values)!r}"
    )


def clone_past_key_values(past_key_values: Any) -> Any:
    """Deep-copy prefix cache so suffix forwards cannot append to the template.

    Hugging Face may call ``Cache.update()`` even when ``use_cache=False``,
    which would grow the prefix cache every diffusion step and break attention
    mask shapes on the next step.
    """
    if past_key_values is None:
        return None

    # HF Cache API (DynamicCache, StaticCache, …)
    if hasattr(past_key_values, "copy"):
        try:
            return past_key_values.copy()
        except Exception:
            pass

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if (
        key_cache is not None
        and value_cache is not None
        and len(key_cache) == len(value_cache)
    ):
        try:
            from transformers.cache_utils import DynamicCache

            cfg = getattr(past_key_values, "_config", None)
            new_cache = DynamicCache(config=cfg) if cfg is not None else DynamicCache()
            for layer_idx, (k, v) in enumerate(zip(key_cache, value_cache)):
                if k is None or v is None:
                    continue
                new_cache.update(k.clone(), v.clone(), layer_idx)
            return new_cache
        except Exception as exc:
            logger.debug("KV cache: DynamicCache clone failed (%s), using deepcopy.", exc)

    # Legacy tuple: ((k, v), ...) per layer
    if isinstance(past_key_values, tuple):
        cloned_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple) and len(layer) == 2:
                k, v = layer
                cloned_layers.append((k.clone(), v.clone()))
            else:
                cloned_layers.append(layer)
        return tuple(cloned_layers)

    return copy.deepcopy(past_key_values)


def llm_forward_prefix_cache(
    llm: torch.nn.Module,
    prefix_embeds: torch.Tensor,
) -> Any:
    """One forward on the fixed prefix; returns ``past_key_values``."""
    bsz, prefix_len, _ = prefix_embeds.shape
    device = prefix_embeds.device
    # 2D mask: HF expands to the correct 4D SDPA shape internally.
    attn_2d = torch.ones(bsz, prefix_len, device=device, dtype=torch.long)

    out = llm(
        inputs_embeds=prefix_embeds,
        attention_mask=attn_2d,
        use_cache=True,
        return_dict=True,
    )
    past = getattr(out, "past_key_values", None)
    if past is None:
        raise RuntimeError("LLM did not return past_key_values with use_cache=True.")
    return past


def llm_forward_suffix_with_past(
    llm: torch.nn.Module,
    suffix_embeds: torch.Tensor,
    prefix_past_template: Any,
) -> torch.Tensor:
    """Forward suffix embeddings attending to a **clone** of prefix K/V."""
    device = suffix_embeds.device
    bsz, suffix_len, _ = suffix_embeds.shape

    past = clone_past_key_values(prefix_past_template)
    past_len = cache_seq_length(past)
    total_len = past_len + suffix_len

    attn_2d = torch.ones(bsz, total_len, device=device, dtype=torch.long)
    cache_pos = torch.arange(
        past_len, past_len + suffix_len, device=device, dtype=torch.long,
    )
    position_ids = cache_pos.unsqueeze(0).expand(bsz, -1)

    out = llm(
        inputs_embeds=suffix_embeds,
        past_key_values=past,
        attention_mask=attn_2d,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    return out[0]

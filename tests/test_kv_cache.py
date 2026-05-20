from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from omnivoice.models.kv_cache import (
    cache_seq_length,
    clone_past_key_values,
    llm_forward_prefix_cache,
    llm_forward_suffix_with_past,
    llm_supports_kv_cache,
)


class _FakeCfg:
    is_decoder = True
    model_type = "qwen2"


class _FakePast:
    def __init__(self, length: int, hidden: int = 8) -> None:
        self.length = length
        self.k = torch.zeros(1, 1, length, hidden)

    def get_seq_length(self) -> int:
        return self.length

    def copy(self) -> "_FakePast":
        other = _FakePast(self.length, self.k.shape[-1])
        other.k = self.k.clone()
        return other


class _FakeLLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeCfg()
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        bsz, seq, hidden = kwargs["inputs_embeds"].shape
        hidden_out = torch.zeros(bsz, seq, hidden, device=kwargs["inputs_embeds"].device)
        past_in = kwargs.get("past_key_values")
        if past_in is not None:
            past_len = past_in.get_seq_length() if hasattr(past_in, "get_seq_length") else 0
            past = _FakePast(past_len + seq, hidden)
        else:
            past = _FakePast(seq, hidden)
        return SimpleNamespace(
            last_hidden_state=hidden_out,
            past_key_values=past,
        )


class KVCacheHelperTest(unittest.TestCase):
    def test_llm_supports_kv_cache(self) -> None:
        self.assertTrue(llm_supports_kv_cache(_FakeLLM()))

    def test_clone_preserves_prefix_length(self) -> None:
        past = _FakePast(10)
        clone = clone_past_key_values(past)
        self.assertEqual(cache_seq_length(clone), 10)
        self.assertIsNot(clone, past)

    def test_prefix_then_suffix_calls(self) -> None:
        llm = _FakeLLM()
        prefix = torch.randn(1, 4, 8)
        past = llm_forward_prefix_cache(llm, prefix)
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(llm.calls[0]["use_cache"])
        self.assertEqual(cache_seq_length(past), 4)

        suffix = torch.randn(1, 3, 8)
        out = llm_forward_suffix_with_past(llm, suffix, past)
        self.assertEqual(out.shape, (1, 3, 8))
        self.assertEqual(len(llm.calls), 2)
        self.assertFalse(llm.calls[1]["use_cache"])
        self.assertIsNot(llm.calls[1]["past_key_values"], past)
        self.assertEqual(cache_seq_length(past), 4)
        attn = llm.calls[1]["attention_mask"]
        self.assertEqual(attn.shape, (1, 7))

        # Second suffix step must not grow the template cache.
        llm_forward_suffix_with_past(llm, suffix, past)
        self.assertEqual(cache_seq_length(past), 4)


if __name__ == "__main__":
    unittest.main()

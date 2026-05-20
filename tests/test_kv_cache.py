from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from omnivoice.models.kv_cache import (
    llm_forward_prefix_cache,
    llm_forward_suffix_with_past,
    llm_supports_kv_cache,
)


class _FakeCfg:
    is_decoder = True
    model_type = "qwen2"


class _FakeLLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeCfg()
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        bsz, seq, hidden = kwargs["inputs_embeds"].shape
        hidden_out = torch.zeros(bsz, seq, hidden, device=kwargs["inputs_embeds"].device)
        past = ("past", seq)
        return SimpleNamespace(
            last_hidden_state=hidden_out,
            past_key_values=past,
        )


class KVCacheHelperTest(unittest.TestCase):
    def test_llm_supports_kv_cache(self) -> None:
        self.assertTrue(llm_supports_kv_cache(_FakeLLM()))

    def test_prefix_then_suffix_calls(self) -> None:
        llm = _FakeLLM()
        prefix = torch.randn(1, 4, 8)
        past = llm_forward_prefix_cache(llm, prefix)
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(llm.calls[0]["use_cache"])

        suffix = torch.randn(1, 3, 8)
        out = llm_forward_suffix_with_past(llm, suffix, past, suffix_len=3, prefix_len=4)
        self.assertEqual(out.shape, (1, 3, 8))
        self.assertEqual(len(llm.calls), 2)
        self.assertFalse(llm.calls[1]["use_cache"])
        self.assertIs(llm.calls[1]["past_key_values"], past)
        attn = llm.calls[1]["attention_mask"]
        self.assertEqual(attn.shape, (1, 1, 3, 7))


if __name__ == "__main__":
    unittest.main()

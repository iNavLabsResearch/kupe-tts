from __future__ import annotations

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tts_server.batcher import DynamicBatcher


class _FakeRuntime:
    def generate(self, texts, cfg, languages=None, voices=None, speeds=None, digit_words_langs=None, digit_words_hints=None, digit_pronunciations=None):
        return [b"WAV" + t.encode("utf-8") for t in texts], 1.0

    def generate_raw(self, texts, cfg, languages=None, voices=None, speeds=None, digit_words_langs=None, digit_words_hints=None, digit_pronunciations=None):
        return [np.zeros(8, dtype=np.float32) for _ in texts], 1.0


class BatcherRuntimeContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_submit_and_submit_raw(self) -> None:
        executor = ThreadPoolExecutor(max_workers=1)
        batcher = DynamicBatcher(executor=executor, max_batch=4, timeout_ms=1, runtime_executor=_FakeRuntime())
        batcher.start()
        try:
            wav = await batcher.submit("hello", {"num_step": 4})
            raw = await batcher.submit_raw("hello", {"num_step": 4})
            self.assertTrue(wav.startswith(b"WAV"))
            self.assertEqual(raw.dtype, np.float32)
        finally:
            batcher.stop()
            await asyncio.sleep(0)
            executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()


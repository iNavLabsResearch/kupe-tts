"""DynamicBatcher — unified priority scheduler for GPU synthesis requests.

Architecture
────────────
A single ``_scheduler`` loop drives ALL GPU submissions through an
``asyncio.PriorityQueue``:

  *First-chunk* requests (priority 0) always dequeue before *rest-chunk*
  requests (priority 1).  Each dispatch is **awaited** — the scheduler
  blocks until the GPU call returns, then immediately re-checks the queue.
  This guarantees that first-chunk items are dispatched as soon as the GPU
  becomes available, regardless of how many rest-chunk items are queued.

Maximum first-chunk latency
───────────────────────────
  ``remaining time of the currently-running GPU call + FC generation time``

To keep this low, rest-chunks are capped at ``MAX_REST_BATCH`` items per
dispatch (default 1) and use ``REST_CHUNK_STEPS`` diffusion steps (default
16, configurable), so each GPU call finishes predictably and the scheduler can
re-check for incoming FC items.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .config import FC_BATCH_TIMEOUT_MS, MAX_REST_BATCH, SORT_BATCH
from .worker import worker_generate, worker_generate_raw

logger = logging.getLogger("omnivoice.batcher")

_PRIO_FC = 0
_PRIO_REST = 1


@dataclass
class _SynthReq:
    text:     str
    cfg:      dict
    language: Optional[str]
    voice:    Optional[str]
    speed:    Optional[float]
    future:   asyncio.Future
    is_fc:    bool  = False
    is_raw:   bool  = False  # True → return np.ndarray instead of WAV bytes
    t_submit: float = field(default_factory=time.perf_counter)
    digit_words_lang:  Optional[str] = None
    digit_words_hint: Optional[str] = None
    digit_pronunciation: Optional[str] = None


@dataclass(order=True)
class _PrioItem:
    priority: int
    seq:      int
    req:      _SynthReq = field(compare=False)


class DynamicBatcher:
    """Priority-aware GPU scheduler for concurrent TTS streams."""

    def __init__(
        self,
        executor:   Executor,
        max_batch:  int,
        timeout_ms: float,
    ) -> None:
        self._executor       = executor
        self._max_batch      = max_batch
        self._timeout        = timeout_ms / 1000.0
        self._fc_timeout     = FC_BATCH_TIMEOUT_MS / 1000.0
        self._max_rest_batch = MAX_REST_BATCH

        self._pq: asyncio.PriorityQueue[_PrioItem] = asyncio.PriorityQueue()
        self._seq  = 0
        self._task: Optional[asyncio.Task] = None

        self.total_requests: int   = 0
        self.total_batches:  int   = 0
        self.total_gen_ms:   float = 0.0

    @property
    def avg_batch_size(self) -> float:
        return (self.total_requests / self.total_batches) if self.total_batches else 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._scheduler(), name="gpu-scheduler")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        text: str,
        cfg: dict,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        digit_words_lang: Optional[str] = None,
        digit_words_hint: Optional[str] = None,
        digit_pronunciation: Optional[str] = None,
    ) -> bytes:
        """Enqueue a rest-chunk request (normal priority)."""
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        self._seq += 1
        await self._pq.put(_PrioItem(
            _PRIO_REST, self._seq,
            _SynthReq(text=text, cfg=cfg, language=language, voice=voice,
                      speed=speed, future=fut, is_fc=False, is_raw=False,
                      digit_words_lang=digit_words_lang,
                      digit_words_hint=digit_words_hint,
                      digit_pronunciation=digit_pronunciation),
        ))
        return await fut

    async def submit_raw(
        self,
        text: str,
        cfg: dict,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        digit_words_lang: Optional[str] = None,
        digit_words_hint: Optional[str] = None,
        digit_pronunciation: Optional[str] = None,
    ) -> np.ndarray:
        """Enqueue a rest-chunk request returning a raw numpy array."""
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        self._seq += 1
        await self._pq.put(_PrioItem(
            _PRIO_REST, self._seq,
            _SynthReq(text=text, cfg=cfg, language=language, voice=voice,
                      speed=speed, future=fut, is_fc=False, is_raw=True,
                      digit_words_lang=digit_words_lang,
                      digit_words_hint=digit_words_hint,
                      digit_pronunciation=digit_pronunciation),
        ))
        return await fut

    async def submit_first_chunk(
        self,
        text: str,
        cfg: dict,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        digit_words_lang: Optional[str] = None,
        digit_words_hint: Optional[str] = None,
        digit_pronunciation: Optional[str] = None,
    ) -> tuple[bytes, float]:
        """Submit a first-chunk request with highest priority.

        Concurrent FC requests are collected within a short window
        (``FC_BATCH_TIMEOUT_MS``, default 30 ms) and dispatched as one
        GPU batch.

        Returns ``(wav_bytes, gen_ms_per_item)``.
        """
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        self._seq += 1
        await self._pq.put(_PrioItem(
            _PRIO_FC, self._seq,
            _SynthReq(text=text, cfg=cfg, language=language, voice=voice,
                      speed=speed, future=fut, is_fc=True, is_raw=False,
                      digit_words_lang=digit_words_lang,
                      digit_words_hint=digit_words_hint,
                      digit_pronunciation=digit_pronunciation),
        ))
        return await fut

    async def submit_first_chunk_raw(
        self,
        text: str,
        cfg: dict,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        digit_words_lang: Optional[str] = None,
        digit_words_hint: Optional[str] = None,
        digit_pronunciation: Optional[str] = None,
    ) -> tuple[np.ndarray, float]:
        """Submit a first-chunk request returning a raw numpy array.

        Returns ``(audio_array, gen_ms_per_item)``.
        """
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        self._seq += 1
        await self._pq.put(_PrioItem(
            _PRIO_FC, self._seq,
            _SynthReq(text=text, cfg=cfg, language=language, voice=voice,
                      speed=speed, future=fut, is_fc=True, is_raw=True,
                      digit_words_lang=digit_words_lang,
                      digit_words_hint=digit_words_hint,
                      digit_pronunciation=digit_pronunciation),
        ))
        return await fut

    async def submit_immediate(
        self,
        text: str,
        cfg: dict,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        digit_words_lang: Optional[str] = None,
        digit_words_hint: Optional[str] = None,
        digit_pronunciation: Optional[str] = None,
    ) -> tuple[bytes, float]:
        """Convenience wrapper — routes through the FC priority path."""
        return await self.submit_first_chunk(
            text, cfg, language=language, voice=voice, speed=speed,
            digit_words_lang=digit_words_lang,
            digit_words_hint=digit_words_hint,
            digit_pronunciation=digit_pronunciation,
        )

    # ------------------------------------------------------------------
    # Unified priority scheduler
    # ------------------------------------------------------------------

    async def _scheduler(self) -> None:
        """Single loop: dequeue by priority, dispatch, await, repeat.

        By awaiting each dispatch (never fire-and-forget), we guarantee
        that after every GPU call the queue is re-checked and FC items
        jump ahead of any queued REST items.
        """
        while True:
            first = await self._pq.get()

            if first.priority == _PRIO_FC:
                batch = await self._collect_fc(first)
                await self._dispatch_fc(batch)
            else:
                batch = await self._collect_rest(first)
                await self._dispatch_rest(batch)

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    async def _collect_fc(self, first: _PrioItem) -> list[_SynthReq]:
        """Collect FC items within a short window (default 30 ms)."""
        batch = [first.req]
        deadline = time.perf_counter() + self._fc_timeout

        while len(batch) < self._max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    self._pq.get(), timeout=max(0.001, remaining),
                )
                if item.priority == _PRIO_FC:
                    batch.append(item.req)
                else:
                    await self._pq.put(item)
                    break
            except asyncio.TimeoutError:
                break

        return batch

    async def _collect_rest(self, first: _PrioItem) -> list[_SynthReq]:
        """Collect REST items, stopping immediately if an FC item appears."""
        batch = [first.req]
        deadline = time.perf_counter() + self._timeout

        while len(batch) < self._max_rest_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    self._pq.get(), timeout=max(0.001, remaining),
                )
                if item.priority == _PRIO_FC:
                    await self._pq.put(item)
                    break
                batch.append(item.req)
            except asyncio.TimeoutError:
                break

        return batch

    # ------------------------------------------------------------------
    # GPU dispatch (no lock — single scheduler serialises all access)
    # ------------------------------------------------------------------

    async def _dispatch_fc(self, batch: list[_SynthReq]) -> None:
        texts     = [r.text for r in batch]
        languages = [r.language for r in batch]
        voices    = [r.voice for r in batch]
        speeds    = [r.speed for r in batch]
        d_langs   = [r.digit_words_lang for r in batch]
        d_hints   = [r.digit_words_hint for r in batch]
        d_pros    = [r.digit_pronunciation for r in batch]
        cfg       = batch[0].cfg
        use_raw   = all(r.is_raw for r in batch)

        logger.info(
            "FC dispatch  size=%d  chars=%d..%d  raw=%s",
            len(batch),
            min(len(t) for t in texts), max(len(t) for t in texts),
            use_raw,
        )

        loop = asyncio.get_running_loop()
        gen_func = worker_generate_raw if use_raw else worker_generate
        try:
            result_list, gen_ms = await loop.run_in_executor(
                self._executor, gen_func,
                texts, cfg, languages, voices, speeds, d_langs, d_hints, d_pros,
            )
            self.total_requests += len(batch)
            self.total_batches  += 1
            self.total_gen_ms   += gen_ms

            per_gen = gen_ms / len(batch)
            logger.info(
                "FC done  size=%d  gen=%.1fms  (%.1fms/item)",
                len(batch), gen_ms, per_gen,
            )
            for req, result in zip(batch, result_list):
                if not req.future.done():
                    req.future.set_result((result, per_gen))
        except Exception as exc:
            logger.exception("FC dispatch error: %s", exc)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)

    async def _dispatch_rest(self, batch: list[_SynthReq]) -> None:
        if SORT_BATCH and len(batch) > 1:
            order = sorted(range(len(batch)), key=lambda i: -len(batch[i].text))
            ordered = [batch[i] for i in order]
        else:
            ordered = batch

        texts     = [r.text for r in ordered]
        languages = [r.language for r in ordered]
        voices    = [r.voice for r in ordered]
        speeds    = [r.speed for r in ordered]
        d_langs   = [r.digit_words_lang for r in ordered]
        d_hints   = [r.digit_words_hint for r in ordered]
        d_pros    = [r.digit_pronunciation for r in ordered]
        cfg       = ordered[0].cfg
        use_raw   = all(r.is_raw for r in ordered)
        avg_q     = sum(
            (time.perf_counter() - r.t_submit) * 1000 for r in ordered
        ) / len(ordered)

        # ---- Length-bucketing metric (opt 2.3) ----
        # Track padding overhead: ratio of max-length to average-length.
        # A ratio of 1.0 means zero padding waste (ideal).
        char_lens = [len(t) for t in texts]
        max_chars = max(char_lens)
        avg_chars = sum(char_lens) / len(char_lens)
        padding_ratio = max_chars / avg_chars if avg_chars > 0 else 1.0

        logger.info(
            "REST dispatch  size=%d  avg_queue=%.1fms  chars=%d..%d  "
            "pad_ratio=%.2f  raw=%s",
            len(ordered), avg_q,
            min(char_lens), max_chars,
            padding_ratio, use_raw,
        )

        loop = asyncio.get_running_loop()
        gen_func = worker_generate_raw if use_raw else worker_generate
        try:
            result_list, gen_ms = await loop.run_in_executor(
                self._executor, gen_func,
                texts, cfg, languages, voices, speeds, d_langs, d_hints, d_pros,
            )
            self.total_requests += len(ordered)
            self.total_batches  += 1
            self.total_gen_ms   += gen_ms

            logger.info(
                "REST done  size=%d  gen=%.1fms  (%.1fms/text)",
                len(ordered), gen_ms, gen_ms / len(ordered),
            )
            for req, result in zip(ordered, result_list):
                if not req.future.done():
                    req.future.set_result(result)
        except Exception as exc:
            logger.exception("REST dispatch error: %s", exc)
            for req in ordered:
                if not req.future.done():
                    req.future.set_exception(exc)



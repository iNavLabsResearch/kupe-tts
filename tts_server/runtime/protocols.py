from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

class RuntimeExecutor(Protocol):
    def generate(
        self,
        texts: list[str],
        cfg: dict,
        languages: Optional[list[Optional[str]]] = None,
        voices: Optional[list[Optional[str]]] = None,
        speeds: Optional[list[Optional[float]]] = None,
        digit_words_langs: Optional[list[Optional[str]]] = None,
        digit_words_hints: Optional[list[Optional[str]]] = None,
        digit_pronunciations: Optional[list[Optional[str]]] = None,
    ) -> tuple[list[bytes], float]: ...

    def generate_raw(
        self,
        texts: list[str],
        cfg: dict,
        languages: Optional[list[Optional[str]]] = None,
        voices: Optional[list[Optional[str]]] = None,
        speeds: Optional[list[Optional[float]]] = None,
        digit_words_langs: Optional[list[Optional[str]]] = None,
        digit_words_hints: Optional[list[Optional[str]]] = None,
        digit_pronunciations: Optional[list[Optional[str]]] = None,
    ) -> tuple[list[np.ndarray], float]: ...


class WorkerRuntimeExecutor:
    """Adapter that delegates to current worker entrypoints."""

    def generate(
        self,
        texts: list[str],
        cfg: dict,
        languages: Optional[list[Optional[str]]] = None,
        voices: Optional[list[Optional[str]]] = None,
        speeds: Optional[list[Optional[float]]] = None,
        digit_words_langs: Optional[list[Optional[str]]] = None,
        digit_words_hints: Optional[list[Optional[str]]] = None,
        digit_pronunciations: Optional[list[Optional[str]]] = None,
    ) -> tuple[list[bytes], float]:
        from ..worker import worker_generate
        return worker_generate(
            texts, cfg, languages, voices, speeds, digit_words_langs, digit_words_hints, digit_pronunciations
        )

    def generate_raw(
        self,
        texts: list[str],
        cfg: dict,
        languages: Optional[list[Optional[str]]] = None,
        voices: Optional[list[Optional[str]]] = None,
        speeds: Optional[list[Optional[float]]] = None,
        digit_words_langs: Optional[list[Optional[str]]] = None,
        digit_words_hints: Optional[list[Optional[str]]] = None,
        digit_pronunciations: Optional[list[Optional[str]]] = None,
    ) -> tuple[list[np.ndarray], float]:
        from ..worker import worker_generate_raw
        return worker_generate_raw(
            texts, cfg, languages, voices, speeds, digit_words_langs, digit_words_hints, digit_pronunciations
        )


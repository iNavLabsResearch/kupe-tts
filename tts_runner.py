"""TTS Runner abstraction layer.

Provides a base class and derived implementations for switching between
different TTS model backends (omnivoice-triton optimized vs standard OmniVoice).

Usage::

    from tts_runner import create_runner

    # Use the Triton-optimised backend (default)
    runner = create_runner("triton")

    # Or the standard PyTorch OmniVoice model
    runner = create_runner("standard", model_name="k2-fsa/OmniVoice")

    result = runner.generate_voice_clone(
        text="Hello, world!",
        ref_audio="reference.wav",
        ref_text="Transcript of the reference audio.",
    )
    print(result.audio.shape, result.sample_rate)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standardised output
# ---------------------------------------------------------------------------


@dataclass
class TTSResult:
    """Standardised container for TTS generation output.

    Attributes:
        audio: 1-D numpy waveform of shape ``(T,)``.
        sample_rate: Sample rate of the audio (e.g. 24000).
    """

    audio: np.ndarray
    sample_rate: int


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseTTSRunner(ABC):
    """Abstract base class that every TTS backend must implement."""

    @abstractmethod
    def generate_voice_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        **kwargs: Any,
    ) -> TTSResult:
        """Generate speech by cloning the voice from a reference audio.

        Args:
            text: The text to synthesize.
            ref_audio: Path to the reference audio file.
            ref_text: Transcript of the reference audio.
            **kwargs: Backend-specific generation parameters.

        Returns:
            A :class:`TTSResult` with the generated waveform and sample rate.
        """
        ...

    @abstractmethod
    def generate(
        self,
        text: str,
        **kwargs: Any,
    ) -> TTSResult:
        """Generate speech without voice cloning (voice design / auto voice).

        Args:
            text: The text to synthesize.
            **kwargs: Backend-specific generation parameters
                      (e.g. ``instruct``, ``language``).

        Returns:
            A :class:`TTSResult` with the generated waveform and sample rate.
        """
        ...

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Return the output sample rate of this backend."""
        ...


# ---------------------------------------------------------------------------
# Triton-optimised runner
# ---------------------------------------------------------------------------


class TritonTTSRunner(BaseTTSRunner):
    """Wraps ``omnivoice_triton.create_runner`` for fast, Triton-optimised inference.

    Requires ``pip install omnivoice-triton``.
    """

    def __init__(self, mode: str = "hybrid") -> None:
        """
        Args:
            mode: Triton runner mode (e.g. ``"hybrid"``, ``"base"``).
        """
        from omnivoice_triton import create_runner as _create_triton_runner

        logger.info("Initialising TritonTTSRunner (mode=%s) ...", mode)
        self._runner = _create_triton_runner(mode)
        self._sample_rate = 24_000  # OmniVoice standard rate

    # -- public interface ---------------------------------------------------

    def generate_voice_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        **kwargs: Any,
    ) -> TTSResult:
        raw = self._runner.generate_voice_clone(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            **kwargs,
        )
        return self._normalise_output(raw)

    def generate(
        self,
        text: str,
        **kwargs: Any,
    ) -> TTSResult:
        raw = self._runner.generate(text=text, **kwargs)
        return self._normalise_output(raw)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    # -- internal helpers ---------------------------------------------------

    def _normalise_output(self, raw: Any) -> TTSResult:
        """Convert the Triton runner's raw output to a :class:`TTSResult`.

        Handles both dict ``{"audio": ..., "sample_rate": ...}``
        and tuple ``(audio, sample_rate)`` return conventions.
        """
        if isinstance(raw, dict):
            audio = np.asarray(raw["audio"]).squeeze()
            sr = int(raw.get("sample_rate", self._sample_rate))
        elif isinstance(raw, (tuple, list)):
            audio = np.asarray(raw[0]).squeeze()
            sr = int(raw[1]) if len(raw) > 1 else self._sample_rate
        else:
            # Assume raw is just the audio array
            audio = np.asarray(raw).squeeze()
            sr = self._sample_rate

        return TTSResult(audio=audio, sample_rate=sr)


# ---------------------------------------------------------------------------
# Standard OmniVoice (PyTorch) runner
# ---------------------------------------------------------------------------


class StandardTTSRunner(BaseTTSRunner):
    """Wraps the original ``OmniVoice.from_pretrained`` PyTorch model."""

    def __init__(
        self,
        model_name: str = "k2-fsa/OmniVoice",
        device: Optional[str] = None,
        dtype: Optional[str] = "float16",
    ) -> None:
        """
        Args:
            model_name: HuggingFace repo id or local checkpoint path.
            device: Device to load the model onto (``"cuda"``, ``"cpu"``,
                ``"mps"``). Auto-detected when ``None``.
            dtype: Model dtype (``"float16"`` or ``"float32"``).
        """
        import torch
        from omnivoice import OmniVoice

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        logger.info(
            "Initialising StandardTTSRunner (model=%s, device=%s, dtype=%s) ...",
            model_name,
            device,
            dtype,
        )
        self._model: OmniVoice = OmniVoice.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch_dtype,
        )

    # -- public interface ---------------------------------------------------

    def generate_voice_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        **kwargs: Any,
    ) -> TTSResult:
        audios = self._model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            **kwargs,
        )
        # model.generate() returns a list of 1-D np.ndarray
        return TTSResult(
            audio=audios[0],
            sample_rate=self._model.sampling_rate,
        )

    def generate(
        self,
        text: str,
        **kwargs: Any,
    ) -> TTSResult:
        audios = self._model.generate(text=text, **kwargs)
        return TTSResult(
            audio=audios[0],
            sample_rate=self._model.sampling_rate,
        )

    def get_sample_rate(self) -> int:
        return self._model.sampling_rate


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_RUNNER_REGISTRY = {
    "triton": TritonTTSRunner,
    "standard": StandardTTSRunner,
}


def create_runner(model_type: str = "triton", **kwargs: Any) -> BaseTTSRunner:
    """Factory function to create a TTS runner.

    Args:
        model_type: Backend to use.
            - ``"triton"``   — Triton-optimised (fast, default).
            - ``"standard"`` — Original OmniVoice PyTorch model.
        **kwargs: Passed to the chosen runner's ``__init__``.

    Returns:
        An instance of :class:`BaseTTSRunner`.

    Raises:
        ValueError: If ``model_type`` is not recognised.
    """
    model_type = model_type.lower()
    runner_cls = _RUNNER_REGISTRY.get(model_type)

    if runner_cls is None:
        valid = ", ".join(sorted(_RUNNER_REGISTRY))
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: {valid}"
        )

    return runner_cls(**kwargs)

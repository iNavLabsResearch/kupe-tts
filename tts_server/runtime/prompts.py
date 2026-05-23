from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt

from ..voice_profiles import VoiceEmbedding


def build_prompt_from_embedding(model: OmniVoice, embedding: VoiceEmbedding) -> VoiceClonePrompt:
    tokens = torch.as_tensor(embedding.ref_audio_tokens, dtype=torch.long, device=model.audio_tokenizer.device)
    return VoiceClonePrompt(ref_audio_tokens=tokens, ref_text=embedding.ref_text, ref_rms=float(embedding.ref_rms))


def build_prompt_from_audio(
    model: OmniVoice, ref_bytes: bytes, ref_sr: int, ref_text: Optional[str]
) -> tuple[VoiceClonePrompt, VoiceEmbedding]:
    ref_np = np.frombuffer(ref_bytes, dtype=np.float32).copy()
    ref_tensor = torch.from_numpy(ref_np).unsqueeze(0)
    prompt = model.create_voice_clone_prompt(ref_audio=(ref_tensor, ref_sr), ref_text=ref_text, preprocess_prompt=True)
    tokens_np = prompt.ref_audio_tokens.detach().cpu().numpy().astype(np.int64)
    embedding = VoiceEmbedding(
        ref_audio_tokens=tokens_np,
        ref_text=prompt.ref_text,
        ref_rms=float(prompt.ref_rms),
        sampling_rate=int(model.sampling_rate),
        model_id="",
        num_codebooks=int(tokens_np.shape[0]),
        num_tokens=int(tokens_np.shape[1]),
    )
    return prompt, embedding


def resolve_voice_prompt(model: OmniVoice, name: str, spec: dict, model_id: str):
    cached = spec.get("cached_embedding")
    if cached is not None:
        embedding = VoiceEmbedding(
            ref_audio_tokens=np.asarray(cached["ref_audio_tokens"], dtype=np.int64),
            ref_text=str(cached["ref_text"]),
            ref_rms=float(cached["ref_rms"]),
            sampling_rate=int(cached["sampling_rate"]),
            model_id=str(cached["model_id"]),
            num_codebooks=int(cached["num_codebooks"]),
            num_tokens=int(cached["num_tokens"]),
        )
        return build_prompt_from_embedding(model, embedding)

    raw_bytes = spec.get("raw_ref_bytes")
    raw_sr = spec.get("raw_ref_sr")
    raw_text = spec.get("raw_ref_text")
    if raw_bytes is None or raw_sr is None:
        raise RuntimeError(f"Voice '{name}' has neither a cached embedding nor raw reference audio.")

    prompt, embedding = build_prompt_from_audio(model, raw_bytes, int(raw_sr), raw_text)
    embedding.model_id = model_id
    cache_save_path = spec.get("cache_save_path")
    if cache_save_path:
        embedding.to_npz(Path(cache_save_path))
    return prompt


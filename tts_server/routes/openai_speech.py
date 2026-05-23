"""OpenAI-compatible Speech API — POST /v1/audio/speech, GET /v1/models."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ..config import MODEL_ID
from ..core.state import get_container
from ..domain.openai_compat import ACCEPTED_SPEECH_MODELS
from ..schemas import OpenAIModelList, OpenAIModelObject, OpenAISpeechRequest

router = APIRouter()


@router.post("/v1/audio/speech")
async def create_speech(req: OpenAISpeechRequest, request: Request) -> Response:
    """Generate speech from text — OpenAI Speech API compatible.

    Example (curl)::

        curl http://127.0.0.1:8000/v1/audio/speech \\
          -H "Content-Type: application/json" \\
          -d '{"model":"tts-1","input":"Hello world","voice":"ajay","response_format":"wav"}' \\
          --output speech.wav

    ``voice`` accepts OmniVoice profile names (``ajay``, ``soham``, …) or OpenAI
    preset names (``alloy``, ``nova``, …) mapped to loaded profiles.
    """
    container = get_container(request)
    try:
        audio_bytes, media_type = await container.synthesis_service.synth_speech(
            req, request.app.state
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    return Response(content=audio_bytes, media_type=media_type)


@router.get("/v1/models", response_model=OpenAIModelList)
async def list_models() -> OpenAIModelList:
    """List models exposed for OpenAI SDK compatibility."""
    created = int(time.time())
    data = [
        OpenAIModelObject(id=model_id, created=created, owned_by="omnivoice")
        for model_id in sorted(ACCEPTED_SPEECH_MODELS, key=str.lower)
    ]
    if MODEL_ID not in {item.id for item in data}:
        data.insert(0, OpenAIModelObject(id=MODEL_ID, created=created, owned_by="omnivoice"))
    return OpenAIModelList(data=data)

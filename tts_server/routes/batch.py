"""POST /api/tts/batch — synchronous batch generation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..core.state import get_container
from ..schemas import BatchTTSRequest, BatchTTSResponse

router = APIRouter()


@router.post("/api/tts/batch", response_model=BatchTTSResponse)
async def batch_tts(req: BatchTTSRequest, request: Request):
    """Generate audio for a list of texts using synthesis service."""
    container = get_container(request)
    try:
        return await container.synthesis_service.synth_batch(req, request.app.state)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

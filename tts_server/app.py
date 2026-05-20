"""FastAPI application factory — lifespan, CORS, router wiring.

This module now keeps the app factory small; the heavy startup lifecycle is
implemented in :mod:`tts_server.startup` and imported here.
"""

from __future__ import annotations

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from .core.container import AppContainer
from .core.settings import load_settings
from .config import (
    TRUST_PROXY_HEADERS,
    FORWARDED_ALLOW_IPS,
)
from .routes import batch_router, health_router, streaming_router, voices_router
from .services.synthesis import DefaultSynthesisService
from .services.voice import DefaultVoiceService
from .startup import lifespan


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
    )

    app = FastAPI(
        title="OmniVoice Production TTS Server",
        version="2.2.0",
        description=(
            "Modular OmniVoice TTS — ProcessPoolExecutor, DynamicBatcher, "
            "SageAttention, chunked-diffusion pipelined streaming, "
            "JSON-based voice profiles with cached numpy embeddings, "
            "INT4 / INT8 / BF16 / FP16 / FP32 weight quantization. "
            "Designed to run behind nginx (see deploy/nginx.conf)."
        ),
        lifespan=lifespan,
    )
    settings = load_settings()
    app.state.container = AppContainer(
        synthesis_service=DefaultSynthesisService(),
        streaming_service=None,
        voice_service=DefaultVoiceService(),
        settings=settings,
    )
    if TRUST_PROXY_HEADERS:
        raw_trusted = FORWARDED_ALLOW_IPS.strip()
        if raw_trusted == "*":
            trusted_hosts: str | list[str] = "*"
        else:
            trusted_hosts = [
                h.strip() for h in raw_trusted.split(",") if h.strip()
            ] or ["127.0.0.1"]
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=trusted_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(batch_router)
    app.include_router(streaming_router)
    app.include_router(voices_router)
    return app

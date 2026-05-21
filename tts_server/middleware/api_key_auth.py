"""API key authentication — OpenAI-style Bearer token from .env / environment."""

from __future__ import annotations

import logging
from urllib.parse import parse_qs

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("omnivoice.auth")

_BEARER_PREFIX = "bearer "


def _header_value(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", ()):
        if key.lower() == name:
            return value.decode("utf-8", errors="replace").strip()
    return None


def extract_api_key(scope: Scope) -> str | None:
    """Read API key from Authorization, x-api-key, or query (WebSocket)."""
    auth = _header_value(scope, b"authorization")
    if auth:
        lowered = auth.lower()
        if lowered.startswith(_BEARER_PREFIX):
            token = auth[len(_BEARER_PREFIX) :].strip()
            if token:
                return token
        elif auth:
            return auth

    api_key = _header_value(scope, b"x-api-key")
    if api_key:
        return api_key

    query = scope.get("query_string", b"").decode("utf-8", errors="replace")
    if query:
        params = parse_qs(query, keep_blank_values=False)
        for name in ("api_key", "api-key", "key"):
            values = params.get(name)
            if values and values[0].strip():
                return values[0].strip()
    return None


class APIKeyAuthMiddleware:
    """Require a valid API key when keys are configured (from ``.env`` / env)."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_keys: frozenset[str],
        public_paths: frozenset[str],
    ) -> None:
        self.app = app
        self.api_keys = api_keys
        self.public_paths = public_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        if not self.api_keys:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.public_paths:
            await self.app(scope, receive, send)
            return

        if scope["type"] == "http":
            method = scope.get("method", "GET").upper()
            if method == "OPTIONS":
                await self.app(scope, receive, send)
                return

        provided = extract_api_key(scope)
        if provided and provided in self.api_keys:
            await self.app(scope, receive, send)
            return

        logger.warning("Unauthorized %s %s", scope["type"], path)
        if scope["type"] == "http":
            response = JSONResponse(
                {
                    "error": {
                        "message": "Incorrect API key provided. Pass it via "
                        "Authorization: Bearer <key> or x-api-key header.",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
                status_code=401,
            )
            await response(scope, receive, send)
            return

        await send({"type": "websocket.close", "code": 1008, "reason": "invalid_api_key"})
        while True:
            message = await receive()
            if message["type"] == "websocket.disconnect":
                break

"""ASGI middleware for the TTS server."""

from .api_key_auth import APIKeyAuthMiddleware

__all__ = ["APIKeyAuthMiddleware"]

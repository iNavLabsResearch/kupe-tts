from __future__ import annotations

from typing import Any, Protocol


class SynthesisService(Protocol):
    async def synth_batch(self, request: Any, app_state: Any) -> Any: ...


class StreamingService(Protocol):
    async def handle_websocket(self, websocket) -> None: ...


class VoiceService(Protocol):
    def get_default_voice(self, app_state: Any) -> str: ...


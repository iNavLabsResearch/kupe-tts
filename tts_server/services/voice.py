from __future__ import annotations

from typing import Any


class DefaultVoiceService:
    def get_default_voice(self, app_state: Any) -> str:
        return getattr(app_state, "default_voice", "")


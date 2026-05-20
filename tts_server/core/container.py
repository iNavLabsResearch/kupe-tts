from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AppContainer:
    """Typed dependency container shared across routes/services."""

    synthesis_service: Any
    streaming_service: Any
    voice_service: Any
    settings: Any


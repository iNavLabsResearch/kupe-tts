"""Route sub-package — aggregate all endpoint routers."""

from .batch import router as batch_router
from .health import router as health_router
from .streaming import router as streaming_router
from .voices import router as voices_router

__all__ = ["batch_router", "health_router", "streaming_router", "voices_router"]

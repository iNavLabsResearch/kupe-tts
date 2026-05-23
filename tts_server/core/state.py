from __future__ import annotations

from fastapi import Request, WebSocket

from .container import AppContainer


def get_container(conn: Request | WebSocket) -> AppContainer:
    container = getattr(conn.app.state, "container", None)
    if container is None:
        raise RuntimeError("AppContainer is not attached to app state.")
    return container


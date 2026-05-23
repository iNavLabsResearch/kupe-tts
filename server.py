#!/usr/bin/env python3
"""OmniVoice Production TTS Server — entry point.

All logic lives in the ``tts_server`` package.  This file is the thin
``uvicorn`` launcher.  Run behind nginx (see ``deploy/nginx.conf``).

    python server.py                        # default (127.0.0.1:8000, nginx on :80)
    HOST=0.0.0.0 python server.py           # direct access without nginx
    OMNIVOICE_SAGE_ATTN=1 python server.py  # SageAttention (default ON)
    OMNIVOICE_COMPILE=1 python server.py    # + torch.compile
"""

from tts_server.app import create_app
from tts_server.config import (
    BIND_HOST,
    BIND_PORT,
    FORWARDED_ALLOW_IPS,
    TRUST_PROXY_HEADERS,
    WS_PING_INTERVAL,
    WS_PING_TIMEOUT,
)

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=BIND_HOST,
        port=BIND_PORT,
        log_level="info",
        ws_ping_interval=WS_PING_INTERVAL,
        ws_ping_timeout=WS_PING_TIMEOUT,
        proxy_headers=TRUST_PROXY_HEADERS,
        forwarded_allow_ips=FORWARDED_ALLOW_IPS,
    )

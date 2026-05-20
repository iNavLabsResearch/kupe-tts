#!/usr/bin/env python3
"""OmniVoice Production TTS Server — entry point.

All logic lives in the ``tts_server`` package.  This file is the thin
``uvicorn`` launcher.  Run behind nginx (see ``deploy/nginx.conf``).

    python server.py                        # default int8 weights, 127.0.0.1:8000
    HOST=0.0.0.0 python server.py           # direct access without nginx
    OMNIVOICE_WEIGHT_DTYPE=fp32 python server.py  # full precision (needs more VRAM)
    OMNIVOICE_SAGE_ATTN=1 python server.py  # SageAttention (default ON)
    OMNIVOICE_COMPILE=1 python server.py    # + torch.compile (fp16/fp32 only)
"""

from tts_server.app import create_app
from tts_server.config import (
    BIND_HOST,
    BIND_PORT,
    FORWARDED_ALLOW_IPS,
    TRUST_PROXY_HEADERS,
)

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=BIND_HOST,
        port=BIND_PORT,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=30,
        proxy_headers=TRUST_PROXY_HEADERS,
        forwarded_allow_ips=FORWARDED_ALLOW_IPS,
    )

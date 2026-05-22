#!/usr/bin/env python3
"""Verify one WebSocket stays open across multiple tts.request messages.

Usage (server must already be running):
    python scripts/test_ws_persistent.py
    python scripts/test_ws_persistent.py --url ws://localhost:8000/ws/tts --requests 3

Success: prints "PASS" and exits 0.
Failure: prints why and exits 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


async def run(url: str, api_key: str | None, requests: int, text: str) -> int:
    try:
        import websockets
    except ImportError:
        print("Install websockets: pip install websockets", file=sys.stderr)
        return 1

    if api_key and "api_key=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api_key={api_key}"

    connect_count = 0
    done_count = 0

    async with websockets.connect(url, ping_interval=20, ping_timeout=120) as ws:
        connect_count = 1
        for i in range(requests):
            await ws.send(json.dumps({"type": "tts.request", "text": f"{text} ({i + 1})"}))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=180)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                mt = msg.get("type", "")
                if mt == "response.audio.done":
                    done_count += 1
                    break
                if mt == "error":
                    print(f"FAIL: server error on request {i + 1}: {msg.get('message')}")
                    return 1

    if connect_count != 1:
        print(f"FAIL: expected 1 connect, got {connect_count}")
        return 1
    if done_count != requests:
        print(f"FAIL: expected {requests} done messages, got {done_count}")
        return 1

    print(
        f"PASS: {requests} tts.request(s) on a single WebSocket "
        f"(1 connect, {done_count} response.audio.done)"
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Test persistent /ws/tts connections")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/tts")
    p.add_argument("--requests", type=int, default=3)
    p.add_argument("--text", default="Hello.")
    p.add_argument("--api-key", default=os.getenv("OMNIVOICE_API_KEY", ""))
    args = p.parse_args()
    code = asyncio.run(
        run(
            args.url,
            args.api_key.strip() or None,
            max(1, args.requests),
            args.text,
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

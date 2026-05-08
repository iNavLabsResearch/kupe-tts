import os
import sys
import json
from typing import Optional
from pyngrok import ngrok, conf

# Define your ngrok auth token here
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2xUQVsPC9sYDraw3qBEzVhevWRL_7noTzjPVqGYXDuTAJJAou")  # Replace with your actual token or set ENV
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2udz3fP5K4xTUfeU5cVk6rwVKyL_67Zo7tAbUBvCRYjKYtSVd")  # Replace with your actual token or set ENV

class NgrokTunnel:
    def __init__(self, port: int, proto: str = "http"):
        """
        Opens an ngrok tunnel to the given port using the provided NGROK_AUTH_TOKEN.
        """
        if not NGROK_AUTH_TOKEN or not isinstance(NGROK_AUTH_TOKEN, str):
            raise EnvironmentError("Please set NGROK_AUTH_TOKEN as an environment variable or in the code")

        conf.get_default().auth_token = NGROK_AUTH_TOKEN

        self.port = port
        self.proto = proto
        self.tunnel_url: Optional[str] = None

    def start(self) -> str:
        tunnel = ngrok.connect(addr=str(self.port), proto=self.proto)
        public_url = tunnel.public_url
        if not public_url:
            raise RuntimeError("Failed to obtain ngrok public URL")
        self.tunnel_url = public_url
        return public_url

    def wait(self) -> None:
        proc = ngrok.get_ngrok_process().proc
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("Interrupted, shutting down ngrok tunnel...")
            self.stop()

    def stop(self) -> None:
        if self.tunnel_url:
            ngrok.disconnect(self.tunnel_url)
        ngrok.kill()


def load_server_port(config_path: Optional[str] = None) -> int:
    if config_path and os.path.isfile(config_path):
        cfg_file = config_path
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cfg_file = os.path.join(project_root, "config.json")

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    port = cfg.get("server", {}).get("port")
    if not isinstance(port, int):
        raise ValueError("Please set 'server.port' (an integer) in config.json")
    return port


if __name__ == "__main__":
    # Usage: python expose_ngrok.py [<port>] [<config_path>]
    port_arg = None
    cfg_arg = None

    if len(sys.argv) >= 2:
        try:
            port_arg = int(sys.argv[1])
        except ValueError:
            cfg_arg = sys.argv[1]

    if len(sys.argv) >= 3:
        cfg_arg = sys.argv[2]

    if port_arg is None:
        port_arg = load_server_port(config_path=cfg_arg)

    tunnel = NgrokTunnel(port=port_arg)
    public_url = tunnel.start()
    print(f"Ngrok tunnel URL: {public_url}")
    tunnel.wait()
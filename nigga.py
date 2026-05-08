import os
from pyngrok import ngrok, conf

PORT = 8000  # Hardcoded port

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2tnJYhVLSHKh9IJlzyTFwAjZlWa_XyFKL7xHN9eAsFwaHefT")
if not NGROK_AUTH_TOKEN:
    raise EnvironmentError("Set NGROK_AUTH_TOKEN environment variable")

conf.get_default().auth_token = NGROK_AUTH_TOKEN
tunnel = ngrok.connect(addr=str(PORT), proto="http")
print(f"Ngrok tunnel URL: {tunnel.public_url}")
ngrok.get_ngrok_process().proc.wait()

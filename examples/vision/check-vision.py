#!/usr/bin/env python3
"""Send a native image through the API; needs only Python's standard library."""
import argparse
import base64
import json
import mimetypes
from pathlib import Path
import time
import urllib.request

fixture = Path(__file__).with_name("smoke.png")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8080", help="Server URL without /v1")
parser.add_argument("--model", default="L0xRE-27b-Low")
parser.add_argument("--image", type=Path, default=fixture)
args = parser.parse_args()
mime = mimetypes.guess_type(str(args.image))[0] or "image/png"
encoded = base64.b64encode(args.image.read_bytes()).decode()
payload = {
    "model": args.model,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        {"type": "text", "text": "Read the printed text exactly, then list the shapes and their colors. Be concise."},
    ]}],
    "temperature": 0, "seed": 42, "max_tokens": 512, "reasoning_effort": "medium",
}
request = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
started = time.monotonic()
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.load(response)
answer = result["choices"][0]["message"]["content"]
print(answer)
print(f"Total response: {time.monotonic() - started:.2f} s")
prompt_ms = result.get("timings", {}).get("prompt_ms")
if prompt_ms is not None:
    print(f"Image + prompt processing: {prompt_ms / 1000:.2f} s")
if args.image.resolve() == fixture.resolve():
    if not all(word in answer.lower() for word in ("vision 742", "red", "blue", "green", "circle", "triangle")):
        raise SystemExit("FAIL: the smoke image's expected text/shapes were not all recognized.")
    if not any(word in answer.lower() for word in ("square", "rectangle", "box")):
        raise SystemExit("FAIL: the red box was not recognized.")
    print("PASS: native image OCR and shapes/colors smoke test.")

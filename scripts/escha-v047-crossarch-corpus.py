#!/usr/bin/env python3
"""Collect matched E3/W2 short-corpus logits on a selected GPU and bridge."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPTS = {
    "control": "Reply with exactly PARITY_OK and nothing else.",
    "fact": "What is the capital of France? Answer with one word.",
    "arithmetic": "Compute 37 times 19. Answer with the number only.",
    "structured": "Return only compact JSON with keys a and b, where a is 3 and b is true.",
    "code": "Write a Python function square(x) that returns x times itself. No explanation.",
    "instruction": "Reply with the second word of this list only: cedar maple willow ash.",
}


def request(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8107)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    cmd = [
        str(args.bin / "llama-server"), "-m", str(args.model),
        "-c", "8192", "-b", "2048", "-ub", "512", "-np", "1",
        "-t", "8", "-ngl", "99", "-fa", "on", "-ctk", "kvarn3",
        "-ctv", "kvarn2", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
        "--spec-draft-ngl", "0", "--spec-draft-type-k", "q4_0",
        "--spec-draft-type-v", "q4_0", "--spec-draft-ubatch-size", "64",
        "--jinja", "--reasoning", "off", "--host", "127.0.0.1",
        "--port", str(args.port), "--alias", "escha-v047-crossarch",
        "--no-webui",
    ]
    (args.out / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    (args.out / "hashes.json").write_text(json.dumps({
        "model": sha256(args.model),
        "server": sha256(args.bin / "llama-server"),
        "libllama": sha256(args.bin / "libllama.so"),
        "bridge": sha256(Path(os.environ["ESCHA_OFFICIAL_BRIDGE_LIBRARY"])),
    }, indent=2) + "\n")
    base = f"http://127.0.0.1:{args.port}"
    result = {"status": "started", "prompt_names": list(PROMPTS),
              "bridge_mapped": False, "responses": {}}
    server = None
    try:
        with (args.out / "server.log").open("w") as log:
            server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      env=os.environ.copy())
            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError(f"server exited before health: {server.returncode}")
                try:
                    if request(base + "/health").get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("health timeout")
            for name, prompt in PROMPTS.items():
                result["responses"][name] = []
                for repeat in (1, 2):
                    response = request(base + "/v1/chat/completions", {
                        "model": "escha-v047-crossarch",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0, "seed": 12345, "max_tokens": 64,
                        "logprobs": True, "top_logprobs": 20,
                    })
                    (args.out / f"{name}-{repeat}.json").write_text(
                        json.dumps(response, indent=2) + "\n")
                    choice = response["choices"][0]
                    result["responses"][name].append({
                        "repeat": repeat, "content": choice["message"]["content"],
                        "finish_reason": choice["finish_reason"],
                        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                    })
            maps = Path(f"/proc/{server.pid}/maps").read_text()
            bridge = os.environ["ESCHA_OFFICIAL_BRIDGE_LIBRARY"]
            result["bridge_mapped"] = bridge in maps
            if not result["bridge_mapped"]:
                raise RuntimeError("expected bridge not mapped after inference")
            result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        result["gpu_after"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            text=True).strip()
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

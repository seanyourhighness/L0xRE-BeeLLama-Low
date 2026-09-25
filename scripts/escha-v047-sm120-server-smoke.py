#!/usr/bin/env python3
"""Bounded Escha server smoke with route, output, memory, and cleanup receipts."""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def gpu_state() -> str:
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.free,pstate",
        "--format=csv,noheader",
    ], text=True).strip()


def request(url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("mtp", "dflash"), default="mtp")
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--draft-ngl", type=int, default=0)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--port", type=int, default=8107)
    parser.add_argument("--needle-lines", type=int, default=0)
    parser.add_argument("--needle-depth", type=float, default=0.5)
    parser.add_argument("--long-repeat", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    if args.mode == "dflash" and not args.draft:
        parser.error("--draft is required for dflash")
    if args.needle_lines and not 0 < args.needle_depth < 1:
        parser.error("--needle-depth must be between zero and one")
    args.out.mkdir(parents=True)

    cmd = [
        str(args.bin / "llama-server"), "-m", str(args.model),
        "-c", str(args.ctx), "-b", "2048", "-ub", "512", "-np", "1",
        "-t", "8", "-ngl", "99", "-fa", "on", "-ctk", "kvarn3",
        "-ctv", "kvarn2", "--spec-type", "draft-mtp" if args.mode == "mtp" else "draft-dflash",
        "--spec-draft-n-max", "2", "--spec-draft-ngl", str(args.draft_ngl),
        "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0",
        "--spec-draft-ubatch-size", "64", "--jinja", "--reasoning", "off",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--alias", "escha-sm120-qualification", "--no-webui",
    ]
    if args.draft:
        cmd.extend(["-md", str(args.draft)])
    (args.out / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    (args.out / "gpu-before.txt").write_text(gpu_state() + "\n")
    base = f"http://127.0.0.1:{args.port}"
    server = None
    result = {"status": "started", "mode": args.mode, "model": str(args.model)}
    try:
        with (args.out / "server.log").open("w") as log:
            server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      env=os.environ.copy())
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during load: {server.returncode}")
                try:
                    health = request(base + "/health")
                    if health.get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("server did not become healthy within 120 seconds")
            (args.out / "gpu-loaded.txt").write_text(gpu_state() + "\n")
            prompt = [{"role": "user", "content": "Reply with exactly PARITY_OK and nothing else."}]
            answers = []
            for index in range(2):
                response = request(base + "/v1/chat/completions", {
                    "model": "escha-sm120-qualification",
                    "messages": prompt, "temperature": 0,
                    "max_tokens": 32, "seed": 12345,
                    "logprobs": True, "top_logprobs": 20,
                })
                (args.out / f"control-{index + 1}.json").write_text(
                    json.dumps(response, indent=2) + "\n")
                answers.append(response["choices"][0]["message"]["content"].strip())
            result["answers"] = answers
            result["control_pass"] = answers == ["PARITY_OK", "PARITY_OK"]
            maps = Path(f"/proc/{server.pid}/maps").read_text()
            bridge = os.environ["ESCHA_OFFICIAL_BRIDGE_LIBRARY"]
            result["bridge_mapped"] = bridge in maps
            (args.out / "bridge-maps.txt").write_text(
                "\n".join(line for line in maps.splitlines() if bridge in line) + "\n")
            if not result["bridge_mapped"]:
                raise RuntimeError(f"bridge library was not mapped: {bridge}")
            result["gpu_after_control"] = gpu_state()
            if not result["control_pass"]:
                raise RuntimeError(f"control output mismatch: {answers}")
            if args.needle_lines:
                lines = [
                    f"Entry {i:05d}: Survey crew recorded a dry road, quiet wind, "
                    "and no signal changes."
                    for i in range(args.needle_lines)
                ]
                lines[int(args.needle_lines * args.needle_depth)] = (
                    "CONFIDENTIAL FIELD CODE: BLUE-FLAMINGO-42. "
                    "This is the only field code in the report."
                )
                long_prompt = (
                    "Read the field report and remember its one confidential field code.\n"
                    + "\n".join(lines)
                    + "\nWhat is the confidential field code? Reply with the code only."
                )
                (args.out / "long-prompt.txt").write_text(long_prompt + "\n")
                long_answers = []
                for index in range(2 if args.long_repeat else 1):
                    response = request(base + "/v1/chat/completions", {
                        "model": "escha-sm120-qualification",
                        "messages": [{"role": "user", "content": long_prompt}],
                        "temperature": 0, "max_tokens": 32, "seed": 12345,
                    })
                    (args.out / f"long-{index + 1}.json").write_text(
                        json.dumps(response, indent=2) + "\n")
                    long_answers.append(response["choices"][0]["message"]["content"].strip())
                result["long_answers"] = long_answers
                result["long_prompt_tokens"] = response.get("usage", {}).get("prompt_tokens")
                result["long_pass"] = all(answer == "BLUE-FLAMINGO-42" for answer in long_answers)
                result["gpu_after_long"] = gpu_state()
                if not result["long_pass"]:
                    raise RuntimeError(f"needle answer mismatch: {long_answers}")
            result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        result["gpu-after.txt"] = gpu_state()
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.out / "gpu-after.txt").write_text(result["gpu-after.txt"] + "\n")


if __name__ == "__main__":
    main()

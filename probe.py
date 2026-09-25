#!/usr/bin/env python3
"""Bounded, repeatable Escha server probe with GPU memory sampling."""

import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone


def gpu_used_mib():
    value = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    return int(value.strip().splitlines()[0])


def run_request(prompt, n_predict, samples):
    first_sample = len(samples) - 1
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 12345,
        "stream": False,
        "cache_prompt": False,
    }).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:30172/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started
    content = result.get("content", "")
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_chars": len(prompt),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_prefix": content[:80],
        "timings": result.get("timings", {}),
        "tokens_predicted": result.get("tokens_predicted"),
        "tokens_evaluated": result.get("tokens_evaluated"),
        "gpu_peak_mib": max(samples[first_sample:]),
    }


def main():
    label = sys.argv[1]
    samples = [gpu_used_mib()]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.2):
            try:
                samples.append(gpu_used_mib())
            except Exception:
                pass

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        control = run_request("Return exactly E3_12GB_OK and nothing else.", 32, samples)
        long_prompt = (
            "A systems engineer records the cache settings, GPU memory, and decode speed "
            "for a local language model. The profile keeps target and draft settings separate.\n"
        ) * 600
        long_prompt += "\nSummarize the main measurement goal in one paragraph."
        long_run = run_request(long_prompt, 128, samples)
    finally:
        stop.set()
        worker.join(timeout=2)
    print(json.dumps({
        "label": label,
        "utc": datetime.now(timezone.utc).isoformat(),
        "gpu_before_mib": samples[0],
        "gpu_peak_mib": max(samples),
        "gpu_after_mib": gpu_used_mib(),
        "control": control,
        "long": long_run,
    }, indent=2))


if __name__ == "__main__":
    main()

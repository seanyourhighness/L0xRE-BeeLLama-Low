#!/usr/bin/env python3
"""Correctness and integration battery driven through the installed launcher."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

def ask(port, prompt, max_tokens=64, seed=1234, timeout=600, force_tokens=None):
    body = {"messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "seed": seed, "max_tokens": max_tokens}
    if force_tokens:
        body["max_tokens"] = force_tokens
        body["min_tokens"] = force_tokens
        body["ignore_eos"] = True
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"], result["usage"]["completion_tokens"]


def needle(port, repeats, code):
    filler = "The quick brown fox jumps over the lazy dog. " * repeats
    prompt = (filler
              + "Important: the vault access code is " + code + ".\n"
              + filler
              + "Question: what is the vault access code? Answer with the code only.")
    text, _ = ask(port, prompt, max_tokens=64)
    return needle_answer_matches(text, code), text[:120].replace("\n", " ")


def needle_answer_matches(text, code):
    """Require the requested code as the complete answer, not a substring."""
    return text.strip() == code


def run(launcher, models, draft, model, profile, port, checks, workspace):
    # The filler appears twice around the needle, so keep the prompt inside the
    # profile's context: about 4K tokens for 8K profiles, 18K for 32K.
    needle_repeats = 900 if profile.endswith("32k") else 180
    command = [str(launcher), "serve", "--model", model, "--model-path", str(models[model]),
               "--profile", profile, "--port", str(port)]
    if profile == "dflash-8k":
        command += ["--draft-path", str(draft)]
    log_path = workspace / f"server-{model}-{profile}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            if process.poll() is not None:
                checks.append({"check": model + "/" + profile + "/startup", "ok": False,
                               "detail": "server exited"})
                return
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        break
            except Exception:
                time.sleep(1)
        else:
            checks.append({"check": model + "/" + profile + "/startup", "ok": False,
                           "detail": "health timeout"})
            return
        checks.append({"check": model + "/" + profile + "/startup", "ok": True})

        prompt_a = ("Write a numbered list from one to forty. Spell out each number and "
                    "add the word sapphire.")
        prompt_b = "Describe how to fold a paper crane in numbered steps."
        first, n1 = ask(port, prompt_a, 256)
        ask(port, prompt_b, 128)
        repeat, n2 = ask(port, prompt_a, 256)
        checks.append({"check": model + "/" + profile + "/aba-identity",
                       "ok": first == repeat and n1 == 256 and n2 == 256,
                       "detail": {"tokens": [n1, n2],
                                  "sha256": hashlib.sha256(first.encode()).hexdigest()}})

        ok, detail = needle(port, needle_repeats, "LARKSPUR-7412")
        checks.append({"check": model + "/" + profile + "/retrieval", "ok": ok,
                       "detail": detail})

        _, tokens = ask(port, "Write a continuous technical paragraph about GPU memory "
                              "bandwidth.", 500, force_tokens=500)
        checks.append({"check": model + "/" + profile + "/long-output", "ok": tokens == 500,
                       "detail": tokens})
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-e3", required=True, type=Path,
                        help="path to the E3 model GGUF")
    parser.add_argument("--model-w2", required=True, type=Path,
                        help="path to the W2 model GGUF")
    parser.add_argument("--model-native", required=True, type=Path,
                        help="path to the native IQ3 control GGUF")
    parser.add_argument("--draft", required=True, type=Path,
                        help="path to the DFlash2 drafter GGUF")
    args = parser.parse_args()
    models = {"e3": args.model_e3.resolve(), "w2": args.model_w2.resolve(),
              "native": args.model_native.resolve()}
    draft = args.draft.resolve()
    for label, path in [*models.items(), ("draft", draft)]:
        if not path.is_file():
            parser.error(f"{label} model file does not exist: {path}")
    launcher = Path(args.launcher).resolve()
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    plan = [("e3", "ordinary-8k"), ("w2", "ordinary-8k"),
            ("e3", "dflash-8k"), ("w2", "dflash-8k"),
            ("native", "ordinary-8k"), ("native", "dflash-8k"),
            ("e3", "ordinary-32k"), ("w2", "ordinary-32k")]
    checks = []
    port = 18095
    for model, profile in plan:
        print(">>>", model, profile, flush=True)
        run(launcher, models, draft, model, profile, port, checks, workspace)
        port += 1
    failures = [c for c in checks if not c["ok"]]
    result = {"launcher": str(launcher), "checks": checks, "failures": failures,
              "status": "pass" if not failures else "fail"}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "failures": [f["check"] for f in failures]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

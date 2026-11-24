#!/usr/bin/env python3
"""Toggle or inspect L0xRE runtime control-vector steering via /l0xre/cvec."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ACTION_ALIASES = {
    "on": "on",
    "true": "on",
    "1": "on",
    "enable": "on",
    "off": "off",
    "false": "off",
    "0": "off",
    "disable": "off",
    "toggle": "toggle",
    "status": "status",
}


def normalize_base_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3].rstrip("/")
    return url


def request_json(url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def format_status(state: dict) -> str:
    label = "ON" if state.get("enabled") else "OFF"
    mode = state.get("mode", "?")
    start = state.get("layer_start", "?")
    end = state.get("layer_end", "?")
    scale = state.get("scale", "?")
    filename = state.get("file", "?")
    return f"Uncensored mode: {label} (mode={mode}, layers={start}..{end}, scale={scale}, file={filename})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="toggle",
        help="on | off | toggle | status (default: toggle)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("L0XRE_URL", "http://127.0.0.1:8080"),
        help="L0xRE server base URL (default: $L0XRE_URL or http://127.0.0.1:8080)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON from /l0xre/cvec")
    args = parser.parse_args(argv)

    action = ACTION_ALIASES.get(args.action.strip().lower())
    if action is None:
        print(f"Invalid action {args.action!r}; expected on, off, toggle, or status.", file=sys.stderr)
        return 2

    base = normalize_base_url(args.url)
    endpoint = f"{base}/l0xre/cvec"
    try:
        current = request_json(endpoint)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        print(f"HTTP {exc.code} from {endpoint}: {body}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"Cannot reach L0xRE control-vector endpoint at {endpoint}: {exc}", file=sys.stderr)
        return 1

    if not current.get("loaded"):
        if args.json:
            print(json.dumps(current))
        else:
            print(
                f"Uncensored control vector is not loaded on {base}. "
                "Start L0xRE with ./l0xre-uncensored (or --uncensored / start-uncensored.cmd).",
                file=sys.stderr if action != "status" else sys.stdout,
            )
        return 0 if action == "status" else 1

    if action == "status":
        print(json.dumps(current) if args.json else format_status(current))
        return 0

    target = (not bool(current.get("enabled"))) if action == "toggle" else (action == "on")
    try:
        updated = request_json(endpoint, {"enabled": target})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        print(f"HTTP {exc.code} from {endpoint}: {body}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"Failed to update control vector at {endpoint}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(updated) if args.json else format_status(updated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

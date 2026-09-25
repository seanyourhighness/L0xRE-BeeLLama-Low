#!/usr/bin/env python3
"""Compare the four E3/W2 llama-bench cells without averaging them together."""

import argparse
import json
import statistics
from pathlib import Path


def samples(directory: Path, model: str, workload: str) -> list[float]:
    path = directory / f"{model}-{workload}.json"
    rows = json.loads(path.read_text())
    if len(rows) != 1:
        raise ValueError(f"expected one benchmark row in {path}, found {len(rows)}")
    values = rows[0]["samples_ts"]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"missing or invalid throughput samples in {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sm89", type=Path)
    parser.add_argument("sm120", type=Path)
    args = parser.parse_args()

    failed = False
    print("model workload  SM89 median  SM120 median  ratio  samples")
    for model in ("e3", "w2"):
        for workload in ("prefill", "decode"):
            reference = samples(args.sm89, model, workload)
            candidate = samples(args.sm120, model, workload)
            base = statistics.median(reference)
            current = statistics.median(candidate)
            ratio = current / base
            print(f"{model:5} {workload:8} {base:11.2f} {current:13.2f} "
                  f"{ratio:6.3f} {len(reference)}/{len(candidate)}")
            failed |= ratio < 1.0
    print("SCREEN PASS" if not failed else "SCREEN REGRESSION")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

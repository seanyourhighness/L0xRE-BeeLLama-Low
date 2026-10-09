#!/usr/bin/env python3
"""Experimental matched-card layer split with per-device fast bridges; untested on physical pairs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command_for(receipt, profile):
    runtime = Path(receipt["runtime"])
    archroot = runtime / "architectures" / receipt["gpu"]["arch"]
    addon = Path(receipt["dual_addon_root"]) / receipt["gpu"]["arch"]
    def expand(v):
        return v.replace("@PACKAGE_ROOT@", str(runtime)).replace("@ARCH_ROOT@", str(archroot)).replace("@ARCH@", receipt["gpu"]["arch"])
    args = [expand(v) for v in profile["argv"]]
    args[args.index("--port") + 1] = str(receipt["port"])
    args += ["--split-mode", "layer", "--tensor-split", "1,1", "--device", "CUDA0,CUDA1", "--spec-draft-device", "CUDA0"]
    models = Path(receipt["models_dir"])
    args += ["-m", str(models / "L0xRE-27b-Low.gguf"), "-md", str(models / "Qwen3.8-27B-DFlash2-Q4_K_M.gguf")]
    if receipt["vision"]:
        args += ["--mmproj", str(models / "mmproj-Qwen3.8-27B-Q8_0.gguf")]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("L0XRE_", "ESCHA_", "GGML_", "LLAMA_ARG_")) and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "CUDA_LAUNCH_BLOCKING")}
    env.update({k: expand(v) for k, v in profile["env"].items()})
    bridge = str(addon / "libbridge-dual.so")
    preloads = env.get("LD_PRELOAD", "").split(":")
    replaced = [bridge if Path(v).name == "libbridge-packed.so" else v for v in preloads if v]
    if bridge not in replaced:
        replaced.append(bridge)
    env.update(CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in receipt["gpus"]),
               LD_PRELOAD=":".join(replaced),
               LD_LIBRARY_PATH=str(addon) + ":" + str(archroot / "bin"),
               GGML_BACKEND_PATH=str(archroot / "bin"),
               ESCHA_OFFICIAL_BRIDGE_LIBRARY=bridge,
               L0XRE_GDN_MASKED_LIBRARY=str(addon / "libgdn-dual.so"),
               L0XRE_DUAL_FAST_EXPERIMENT="1")
    return args, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    opt = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads((root / "INSTALLATION.json").read_text())
    if receipt.get("mode") != "dual-fast-experimental":
        raise ValueError("This installation is not configured for dual fast layer split")
    for name, expected in receipt["launcher_files"].items():
        if sha(root / name) != expected:
            raise ValueError("Dual launcher integrity mismatch: " + name)
    profile = json.loads((Path(receipt["runtime"]) / "tools/universal/r6-profile.json").read_text())
    runtime = Path(receipt["runtime"])
    for package in [runtime, Path(receipt["dual_addon_root"])]:
        for line in (package / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = package / name
            if not path.resolve().is_relative_to(package.resolve()) or sha(path) != expected:
                raise ValueError("Runtime/add-on integrity mismatch: " + name)
    manifest = json.loads((runtime / "architectures" / receipt["gpu"]["arch"] / "R6-MANIFEST.json").read_text())
    for name, target in manifest.get("loader_links", {}).items():
        if os.readlink(runtime / "architectures" / receipt["gpu"]["arch"] / name) != target:
            raise ValueError("Runtime loader link changed: " + name)
    for gpu in receipt["gpus"]:
        row = next(csv.reader(subprocess.check_output(["nvidia-smi", "--id=" + gpu["uuid"], "--query-gpu=uuid,name,compute_cap,memory.total", "--format=csv,noheader,nounits"], text=True).splitlines()))
        uuid, name, cc, memory = [s.strip() for s in row]
        if uuid != gpu["uuid"] or name != gpu["name"] or "sm" + cc.replace(".", "") != gpu["arch"] or int(memory) < 12000:
            raise ValueError("Selected matched GPU is unavailable or changed")
    for key, name in [("target", "L0xRE-27b-Low.gguf"), ("draft", "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"), *([("vision", "mmproj-Qwen3.8-27B-Q8_0.gguf")] if receipt["vision"] else [])]:
        if sha(Path(receipt["models_dir"]) / name) != profile["model_sha256"][key]:
            raise ValueError("Pinned model mismatch: " + name)
    args, env = command_for(receipt, profile)
    if opt.dry_run:
        print(json.dumps({"mode": "dual-fast-experimental", "hardware_qualified": False, "argv": args,
                          "env": {k: v for k, v in env.items() if k.startswith(("ESCHA_", "L0XRE_", "GGML_", "LD_", "CUDA_"))},
                          "external_fast_bridges": True}, indent=2))
        return
    os.sched_setaffinity(0, set(range(8)))
    os.execvpe(args[0], args, env)


if __name__ == "__main__":
    main()

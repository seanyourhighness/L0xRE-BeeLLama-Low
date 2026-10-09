#!/usr/bin/env python3
"""Install a SHA-pinned L0xRE Linux/WSL package without altering its profile."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

CATALOG_URL = "https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/catalog.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_catalog(path=None):
    local = Path(path) if path else Path(__file__).with_name("catalog.json")
    if local.is_file():
        data = json.loads(local.read_text())
    elif path:
        raise ValueError(f"Catalog not found: {local}")
    else:
        with urllib.request.urlopen(CATALOG_URL, timeout=60) as response:
            data = json.load(response)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported installer catalog version; download the installer again")
    return data


def detect_gpu(selector=None):
    selected = selector if selector is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not selected or selected == "-1":
        raise ValueError("No GPU selected by CUDA_VISIBLE_DEVICES; supply --gpu INDEX or UUID")
    command = ["nvidia-smi", "--id=" + selected,
               "--query-gpu=index,uuid,name,compute_cap,memory.total", "--format=csv,noheader,nounits"]
    try:
        rows = list(csv.reader(subprocess.check_output(command, text=True).strip().splitlines()))
        if len(rows) != 1 or len(rows[0]) != 5:
            raise ValueError("Expected exactly one GPU")
        index, uuid, name, cc, memory = (s.strip() for s in rows[0])
        return {"index": int(index), "uuid": uuid, "name": name,
                "arch": "sm" + cc.replace(".", ""), "memory_mib": int(memory)}
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError("Cannot detect the GPU; check nvidia-smi and the NVIDIA driver") from error


def make_plan(catalog, gpu, allow_candidate=False, runtime_archive=None):
    package = catalog["packages"]["linux"].get(gpu["arch"])
    if not package:
        raise ValueError(f"Unsupported GPU architecture: {gpu['arch']}")
    if gpu["memory_mib"] < package["min_memory_mib"]:
        raise ValueError(f"{gpu['name']} has insufficient VRAM for this packaged 80K profile")
    if package["gpu_name_contains"] and package["gpu_name_contains"] not in gpu["name"]:
        raise ValueError(f"The available {gpu['arch']} setup is scoped to {package['gpu_name_contains']}")
    if package["status"] == "candidate" and not allow_candidate:
        raise ValueError(f"{gpu['arch']} Linux is a candidate. Use --allow-candidate only to opt into testing it")
    if not package["url"] and not runtime_archive:
        raise ValueError("The certified Linux SM120 archive is not published yet. Supply --runtime-archive PATH to the sealed archive; Windows SM120 is available now")
    return {"gpu": gpu, "package": package, "platform": "linux"}


def verify_artifact(path, record):
    path = Path(path)
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Size or SHA-256 mismatch: {path}. Existing files are never overwritten")


def get_artifact(record, directory):
    name = record["filename"]
    if Path(name).name != name or name in (".", "..") or "\\" in name:
        raise ValueError("Invalid artifact filename")
    directory = Path(directory)
    destination = directory / record["filename"]
    if destination.exists():
        verify_artifact(destination, record)
        print(f"Reusing verified {destination.name}", flush=True)
        return destination
    directory.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.is_file() and partial.stat().st_size == record["bytes"]:
        try:
            verify_artifact(partial, record)
            partial.replace(destination)
            return destination
        except ValueError:
            partial.unlink()
    print(f"Downloading {destination.name} ({record['bytes'] / 1e9:.2f} GB)", flush=True)
    command = ["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
               "--output", str(partial), record["url"]]
    subprocess.run(command, check=True)
    try:
        verify_artifact(partial, record)
    except ValueError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination


def contained_path(root, relative):
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe package manifest path: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Package path escapes extraction root: {relative}")
    return result


def verify_package(root):
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError("Package is missing SHA256SUMS")
    count = 0
    for line in manifest.read_text().splitlines():
        if not line:
            continue
        match = re.fullmatch(r"([a-fA-F0-9]{64}) [ *](.+)", line)
        if not match:
            raise ValueError("Invalid package checksum entry")
        expected, relative = match.groups()
        path = contained_path(root, relative)
        if not path.is_file() or sha256(path) != expected.lower():
            raise ValueError(f"Package checksum mismatch: {relative}")
        count += 1
    if not count:
        raise ValueError("Empty package checksum manifest")
    print(f"Verified {count} package files", flush=True)


def unpack(archive, destination, package):
    if destination.exists():
        verify_package(destination)
        return destination
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        process = subprocess.Popen(["zstd", "--decompress", "--stdout", str(archive)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
                tar.extractall(stage, filter="data")
            process.stdout.close()
            if process.wait() != 0:
                raise ValueError("Archive decompression failed")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        roots = [p for p in [stage, *stage.iterdir()] if p.is_dir() and (p / package["launcher"]).is_file()]
        if len(roots) != 1:
            raise ValueError("Cannot identify the package launcher in this archive")
        root = roots[0]
        verify_package(root)
        root.rename(destination)
    return destination


def disk_parent(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def check_space(install_dir, models_dir, records, runtime_reserve):
    required = {}
    for path, amount in [(install_dir, runtime_reserve),
                         (models_dir, sum(r["bytes"] for r in records if not (models_dir / r["filename"]).exists()))]:
        parent = disk_parent(path)
        key = parent.stat().st_dev
        old_parent, old_amount = required.get(key, (parent, 0))
        required[key] = (old_parent, old_amount + amount)
    for parent, amount in required.values():
        if shutil.disk_usage(parent).free < amount:
            raise ValueError(f"Insufficient free disk space on {parent}: need about {amount / 1e9:.1f} GB")


def write_launcher(install_dir, runtime, models_dir, plan, vision, port):
    args = [sys.executable, str(runtime / plan["package"]["launcher"]), "serve", "--arch", plan["gpu"]["arch"],
            "-m", str(models_dir / "L0xRE-27b-Low.gguf"), "-md", str(models_dir / "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"),
            "--host", "127.0.0.1", "--port", str(port)]
    if plan["package"]["status"] == "candidate":
        args += ["--qualification-probe"]
    if vision:
        args += ["--mmproj", str(models_dir / "mmproj-Qwen3.8-27B-Q8_0.gguf")]
    script = install_dir / "start.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=" +
                      shlex.quote(plan["gpu"]["uuid"]) + "\nexec " + shlex.join(args) + ' "$@"\n')
    script.chmod(0o755)
    (install_dir / "INSTALLATION.json").write_text(json.dumps({**plan, "models_dir": str(models_dir),
                                                               "runtime": str(runtime), "vision": vision, "port": port}, indent=2) + "\n")
    return script


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dir", type=Path, default=Path.home() / "L0xRE")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--gpu", help="Physical GPU index or UUID; defaults to first CUDA_VISIBLE_DEVICES entry or GPU 0")
    parser.add_argument("--runtime-archive", type=Path)
    parser.add_argument("--allow-candidate", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show the selection without writing files or downloading artifacts")
    parser.add_argument("--yes", action="store_true", help="Accept the planned downloads without a prompt")
    parser.add_argument("--port", type=int, default=8080)
    options = parser.parse_args(argv)
    if not 1 <= options.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if sys.platform != "linux" or sys.version_info < (3, 12):
        parser.error("Linux / WSL with Python 3.12+ is required; use install.ps1 on Windows")
    catalog = load_catalog(options.catalog)
    plan = make_plan(catalog, detect_gpu(options.gpu), options.allow_candidate, options.runtime_archive)
    install_dir = options.dir.expanduser().resolve()
    models_dir = (options.models_dir or install_dir / "models").expanduser().resolve()
    plan.update({"install_dir": str(install_dir), "models_dir": str(models_dir), "vision": options.vision, "port": options.port})
    if options.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    for command in ("curl", "zstd", "numactl"):
        if not shutil.which(command):
            raise ValueError(f"Missing {command}. On Ubuntu / Debian: sudo apt-get install curl zstd numactl")
    if len(os.sched_getaffinity(0).intersection(range(8))) != 8:
        raise ValueError("The packaged profile requires CPU cores 0–7 to be accessible")
    records = [catalog["models"][k] for k in (["target", "draft", "vision"] if options.vision else ["target", "draft"])]
    if not options.runtime_only:
        for record in records:
            path = models_dir / record["filename"]
            if path.exists():
                verify_artifact(path, record)
    print(f"GPU: {plan['gpu']['name']} / {plan['gpu']['arch']}\nPackage: {plan['package']['status']}\nProfile: {plan['package']['profile']}\nInstall: {install_dir}\nModels: {models_dir}")
    if not options.yes:
        print("Runtime-only setup; model downloads are skipped." if options.runtime_only else
              "Setup may download about 10 GB of models plus the runtime. Existing matching files will be reused.")
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            return 0
    check_space(install_dir, models_dir, [] if options.runtime_only else records, plan["package"]["install_reserve_bytes"])
    install_dir.mkdir(parents=True, exist_ok=True)
    archive = options.runtime_archive.expanduser().resolve() if options.runtime_archive else get_artifact(plan["package"], install_dir / "downloads")
    verify_artifact(archive, plan["package"])
    runtime = install_dir / (plan["gpu"]["arch"] + "-" + plan["package"]["sha256"][:12])
    runtime = unpack(archive, runtime, plan["package"])
    if not options.runtime_only:
        for record in records:
            get_artifact(record, models_dir)
    script = write_launcher(install_dir, runtime, models_dir, plan, options.vision, options.port)
    print(f"Setup complete. Start: {shlex.quote(str(script))}\nCheck the launcher: {shlex.quote(str(script))} --dry-run\nAPI: http://127.0.0.1:{options.port}/v1")
    if options.runtime_only:
        print("Runtime-only setup: download the pinned models or rerun without --runtime-only before starting.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError, tarfile.TarError, EOFError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        raise SystemExit(2)

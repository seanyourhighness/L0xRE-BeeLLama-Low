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
    if gpu["arch"] == "sm120" and (gpu["memory_mib"] < 30000 or "RTX 5090" not in gpu["name"]):
        package = catalog["packages"]["linux"].get("sm120-12gb", package)
    if not package:
        raise ValueError(f"Unsupported GPU architecture: {gpu['arch']}")
    if gpu["memory_mib"] < package["min_memory_mib"]:
        raise ValueError(f"{gpu['name']} has insufficient VRAM for this packaged profile")
    if package["gpu_name_contains"] and package["gpu_name_contains"] not in gpu["name"]:
        raise ValueError(f"The available {gpu['arch']} setup is scoped to {package['gpu_name_contains']}")
    if package["status"] == "candidate" and not allow_candidate:
        raise ValueError(f"{gpu['arch']} Linux is a candidate. Use --allow-candidate only to opt into testing it")
    if not package["url"] and not runtime_archive:
        raise ValueError("The certified Linux SM120 archive is not published yet. Supply --runtime-archive PATH to the sealed archive; Windows SM120 is available now")
    measured = package.get("certified_gpu_name_contains", package["gpu_name_contains"])
    return {"gpu": gpu, "package": package, "platform": "linux",
            "hardware_qualified_for_gpu": package["status"].startswith("certified") and bool(measured) and measured in gpu["name"]}


def make_dual_plan(catalog, selectors, allow_candidate=False, runtime_archive=None):
    if not allow_candidate:
        raise ValueError("Dual-GPU native layer split is experimental; use --allow-candidate")
    ids = selectors.split(",")
    if len(ids) != 2 or any(not s.strip() for s in ids):
        raise ValueError("Supply exactly two GPU indices or UUIDs with --gpus 0,1")
    gpus = [detect_gpu(s.strip()) for s in ids]
    a, b = gpus
    if a["uuid"] == b["uuid"] or a["arch"] != b["arch"] or a["name"] != b["name"] or abs(a["memory_mib"] - b["memory_mib"]) > 256:
        raise ValueError("Dual mode requires two distinct matched cards of the same model and VRAM size")
    if min(g["memory_mib"] for g in gpus) < 12000:
        raise ValueError("Dual mode requires at least 12GB VRAM per card")
    plan = make_plan(catalog, a, True, runtime_archive)
    addon = catalog.get("dual_addons", {}).get("linux")
    if not addon:
        raise ValueError("This catalog does not contain the experimental dual fast-path add-on")
    plan.update(gpus=gpus, mode="dual-fast-experimental", hardware_qualified_for_gpu=False, dual_addon=addon)
    return plan


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
    receipt_path = install_dir / "INSTALLATION.json"
    args = [sys.executable, str(runtime / plan["package"]["launcher"]), "serve", "--arch", plan["gpu"]["arch"],
            "-m", str(models_dir / "L0xRE-27b-Low.gguf"), "-md", str(models_dir / "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"),
            "--host", "127.0.0.1", "--port", str(port)]
    if plan["package"]["status"] == "candidate":
        args += ["--qualification-probe"]
    if vision:
        args += ["--mmproj", str(models_dir / "mmproj-Qwen3.8-27B-Q8_0.gguf")]
    if plan.get("mode") == "dual-fast-experimental":
        source = Path(__file__).with_name("dual-launch.py")
        helper = install_dir / "launchers" / sha256(source)[:16] / source.name
        helper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, helper)
        plan["launcher_files"] = {str(helper.relative_to(install_dir)): sha256(helper)}
        args = [sys.executable, str(helper)]
    script = install_dir / "start.sh"
    script_text = ("#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=" +
                   shlex.quote(plan["gpu"]["uuid"]) + "\nexec " + shlex.join(args) + ' "$@"\n')
    previous = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    if previous and script.is_file() and (script.read_text() != script_text or previous.get("runtime") != str(runtime)
                                         or previous.get("dual_addon_root") != plan.get("dual_addon_root")):
        rollback = install_dir / "previous"
        rollback.mkdir(exist_ok=True)
        for name in ("INSTALLATION.json", "start.sh"):
            shutil.copy2(install_dir / name, rollback / name)
    script.write_text(script_text)
    script.chmod(0o755)
    updater = install_dir / "update.sh"
    updater.write_text('#!/usr/bin/env bash\nset -euo pipefail\nstage="$(mktemp -d)"\ntrap \'rm -rf -- "$stage"\' EXIT\n'
        'curl -fL --retry 3 https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh -o "$stage/install.sh"\n'
        'bash "$stage/install.sh" --update --dir ' + shlex.quote(str(install_dir)) + ' "$@"\n')
    updater.chmod(0o755)
    (install_dir / "INSTALLATION.json").write_text(json.dumps({**plan, "models_dir": str(models_dir),
                                                               "runtime": str(runtime), "vision": vision, "port": port}, indent=2) + "\n")
    return script


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dir", type=Path, default=Path.home() / "L0xRE")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--gpu", help="Physical GPU index or UUID; defaults to first CUDA_VISIBLE_DEVICES entry or GPU 0")
    parser.add_argument("--gpus", help="Experimental matched pair, e.g. 0,1; requires --allow-candidate")
    parser.add_argument("--runtime-archive", type=Path)
    parser.add_argument("--allow-candidate", action="store_true")
    parser.add_argument("--update", action="store_true", help="Keep installed GPU, models, vision and port while selecting the current release")
    parser.add_argument("--rollback", action="store_true", help="Restore the previous verified runtime and start script")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show the selection without writing files or downloading artifacts")
    parser.add_argument("--yes", action="store_true", help="Accept the planned downloads without a prompt")
    parser.add_argument("--port", type=int)
    options = parser.parse_args(argv)
    install_dir = options.dir.expanduser().resolve()
    if options.rollback:
        previous = install_dir / "previous"
        receipt = json.loads((previous / "INSTALLATION.json").read_text())
        verify_package(Path(receipt["runtime"]))
        if receipt.get("dual_addon_root"):
            verify_package(Path(receipt["dual_addon_root"]))
        for name, expected in receipt.get("launcher_files", {}).items():
            path = install_dir / name
            if not path.resolve().is_relative_to(install_dir.resolve()) or sha256(path) != expected:
                raise ValueError("Previous launcher integrity mismatch: " + name)
        if not options.dry_run:
            for name in ("INSTALLATION.json", "start.sh"):
                shutil.copy2(previous / name, install_dir / name)
        print("Previous runtime: " + receipt["runtime"])
        return 0
    if options.update:
        previous = json.loads((install_dir / "INSTALLATION.json").read_text())
        if not options.gpu and not options.gpus:
            if previous.get("mode") == "dual-fast-experimental":
                options.gpus = ",".join(g["uuid"] for g in previous["gpus"])
            else:
                options.gpu = previous["gpu"]["uuid"]
        options.models_dir = options.models_dir or Path(previous["models_dir"])
        options.vision = options.vision or previous["vision"]
        options.port = options.port if options.port is not None else previous["port"]
        options.allow_candidate = options.allow_candidate or previous["package"]["status"] == "candidate" or previous.get("mode") == "dual-fast-experimental"
    options.port = options.port if options.port is not None else 8080
    if not 1 <= options.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if sys.platform != "linux" or sys.version_info < (3, 12):
        parser.error("Linux / WSL with Python 3.12+ is required; use install.ps1 on Windows")
    if options.update and not options.catalog:
        with urllib.request.urlopen(CATALOG_URL, timeout=60) as response:
            catalog = json.load(response)
        if catalog.get("schema_version") != 1:
            raise ValueError("Download the current installer to read this catalog")
    else:
        catalog = load_catalog(options.catalog)
    if options.gpu and options.gpus:
        raise ValueError("Choose --gpu or --gpus, not both")
    plan = (make_dual_plan(catalog, options.gpus, options.allow_candidate, options.runtime_archive) if options.gpus else
            make_plan(catalog, detect_gpu(options.gpu), options.allow_candidate, options.runtime_archive))
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
    print(f"Certified on selected GPU: {plan['hardware_qualified_for_gpu']}")
    if plan.get("mode"):
        print("Dual GPU: EXPERIMENTAL / UNTESTED / UNCERTIFIED; performance parity is unmeasured.")
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
    if plan.get("mode") == "dual-fast-experimental":
        addon = plan["dual_addon"]
        addon_archive = get_artifact(addon, install_dir / "downloads")
        addon_root = install_dir / ("dual-" + addon["sha256"][:12])
        plan["dual_addon_root"] = str(unpack(addon_archive, addon_root, addon))
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

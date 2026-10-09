#!/usr/bin/env bash
set -euo pipefail
command -v python3 >/dev/null || { echo 'Install Python 3.12+ first.' >&2; exit 2; }
python3 -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ is required"'
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$root/install.py" && -f "$root/catalog.json" ]]; then
  exec python3 "$root/install.py" "$@"
fi
command -v curl >/dev/null || { echo 'Install curl first: sudo apt-get install curl' >&2; exit 2; }
stage="$(mktemp -d)"
trap 'rm -rf -- "$stage"' EXIT
ref="${L0XRE_INSTALL_REF:-main}"
base="https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/$ref/install"
curl -fL --retry 3 "$base/install.py" -o "$stage/install.py"
curl -fL --retry 3 "$base/catalog.json" -o "$stage/catalog.json"
curl -fL --retry 3 "$base/dual-launch.py" -o "$stage/dual-launch.py"
python3 "$stage/install.py" "$@"

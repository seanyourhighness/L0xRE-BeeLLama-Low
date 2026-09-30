#!/usr/bin/env bash
# Opt-in SM86 CPU-vision recipe for the extracted universal runtime.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'VISION_RUNTIME_DIR=/path/to/extracted/runtime VISION_WORKERS=32 bash serve-cpu-sm86.sh [server options]'
  exit 0
fi
vision_runtime="${VISION_RUNTIME_DIR:-$PWD}"
vision_models="${VISION_MODELS_DIR:-$vision_runtime/models}"
vision_workers="${VISION_WORKERS:-32}"
[[ "$vision_workers" =~ ^[1-9][0-9]*$ ]] || { echo 'VISION_WORKERS must be a positive integer.' >&2; exit 2; }
[[ -x "$vision_runtime/l0xre" ]] || { echo 'Set VISION_RUNTIME_DIR to the extracted universal runtime directory.' >&2; exit 2; }
for vision_file in L0xRE-27b-Low.gguf Qwen3.8-27B-DFlash2-Q4_K_M.gguf mmproj-Qwen3.8-27B-Q8_0.gguf; do
  [[ -f "$vision_models/$vision_file" ]] || { echo "Missing model: $vision_models/$vision_file" >&2; exit 2; }
done
exec "$vision_runtime/l0xre" serve --profile 12gb-b84 \
  -m "$vision_models/L0xRE-27b-Low.gguf" \
  -md "$vision_models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf" \
  --mmproj "$vision_models/mmproj-Qwen3.8-27B-Q8_0.gguf" \
  --no-mmproj-offload --image-min-tokens 1024 --image-max-tokens 1024 \
  -t "$vision_workers" -tb "$vision_workers" --reasoning-effort medium \
  --host 127.0.0.1 --port 8080 "$@"

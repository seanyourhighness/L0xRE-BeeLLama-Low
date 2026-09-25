#!/usr/bin/env bash
# Same-card native Qwen GGUF reference for the Escha four-cell screen.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "usage: BIN_DIR=... MODEL=... $0 OUT_DIR REPS KV_K KV_V" >&2
    exit 2
fi
out=$1
reps=$2
kv_k=$3
kv_v=$4
: "${BIN_DIR:?set BIN_DIR}"
: "${MODEL:?set MODEL}"
[[ $reps =~ ^[1-9][0-9]*$ ]] || exit 2
[[ ! -e $out ]] || { echo "refusing to overwrite $out" >&2; exit 2; }
mkdir -p "$out"
export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset ESCHA_OFFICIAL_BRIDGE ESCHA_OFFICIAL_BRIDGE_LIBRARY \
      ESCHA_OFFICIAL_RAW_BRIDGE ESCHA_OFFICIAL_RAW_DECODE_BRIDGE \
      ESCHA_OFFICIAL_RAW_DECODE_F32_INPUT ESCHA_GDN_CHUNK_BRIDGE
printf 'model=%s\nbinary=%s\nreps=%s\nkv_k=%s\nkv_v=%s\n' \
    "$MODEL" "$BIN_DIR/llama-bench" "$reps" "$kv_k" "$kv_v" > "$out/config.txt"
if [[ -n ${MODEL_SHA256:-} ]]; then
    [[ $MODEL_SHA256 =~ ^[0-9a-f]{64}$ ]] || { echo "invalid MODEL_SHA256" >&2; exit 2; }
    printf '%s  %s\n' "$MODEL_SHA256" "$MODEL" > "$out/SHA256SUMS.txt"
else
    sha256sum "$MODEL" > "$out/SHA256SUMS.txt"
fi
sha256sum "$BIN_DIR/llama-bench" "$BIN_DIR/libllama.so" \
    "$BIN_DIR/libggml-cuda.so" >> "$out/SHA256SUMS.txt"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free,pstate,power.draw,clocks.sm \
    --format=csv,noheader > "$out/gpu-before.txt"
"$BIN_DIR/llama-bench" -m "$MODEL" -p 2048 -n 0 -b 2048 -ub 2048 \
    -t 8 -ngl 99 -fa on -ctk "$kv_k" -ctv "$kv_v" -r "$reps" -o json \
    > "$out/prefill.json" 2> "$out/prefill.log"
"$BIN_DIR/llama-bench" -m "$MODEL" -p 0 -n 256 -b 2048 -ub 512 \
    -t 8 -ngl 99 -fa on -ctk "$kv_k" -ctv "$kv_v" -r "$reps" -o json \
    > "$out/decode.json" 2> "$out/decode.log"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free,pstate,power.draw,clocks.sm \
    --format=csv,noheader > "$out/gpu-after.txt"
echo "receipts: $out"

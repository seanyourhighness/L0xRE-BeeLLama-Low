#!/usr/bin/env bash
# Short, comparable E3/W2 bridge prefill/decode screen on SM89 or SM120.
set -euo pipefail

if [[ $# -ne 3 || ( $1 != sm89 && $1 != sm120 ) ]]; then
    echo "usage: BIN_DIR=... E3_MODEL=... W2_MODEL=... $0 sm89|sm120 OUT_DIR REPS" >&2
    exit 2
fi

arch=$1
out=$2
reps=$3
: "${KV_K:=f16}"
: "${KV_V:=f16}"
: "${BIN_DIR:?set BIN_DIR to the matching build/bin directory}"
: "${E3_MODEL:?set E3_MODEL to the merged E3 GGUF}"
: "${W2_MODEL:?set W2_MODEL to the merged W2 GGUF}"
[[ $reps =~ ^[1-9][0-9]*$ ]] || { echo "REPS must be positive" >&2; exit 2; }
[[ ! -e $out ]] || { echo "refusing to overwrite $out" >&2; exit 2; }
mkdir -p "$out"

case $arch in
    sm89)
        bridge_root=${BRIDGE_ROOT:-/home/sean/kernel-lab/escha-sprint-rc-sm89-fast/bridge}
        export ESCHA_OFFICIAL_F32_DECODE_CUBIN="$bridge_root/code-gemm-sm89.cubin"
        unset ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI
        ;;
    sm120)
        bridge_root=${BRIDGE_ROOT:-/home/sean/kernel-lab5090/preview-bridge-sm120}
        export ESCHA_OFFICIAL_F32_DECODE_CUBIN="$bridge_root/f32_input.sm120.cubin"
        export ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32
        ;;
esac

export ESCHA_OFFICIAL_BRIDGE_LIBRARY="$bridge_root/libescha_official_bridge_cuda_${arch}.so"
export ESCHA_OFFICIAL_CODE_GEMM_CUBIN="$bridge_root/code-gemm-${arch}.cubin"
export ESCHA_GDN_CHUNK_BRIDGE_LIBRARY="$bridge_root/libescha_gdn_chunk_bridge_${arch}_ported.so"
export ESCHA_GDN_CHUNK_CUBIN_ROOT="$bridge_root/gdn-cubins-${arch}-exact"
export ESCHA_OFFICIAL_RAW_BRIDGE=1
export ESCHA_OFFICIAL_RAW_SWIGLU_FUSION=1
export ESCHA_OFFICIAL_RAW_DECODE_BRIDGE=1
export ESCHA_OFFICIAL_RAW_DECODE_F32_INPUT=1
export ESCHA_OFFICIAL_RAW_DECODE_SWIGLU_FUSION=1
export ESCHA_OFFICIAL_RAW_DECODE_K3_SPLIT_DIV2=0
export ESCHA_OFFICIAL_BRIDGE_ACCUMULATION=mixed
export ESCHA_GDN_CHUNK_BRIDGE=1
export ESCHA_E3_EMBED_GPU=1
export ESCHA_OFFICIAL_RAW_DECODE_DOWN_ADD_RMS_FUSION=1
export ESCHA_OFFICIAL_RAW_DECODE_K3_DOWN_SPLITS=12
export ESCHA_OFFICIAL_RAW_DECODE_QKV_Z_BETA_ALPHA_OVERLAP=1
export ESCHA_DIRECT_RECURRENT_STATE_R258=1
export ESCHA_DIRECT_RECURRENT_CONV_R260=1
export ESCHA_FUSE_GDN_DECODE_PREP_R262=1
export ESCHA_FUSE_GDN_NORM_GATE_R264=1
export ESCHA_W2_I8_HEAD=0
export ESCHA_OFFICIAL_RAW_DECODE_GATE_UP_OVERLAP=0
export ESCHA_OFFICIAL_RAW_DECODE_OVERLAP_GATE_SPLITS=4
export ESCHA_OFFICIAL_RAW_DECODE_OVERLAP_UP_FIRST=0
export ESCHA_OFFICIAL_RAW_DECODE_OVERLAP_UP_AUX=0
export ESCHA_OFFICIAL_RAW_DECODE_OVERLAP_AUX_HIGH_PRIORITY=0
export ESCHA_OFFICIAL_RAW_DECODE_QKV_Z_OVERLAP=1
export ESCHA_OFFICIAL_RAW_DECODE_ATTN_QKV_OVERLAP=1
unset ESCHA_OFFICIAL_BRIDGE GGML_CUDA_DISABLE_GRAPHS
export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for file in "$BIN_DIR/llama-bench" "$E3_MODEL" "$W2_MODEL" \
            "$ESCHA_OFFICIAL_BRIDGE_LIBRARY" "$ESCHA_OFFICIAL_CODE_GEMM_CUBIN" \
            "$ESCHA_OFFICIAL_F32_DECODE_CUBIN" "$ESCHA_GDN_CHUNK_BRIDGE_LIBRARY"; do
    [[ -s $file ]] || { echo "missing artifact: $file" >&2; exit 1; }
done

{
    printf 'arch=%s\nreps=%s\n' "$arch" "$reps"
    printf 'binary=%s\nbridge=%s\nf32_cubin=%s\nf32_abi=%s\n' \
        "$BIN_DIR/llama-bench" "$ESCHA_OFFICIAL_BRIDGE_LIBRARY" \
        "$ESCHA_OFFICIAL_F32_DECODE_CUBIN" "${ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI:-f16-convert}"
    printf 'prefill=-p 2048 -n 0 -b 2048 -ub 2048 -t 8 -ngl 99 -fa on -ctk %s -ctv %s\n' "$KV_K" "$KV_V"
    printf 'decode=-p 0 -n 256 -b 2048 -ub 512 -t 8 -ngl 99 -fa on -ctk %s -ctv %s\n' "$KV_K" "$KV_V"
    "$BIN_DIR/llama-bench" --help
} > "$out/config.txt" 2>&1
sha256sum "$BIN_DIR/llama-bench" "$BIN_DIR/libllama.so" \
    "$BIN_DIR/libggml-cuda.so" "$E3_MODEL" "$W2_MODEL" \
    "$ESCHA_OFFICIAL_BRIDGE_LIBRARY" "$ESCHA_OFFICIAL_CODE_GEMM_CUBIN" \
    "$ESCHA_OFFICIAL_F32_DECODE_CUBIN" "$ESCHA_GDN_CHUNK_BRIDGE_LIBRARY" \
    > "$out/SHA256SUMS.txt"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free,pstate,power.draw,clocks.sm \
    --format=csv,noheader > "$out/gpu-before.txt"

for model_name in e3 w2; do
    if [[ $model_name == e3 ]]; then model=$E3_MODEL; else model=$W2_MODEL; fi
    "$BIN_DIR/llama-bench" -m "$model" -p 2048 -n 0 -b 2048 -ub 2048 \
        -t 8 -ngl 99 -fa on -ctk "$KV_K" -ctv "$KV_V" -r "$reps" -o json \
        > "$out/${model_name}-prefill.json" 2> "$out/${model_name}-prefill.log"
    "$BIN_DIR/llama-bench" -m "$model" -p 0 -n 256 -b 2048 -ub 512 \
        -t 8 -ngl 99 -fa on -ctk "$KV_K" -ctv "$KV_V" -r "$reps" -o json \
        > "$out/${model_name}-decode.json" 2> "$out/${model_name}-decode.log"
done

nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free,pstate,power.draw,clocks.sm \
    --format=csv,noheader > "$out/gpu-after.txt"
echo "receipts: $out"

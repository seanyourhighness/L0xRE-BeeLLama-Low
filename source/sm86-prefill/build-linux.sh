#!/usr/bin/env bash
set -euo pipefail
# Full source checkout and retained backend/libggml-base must match the source ABI.
source_root="$(cd "$(dirname "$0")/../.." && pwd)"
output_dir="${1:?output directory}"; backend_dir="${2:?matching backend library directory}"
nvcc_path="${NVCC:-nvcc}"
mkdir -p "$output_dir"
include=(-I"$source_root/ggml/include" -I"$source_root/ggml/src" -I"$source_root/ggml/src/ggml-cuda")
flags=(-O3 -std=c++17 -arch=sm_86 --cudart=shared -Xcompiler=-fPIC)
"$nvcc_path" "${flags[@]}" -shared "${include[@]}" "$(dirname "$0")/bridge.cu" -lcublas -lcuda -o "$output_dir/libbridge-int8-allproj.so"
"$nvcc_path" "${flags[@]}" --use_fast_math --extended-lambda -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_KVARN -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -Dggml_cuda_EXPORTS -DNDEBUG "${include[@]}" -c "$(dirname "$0")/qk16-kernels.cu" -o "$output_dir/qk16-kernels.o"
"$nvcc_path" "${flags[@]}" --use_fast_math --extended-lambda -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_KVARN -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -Dggml_cuda_EXPORTS -DNDEBUG "${include[@]}" -c "$(dirname "$0")/qk16-host-wrapper.cu" -o "$output_dir/qk16-host-wrapper.o"
"$nvcc_path" "${flags[@]}" -shared -Xlinker=--no-undefined "$output_dir/qk16-kernels.o" "$output_dir/qk16-host-wrapper.o" -L"$backend_dir" -lggml-cuda -lggml-base -lcublas -lcuda -ldl -o "$output_dir/libqk16-context-gate.so"

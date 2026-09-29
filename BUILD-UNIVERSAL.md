# Universal B84 source and build provenance

Release: `beellama-v0.4.7-sm86-b84-universal-r1`.

The [committed SM86 / Windows source](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/tree/be302741801080a4fa2ea713fdf9409e2ef599d5) is `be302741801080a4fa2ea713fdf9409e2ef599d5` on `release/sm86-b84-universal`. It contains the opt-in head128 patch, B74 CUDA bridge and PTX reconstruction inputs, Windows export/build support, and the universal architecture-list build-script fix. The source worktree was clean when published.

## Linux / WSL

SM86 preserves the measured champion: BeeLLama baseline `ab1698c739b3bc13b9e06fb8febdc774ff106d23` plus the opt-in `head128-lowgpu.patch`, B74 bridge and vector cubin. The server itself remains the measured baseline executable; the head change resides in its CUDA backend. The portable launcher incorporates the B84 Q4-drafter overrides. Every component hash is in the Linux manifest and SM86 receipt.

Original SM86 compiler: CUDA 12.8.93 / GCC 13.3.0; Release, shared libraries, `CMAKE_CUDA_ARCHITECTURES=86`, `GGML_CUDA_FA=ON`, `GGML_CUDA_KVARN=ON`, `GGML_BACKEND_DL=OFF`, `GGML_NATIVE=ON`, prebuilt UI enabled. The original host was Z840. Native CPU compilation is a portability constraint; the release target requires AVX2-capable x86-64 hosts.

To build the committed source with an explicit compiler and portable CPU setting:

```bash
git clone --branch release/sm86-b84-universal https://github.com/seanyourhighness/L0xRE-BeeLLama-Low.git l0xre-source
cd l0xre-source
git checkout be302741801080a4fa2ea713fdf9409e2ef599d5
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_KVARN=ON -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF
cmake --build build-sm86 --parallel 8 --target llama-server llama-cli llama-bench
cd pocs/sm86-b74
python3 b71_vectorize_f32.py
python3 b72_vectorize_half.py
/path/to/cuda-12.8/bin/ptxas -arch=sm_86 k2k3-sm86-k3-f32-v4-half-v4.ptx -o k3-vector-all.cubin
/path/to/cuda-12.8/bin/nvcc -shared -Xcompiler=-fPIC -O3 -std=c++17 -arch=sm_86 \
  official_bridge_b74_k3_vector.cu -o libbridge-b74-k3-vector.so -lcuda
```

This gives a new build for qualification; it does not promise byte identity to the retained champion. The exact original head-backend relink replaced only the patched `lowgpu.cu` object, kept the original object list/flags, and compared stripped rebuilt loadable code/data. The package carries the source patch and bridge reconstruction material, not compiler caches.

SM89 uses the existing `beellama-sm89-v0.4.7-r1` payload and receipts; SM120 uses `beellama-sm120-v0.4.7-r9`. Their original build notes, source patches, model identities and license notices remain under their architecture directories. See [the original SM120 build](BUILD.md) for that separate baseline. The SM120 historical E3 model includes MTP tensors; the new Low adapter does not inherit a throughput claim for its different target file.

## Windows

Toolchain: CUDA 13.3, MSVC 19.44.35228.0, CMake / Ninja, Release, shared libraries, `GGML_NATIVE=OFF`, CUDA FlashAttention/KVarN enabled, default quant matrices, backend dynamic loading enabled, architectures `86;89;120` (the backend selects `120a` during configuration).

```powershell
git clone --branch release/sm86-b84-universal https://github.com/seanyourhighness/L0xRE-BeeLLama-Low.git l0xre-source
Set-Location l0xre-source
git checkout be302741801080a4fa2ea713fdf9409e2ef599d5
powershell -ExecutionPolicy Bypass -File scripts\build-win-l0xre.ps1 `
  -CudaArch '86;89;120' -Parallel 8 -Package
powershell -ExecutionPolicy Bypass -File pocs\sm86-b74\windows\build.ps1 `
  -OutDir C:\work\l0xre-bridge-build
```

The published Windows build reused the retained universal object cache after verifying the baseline flags/toolchain and matching 1,604 compiled-source/build files to the committed tree (CRLF normalized to LF). It rebuilt `lowgpu.cu` and the CUDA DLL, compiled the new bridge, then rebuilt/relinked all affected host tools with explicit source-export metadata. The executable reports build 11 / commit `be302741801080a4fa2ea713fdf9409e2ef599d5` without a dirty suffix. The verified source-export CMake flags were `LLAMA_BUILD_COMMIT=<full SHA>`, `LLAMA_BUILD_NUMBER=11`, `LLAMA_BUILD_DIRTY=0`; a fresh Git checkout obtains its metadata normally.

The new bridge's CUDA source matches Linux SHA-256 `40b162dd48d41ca585968043a492d632ec0395be2029f3535c797474b3ce59c8`. It exports the existing 10-symbol C ABI and contains two CUDA ELF entries per architecture. The build script checks actual `nvcc --list-gpu-code` capabilities and fails if a requested architecture or export is missing. MSVC dynamic CRT linkage is kept consistent with `/MD` and `/NODEFAULTLIB:LIBCMT`; the final bridge build emitted no compiler/linker warnings.

Package the new bridge at `bridge/sm86/bridge-b74-k3-vector.dll` and the SM86 vector cubin alongside it. Only the SM86 launcher sets `ESCHA_E3_HEAD_RT_BLOCK128` and `L0XRE_K3_VECTOR_CUBIN`. The other architectures use their retained bridge paths. Bundle the CUDA, MSVC and OpenMP runtime DLLs, including `vcomp140.dll`.

## Validation and release gates

See [VALIDATION.json](evidence/universal/VALIDATION.json). CPU regression checks cover KVarN, LowGPU metadata, KV-tail requests and fit/tail logic. Launcher integration tests use stub executables and test routing, arguments, paths with spaces, invalid input and architecture isolation. ELF dependency/version and PE import/export/architecture checks are separate from GPU execution.

Archive integrity checks run after sealing: clean extraction, internal file SHA-256 verification, safe paths/symlinks, and exact archive checksums. Public model LFS hashes are checked at the pinned HF revision. No model weights are changed by this release.

Hardware gates still pending: Windows inference/performance/headroom on SM86/SM89/SM120; clean-extract SM86 inference through the portable wrapper; SM120 common-CLI inference with the non-MTP Low target. The release remains a candidate. Prior benchmark evidence is attributed to the exact components and settings, never to all GPUs or Windows.

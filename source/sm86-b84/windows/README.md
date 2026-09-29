# Windows bridge: `bridge-b74-k3-vector.dll`

Windows build of the L0xRE SM86 B74 decode bridge used by the universal Windows
runtime package.

## Purpose

The v0.4.7 SM86 champion is not a preset-only change. It pairs the head128
`lowgpu.cu` tuning with a bridge that can route K3 decode to the vectorized
FP32-input cubin selected by `L0XRE_K3_VECTOR_CUBIN`. This directory builds that
bridge for Windows and verifies the result before it is packaged.

The CUDA source is byte-identical to the reviewed Linux source, so the CUDA math
is shared rather than re-implemented:

| File | Role | SHA-256 |
| --- | --- | --- |
| `official_bridge_b74_k3_vector.cu` | bridge source, identical to the Linux source | `40b162dd48d41ca585968043a492d632ec0395be2029f3535c797474b3ce59c8` |
| `escha_official_bridge_v1.h` | shared ABI header, identical to the retained Windows tree | `6cc9fae65e2d68cacdf78a001c8eda78004594277fcd820815dc3d894fa423db` |
| `bridge-b74-k3-vector.def` | export list, the same 10 symbols as the retained `official.def` | - |
| `build.ps1` | build and verification script | - |

## Requirements

- Windows x64 with the MSVC toolchain (Visual Studio 2022 Build Tools tested).
- CUDA toolkit: 12.8 or newer when targeting `sm_120`; 13.3 is used here. The
  script confirms support with `nvcc --list-gpu-code`, so an older toolkit fails
  with the list of supported codes instead of a linker error.
- PowerShell 5.1 or newer.

## Build

Build all shipped architectures into an isolated output directory:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 `
  -OutDir C:\work\l0xre-win\sm86-b74-port-build\out
```

Build a single architecture, or point at a specific toolkit:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -CudaArch 86 -OutDir .\out-sm86
powershell -ExecutionPolicy Bypass -File build.ps1 -CudaRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
```

`-CudaArch` accepts `86`, `89` and `120`, and defaults to all three. The script
locates the newest installed CUDA toolkit and `vcvarsall.bat` automatically
(`vswhere` first, then well-known paths); pass `-CudaRoot` or `-VcVarsAll` to
override. The equivalent raw command is produced in `build-bridge.bat` inside
`-OutDir` when `-KeepBat` is given.

## What the script verifies

1. Every requested architecture is supported by the selected toolkit, checked
   against `nvcc --list-gpu-code` rather than a version guess.
2. The DLL exports all 10 bridge ABI symbols (`dumpbin /exports`); a missing
   symbol fails the build.
3. `cuobjdump -lelf` lists every requested architecture; an architecture with no
   device code fails the build. Each architecture is expected to show two ELF
   entries, one per device kernel.
4. The import table and any compiler warnings are printed. The script exits
   non-zero on any build or verification failure.

## Build record (verified on this workstation)

```
dll    : out3\bridge-b74-k3-vector.dll
bytes  : 147968
sha256 : 0c4b87fb0805c74fd230596ab86e4121a589b506a7a380bc568243b6e5888d11
arch   : 86;89;120
compile warnings: none
exports: 10/10 present
arch sm_86: 2 ELF entries
arch sm_89: 2 ELF entries
arch sm_120: 2 ELF entries
imports: MSVCP140.dll, VCRUNTIME140.dll, VCRUNTIME140_1.dll, KERNEL32.dll, api-ms-win-crt-*.dll
```

Toolchain: CUDA 13.3.33 nvcc, MSVC 14.44.35207, `-O2 -shared -Xcompiler /MD`.
The `nvcc`/`link.exe` output is not bit-reproducible: repeated builds produce the
same size, exports, architectures and imports with different file hashes. Record
the hash of the artifact that is packaged.

### CRT note

The generated host object carries `/DEFAULTLIB:MSVCRT` (dynamic CRT, consistent
with `/MD`), but a CUDA static library also requested `LIBCMT`, which raised
`LNK4098: defaultlib LIBCMT conflicts`. The DLL imports the dynamic CRT set and
none of the static CRT, so the build adds `-Xlinker /NODEFAULTLIB:LIBCMT`; the
link is then verified through the reported import list. `--cudart shared` is
deprecated in CUDA 13.3 and was measured to change neither the imports nor the
warning, so it is not used.

## Integration

Deploy the DLL as `bridge\sm86\bridge-b74-k3-vector.dll` (and
`bridge\sm86\k3-vector-all.cubin`) in the universal package, alongside the
retained `bridge\escha_official_bridge_cuda.dll`. The SM86 profile of the
launcher sets:

```powershell
$env:ESCHA_OFFICIAL_BRIDGE_LIBRARY = "$BRIDGE\sm86\bridge-b74-k3-vector.dll"
$env:L0XRE_K3_VECTOR_CUBIN          = "$BRIDGE\sm86\k3-vector-all.cubin"
```

Leave `ESCHA_OFFICIAL_CODE_GEMM_CUBIN` and `ESCHA_OFFICIAL_F32_DECODE_CUBIN`
pointing at `code-gemm.cubin`.

`L0XRE_K3_VECTOR_CUBIN` is SM86-only: the cubin is `sm_86` SASS with no PTX
fallback, so on SM89 or SM120 the driver rejects it. When the variable is unset
the bridge loads `ESCHA_OFFICIAL_F32_DECODE_CUBIN` and behaves exactly like the
retained bridge, which is the rollback path.

## Verification scope

Source, build and PE verification only. No GPU execution was performed for this
artifact, so Windows SM86 B84 remains hardware-unqualified until the
deterministic decode and the 92,879-token greedy output hash
`8f9908604bc899ee40ed5dcb4136c70807edca6438c7320f050e5be99c9b69d2` are
reproduced on a 12 GB RTX 3060.

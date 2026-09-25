# BeeLlama Escha multi-architecture release roadmap

Status: planning and offline preparation; no release has been published.
Updated: 2026-09-23. The v0.4.7 Preview SM89 E3 build and draft-memory probe are
recorded in [the port note](escha-v047-preview-port.md). The concrete staged
Linux/WSL prerelease, per-model five-percent gate, user install/report contract,
and future BeeLLaMA rebase loop are in
[the staged prerelease plan](escha-v047-staged-prerelease.md).

## Goal

Publish the BeeLlama fork with first-class Escha E3/W2 support for SM86, SM89,
and SM120. Put out a Linux/WSL Preview first. After the Preview, make the
Windows prebuilt runtime and the bundled 0xRCA/agent executables the main
release target.

This is one BeeLlama/Escha source line with architecture-aware runtime assets.
SM86, SM89, and SM120 backend and bridge objects must be identified by their
actual target architecture. A successful source build is not a hardware
qualification claim.

## Release sequence

### 1. Publish the source fork

Prepare one reviewable source snapshot and publish the repository/fork before
publishing consumer binaries. The source publication should include the Escha
loader, graph, GGUF conversion path, CPU reference path, CUDA kernels, build
instructions, third-party notices, and the architecture support matrix.

The working tree is not ready to tag today:

- Branch: `escha-rc-v046-full`, HEAD `2d1411c2c9eaca7d92bdd6a84279e34e580c4cf7`.
- Tracked local changes are in `ggml/src/ggml-cuda/escha-moe.cu`,
  `ggml/src/ggml-cuda/lowgpu.cu`, and `src/models/qwen35.cpp` (513 added and 38
  removed lines across those files).
- The SM120 preview project and ledger are untracked local documents.
- The SM89 champion receipt names a source manifest and archive, but its Git
  commit is unavailable because the parent worktree metadata was deleted.

Before publishing, review and preserve those changes in a clean commit, record
the full source hash/manifest, and connect each build receipt to that exact
source identity. Do not retag the dirty working tree.

The repository's current release automation dispatches preview builds from
`v*` branches and stable builds from `v*` tags. Its Ubuntu and Windows CUDA
jobs do not set `CMAKE_CUDA_ARCHITECTURES`; a runner without a GPU therefore
does not prove that the resulting backend contains SM86, SM89, or SM120 code.
The release workflow needs explicit architecture targets and architecture
names in its cache keys and asset manifests before its CUDA assets can serve as
the multi-architecture release pipeline.

The current preview dispatcher also launches the full platform build. Add a
Linux-only Preview scope (or a dedicated Linux Preview workflow) so the first
binary Preview does not publish Windows assets ahead of the Windows release
work.

### 2. Publish a Linux/WSL Preview

Make the first consumer Preview a Linux x86-64 package usable from WSL. Keep
the Windows package out of this Preview. Build from the published source SHA,
not from a neighboring lab tree.

Package the runtime and each bridge as a matched set. Use separate
architecture-specific backend and bridge payloads for the first Preview;
reconsider a combined CUDA fat binary only after the separate artifacts qualify.
The official bridge files currently available are Linux `.so` files and
architecture-specific cubins. Keep each wrapper, cubin, GDN bridge, manifest,
and selector together. Do not pair a wrapper from one Preview with receipts
from another.

Keep model files outside the source archive. Publish model URLs, license and
provenance, hashes, supported runtime version, and tested configuration in the
Preview notes.

Preview qualification must be attached to the exact source, build, runtime,
model, bridge, profile, and launcher hashes. E3 and W2 must load and pass
deterministic correctness; DFlash2/KVarN paths must pass where advertised;
memory/context behavior and package startup must match the published profile.
Label any untested architecture as build-only or pending hardware validation.

### 3. Make Windows the main post-Preview target

Use the existing 0xRCA Windows product as the package shell. Its build script
produces the root `0xrca.exe` supervisor, the Goose `goose.exe` backend, and the
packaged desktop executable at
`agent/goose/ui/desktop/out/0xrca-win32-x64/0xrca.exe`. The current script does
not build BeeLlama; the qualified BeeLlama runtime and matching CUDA backend
must be staged into the package before producing the final Windows archive.

The Windows package needs a GPU capability selector that maps SM86, SM89, and
SM120 to a matching CUDA backend and Escha bridge payload, with a clear error
if no matching payload is installed. Keep memory profile choice separate from
compute capability: for example, an SM86 card can have 12 GB or 24 GB of VRAM.
Treat the existing 12 GB and 16 GB 0xRCA profiles as presets until measured on
their target hardware.

The 0xRCA shell already detects compute capability and loads architecture/VRAM
profiles, but its current profiles all point to the same runtime path and the
supervisor does not set `GGML_BACKEND_PATH`. BeeLlama's dynamic backend loader
does accept that variable for loading an out-of-tree CUDA backend. The new
package can therefore keep one server executable and select a staged backend
DLL by the detected SM. Update the supervisor/profile contract and ensure the
generic runtime directory does not also load a second CUDA backend.

There is a profile-selection edge case to fix with that work: 0xRCA first
looks for an exact SM and exact VRAM bucket, then falls back to generic
`16gb`/`12gb`. A 32 GB SM120 card does not currently select
`sm120-16gb.yaml`; it falls back to a generic profile. Select the nearest lower
VRAM profile for the exact SM before using generic profiles.

The official Escha bridge currently rejects Windows and is distributed as
Linux `.so` files. Before claiming bridge acceleration on Windows, either port
the bridge ABI to a Windows DLL and qualify it, or ship a documented native
runtime path that does not claim bridge acceleration. The Linux Preview can
proceed independently of that Windows blocker.

Build the Windows runtime, CUDA DLLs, 0xRCA supervisor, and desktop/agent
package from the same published source release. Include per-file SHA-256
checksums, a `MANIFEST.json` with source SHA, toolchain, target architecture,
runtime and bridge hashes, config hash, and receipt IDs, plus a consumer
quickstart and model download/verification instructions.

## Current evidence by architecture

| Target | Evidence already available | Still open |
|---|---|---|
| SM86 / RTX 30-series | The Escha and LowGPU CUDA files compiled for `sm_86`, `sm_89`, and `sm_120a`; the changed Qwen35 host file also compiled. The retained Escha wheel has 56 SM86 `escham_code_gemm_kernel` variants. Its direct-F32 decode PTX was retargeted and assembled for SM86. Both bridge wrapper sources compile into SM86-targeted shared libraries, and six of seven GDN payloads compiled offline. | The solve GDN cubin is still missing from the SM86 set. The candidate artifacts are unqualified: no module-load/model run, linked BeeLlama server, or GPU runtime receipt exists. Qualify on actual SM86 hardware before claiming runtime support. |
| SM89 / RTX 40-series | The 4090 champion release has extensive E3/W2 correctness and performance receipts. The current RC also has an SM89 build/correctness record. | The champion's source commit is unavailable; its CUDA 12.8 artifact is not the same source/build identity as the current SM120 Preview. Rebuild from the published source SHA and attach fresh matched receipts. |
| SM120 / RTX 50-series | `build-sm120-preview` is configured for architecture 120 with CUDA 13.0. Existing RTX 5090 receipts show bounded W2/E3 bridge controls and 20/20 top-20 logit overlap at all four probed positions. Direct-F32 decode measured 82.37 +/- 2.01 tok/s W2 and 84.63 +/- 3.38 E3; p2048 prefill measured 2902.77 +/- 80.69 W2 and 2758.21 +/- 83.78 E3. | These receipts predate the current dirty source edits. The p2048 results are below the 3000 tok/s Preview target for both models, especially E3. Full long-context/quality gates remain pending; reconcile the wrapper/manifest mismatch and rerun from the published SHA before publishing. |

The Windows 0xRCA prototype has an older BeeLlama runtime smoke on an RTX 5090.
That establishes the product shell's SM120 detection/lifecycle path for that
older runtime; it does not qualify the Escha v0.4.6 runtime or the 12/16 GB
profiles.

## Offline work queue

1. **Initial source-diff review complete (2026-09-22):** the tracked changes add
   advanced Escha generation kernels, a decode-only W2 INT8 head path, optional
   GDN decode preparation/overlap, and E3 LowGPU embedding/output routes for MTP.
   Preserve the working changes while the two SM120 preview documents and open
   RC gates are reconciled before making the public source snapshot.
2. **Partially complete (2026-09-22):** configured the current dirty source at
   `/tmp/escha-sm86-offline-20260922` with CUDA 13.0 and
   `CMAKE_CUDA_ARCHITECTURES=86;89;120`. CMake resolved this to
   `86;89;120a`. The changed Escha and LowGPU CUDA translation units compiled
   and each object contains `sm_86`, `sm_89`, and `sm_120a` cubins; the changed
   `src/models/qwen35.cpp` also compiled as a host object. The three-file tracked
   source diff used by this compile hashes to
   `3686d40b6a2d66fc13c997b4959818f91d9a1f950d95f63a365a33a7ca229f56`.
   A broader backend build was interrupted during the large KVarN FlashAttention
   template matrix; there is no linked server and this is not a complete build
   receipt. No GPU was used.
3. **Bridge inventory and partial SM86 port (2026-09-22):** the retained Escha wheel at
   `/home/sean/escha-venv/lib/python3.12/site-packages/escha/_C.cpython-312-x86_64-linux-gnu.so`
   contains SM86, SM89, and SM120 code-GEMM ELF cubins. `cuobjdump` found 56
   SM86 `escham_code_gemm_kernel` variants and extracted nine SM86 cubins to
   `/tmp/escha-sm86-bridge-extract-20260922`. The retained direct-F32 PTX was
   retargeted from `.target sm_89` to `.target sm_86` and assembled by CUDA 13.0
   `ptxas`. Both the official bridge wrapper and GDN wrapper compiled as
   SM86-targeted shared libraries. Six of seven GDN cubins compiled offline
   from the retained SM89 PTX/state source; the generated `solve_tril` cubin is
   still missing. All artifacts are scratch candidates only; no bridge was
   loaded or launched on a GPU.
4. Add a Linux-only Preview path to the release workflow. The existing
   `package-assets` job requires every OS/backend job and `release` also waits
   for Docker publishing, so skipping Windows jobs would currently block the
   aggregate. Make CUDA builds target explicit SM86, SM89, and SM120
   architectures and emit distinguishable artifact names/manifests. Schedule
   Windows product packaging after the Preview gate.
5. Update 0xRCA profile selection to prefer the nearest lower VRAM bucket for
   the exact SM; add a backend path to the launch profile and set
   `GGML_BACKEND_PATH` to the matching staged CUDA DLL. Keep VRAM profile
   choice independent from architecture selection.
6. Reconcile the SM89 and SM120 build environments and bridge manifests against
   one published source SHA. Produce a source manifest and release manifest
   template before preparing a public Preview.
7. After the source fork and Linux/WSL Preview are published, stage the
   matching Windows runtime, CUDA payloads, and Agent/desktop executables in
   the 0xRCA package, then complete Windows bridge/fallback and hardware gates.

## Promotion gates

- Every asset identifies the exact source SHA and all binary/model/bridge/config
  hashes.
- E3 and W2 load and produce deterministic, parity-checked output on every
  architecture claimed as supported.
- DFlash2, KVarN3/KVarN2, context, and memory behavior match the advertised
  profile.
- Performance is measured on the target hardware with production settings and
  without profiling instrumentation.
- Bridge loading is proven on the target OS and architecture; an unavailable
  bridge fails clearly or uses the explicitly documented native path.
- The consumer package starts through its shipped launcher and cleanly stops
  BeeLlama and the bundled agent.
- SM86 support and Windows bridge acceleration are not claimed from compile-only
  or cross-architecture evidence.

## Related plans and runbooks

- [v0.4.7 staged prerelease plan](escha-v047-staged-prerelease.md)
- [v0.4.7 Preview SM89 port and tuning receipt](escha-v047-preview-port.md)
- [SM120 Preview project](escha-rc-v046-sm120-preview-project.md)
- [SM120 Preview ledger](escha-rc-v046-sm120-preview-ledger.md)
- [RC gate runbook](escha-rc-v046-gate-runbook.md)
- [BeeLlama release workflow](release.md)
- GBrain: “Escha BeeLlama v0.4.6 unified multi-architecture release plan”
- GBrain: “Escha SM120 stretch plan — 90 decode / 3500 prefill”
- GBrain: “0xrca release — M1 core verified, project organized”

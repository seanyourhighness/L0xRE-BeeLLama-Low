# Escha staged prerelease on BeeLLaMA v0.4.7 Preview

Status: release design and execution order, 2026-09-23. No assets have been published.
This narrows the [multi-architecture roadmap](escha-multiarch-release-roadmap.md)
to a first Linux/WSL user Preview. The SM89 E3 build and draft-memory probe are
recorded in [the v0.4.7 port note](escha-v047-preview-port.md).

## Release contract

One pinned BeeLLaMA commit plus one reviewable Escha patch stack produces one
source SHA. Every runtime asset for a prerelease points to that source SHA.
Build jobs may use architecture-specific CUDA toolchains, but each asset records
its exact compiler, CUDA, target cubin architecture, runtime and bridge hashes,
model revisions, and qualification receipt IDs. Never select a binary solely
because its filename says `sm89` or because a previous release used the same
library name. The first Preview bundles a complete runtime and matching bridge
set per architecture; the small duplication avoids cross-package ABI mistakes.

The present candidate is an uncommitted merge and cannot be a reproducible
public source release yet. First convert its Escha-only changes and the recovered
SM89 champion patches into a small, reviewed sequence of commits on the pinned
BeeLLaMA commit. Do not copy the champion's whole source tree into a future
update. Preserve the frozen v0.4.6 champion and current v0.4.7 receipts.

## Stages

| Stage | User-facing claim | Gate before publication |
|---|---|---|
| P0: source + SM89 Linux/WSL | E3 and W2 on qualified RTX 40-series profiles | Rebuild both from the committed v0.4.7 snapshot; complete matched model/bridge, parity, speed, memory and packaged-install receipts. The current SM89 E3 18K probe is an input, not the W2/full-context gate. |
| P1: add SM120 Linux/WSL | E3 and W2 on tested RTX 50-series | Build CUDA 13.x `120a` payload from the same source SHA and selected wrapper/cubins. On RTX 5090 rerun both models and the bridge ABI/control gates; compare against the frozen same-card SM120 reference. A previous v0.4.6 SM120 preview cannot qualify v0.4.7. |
| P2: add SM86 Linux/WSL | E3 and W2 on tested RTX 30-series | Build `sm_86` runtime and bridge from distributable sources, complete the missing GDN `solve_tril` payload, then run both models on real SM86 hardware. The current extracted/retargeted scratch cubins are build research, not a shippable bridge. |
| P3: Windows + 0xRCA | Consumer package with architecture selection | Stage native Windows runtime, CUDA DLL and bridge DLL or an explicitly labeled unaccelerated path; finish supervisor backend selection and Windows fit/performance gates. |

Publish P0, P1 and P2 as separate **immutable prerelease tags**, each with a
complete manifest and only supported architecture assets. Keep the source SHA
the same if only qualified architecture coverage expands. If code or profile
changes, mint a new source SHA and prerelease tag; never replace an old asset
in place. Do not use the upstream rolling `preview-v0.4.7` tag, moving Docker
image, or this repository's current force-updated `preview-*` tag as the
consumer version identity.

Recommended names: `escha-v0.4.7-p0`, `-p1`, `-p2`, then a new series when the
BeeLLaMA base moves. Labels should state **verified**, **experimental**, or
**build-only** per architecture and model. A build-only payload is downloadable
for developers but excluded from the default installer and support claim.

## Compatibility and the five-percent gate

Correctness and speed are different gates. For E3 and W2 **independently**:

1. Confirm model and bridge hashes, model load, explicit bridge selection,
   DFlash2/MTP applicability, cache route, fixed-prompt output and selected
   logits against the frozen same-model reference. Compare known nondeterministic
   long-prompt cases by behavior/logit criteria rather than a single text hash.
2. Run packaged executable and bundled launcher, then short, 2K, 18K and a
   long-context request, plus multi-turn/prompt-cache and stop/restart cases.
   Include the actual consumer VRAM profile; record peak GPU memory, host RSS,
   driver, card, context, batch/ubatch, draft acceptance, bridge load and errors.
3. Measure production prefill and decode on the **same GPU, model files,
   profile, prompt set and harness** as a frozen reference, using warmups and
   at least five paired runs. Compare each model/workload cell separately.
   Target a candidate/reference throughput ratio of at least 0.95; rerun a
   borderline cell with more pairs rather than hiding it in an E3/W2 mean.
   A large gain in one cell does not excuse a greater-than-five-percent loss
   in another advertised profile. Document any intentional slower low-memory
   preset separately from the speed preset.

SM89 has a v0.4.6 champion on the same RTX 4090. SM120 has older bounded
v0.4.6 receipts but needs a newly frozen full reference for each claimed
profile. SM86 has no qualified same-card Escha baseline yet; establish one
before claiming a five-percent regression result. Raw RTX 4090 versus RTX
5090 token rates, and raw E3 versus W2 rates, are not parity comparisons.
The user's five-percent target is a **per-model same-hardware speed guardrail**;
output correctness remains a separate hard gate.

## Build and package shape

Use a small dedicated `escha-preview` workflow, separate from the existing
all-platform BeeLLaMA release graph. Its inputs are an immutable source SHA,
prerelease stage, and publish=false by default. Matrix rows specify `sm_89`,
`sm_120a`, and eventually `sm_86` explicitly, with CUDA toolchain and cache key
including architecture. Keep CUDA 12.8 for the currently working SM89 line and
CUDA 13.x for SM120 initially; qualify any later toolchain unification as a
separate change. Build-only CI jobs can cross-compile; a hardware receipt must
come from the real target GPU before that row becomes verified. Assemble only
the verified rows, then make a draft release for human review before publish.

Each Linux asset is a self-contained directory:

```text
escha-preview-<tag>-linux-x86_64-sm89/
  escha                 # launcher/doctor entry point
  bin/                  # llama-server, llama-cli and matching shared libraries
  bridge/               # wrapper, code-GEMM/F32 and GDN cubins for this SM
  profiles/             # E3/W2 profile presets; context and VRAM are separate from SM
  MANIFEST.json         # source/toolchain/file/model/profile/receipt identities
  SHA256SUMS
  QUICKSTART.md
  LICENSES/
```

SM120 uses its own wrapper and `120a` cubins; SM86 uses its own independently
built bridge. Do not mix wrappers or cubins across assets. Use `$ORIGIN` runtime
paths and paths relative to the installed bundle, not lab-home absolute paths.
Preflight every required bridge file and expected SHA before starting a model;
fail with the missing component and expected architecture. Download E3/W2 model
files separately with pinned revision and SHA from the manifest. The user picks
model and memory profile after GPU detection; GPU architecture never implies
12 GiB versus 24/32 GiB memory. The v0.4.7 E3 96K low-memory candidate sets
`--spec-draft-ubatch-size 64`; the 80K Q4 profile should explicitly retain
`64` until its own comparison is complete.

## Install and feedback experience

The release page should put one short Linux/WSL path first: download the asset
for the detected GPU, verify `SHA256SUMS`, unpack, run `./escha doctor`, add an
E3 or W2 model with the manifest's revision/hash, then run `./escha serve
--model e3 --profile auto`. The proposed `escha` entry point should implement
those commands before this quickstart is published. Provide a manual asset
selection path for systems where `nvidia-smi` is unavailable. The package
should print the selected SM, model, VRAM profile, runtime hash, bridge hash,
and source SHA at startup.

`./escha doctor --report report.json` should create a **local, opt-in** support
report containing version/source SHA, asset and profile IDs, OS/WSL, GPU model
and VRAM, driver/CUDA, loaded backend and bridge hashes, model revision/hash
status, startup/health result, peak GPU memory, a short local performance
smoke, and sanitized error categories. Omit prompts, generated content, API
keys, environment values, usernames, full file paths, GPU serials and IPs.
The report is never uploaded automatically. Release notes and the issue form
should ask users to attach it, choose E3/W2 and stage, report install versus
runtime failure, and include actual versus expected behavior. Add a separate
opt-in quality/performance issue template so install problems remain easy to
triage. Include a 30-second rollback instruction to reinstall the previous
immutable asset.

## Rebase loop for future BeeLLaMA updates

1. Fetch the upstream release and pin its **commit SHA**. Create a fresh
   disposable integration worktree; keep published branches/tags, manifests and
   reference binaries untouched. Rebasing is confined to this unpublished
   integration line, so users' source links and old receipts never move.
2. Replay the Escha feature-commit stack on that base and inspect
   `git range-diff` plus conflicts in model loading, DFlash, graph building,
   KVarN, CUDA, and backend ABI. The v0.4.7 LowGPU DFlash and raw bridge
   repairs show why a clean compile alone is insufficient.
3. Build all three CUDA targets from the same new source SHA. Run the common
   E3/W2 correctness matrix first, then paired same-card performance and
   memory gates. Publish a new immutable prerelease tag only for rows that
   pass, with updated per-asset receipts. Keep prior release available for
   rollback and for a direct user comparison.

The current `.github/workflows/release.yml` sets no explicit CUDA architecture
in its Ubuntu/Windows CUDA jobs, and its `package-assets` and `release` jobs
depend on all platforms and Docker. The dedicated workflow avoids blocking a
small Linux Preview on unrelated jobs. The current preview publication also
force-updates its tag/assets; the Escha prerelease must use immutable tags.

# S71 fast runtime: one source overlay for SM86, SM89 and SM120

This directory contains the exact frozen S71 source additions: modified rejection verification, fused output head, projection-transform reuse, GDN512 masked-prefill scratch, and the FP16-attention/INT8-prefill bridge. The target weights and original reasoning guard are unchanged. The source profile is shared across all three architectures; hardware certification is tracked separately.

## Reproduce the source

Download `l0xre-fast-runtime-source-v0.4.7.tar.xz` from the `beellama-v0.4.7-universal-r5-fast` release. Its SHA-256 is `a9dd23bfa3ed38c040ffed6447e7085e083cb6abbb640a5b0cb3cf48f7de7263`. The universal archive also contains that base snapshot at `source/sm86-r5/runtime-source.tar.xz`.

```bash
python3 source/fast/prepare-source.py \
  --archive l0xre-fast-runtime-source-v0.4.7.tar.xz \
  --destination /path/to/new-fast-source
```

The destination must be empty. Preparation verifies the base archive and writes `FAST-SOURCE.json`, pinning every overlaid file. The same source preparation serves SM86, SM89 and SM120.

## Compile an architecture

CUDA 13.0 was used for the three architecture compilation checks. CUDA 12.8 remains the toolchain for the certified SM86 runtime bytes. A cross-build is a new artifact and needs its own hardware qualification.

```bash
# Bounded kernel/source preparation; no GPU execution, no installable runtime.
python3 source/fast/build-parity.py --arch sm89 \
  --source /path/to/new-fast-source --cuda /usr/local/cuda \
  --output /path/to/sm89-fast

# Full runtime build from that same prepared source.
python3 source/fast/build-parity.py --arch sm89 --runtime \
  --source /path/to/new-fast-source --build /path/to/build-sm89 \
  --cuda /usr/local/cuda --output /path/to/sm89-fast --jobs 4
```

Use `--arch sm86` or `--arch sm120` for the other targets. The build retargets and assembles the projection and masked-GDN PTX, compiles both bridges and the fused-head templates, and writes `BUILD-PARITY.json` with artifact hashes. A full runtime build also compiles the overlaid sampler/server and CUDA backend, then the attention interposer against that matching backend ABI.

The output still needs its inherited, architecture-matching code-GEMM and unmasked GDN fallback bridge/cubins, matching CUDA runtime libraries, and a sealed `FAST-MANIFEST.json`. Do not mix the SM86 sampler/server libraries with an older SM89/SM120 core. `installable_fast_payload` and `hardware_qualified` remain false in build receipts until packaging and hardware gates are complete.

## SM89 update handoff

The SM89 transform bridge, GDN512 bridge, fused-head templates, projection PTX and all seven masked-GDN kernels compiled successfully. See `evidence/parity/sm89-compile.json`. SM86 and SM120 passed the same checks. The RTX4090 was absent from Z840 during preparation, so no SM89 inference or throughput result is claimed.

Next gates for each newly built architecture:

1. Seal all executable/library/cubin hashes and verify ELF dependencies and GPU architecture coverage.
2. Test a clean package launch, greedy code/prose, production sampling, and CPU vision. Require the expected target/draft file hashes and one-slot80K profile.
3. Run the frozen Bench.sh decode-only reference with seeds42/1234, three warmups and five measured requests per code/narrative prompt per boot, with balanced baseline/candidate ordering. Keep both seeds and all failures.
4. Test10K/77K prefill and81,916-token capacity without truncation or CUDA errors. Compare prefill against that architecture's baseline.
5. Run the same full150 cases, production medium effort .7/.95/k20/minp.05, up to three attempts. Report pass@1, pass@3, per-pack losses, guard events and invalid outputs independently.
6. Install only after package, performance and quality review pass. Preserve that architecture's prior unit and package for rollback.

## Parity boundary

Source/profile parity and compilation parity are prepared across SM86/89/120. Only the original frozen SM86 S71 payload is hardware-certified in this release. Inherited SM89/SM120 binaries are retained for existing profiles; `--profile fast` fails explicitly on those routes until a complete fast payload is installed. The release does not silently run SM86 SASS on another GPU or claim old SM89/SM120 measurements for the new code.

Windows remains at its previous release. This preparation targets Linux/WSL; Windows needs the corresponding native bridge/backend build and Windows hardware checks.

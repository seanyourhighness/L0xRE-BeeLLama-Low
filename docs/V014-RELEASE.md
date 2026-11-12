# L0xRE v0.1.4 — P2 runtime (draft)

This release packages the P2 stack promoted on the RTX 3060 on November 10: decode fusions and CUDA copy batching, backend-sampling reserve correction, reasoning-loop guard correction, cache reuse Fix 1, incremental tokenization and P1–P4/P2 append and render improvements. Model weights are unchanged.

Frozen source: `e7c3a18defbfadb96a877d923e4b1c9c3157d530`.

## Available packages and coverage

| Package | Current coverage | Draft availability |
|---|---|---|
| Windows SM86 | Build-verified: checksums, launcher tests, install/update/rollback, every DLL import and native load; no matching hardware run | Available |
| Windows SM89 | Build-verified: checksums, launcher tests, install/update/rollback, every DLL import and native load; no matching hardware run | Available |
| Linux SM86 | Production bytes packaged; GPU reference checks pending | Pending |
| Linux SM89 | Selective CUDA/bridge rebuild; 4090 smoke checks pending | Pending |
| Linux SM120 (WSL) | Selective CUDA/bridge rebuild; 5090 reference checks pending | Pending |
| Windows SM120 | Host, selective CUDA and bridge rebuild; 5090 reference checks pending | Pending |

The catalog retains v0.1.3.2 for unfinished targets. No v0.1.4 speed or quality improvement is claimed from build-only coverage.

## Default flags

All presets use `--backend-sampling`, `--cache-ram 8192`, `--temp 1.0` (previously 0.7) and `--min-p 0.0` (previously 0.05). Existing `--top-p 0.95`, `--top-k 20`, `--reasoning off`, `--reasoning-effort none` and `--reasoning-budget 0` remain. The control vector is opt-in. Architecture-specific arenas, context capacity, thread/affinity selection and GPU vision settings remain as specified by each preset. These defaults reproduce the production 3060 behavior; the owner can object before publication.

## Validation and limits

Windows SM86/SM89 carry the new code with build-only qualification. Native launcher tests retain the two known stale preflight/consent failures named in the release plan; the remaining checks pass. No Windows recovery-profile claim is made (existing platform exception).

The three Linux packages share the exact production host libraries. A source-archive host rebuild completed, but its three library hashes differ from production; they are not presented as reproducible matches. SM86's CUDA backend and bridge are the promoted incremental builds. SM89/SM120 recompile two CUDA units and the main bridge with their pinned CUDA toolchains; unchanged objects are hash-verified against the retained build witness.

P2 CPU render/tokenization proofs cover the shared Linux host bytes. Carry-over of 3060 quality/speed still requires the release checks to confirm the packaged runtime and effective arguments. The P2.1 warm-overhead waiver stays bound to the promoted production bytes; it is not transferred to Windows or other newly compiled CUDA backends.

## Install and rollback

Windows: run `.\install.cmd`; an existing installation uses `update.cmd`. Linux/WSL: use `install.sh` or `install.py --update`. The previous verified runtime is v0.1.3.2; `--rollback` on Linux/WSL or `-Rollback` on Windows restores it after an update. Published downloads use the public L0xRE repository.

This remains a draft. Sean publishes at the catalog PR merge commit, merges the PR and marks the release Latest after reviewing the final lane results and any parked targets.

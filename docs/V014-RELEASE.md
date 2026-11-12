# L0xRE v0.1.4 — P2 runtime (draft)

This release packages the P2 stack promoted on the RTX 3060 on November 10: decode fusions and CUDA copy batching, backend-sampling reserve correction, reasoning-loop guard correction, cache reuse Fix 1, incremental tokenization and P1–P4/P2 append and render improvements. Model weights are unchanged.

Frozen source: `e7c3a18defbfadb96a877d923e4b1c9c3157d530`.

## Available packages and coverage

| Package | Current coverage | Draft availability |
|---|---|---|
| Windows SM86 | Build-verified: checksums, launcher tests, install/update/rollback, every DLL import and native load; no matching hardware run | Available |
| Windows SM89 | Build-verified: checksums, launcher tests, install/update/rollback, every DLL import and native load; no matching hardware run | Available |
| Linux SM86 | Fully tested on RTX3060: deep selftest, changed-path, recovery, OFF quality carried/Mirai117, same-session ABBA and unlisted-card stock start | Available |
| Linux SM89 | RTX4090 smoke-only:12GB/16GB deep, changed-path, one canonical pass per version; no N150 | Available |
| Linux SM120 (WSL) | Parked:12GB ABBA short TTFT+21.25%, depth10K decode-5.73%, depth32K decode-3.64% vs0.1.3.2; selftests/smoke/recovery/quality passed | Withheld |
| Windows SM120 | Parked: clean12GB ABBA short TTFT+12.94%; both deep/changed-path/OFF quality passed; one clean diagnostic used | Withheld |

The catalog retains v0.1.3.2 for unfinished targets. Windows build-only binaries have corrected profile/component hash bindings and launcher status metadata; final checks include every advertised preset launcher dry-run. No v0.1.4 speed or quality improvement is claimed from build-only coverage.

## Default flags

All presets use `--backend-sampling`, `--cache-ram 8192`, `--temp 1.0` (previously 0.7) and `--min-p 0.0` (previously 0.05). Existing `--top-p 0.95`, `--top-k 20`, `--reasoning off`, `--reasoning-effort none` and `--reasoning-budget 0` remain. The control vector is opt-in. Architecture-specific arenas, context capacity, thread/affinity selection and GPU vision settings remain as specified by each preset. These defaults reproduce the production 3060 behavior; the owner can object before publication.

## Validation and limits

Windows SM86/SM89 carry the new code with build-only qualification. Hardware selftest references for the new Windows runtime remain deferred to matching-hardware qualification; a launcher dry-run does not qualify a GPU. Native launcher tests retain the two known stale preflight/consent failures named in the release plan; the remaining checks pass. No Windows recovery-profile claim is made (existing platform exception).

The three Linux packages share the exact production host libraries. A source-archive host rebuild completed, but its three library hashes differ from production; they are not presented as reproducible matches. SM86's CUDA backend and bridge are the promoted incremental builds. SM89/SM120 recompile two CUDA units and the main bridge with their pinned CUDA toolchains; unchanged objects are hash-verified against the retained build witness.

P2 CPU render/tokenization proofs cover the shared Linux host bytes. Carry-over of 3060 quality/speed still requires the release checks to confirm the packaged runtime and effective arguments. The P2.1 warm-overhead waiver stays bound to the promoted production bytes; it is not transferred to Windows or other newly compiled CUDA backends.

## Install and rollback

Windows: run `.\install.cmd`; an existing installation uses `update.cmd`. Linux/WSL: use `install.sh` or `install.py --update`. The previous verified runtime is v0.1.3.2; `--rollback` on Linux/WSL or `-Rollback` on Windows restores it after an update. Published downloads use the public L0xRE repository.

This remains a draft. Sean publishes at the catalog PR merge commit, merges the PR and marks the release Latest after reviewing the final lane results and any parked targets.

## Parked target

Linux SM120 remains on v0.1.3.2 in the catalog after the speed regressions above. Its OFF quality result is120/150 versus119/150 with no worsened cases (HA-02 improved), and both preset deep selftests passed. Those results do not waive the2% performance floor. All ABBA samples are retained for owner review. Native Windows is also parked after its clean diagnostic short-TTFT failure.

## RTX3060 measured packaging comparison

One completed ABBA block against v0.1.3.2, default OFF, same card/session. The earlier block stopped on an output-pipe failure; partial samples were retained and not mixed into this diagnostic block.

| Metric | v0.1.3.2 | v0.1.4 | Change |
|---|---:|---:|---:|
| narrative_decode | 42.758 | 44.253 | +3.50% |
| code_decode | 75.567 | 76.947 | +1.83% |
| prefill_10k | 732.722 | 733.567 | +0.12% |
| short_ttft_ms | 442.950 | 425.548 | -3.93% |
| cached_prompt_ttft_ms | 636.616 | 324.012 | -49.10% |
| followup_ttft_ms | 758.288 | 491.850 | -35.14% |
| depth_10k_decode | 43.360 | 43.085 | -0.64% |
| depth_32k_decode | 37.924 | 39.139 | +3.20% |

Decode/prefill values are tokens/s; TTFT values are milliseconds (lower is better). Every metric passes the2% no-regression rule. Mirai quality:117/150 vs116, no worsened cases; HA-17 improved. OFF120/150 and P2 CPU render/tokenization proofs carry only with the verified production bytes and effective argv. A fresh public-branch installer with the3060 name removed from its local catalog generated a stock start.sh without a probe flag; startup, reply, effective argv and all five loaded production hashes passed. Final archive hashes and install/update/rollback passed. The historical P2.1 warm-overhead waiver remains bound to its original production proof; it was not needed to clear this ABBA block.

## RTX4090 smoke benchmark

One canonical pass per version, v0.1.3.2 → v0.1.4: narrative95.149 →95.957 tokens/s(+0.85%); code157.128 →164.718 tokens/s(+4.83%). Both advertised preset deep selftests and changed-path smoke passed. No full reference-lane, N150 or ABBA claim is made. NUMA0 hugepages were0 during the lane and the original24576 plus Strata4090 service identity/health were restored.

## RTX5090 native Windows — clean12GB ABBA, parked

Both advertised deep selftests, changed-path smoke and OFF quality120/150 versus120/150(zero flips,150single attempts/no retries) passed. The original12GB timing block overlapped package generation and is retained separately as contaminated. One clean diagnostic repeat ran after all host builds/copies/uploads finished, with actual loaded DLL identities and zero Windows5090GPU Engine frames before each arm.

| Metric | v0.1.3.2 | v0.1.4 | Change |
|---|---:|---:|---:|
|narrative_decode|125.748|129.778|+3.20%|
|code_decode|219.578|222.750|+1.44%|
|prefill_10k|3461.256|3466.998|+0.17%|
|short_ttft_ms|143.775|162.379|+12.94%|
|cached_prompt_ttft_ms|239.320|160.542|-32.92%|
|followup_ttft_ms|252.348|144.419|-42.77%|
|depth_10k_decode|116.409|114.239|-1.86%|
|depth_32k_decode|109.910|108.727|-1.08%|

Short TTFT fails the mandatory2% floor; all other seven metrics pass. No further repeat,16GB speed, final promotion or unlisted-card stock start was run. The package is withheld and its catalog row remains on v0.1.3.2. Next owner investigation: short-request setup/variance. Semantic and quality passes do not waive the performance failure.

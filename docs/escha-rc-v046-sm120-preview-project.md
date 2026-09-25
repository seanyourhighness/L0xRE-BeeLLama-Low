# Escha BeeLlama v0.4.6 RC SM120 preview project

**Owner:** Sean / Codex  
**Date opened:** 2026-09-14  
**Tree:** `/home/sean/kernel-lab5090/beellama-escha/rc-v046`  
**Commit under test:** `2d1411c2c9eaca7d92bdd6a84279e34e580c4cf7` (`escha-rc-v046-full`)  
**Hardware:** NVIDIA GeForce RTX 5090 (SM120), CUDA build `build-sm120-preview`

This document defines the RC preview scope and the reproducibility contract. The
companion [SM120 preview ledger](escha-rc-v046-sm120-preview-ledger.md) is the
authoritative run-by-run receipt.

## Objective

Qualify the SM120 Preview bridge against a named SM89 Sprint oracle: approximately
80 tok/s decode and 3,000 tok/s
prefill in SGLang-shaped workloads, while preserving model-output correctness,
MTP behavior, cache reuse, and GPU stability. A performance result is not a
release result until its exact artifact hashes, environment, command, samples,
exit status, and post-run GPU state are recorded.

## Frozen performance profile

Use a fresh process and the following environment for every comparable run.
Do not add undocumented selectors to a receipt.

```sh
unset ESCHA_OFFICIAL_BRIDGE
export ESCHA_OFFICIAL_RAW_BRIDGE=1
export ESCHA_OFFICIAL_RAW_SWIGLU_FUSION=1
export ESCHA_OFFICIAL_RAW_DECODE_BRIDGE=1
export ESCHA_OFFICIAL_RAW_DECODE_F32_INPUT=1
export ESCHA_OFFICIAL_RAW_DECODE_SWIGLU_FUSION=1
export ESCHA_OFFICIAL_RAW_DECODE_K3_SPLIT_DIV2=0
export ESCHA_OFFICIAL_BRIDGE_ACCUMULATION=mixed
export ESCHA_OFFICIAL_BRIDGE_LIBRARY=/home/sean/kernel-lab5090/preview-bridge-sm120/libescha_official_bridge_cuda_sm120.so
export ESCHA_OFFICIAL_CODE_GEMM_CUBIN=/home/sean/kernel-lab5090/preview-bridge-sm120/code-gemm-sm120.cubin
export ESCHA_OFFICIAL_F32_DECODE_CUBIN=/home/sean/kernel-lab5090/preview-bridge-sm120/code-gemm-sm120.cubin
export ESCHA_GDN_CHUNK_BRIDGE=1
export ESCHA_GDN_CHUNK_BRIDGE_LIBRARY=/home/sean/kernel-lab5090/preview-bridge-sm120/libescha_gdn_chunk_bridge_sm120_ported.so
export ESCHA_GDN_CHUNK_CUBIN_ROOT=/home/sean/kernel-lab5090/preview-bridge-sm120/gdn-cubins-sm120-exact
export ESCHA_DIRECT_RECURRENT_STATE_R258=1
export ESCHA_DIRECT_RECURRENT_CONV_R260=1
export ESCHA_FUSE_GDN_DECODE_PREP_R262=1
export ESCHA_FUSE_GDN_NORM_GATE_R264=1
export ESCHA_OFFICIAL_RAW_DECODE_QKV_Z_OVERLAP=1
export ESCHA_OFFICIAL_RAW_DECODE_ATTN_QKV_OVERLAP=1
```

The profile intentionally excludes `ESCHA_W2_I8_HEAD=1` and
`ESCHA_OFFICIAL_RAW_DECODE_GATE_UP_OVERLAP=1`: the former is a decode-only
experiment and the latter is not valid for the current Qwen graph. CUDA graphs
are enabled by leaving `GGML_CUDA_DISABLE_GRAPHS` unset. Qwen's managed service
remains stopped during qualification (`llama-server.service` inactive; port
8082 unused).

The code-GEMM cubin is the frozen parity-safe profile. The faster direct-F32
profile uses the same selectors but replaces the decode artifact and adds the
explicit activation-ABI selector:

```sh
export ESCHA_OFFICIAL_F32_DECODE_CUBIN=/home/sean/kernel-lab5090/preview-bridge-sm120/f32_input.sm120.cubin
export ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32
```

The first A/B of that cubin produced `PAR///////////////` because the wrapper
was converting the F32 activation to F16. After the wrapper ABI fix, W2 and E3
both return `PARITY_OK`; all four generated positions have identical top-20
IDs against the code-GEMM control, with maximum observed log-probability
deltas of `1.19223841e-07` (W2) and `2.38447683e-07` (E3). The direct-F32 lane
measures `82.37 +/- 2.01` tok/s (W2) and `84.63 +/- 3.38` tok/s (E3), so it is
the current fast-build candidate. The code-GEMM profile remains the rollback
and parity reference until the longer release gates are completed.

## KVarN SM120 status

DeepSeek Flash 4.1 reviewed the route traces and the post-change receipts
through Hermes/OpenRouter. The native specialized path is now the RC default.
The controlled intrinsic-tail A/B (250 -> 254 cached tokens) exercised the
reachable `nq=128,nkv=256` prompt boundary and `nq=1,nkv=512` decode boundary:
fast record-loader vs generic rotated-loader selected-token log-probability
delta was at most `2.648e-5` (mean `6.531e-6`), with no top-1 changes and
identical fixed-seed sampled text. A long-specialized opt-in full verification
also passed at `nkv=512/768/1024`.

The release route in `ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu` therefore
keeps native split/vector dispatch enabled by default. The test-only
`GGML_KVARN_TEST_DISABLE_LONG_SPECIALIZED_DECODE=1` switch is an emergency
portable rollback for long-context shapes; it disables both specialized decode
eligibility and falls through to portable or generic/materialize handling so it
does not silently re-enter the record-backed route. The separate
`GGML_KVARN_TEST_DISABLE_FAST_RECORD=1` switch remains a diagnostic loader A/B
only. No F16-KV testing is included.

Final native-default evidence is recorded in the ledger under
`escha-sm120-kvarn-native-final-full-20260914-170000`: full verify exit 0,
all applicable Club-3090 checks passed, quality variety `0.580`, routes were
303 `decode-split`, 16 `generic-mma`, and 160 `prompt-generic-mma`, with no
portable fallback or Xid. Separate native-default KVarN 3/3 and regular q4
probes returned `PARITY_OK` with clean 7 MiB idle recovery. DeepSeek's final
review accepts this as conditional RC ship and recommends one bounded future
3-depth native-vs-rollback long-generation parity run before calling the
rollback fully proven.

The rollback branch itself has now been exercised end-to-end at
`/home/sean/kernel-lab5090/runs/escha-sm120-kvarn-rollback-probe-20260914-171100`:
64 decode steps routed `portable-native` with no specialized re-entry, the
completion exited 0, and the GPU returned to the 7 MiB idle baseline without
an Xid. This closes the runtime evidence gap for portable-supported long
shapes; the deeper native-vs-rollback logit A/B remains a follow-up.

## Immutable inputs

| Input | SHA-256 |
|---|---|
| `escha-w2-firstclass.gguf` | `5ac8d3439d829ae616529e9f3e8448c5665622a184112cd9055cf560d2aa9195` |
| `escha-e3-firstclass-v2.gguf` | `b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543` |
| `libescha_official_bridge_cuda_sm120.so` | `459e04d492a32a1d46df078341276642962dcac50f59846410df8d555551a0f9` |
| `code-gemm-sm120.cubin` | `6cdbf1ab440958d570ef0b3425df5fcea14c40f48c02d5774da983514d676d8b` |
| `f32_input.sm120.cubin` | `04692f328b386967c02b68a2061e6ec846675f82e516978a7f1ab9555c99e39b` |
| `libescha_gdn_chunk_bridge_sm120_ported.so` | `32ad67be3344759298940db0e1b70315ea1d9c43b5ea3491b939f57ec90b0544` |

The worktree is intentionally dirty with the RC source changes. Every receipt
must include both the commit and a hash of `git diff --stat` (or an equivalent
full worktree manifest); the commit alone does not pin the tested source.

## Gate order and safety envelope

1. Artifact/hash/Xid preflight, build, focused tests, and `git diff --check`.
2. Bounded W2 control prompt, cache reuse, and fixed-output/logit parity.
3. Bounded MTP receipt for W2, then E3 if its artifact supports the route.
4. Short decode/prefill measurements using the frozen profile (performance
   smoke rows are not release gates until functional gates are green).
5. One fresh 32k NIAH case at a time; stop if `nvidia-smi` or `/dev/nvidia*`
   changes. Do not jump to 128k/260k after a cancellation or device fault.
6. Club-3090 benchmark and quality packs only after the functional gates are
   green.

Every GPU run must capture, before and after: `nvidia-smi --query-gpu=name,
memory.used,memory.free,pstate,power.draw,clocks.sm --format=csv,noheader`,
driver/CUDA versions, `dmesg`/journal Xid evidence, process exit code, and the
server log tail. The idle tolerance is at most 64 MiB above the recorded 7 MiB
baseline. A nonzero exit, missing device, service collision, Xid, or memory
outside that tolerance leaves the gate open and blocks promotion.

The recovery policy is: stop the current process; do not retry on the same
context; capture `nvidia-smi -q`, `dmesg -T | grep -iE 'NVRM|Xid'`, and the
user-service journal; verify `/dev/nvidia*` and a fresh idle `nvidia-smi`; then
obtain a human go-ahead before another GPU test. An unresolved Xid or missing
device permanently defers the long-context ladder for this run.

## Release bar

The Preview is not promoted until W2 and E3 have: deterministic control output;
cache-reuse evidence; MTP acceptance evidence where supported; fixed-prompt
output/logit parity against the same-binary control/reference; the NIAH ladder;
and two complete no-thinking Club-3090 quality receipts. The prefill target is
3,000 tok/s; a result below it is reported as a target shortfall, not a pass on
the target. Performance is reported
separately for prefill and decode, with mean, standard deviation, sample count,
and exact command. Partial or interrupted tests remain `PENDING`, `DEFERRED`, or
`BLOCKED`, never `PASS`.

## SM89 transfer validation (4090)

The KVarN source fix transfers cleanly to SM89 as source, not as a SM120 binary:
the exact RC source was staged in `/home/sean/kernel-lab/beellama-escha-rc046-sm89`,
configured for `CMAKE_CUDA_ARCHITECTURES=89`, and built successfully with CUDA
13.3/GCC 13. The resulting `llama-cli` and `llama-server` are separate SM89
artifacts. Both base Escha models returned deterministic `PARITY_OK` on the
4090. The merged W2 and E3 MTP artifacts also returned `PARITY_OK` with
`--spec-type draft-mtp`, `n_max=2`, q4_0 draft K/V, and the draft model kept on
CPU to avoid loading a second full merged GGUF into 24 GB VRAM. No F16 KV was
used.

The bounded NIAH ladder passed for both models at 10%, 50%, and 90% needle
depths (approximately 8K-class prompts, `ctx=16384`): every case returned the
exact `BLUE-FLAMINGO-42` answer and exit 0. The full receipt table and hashes
are in the companion ledger under “SM89 transfer and 4090 correctness start.”
This validates correctness portability and cache behavior at the tested context;
it does not yet claim SM89 performance parity or MTP throughput parity.

## 4090 storage layout

After the correctness runs, large models, activation captures, run trees,
builds, caches, and the Escha virtualenv were moved to `/mnt/storage/ai-*` with
compatibility symlinks at their former paths. The active RC source remains on
the system volume, while its SM89 build and evidence are storage-backed. A
small read-only remainder is preserved separately for rollback; no protected
files were force-deleted, and no CUDA/Conda system dependency was relocated.

## Named 4090 fast profile

The fastest optimized SM89 test profile is frozen as **4090 Escha Sprint RC —
KVarN/MTP Fast**. It uses the SM89 bridge package, the retained R248/R251/R253/
R258/R260/R262/R264 route environment, `ESCHA_E3_EMBED_GPU=1`, KVarN3/2, and
MTP n2 with CPU q4_0 draft at 256K context and batch 2048/ubatch 1024. The
profile manifest and bounded W2/E3 receipts are at
`/home/sean/kernel-lab5090/4090-escha-sprint-rc-fast.md` and the adjacent
`.env` file. It explicitly excludes F16 KV and never runs on the 5090.

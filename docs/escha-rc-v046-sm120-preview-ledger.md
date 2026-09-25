# Escha BeeLlama v0.4.6 RC SM120 preview ledger

This is the reproducibility ledger for the 5090 Preview. All times are
`America/Los_Angeles` on 2026-09-14. A receipt is valid only when it includes
the frozen profile from the companion project document, the exact binary/model
hashes, command, sample list, exit status, and post-run GPU state.

## Baseline and environment

- Source: `/home/sean/kernel-lab5090/beellama-escha/rc-v046`
- Commit: `2d1411c2c9eaca7d92bdd6a84279e34e580c4cf7`
- Build: `build-sm120-preview` (`llama-bench`, `llama-cli`, `llama-server`)
- GPU idle after last check: `RTX 5090, 7 MiB used, 32181 MiB free, P8`
- Managed Qwen service: `llama-server.service` inactive; port 8082 closed
- Historical Sprint/SM89 reference: W2 `78.95` tok/s decode / `2716.81`
  tok/s prefill, from the named Sprint receipt `escha-v0.4.6-rc2` on the 4090.
  Its binary/cubin hashes are not present on this host, so this remains a
  comparison target rather than a reproducible oracle until those artifacts are
  imported and hashed. The older RC native snapshot (`54.814` / `1835.843`) is
  a separate, non-Sprint baseline and must not be mixed into the target row.

## Receipt table

| ID | Scope / exact command | Result | Evidence / next action |
|---|---|---|---|
| BLD-01 | Build `cmake --build build-sm120-preview --target llama-bench llama-cli llama-server -j4` | PASS | `llama-bench` `c7c1b22e...`, `llama-cli` `2fcd11db...`, `llama-server` `b3a355e9...`; manifest/source and wrapper hashes are recorded below. |
| TST-01 | `ctest --test-dir build-sm120-preview --output-on-failure -R '^(test-escha-arch|test-escha-cpu|test-escha-tensor)$'` | PASS | 3/3 tests passed. |
| DEC-01 | Earlier `llama-bench -p 0 -n 256 ... -r 5` with the pre-fix F32-input wrapper, W2 | PREVIEW SMOKE (superseded) | `81.934606 +/- 1.067854` tok/s, but fixed control failed. Retained only as a diagnostic history row. |
| DEC-02 | `llama-bench -p 0 -n 256 -b 2048 -ub 512 -t 8 -ngl 99 -fa on -ctk f16 -ctv f16 -r 5`, code-GEMM decode profile | PASS / REFERENCE | W2 `78.22 +/- 1.14` tok/s; E3 `82.79 +/- 1.82` tok/s. Both exit 0 and returned to 7/32181 MiB. |
| DEC-03 | Same decode command with patched direct-F32 ABI (`f32_input.sm120.cubin`, `ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32`) | PASS / FAST CANDIDATE | W2 `82.37 +/- 2.01` tok/s; E3 `84.63 +/- 3.38` tok/s. Both exit 0; post-run GPU 7/32181 MiB, no Xid. |
| PRE-01 | `llama-bench -p 2048 -n 0 -b 2048 -ub 2048 -t 8 -ngl 99 -fa on -ctk f16 -ctv f16 -r 5`, W2 frozen profile | PASS / SMOKE | `2902.77 +/- 80.69` tok/s; exit 0; GPU returned to 7/32181 MiB. Below the 3,000 tok/s target. |
| PRE-02 | Same prefill command, E3 frozen profile | PASS / SMOKE | `2758.21 +/- 83.78` tok/s; exit 0; GPU returned to 7/32181 MiB. Below the 3,000 tok/s target. |
| PRE-ABI-01 | Fresh W2/E3 A/B of `ESCHA_OFFICIAL_BRIDGE_ACCUMULATION={mixed,fp32,k2-fp32}` plus `mixed` with historical `ESCHA_NO_PREFILL_MIXEDACC=1`; `-p 2048 -n 0 -b 2048 -ub 2048 -t 8 -ngl 99 -fa on -ctk f16 -ctv f16 -r 5` | PASS / NO GAIN | Mixed is fastest: W2 `2801.799 +/- 308.626` (warm `2939.820 +/- 0.748`), E3 `2659.670 +/- 84.407` (warm `2695.048 +/- 34.011`). `fp32` is `-44.39%` W2 / `-40.67%` E3 warm; `k2-fp32` is `-35.10%` / `-31.80%`. The `NO_PREFILL_MIXEDACC` control is effectively unchanged (W2 `-0.01%`, E3 `+1.49%`, warm). All 8 bench processes exited 0, returned to 7/32181 MiB, and had empty Xid receipts. |
| PAR-ABI-01 | Six fresh `llama-cli --single-turn --reasoning off` fixed controls under the same three accumulation policies, W2/E3 | PASS / DIAGNOSTIC | Every arm exited 0 and emitted exactly `PARITY_OK`; output, exit, GPU, and Xid receipts are under the PRE-ABI-01 run root. |
| PAR-01 | W2 fixed control: native decode vs pre-fix retargeted F32-input vs code-GEMM decode | FAIL / DIAGNOSTIC | Native and code-GEMM: `PARITY_OK`; pre-fix F32-input: `PAR///////////////`. Root cause was an activation-ABI mismatch. |
| PAR-02 | Patched direct-F32 W2/E3 control plus four generated top-20 logit positions against code-GEMM | PASS / FAST CUBIN | Both models return `PARITY_OK`; top-20 overlap is 20/20 at 4/4 positions. Max log-prob delta W2 `1.19223841e-07`, E3 `2.38447683e-07`. |
| EXP-01 | W2 INT8 head, fresh decode-only process | EXCLUDED | About 7.37 tok/s; direct head is one-token-only and p2048 is guarded. Not in parity profile. |
| FUNC-W2 | Code-GEMM parity-safe control, cache reuse, bridge load, graph reuse | PASS | Control `PARITY_OK`; cache second `cache_n=2688`, `cache_lcp_n=2801`, source `checkpoint`, reason `committed`; log SHA `4a7565f07eaa33ca219c423ffab7f362e7d3b35cf150e06bfd6e9ce323dca576`. |
| FUNC-E3 | Code-GEMM parity-safe control, cache reuse, bridge load, graph reuse | PASS | Control `PARITY_OK`; cache second `cache_n=2944`, `cache_lcp_n=3001`, source `checkpoint`, reason `committed`; log SHA `33d5d839137ba10dc41cfff0b6cea3a7328fb73aed663a760299206b6e8c7c56`. |
| FAST-W2/E3 | Patched direct-F32 bounded control with packaged-wrapper smoke | PASS | W2/E3 `PARITY_OK`; packaged W2 loader evidence and graph reuse captured; packaged smoke log SHA `f4d66e6652899361741f388dac047dfd89b3b2790b2f48356dbe304a437a6b2a`. |
| R248-CONF-W2 | Fresh five-pair alternating W2 confirmation with only `ESCHA_OFFICIAL_RAW_DECODE_DOWN_ADD_RMS_FUSION=1` enabled for the candidate | PASS / HOLD | 4/5 wins; mean paired delta `+1.034% +/- 1.006%`; all explicit exit files `0`, all Xid receipts empty. Positive but not promotion-qualified; W2-scoped only. |
| MTP-BASE-W2/E3 | `--spec-type draft-mtp --spec-draft-n-max 2` on base GGUFs | NOT_APPLICABLE | Both models report no embedded MTP layers; each exited safely with GPU back at 7 MiB. |
| MTP-W2 | Merged `/home/sean/kernel-lab5090/escha-mtp/escha-w2-with-mtp.gguf` | PASS | `draft_n=18`, `draft_n_accepted=11`, `predicted_n=21`, exit 0; log SHA `4c63071ed7d7fdbcda80d281528efcb76803b6e680c63cdde77bd801e26a5b2e`. |
| MTP-E3 | Merged `/home/sean/kernel-lab5090/escha-mtp/escha-e3-with-mtp.gguf` | PASS | `draft_n=20`, `draft_n_accepted=12`, `predicted_n=23`, exit 0; log SHA `271184f8bd199901d673b240ee7cdc6a6e85665c8b890149ee7c9dcc60ff0c0c`. |
| NIAH-32K | 32k tokens at 10/50/90% needle depths | PENDING | One case per fresh process; stop on any device instability. Recovery policy and Xid capture are mandatory. |
| NIAH-128K/260K | 128k and 260k ladder | DEFERRED | Do not run until 32k and recovery policy are green; prior cancellation detached the device. |
| BENCH-W2/E3 | Club-3090 `bench.sh`, fixed endpoint and 5+3 samples | PENDING | Save stdout and GPU receipt; no managed Qwen service. |
| QUAL-W2/E3 | Club-3090 `quality-test.sh --full --no-thinking --repeat 2` | PENDING | Partial runs are not passes. |

### 2026-09-14/15 SM89 transfer and 4090 correctness start

The RC source was copied unchanged to `z840:/home/sean/kernel-lab/beellama-escha-rc046-sm89`
and configured with `CMAKE_CUDA_ARCHITECTURES=89`, CUDA 13.3, GCC 13, and
`GGML_CUDA_KVARN=ON`. The three KVarN source hashes match the SM120 RC source:

- `fattn-kvarn-dispatch.cu`: `3c0d474967247c533c2e007725c00b3bdea1ce7068a5fb7884b4fb05fadb0759`
- `fattn-mma-kvarn-decode-decl.cuh`: `b7b77c23144441ca4822e591efa3d4c1d1a316b9e3468eddfd579ec8704ac7fd`
- `fattn-mma-kvarn-decode.cuh`: `57fd293daef2804d927e20e00342577706c46a05a4324fbf1d2f0463b6f0eb9b`

SM89 binaries: `llama-server` SHA-256
`2ccb27f8bfe063fcd187db613e0f2eea763ea324b69ba571a3193c824174f7b6` and
`llama-cli` SHA-256
`4d6432baa3552b1f386282237b35e74337d08dc7d01c98c79551b16bfdca6eb4`.
The build completed `445/445`; the binary reports `0.4.6-dev` and was built
for SM89. The 4090 production port 8080 and managed Qwen service remained off.

Run root: `/home/sean/kernel-lab5090/runs/escha-sm89-4090-correctness-20260915-003700`.
All runs used fresh bounded `llama-cli` processes, target KVarN3/KVarN2, no
F16 KV, `--single-turn`, `temperature=0`, seed 42, and the no-thinking chat
option. The 4090 returned to 4 MiB used / 24104 MiB free after every case;
the final idle snapshot was P8 with no compute processes and no journal Xid or
CUDA-error entries.

| ID | Test | Result | Receipt |
|---|---|---|---|
| SM89-CONTROL-W2 | Base W2 deterministic control | PASS | `PARITY_OK`, exit 0; log SHA `030aa8c24495826f73d203bad131650fcad1d464a265092c9eb6823358ede71c` |
| SM89-CONTROL-E3 | Base E3 deterministic control | PASS | `PARITY_OK`, exit 0; log SHA `8a0feb8b394bd82fa9ed2a893f0b62a09eb2ba08cf5c23a2530f568525db1180` |
| SM89-MTP-W2 | Merged W2 MTP, `draft-mtp`, `n_max=2`, CPU draft, q4_0 draft K/V | PASS | `PARITY_OK`, exit 0; log SHA `cec9b3a9e629afe529463fdf0f2e16cd192658e4fefc1c2cfad05fc9dd4db4e1` |
| SM89-MTP-E3 | Merged E3 MTP, `draft-mtp`, `n_max=2`, CPU draft, q4_0 draft K/V | PASS | `PARITY_OK`, exit 0; log SHA `1d1ffa2048904397d029792ea87c73f86d6f9b2277d9819672da783854e662db` |
| SM89-MTP-VRAM | Same merged model as target and GPU draft | EXPECTED FAIL-CLOSED | Loader rejected the unaudited Escha MTP KVarN draft route; q4 GPU-draft then hit a second-copy CUDA OOM before inference. No GPU state change; log SHA `40efd112754a83cba236902bc094769c3bf62f1a379fd4883b5f5fc1450c2d8d` |
| SM89-NIAH-8K | W2/E3, 10/50/90% needle depth, ~77.6 KiB prompt, `ctx=16384`, KVarN3/KVarN2 | PASS | All six returned exactly `BLUE-FLAMINGO-42`, exit 0. W2 depth-10 log SHA `09e9f8d41cab9c704603ff4ad6ccd8852adc9e322646a73fc59165b21356ae88`; remaining five ladder log SHA `e2c6cecbd9d5c421d7a173d18f8a918c05f9ad2ef3c972e2222edc8c9028a644` |

MTP merged-artifact hashes transferred to the 4090 storage volume:

- W2 `escha-w2-with-mtp.gguf`: `3f93cbe77a20f1fa7272741757596cac66a66457d7ecaed1e5a6e4baa409535e`
- E3 `escha-e3-with-mtp.gguf`: `746bd40841fb18b9c1918923e89c007df70b5e4f3de64290d1399593e70c96b0`

The initial NIAH attempts with an 8-token cap and an empty `-p` are retained as
aborted harness receipts (`niah-8k-ladder-aborted-n8.log`, SHA
`62e6b5325a114f568c874b4602e5bd2b7c2e41a80c48bd681b2a0194ff6ec9b3`, and
`niah-8k-ladder-aborted-empty-p.log`, SHA
`a5668771d58ecac1702de8cca183f927ce7807ca58544f81df22a0e3ff6ce833`); they
were stopped before being treated as test results. The corrected file-only,
24-token-cap ladder is the authoritative NIAH result.

### 2026-09-15 4090 storage cleanup

After correctness testing, large models, captures, caches, builds, and the
Escha virtualenv were moved from the 233 GB system volume to organized
`/mnt/storage/ai-models`, `/mnt/storage/ai-artifacts`, `/mnt/storage/ai-builds`,
`/mnt/storage/ai-cache`, and `/mnt/storage/ai-venvs` trees. Compatibility
symlinks preserve the original script paths. Protected read-only material that
could not be unlinked by an unprivileged cross-device move remains as
`kernel-lab5090/runs-root-remainder` and `kernel-lab5090/lab-root-remainder`;
the storage copies were verified with rsync dry-runs. Root usage fell to 49 GB
(23%) with 172 GB free; `/mnt/storage` has 6.8 TB free. No CUDA/Conda system
dependencies were moved, no service was started, and the 4090 remained idle at
4 MiB/P8 after cleanup. Final health receipt:
`cleanup-final-health.txt`, SHA-256
`ba8e119f7e288d66aa180ac2defe83e383826a577b7e808f0bd01bf59be9699c`.

## Commands for open gates

Before every row, re-hash all six immutable inputs and capture the worktree
manifest, driver/CUDA, graph selector state, power/clock state, and Xid-clean
precondition. Assert bridge engagement by capturing the loader's path-specific
`dlopen` evidence (or a fail-closed symbol/load probe) in the server stderr.
Use the exact server command documented in the runbook, a fresh process, and a
bounded context for functional checks. For each response save raw JSON, server
stderr, and a SHA-256 of each evidence file. The NIAH ladder must use a fresh
nonce per prompt and record actual prompt tokens, depth, answer, and exit code.

## Interpretation rules

- Compare only rows with the same model hash, bridge/cubin hashes, source
  commit, quant/KV types, context, batch/ubatch, flash-attention setting,
  thread count, GPU layers, and sample protocol.
- Report decode and prefill separately; never average them into one score.
- A performance win without output parity is a diagnostic, not a promotion.
- The SM120 profile is not literally Sprint-identical while the Sprint INT8
  head is excluded; report it as bridge/profile parity, not proof of full oracle
  equivalence. QKV/Z and attention-QKV overlap are retained only because their
  graph-safe A/B receipts are still required before promotion.
- `PASS` means the command completed and its acceptance predicate passed;
  `EXCLUDED` means intentionally outside the release profile; `DEFERRED` means
  safety sequencing, not success; `BLOCKED` means an external failure.

## Append-only run records

Add one dated subsection per execution. Never rewrite an earlier receipt; if a
profile or artifact changes, create a new ID and explain the delta.

### 2026-09-14 fast cubin ABI correction and SM120 smoke

- Source wrapper: `official_bridge_cuda.cu` SHA-256
  `741335d8a10f9580f7c176b61b1dee0bb4aacb41ca740d94b9845f53c9e9f197`.
- Packaged wrapper: `libescha_official_bridge_cuda_sm120.so` SHA-256
  `459e04d492a32a1d46df078341276642962dcac50f59846410df8d555551a0f9`.
- Cubins: code-GEMM
  `6cdbf1ab440958d570ef0b3425df5fcea14c40f48c02d5774da983514d676d8b`;
  direct-F32 `04692f328b386967c02b68a2061e6ec846675f82e516978a7f1ab9555c99e39b`.
- ABI correction: `ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32` passes F32 directly
  to `f32_input.sm120.cubin`; the selector is unset for code-GEMM's F16 ABI.
- Logit receipts: W2 response SHA
  `4b2d680b725f6f26b99ca321149ba3f8e4396aec02ebf479d3f089935cfbb9bc` (top-20 A/B response
  SHA `31d33f09d849c31db00884cb7b093568666c75b64d44bdbeec5d033b04636550`);
  E3 direct-F32 response SHA
  `41814524d41d1d02317df4a929b9ba0bd5bd2fcfe5c49cf9c95c83305d3381fe`.
- Throughput receipts: W2 log SHA
  `26bae6e5493b004dd71f588635c1faabf1dad52c0976d2be10618a0abae2d8cc`;
  E3 log SHA `dcd8071da01e32ac9dd9fa27694a71655ddf33ff622e040eae336f9109b26179`.
- Packaged-wrapper control: W2 `PARITY_OK`, loader path and CUDA graph reuse;
  log SHA `f4d66e6652899361741f388dac047dfd89b3b2790b2f48356dbe304a437a6b2a`.
- Every fast run returned to `RTX 5090, 7 MiB used, 32181 MiB free`; no Xid was
  observed. Slow NIAH/Club quality packs remain intentionally deferred.

### DeepSeek Flash 4.1 post-review

Hermes review receipt: `/tmp/escha-hermes-postreview.txt`, SHA-256
`5a0a27f229b159f1ddab78c70f2eb4fa78939f5be148f64affa53c66eb74dd3d`.
The review confirms the direct-F32 ABI fix and W2/E3 top-20 logit A/B are
numerically consistent with the code-GEMM reference and that the GPU safety
receipt is clean. It keeps the candidate open rather than promoted because
both prefill point estimates remain below the 3,000 tok/s target and the slow
NIAH/Club quality suites are deferred.

### 2026-09-14 approved SM120 decode selector screen

All arms used the patched direct-F32 lane (`f32_input.sm120.cubin` with
`ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32`), the retained R258/R260/R262/R264
stack, valid QKV/Z and attention-QKV overlap, `-p 0 -n 256 -b 2048 -ub 512
-t 8 -ngl 99 -fa on -ctk f16 -ctv f16 -r 3`, and fresh processes. Run root:
`/home/sean/kernel-lab5090/runs/escha-sm120-decode-approved-20260914`.

| ID | Candidate | W2 | E3 | Fixed control | Decision |
|---|---|---:|---:|---|---|
| DEC-R244-S | `ESCHA_OFFICIAL_RAW_DECODE_K3_DOWN_SPLITS=12` screen | 84.704933 +/- 0.362362 | 85.737933 +/- 2.354941 | all six candidate parity smokes returned `PARITY_OK` | paired screen required |
| DEC-R248-S | `ESCHA_OFFICIAL_RAW_DECODE_DOWN_ADD_RMS_FUSION=1` screen | 84.668103 +/- 0.945289 | 86.406220 +/- 1.723926 | `PARITY_OK` on W2/E3 | paired screen required |
| DEC-R234-S | `ESCHA_OFFICIAL_RAW_DECODE_QKV_Z_BETA_ALPHA_OVERLAP=1` screen | 83.851126 +/- 0.998306 | 86.607947 +/- 0.366197 | `PARITY_OK` on W2/E3 | paired screen required |

Three alternating pairs per model then produced:

| Candidate | W2 paired delta | E3 paired delta | Decision |
|---|---:|---:|---|
| R244 / K3 split 12 | -1.536% +/- 1.598% | -1.448% +/- 2.770% | **REJECT**; no two-model gain |
| R248 / down-add-RMS | +2.061% +/- 0.949% | +0.609% +/- 1.258% | **HOLD**; W2 lead only, not promotion-qualified |
| R234 / QKV beta-alpha | -0.725% +/- 1.011% | +0.143% +/- 0.929% | **REJECT**; noise/regression |

Every arm exited 0, returned to `RTX 5090, 7 MiB used, 32181 MiB free`, and
produced no Xid. Candidate fixed-control logs all contained `PARITY_OK`; these
are functional smoke checks, not a replacement for the existing full top-20
logit certificate. R248 remains an opt-in W2 investigation only; it is not in
the frozen Preview profile and must not be enabled by default.

### DeepSeek Flash 4.1 decode post-review — 2026-09-14

Hermes artifact: `/tmp/escha-hermes-decode-postreview-20260914.txt`, SHA-256
`f34133400898ab208613516c17119f665b80fd538e13169154130157e00fa931`.
DeepSeek independently re-parsed the 52 receipts and reproduced the paired
deltas. Its recommendation is to keep R244 and R234 off, and run one fresh
5--7-pair confirmation batch for R248 on W2 only before any scoped promotion;
E3 does not yet show a meaningful R248 effect. It also flagged that future
receipts should store explicit exit-code files, not infer exit 0 from logs.

### 2026-09-14 fresh R248 W2 confirmation

Fresh run root: `/home/sean/kernel-lab5090/runs/escha-sm120-r248-confirm-20260914-123932`.
The exact fixed-control profile was reused, with only
`ESCHA_OFFICIAL_RAW_DECODE_DOWN_ADD_RMS_FUSION=1` changed for the candidate.
Each arm ran a fresh `llama-bench -m escha-w2-firstclass.gguf -p 0 -n 256
-b 2048 -ub 512 -t 8 -ngl 99 -fa on -ctk f16 -ctv f16 -r 3 -o json` process.

| Pair | R248 candidate tok/s | Control tok/s | Paired delta |
|---|---:|---:|---:|
| 1 | 85.034109 | 83.718199 | +1.572% |
| 2 | 84.762515 | 83.722153 | +1.243% |
| 3 | 84.817467 | 83.369179 | +1.737% |
| 4 | 84.576423 | 83.449722 | +1.350% |
| 5 | 84.435352 | 85.059160 | -0.733% |

Aggregate: **+1.034% +/- 1.006%** sample SD, with 4/5 candidate wins. All ten
bench processes have explicit exit-code files containing `0`; all ten Xid
receipt files are empty; every post-run GPU snapshot shows the RTX 5090 at
7 MiB used / 32181 MiB free. Complete receipt-list SHA-256:
`6481c6e42cf25120b94ae12e9e04eeb1787c92c27eff99eae82fffd72b8d1a3a`.
R248 remains **HOLD / W2-scoped only**: the positive effect is small and not
promotion-qualified, and no default or frozen-profile setting was changed.

### 2026-09-14 SM89 ABI-to-SM120 prefill reconciliation

Static comparison of `preview-bridge-sm89/README.md`,
`preview-bridge-sm120/README.md`, the RC frozen profile, and both bridge
manifests found no missing SM89 prefill selector on SM120. The profile already
matches on raw bridge/SwiGLU fusion, code-GEMM cubin, mixed accumulation, GDN
chunk bridge, and the R258/R260/R262/R264 stack; only artifact paths and the
decode-only direct-F32 ABI selector differ. The source confirms why the native
SM89 fallback knob is not a hidden SM120 opportunity: raw prefill enters the
official bridge early-return at `ggml/src/ggml-cuda/escha-moe.cu:2960-2982`,
and passes `escha_official_bridge_acc_mode()` (`:278-291`) directly to the
code-GEMM bridge. `ESCHA_NO_PREFILL_MIXEDACC` is below that return and is not
consulted by this route.

Fresh run root:
`/home/sean/kernel-lab5090/runs/escha-sm120-prefill-abi-20260914-125807`.
Every arm used commit `2d1411c2c9eaca7d92bdd6a84279e34e580c4cf7`, the frozen
SM120 wrapper/cubin/GDN hashes above, `-p 2048 -n 0 -b 2048 -ub 2048 -t 8
-ngl 99 -fa on -ctk f16 -ctv f16 -r 5 -o json`, and a fresh process. The table
reports llama-bench's five-sample mean/SD; warm values remove the first cold
sample only for diagnosis, not for release scoring:

| Model / policy | Reported tok/s | Warm tok/s | Delta vs mixed warm |
|---|---:|---:|---:|
| W2 / mixed | `2801.799 +/- 308.626` | `2939.820 +/- 0.748` | baseline |
| W2 / fp32 | `1620.726 +/- 31.825` | `1634.898 +/- 3.397` | `-44.39%` |
| W2 / k2-fp32 | `1889.842 +/- 40.441` | `1907.923 +/- 1.440` | `-35.10%` |
| W2 / mixed + `NO_PREFILL_MIXEDACC=1` | `2903.466 +/- 80.699` | `2939.555 +/- 0.508` | `-0.01%` |
| E3 / mixed | `2659.670 +/- 84.407` | `2695.048 +/- 34.011` | baseline |
| E3 / fp32 | `1583.686 +/- 34.627` | `1599.080 +/- 4.322` | `-40.67%` |
| E3 / k2-fp32 | `1825.189 +/- 29.917` | `1838.068 +/- 9.362` | `-31.80%` |
| E3 / mixed + `NO_PREFILL_MIXEDACC=1` | `2720.845 +/- 32.942` | `2735.283 +/- 7.541` | `+1.49%` |

The mixed-vs-fp32 gap is a large regression, not a gain; the k2-fp32 policy
also regresses. The native fallback selector is within run noise, confirming it
is bypassed by the raw bridge. Six additional bounded fixed controls used
`llama-cli --single-turn --reasoning off` and all emitted exactly `PARITY_OK`
(W2/E3 × mixed/fp32/k2-fp32), with exit code 0 and no Xid. All timing and
control arms returned to `RTX 5090, 7 MiB used, 32181 MiB free`; the final
idle check was P8 with no Xid and Qwen still inactive. Receipt-list SHA-256
(197 evidence files, including explicit `env-final.txt` snapshots and the
artifact manifest; exploratory invalid CLI directories excluded):
`63c3aab92419ca898d09f404b9a2058b45da095181ac5fb18c5af670764ffcc2`.

Two exploratory CLI attempts were invalid/non-results: the first, without
`--single-turn --reasoning off`, grew `w2-mixed-control/stdout.txt` to
12,030,051,388 bytes; the follow-up ended after a partial thinking trace. The
generated stdout files were truncated to zero bytes and both directories are
excluded from the receipt manifest; no model/source/artifact file was removed,
and the subsequent bounded controls are the only correctness results counted
above.

Decision: **NO ABI PREFILL GAIN / KEEP MIXED**. Do not change the frozen profile.
The remaining path toward 3,500 tok/s is a new specialized/fused kernel (most
likely GDN/chunk or equivalent), not an SM89 bridge-environment toggle.

### 2026-09-14 KVarN variant and boundary diagnosis

The short direct controls for KVarN 3/2, KVarN 3/3, and regular `q4_0` all
returned `PARITY_OK` on the RTX 5090. The KVarN 3/2 full Club-3090 verification
was then run with route tracing enabled. It passed at the short n_kv=256 control
but failed at the first longer boundary (n_kv=512), producing repeated `/`
tokens and failing tool-call checks. Route receipts show 383
`KVARN_DECODE_SPLIT` dispatches and 160 windowed/prompt-generic dispatches;
the GPU returned to 7 MiB with no Xid. Evidence roots:
`/home/sean/kernel-lab5090/runs/escha-sm120-club3090-gates-20260914-143704/kvarn32-verify`
and `.../kvarn33-probe`, `.../q4-probe`.

As a correctness oracle, the same KVarN 3/2 full verification was repeated with
`GGML_KVARN_TEST_FORCE_PORTABLE_CAPABILITY=1`, which disables the specialized
matrix routes while retaining KVarN quantization. That run passed all applicable
checks, including coherent streaming, tool calls, reasoning, and output quality
(lexical variety `0.630`). This isolates the defect to the specialized CUDA
split/record-backed path rather than the model, bridge, or quantized cache.
The portable receipt is under
`/home/sean/kernel-lab5090/runs/escha-sm120-club3090-gates-20260914-143704/kvarn32-portable-verify`.

DeepSeek Flash 4.1 reviewed the exact evidence through Hermes/OpenRouter
(review receipt `/tmp/deepseek-kvarn-final-review-20260914.usage.json`). Its
recommendation is: A/B the record-backed and generic rotated K/V loads on the
same n_kv=512 tensor, force split-on at 256 and split-off at 512, then apply an
auditable runtime gate that keeps the specialized split route only in the
verified envelope (currently n_kv <= 256) and falls back to the portable rotated
loader above it. After that containment, repair the tile/record indexing and
raise the gate only after 256/384/512/768/1024 boundary regressions pass.

Decision at review time: **SPECIALIZED KVARN PATH NOT RELEASE-QUALIFIED**. The
safe next implementation was the bounded fallback gate plus a tensor-level
loader diff. F16-KV testing remains stopped per request.

### 2026-09-14 KVarN conservative fallback gate

DeepSeek's containment recommendation was implemented in
`ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu`. The default route now keeps the
specialized path only for the directly verified Q=1, n_kv=256 split shape. It
falls back to portable/native KVarN for all other shapes at n_kv=256 and above,
covering both prompt/windowed prefill and decode. A diagnostic-only opt-in,
`GGML_KVARN_TEST_ENABLE_LONG_SPECIALIZED_DECODE=1`, is required to re-enter the
unverified specialized paths; the frozen profile does not set it.

The first decode-only gate was intentionally retained as a diagnostic result,
not a release result: it still failed because the Q=13 generic-matrix path at
n_kv=256 poisoned the cache before long decode. The tightened all-shapes gate
was rebuilt and passed the complete Club-3090 functional verification:

| Result | Value |
|---|---|
| Evidence root | `/home/sean/kernel-lab5090/runs/escha-sm120-kvarn-gated-q1-20260914-151755` |
| Verify exit | `0` — all applicable checks passed |
| Output quality | 9,575 chars; lexical variety `0.620`; no repeated-line cascade |
| Tool/stream/reasoning | tool calls, streaming, streaming tool calls, and reasoning all passed |
| Route evidence | 96 specialized `decode-split`; 335 `portable-native`; no long specialized route |
| GPU safety | RTX 5090 returned to 7 MiB / 32181 MiB free, P8; after-Xid receipt empty |

Build validation completed with `cmake --build build-sm120-preview --target
llama-server llama-cli -j4`, the route-policy unit test passed via CTest, and
`git diff --check` is clean. Final artifact hashes are:

```text
b3a355e9dfcdb0ab9666792603b91138450473bda021596fbffc85cbedced1ed  build-sm120-preview/bin/llama-server
3a39d93c4b70eb34e94d702540789de5e9b76aa0e848ae0e1470e70042fb6411  ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu
```

This is a correctness containment, not a performance claim: long-context KVarN remains portable until
the planned tensor-level record-loader A/B and 256/384/512/768/1024 regressions
identify and repair the indexing defect.

### 2026-09-14 DeepSeek post-fix audit

DeepSeek Flash 4.1 reviewed the final gate and full verification after the
source change. It judged the containment adequate for a correctness-safe RC
preview, with the explicit qualifier that this is containment rather than a
fast-path repair and that the passing cell does not certify wider Q/n_kv
invariants. It called out these residual risks: an empty Xid receipt rules out
hardware faults but not silent wrong data; the diagnostic opt-in must remain
non-shipping and ideally assert-fail outside debug builds; route-selection
coverage should be locked with tests; Q=1 prefill/windowed coverage at n_kv=256
needs confirmation; and every alternate dispatch site must be checked for
specialized-route reachability.

Its exact next experiment is a fixed-seed two-factor bisect against portable
KVarN: sweep Q in `{1,2,4,8}` at n_kv=256, then sweep n_kv in
`{128,256,512,1024}` at Q=1. Diff per-token logits, record the first-divergence
token, and hash the record buffer per CTA to locate the split-boundary
corruption.

The short post-fix receipt is preserved at
`/tmp/deepseek-kvarn-postfix-audit-20260914.txt` with usage metadata at
`/tmp/deepseek-kvarn-postfix-audit-20260914.usage.json`. Its follow-up narrows
the first repair probe to the exact slash-loop prompt with Q=1 and
n_kv `{256,272,320,384}` under the old dispatch, diffing specialized versus
portable logits per step to find the first divergence.

### 2026-09-14 W2 intrinsic-tail boundary A/B and long-specialized verify-full

The KVarN intrinsic-tail boundary was forced on W2 KVarN 3/2 by caching a
250-token prefix and appending 4 tokens (250 -> 254; the cache reprocessed
exactly 4). The prompt path logged `route=prompt-generic-mma` at `nkv=256`,
`nq=128`, `entry=compact-tail` -- the KVarN 128-token exact-tail recompute on
the fail-closed fallback path, not the record-backed specialized path. Decode
logged `route=decode-split` at `nkv=256` (80) and `nkv=512` (64) for a 64-token
generation. Three fixed-seed arms produced byte-identical 64-token text: fast
record loader; split MMA with `GGML_KVARN_TEST_DISABLE_FAST_RECORD=1`;
portable-native. Receipts:
`/home/sean/kernel-lab5090/runs/escha-sm120-kvarn-q4-boundary-{fast-163000,generic-163100,portable-163200}`.

Verifier caveat: the 64-token output is a degenerate repeated-token attractor
(`" x"` x64), so token identity is weak evidence. The discriminating signal is
the same-math fast-vs-generic selected-token logprob delta (max `2.65e-5`, mean
`6.5e-6`); fast-vs-portable is max `1.09e-2`, mean `5.16e-4` (algorithm
differs, expected). No top-1 changes; temp 0.7 with a different fixed seed also
matched. No Xid; GPU returned to 7 MiB.

A separate full Club-3090 verify-full with long specialized enabled and fast
records passed all 9 checks (exit 0): 4671 chars, variety 0.615,
max_line_repeat 0, tools/streaming/reasoning pass, routes exercised at
`nkv=512/768/1024`. Receipt:
`/home/sean/kernel-lab5090/runs/escha-sm120-kvarn-specialized-full-fast-20260914-161500`.

Verdict update: **containment still required.** The n_kv=512 fast-loader
corruption did not reproduce under fixed-seed 3-arm A/B, but (i) the Q>1
generic-matrix cell at n_kv=256 -- the historically failing cell -- remains
fail-closed and unverified on the specialized path; (ii) the 512 pass is
behavioral-only (one prompt, degenerate output), with no tensor-level per-step
full-vocab logit diff or per-CTA record-buffer hash; (iii) unaligned
272/320/384 boundaries were unexercised. Do not remove the portable containment
gate. Any widening must first generalize the Q>1 clause (`n_kv==256 && Q>1` ->
`Q>1 && n_kv>=256`) before raising
`GGML_CUDA_FATTN_KVARN_SPECIALIZED_DECODE_MAX_KV` 256 -> 512. F16-KV testing
remains stopped per request.

### 2026-09-14 native-default KVarN RC qualification

Following the intrinsic-tail A/B, the specialized long-context route was made
native-by-default in `fattn-kvarn-dispatch.cu`. The previous diagnostic opt-in
was inverted into the test-only emergency rollback
`GGML_KVARN_TEST_DISABLE_LONG_SPECIALIZED_DECODE=1`. DeepSeek Flash 4.1's final
Hermes review accepted this for conditional RC shipment and identified one
remaining bounded follow-up: compare native versus rollback generations at
`nkv=512/768/1024` for 256 greedy tokens, with route attribution and per-step
log-probability checks. It explicitly recommended no F16-KV testing.

The final source tightened rollback semantics as well: when portable-native is
unsupported, both specialized vector and split eligibility are disabled so the
normal generic/materialize fallback handles the shape (no fail-open return to
record-backed decode). Build and the focused route-policy CTest passed.

Final native-default W2 KVarN 3/2 full Club-3090 receipt:
`/home/sean/kernel-lab5090/runs/escha-sm120-kvarn-native-final-full-20260914-170000`.
The verify exit is `0`; all nine applicable checks passed (Paris, tools,
streaming, streaming tool-calls, reasoning, and quality). Quality output was
4,621 chars with lexical variety `0.580` and max repeated line `0`. Route trace
counts were 303 `decode-split`, 16 `generic-mma`, and 160
`prompt-generic-mma`; no portable fallback, CUDA error, or Xid was observed.
The RTX 5090 returned to P8 with 7 MiB used / 32181 MiB free.

The requested KVarN 3/3 variant also passed the full Club-3090 functional gate
under the same native-default binary:
`/home/sean/kernel-lab5090/runs/escha-sm120-kvarn-native-final-kvarn33-full-20260914-171200`.
Verify exit was `0`; all nine applicable checks passed, quality was 4,598 chars
with lexical variety `0.610` and max repeated line `0`, and route counts again
were 303 `decode-split`, 16 `generic-mma`, and 160 `prompt-generic-mma`. The
RTX 5090 returned to P8 at 7 MiB with no Xid/CUDA error.

Variant receipts under the final binary:

| Variant | Receipt | Result | Route/GPU notes |
|---|---|---|---|
| KVarN 3/3 | `/home/sean/kernel-lab5090/runs/escha-sm120-kvarn-native-default-kvarn33-20260914-165000` | `PARITY_OK`, exit 0 | 32 `decode-split` + 16 prompt-generic; 7 MiB idle; no Xid |
| regular q4 (`q4_0`) | `/home/sean/kernel-lab5090/runs/escha-sm120-kvarn-native-default-q4-20260914-165100` | `PARITY_OK`, exit 0 | no KVarN route selected for the 21-token control; 7 MiB idle; no Xid |

Final artifact hashes after the rollback hardening:

```text
3c0d474967247c533c2e007725c00b3bdea1ce7068a5fb7884b4fb05fadb0759  ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu
b7b77c23144441ca4822e591efa3d4c1d1a316b9e3468eddfd579ec8704ac7fd  ggml/src/ggml-cuda/fattn-mma-kvarn-decode-decl.cuh
57fd293daef2804d927e20e00342577706c46a05a4324fbf1d2f0463b6f0eb9b  ggml/src/ggml-cuda/fattn-mma-kvarn-decode.cuh
b3a355e9dfcdb0ab9666792603b91138450473bda021596fbffc85cbedced1ed  build-sm120-preview/bin/llama-server
2fcd11db886b2ef6ea2a9a9bd4bb023df9394f63be151203022014bd2d8fe882  build-sm120-preview/bin/llama-cli
```

The RC remains conditional rather than a claim of full mathematical parity:
the earlier un-gated failure is not reproduced by the fixed-seed record-loader
A/B or the long native full verify, but the DeepSeek-recommended 3-depth,
256-token native-vs-rollback generation comparison is still a scheduled
follow-up. F16-KV work remains explicitly excluded.

### 2026-09-14 DeepSeek post-hardening review

DeepSeek Flash 4.1 re-read the final dispatch source and raw receipts after the
rollback hardening. It verified the description and all headline numbers,
confirmed that the final full run executed 175 `decode-split` calls at
`nkv>=512` (111 at 512, 32 at 768, 32 at 1024), and found no reproducible
blocker. Verdict: **ship as RC**, with the residual note that the original
`nkv=512` slash event's root cause remains unidentified and the long-context
native path has no independent fixed-seed full-vocabulary equivalence receipt.
It also confirmed there is no performance claim in this evidence; route counts
are not tokens/sec. Its single recommended follow-up is a bounded two-arm,
fixed-seed A/B at `nkv=512` and `nkv=768` comparing default specialized decode
against `GGML_KVARN_TEST_DISABLE_LONG_SPECIALIZED_DECODE=1`, including route
proof, selected-token log-probability deltas, and sampled-text equality. The
review receipt is `/tmp/deepseek-kvarn-native-final-posthardening.txt` (Hermes
usage metadata `/tmp/deepseek-kvarn-native-final-posthardening-20260914.usage.json`).

The rollback switch was then exercised end-to-end in
`/home/sean/kernel-lab5090/runs/escha-sm120-kvarn-rollback-probe-20260914-171100`.
A bounded 763-token fixed prompt plus 64-token greedy completion exited 0 and
returned coherent text with per-token logprobs. Route traces showed 64
`portable-native` decode calls and 48 `prompt-generic-mma` calls (no split
decode re-entry); the RTX 5090 returned to 7 MiB with no Xid/CUDA error. This
closes the runtime-branch evidence gap for portable-supported long shapes.

# Escha on BeeLLaMA v0.4.7 Preview

## Source and build identity

- Escha starting point: `escha-rc-v046-full` at `2d1411c2c9eaca7d92bdd6a84279e34e580c4cf7`.
- BeeLLaMA Preview: `preview-v0.4.7` at `1156b183630eb8bd76ff931348d0b1a258ae171b` (rolling prerelease; pin this commit, not the moving image tag).
- The candidate is an uncommitted merge in `/home/sean/work/escha-beellama-v047-candidate`, with the original working changes from `/home/sean/kernel-lab5090/beellama-escha/rc-v046` applied. The original tree and release binaries remain intact.
- The saved original working patch is `/home/sean/work/escha-v046-working-20260923.patch` (SHA-256 `1e18ca71cc888d2c5d1c08b0d6399442a436dd576494e6d427c2224ba4c45a96`).
- The only patch conflict was in `fattn-mma-kvarn-decode.cuh`: the resolution retains Preview's `disable_fast_records` gate and Escha's `RECORD_DIM` payload sizes for D64 rectangular records.
- The frozen SM89 champion carried additional release-critical source absent from the Git RC branch. Its `escha-moe.cu` (SHA-256 `7043844d73216d4d721945948eb8612332470605691245d260998e99a7c46ea3`) was copied into this checkout, preserving exact-row generation tiles and the all-output, at-most-eight-row raw bridge gate. The corresponding graph output flag was carried into both `build_escha_mm` and `build_escha_mm_aux` without replacing Preview's other `llama-graph.cpp` changes.
- The first GPU launch exposed another omitted champion patch: the DFlash2 E3 draft failed to load at `src/models/dflash.cpp:692` because the Preview base lacked LowGPU placement for its embeddings and output projection. The frozen champion's `src/models/dflash.cpp` was copied exactly (SHA-256 `3465774d5bcf57d492942031bf6743707eb9f235a469eb8c09221fc52316e5c5`) and the server was rebuilt. The subsequent E3 control and long-prompt requests completed.
- Linux SM89 build source is copied to `z840:/home/sean/work/escha-beellama-v047-candidate`; configure uses CUDA 12.8, `CMAKE_CUDA_ARCHITECTURES=89`, Release, CUDA FlashAttention, and KVarN. Local SM120a configuration uses CUDA 13.0. Both compile and runtime status must be recorded separately below.

## Draft memory feature and tuning

Preview introduces `--spec-draft-ubatch-size` (`-ubd`), default 128. It controls the model-backed draft context's physical batch size independently of target `-ub`, while the draft keeps target `-b` as logical capacity. The startup log prints both contexts' actual `n_batch` and `n_ubatch`. This knob affects draft prefill and its graph/workspace allocation; it does not shrink Escha model weights or target KV.

For the existing E3 12 GiB profiles, hold model files, context, slots, K/V types, exact tail, draft depth, sampler, and bridge flags constant when comparing binaries:

| Profile | Existing target `-ub` | First draft `-ubd` arms | Why |
|---|---:|---:|---|
| 80K / N3 / Q4 draft | 64 | 64, 32, 128 | 64 reproduces the old geometry; the new default 128 may increase memory. |
| 96K / N3 / Q2 draft / KVarN window 16K | 256 | 128, 64, 256 | 128 is the Preview default and may reduce draft workspace; 256 is the old coupled geometry. |
| 16 GiB or larger | profile specific | 128, 64, 256 | Select by measured peak memory and real prefill/decode throughput. |

Use fresh server processes and the same prompt set for each arm. Record startup target/draft `n_ubatch`, peak GPU allocation, free VRAM after load and after long prefill, host RSS, prefill tok/s, decode tok/s, draft acceptance, and deterministic output hashes. The existing 96K conservative control is RC047 E3, target B1024/UB256, DFlash2 N3 Q2, q2_0 draft KV, and `GGML_KVARN_WINDOW_CHUNK=16384`; its three-run reference was 1115.35 client prefill tok/s at 50K and 83.75 narrative decode tok/s. These are historical reference values, not Preview measurements.

Keep `--spec-draft-kvarn-window-chunk` out of this comparison: it only changes owned KVarN draft caches, whereas these profiles use q4_0 or q2_0 draft KV.

On z840, the 96K Preview arm can be launched from a separate shell with the existing research profile and a candidate `BIN` override:

```bash
BIN=/home/sean/work/escha-beellama-v047-candidate/build-sm89/bin \
CTX=98304 UB=256 BATCH=1024 NMAX=3 \
DRAFT=/home/sean/models/qwen3.8-27b-dflash2/Qwen3.8-27B-DFlash2-Q2_K.gguf \
DRAFT_KV_K=q2_0 DRAFT_KV_V=q2_0 \
GGML_KVARN_WINDOW_CHUNK=16384 \
bash /home/sean/escha-release/launch-e3-12gb-candidate.sh \
  --spec-draft-ubatch-size 128
```

For the measured 96K profile, use `--spec-draft-ubatch-size 64`. Change only that value to 256 for the coupled-geometry control. For the 80K profile, explicitly set 64 to retain its old target/draft geometry; the Preview default of 128 has not been qualified there. Keep the old champion binary and all its source manifests untouched.

### 2026-09-23 SM89 96K tuning receipt

Hardware: z840 RTX 4090, CUDA 12.8, 24 GiB physical VRAM. Candidate: `build-sm89/bin/llama-server` from the source snapshot above. All arms used the existing E3 12 GiB candidate launcher with CTX98304, B1024, target UB256, one slot, DFlash2 N3 Q2, q2_0/q2_0 draft KV, target KVarN3/KVarN2, exact tail128, and a 16K KVarN window. The only changed argument was `--spec-draft-ubatch-size`. Each arm used a fresh server, a temperature-zero control request, and the same deterministic 18,011-token prompt with 128 generated tokens, twice. The server log confirms target `n_ubatch=256` and draft `n_ubatch` matching the arm. Raw JSON receipts and server logs are in `z840:/home/sean/work/escha-beellama-v047-candidate/tuning/`; the exact probe is `probe.py` in the candidate directory on z840.

| Draft `-ubd` | Peak GPU used (MiB), two runs | Prefill tok/s, two runs | Decode tok/s, two runs | Draft/accepted | Output SHA-256 |
|---:|---:|---:|---:|---:|---|
| 256 | 11862, 11862 | 1186.23, 1186.35 | 65.75, 64.56 | 147/76 | `33633ca4154a10f28ddece27314104cb213b7b62f5504454f70a8d9bbdbfa935` |
| 128 | 11596, 11596 | 1231.71, 1241.81 | 69.83, 69.88 | 141/78 | same |
| 64 | 11460, 11462 | 1277.39, 1284.69 | 69.50, 69.64 | 141/78 | same |

`-ubd 64` saved 400–402 MiB of observed peak GPU memory versus the old coupled value 256, and 134–136 MiB versus the new default 128. Mean prefill rose about 8.0% versus 256; mean decode rose about 6.8% versus 256. Decode was about 0.4% slower than 128 in these short runs, so the main reason to select 64 is memory headroom and prefill. These numbers are total `nvidia-smi` used memory, not an isolated draft allocator trace. The control returned `E3_12GB_OK` in each arm, and all six long-request output hashes matched. They are a bounded 18K-prompt probe on a 24 GiB card, not a physical 12 GiB or full 96K qualification. Host RSS was not sampled. The 80K Q4 profile was not rerun.

Do not promote an arm on startup fit alone. Check the long-prompt transition to decode, output parity, throughput, and 12 GiB headroom on a physical target card before updating the consumer profile. The existing 80K N3 arm used about 11,535 MiB under a simulated 12 GiB envelope; the 96K arm's 16K KVarN window is the conservative choice from its previous paired sweep.

## Qualification status

Both z840 Release CPU and CUDA 12.8 SM89 `llama-server` builds linked successfully from this merged source, including an incremental rebuild after the DFlash LowGPU patch. The SM89 CUDA `llama-cli` also linked. The server `--help` exposes `--spec-draft-ubatch-size` with default 128. The CUDA link required `LIBRARY_PATH` and `LD_LIBRARY_PATH` to include `/home/sean/cuda-12.8/targets/x86_64-linux/lib`, matching the frozen release build script. The copied remote source lacks Git metadata, so its generated build revision is `unknown`; use the pinned source references and file hashes above, not that generated string, for provenance.

SM89 binary SHA-256: `llama-server` `e1e8a4b4a034e97e383282813ea83b3679527423dce56fa9e955bfc9e535ceb1`, `libllama.so.0.4.7` `34b79eff3ded96aafc80467ff01e039102a6e0464f6c95f7e58bd4a845a1f7f8`, and `libggml-cuda.so.0.23.0` `58f9fc4a8362c2056d1af3522b90334c6ef28694bb5d3bdf466de53345228610`.

The 96K SM89 runtime probe passed and favors `-ubd 64`. A local SM120a build directory was configured for CUDA 13.0, but no SM120a build or runtime result exists yet; the local 5090 host had only 1.6 GiB available system RAM when this work reached that step. Physical 12 GiB, full 96K context, 80K Q4, W2, other architectures, bridge packaging, and release gate checks remain open. Keep the v0.4.6 champion and its manifests frozen until an architecture-specific Preview candidate passes the release gates in `docs/escha-multiarch-release-roadmap.md`. The original port-8080 server was restored with its captured command and returned `{"status":"ok"}` after tuning.

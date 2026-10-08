# L0xRE BeeLLama

**R6 is the fastest certified SM86 runtime yet — 37 t/s prose and 65 t/s code on the RTX 3060.** Run **L0xRE-27b-Low** on NVIDIA RTX 30 / 40 / 50 series GPUs with an 80K context, INT8 prefill, and the Q4_K_M DFlash2 drafter. One archive contains the SM86 / SM89 / SM120 runtime paths; model weights are a separate download.

## TLDR

- **Fastest SM86 speed to date:** **37 t/s prose** · **65 t/s code** (measured on RTX 3060, 12 GB).
- **Certified:** the numbers above are the certified **Linux / WSL SM86** result.
- **Windows is a candidate:** installable, but no certified GPU throughput yet (run `--qualification-probe`).
- **80K context** (81,920 tokens) with INT8 all-projection prefill and gated FP16 long attention.
- **12 GB VRAM minimum** (not RAM). RTX 30 / 40 / 50 series.

## Speed card (measured, RTX 3060 / 12 GB)

| Metric | Value |
| --- | --- |
| **Prose (narrative)** | **37 t/s** |
| **Code** | **65 t/s** |
| Cold prefill 10K | 566 t/s |
| Cold prefill 77K | 469 t/s |
| Context capacity | 81,920 tokens |
| vs R5 (29 / 54 t/s) | **+27% prose · +20% code** |

These are the frozen seed-42 certified measurements (37.16 / 65.06 sustained; 37.27 / 65.21 frozen screen). Balanced across seeds 42 / 1234: 34.95 / 62.69, versus R5's 29.21 / 54.09.

## Quick start (Linux / WSL — certified SM86)

Download the R6 Linux archive and its checksum into one directory:

| File | Download |
| --- | --- |
| Runtime | [L0xRE-BeeLLama-Low-R6-sm86-linux.tar.zst](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r6/L0xRE-BeeLLama-Low-R6-sm86-linux.tar.zst) |
| Checksum | [SHA-256](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r6/L0xRE-BeeLLama-Low-R6-sm86-linux.tar.zst.sha256) |
| Model + drafter | [L0xRE-27b-Low on Hugging Face](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low) |

```bash
sha256sum -c L0xRE-BeeLLama-Low-R6-sm86-linux.tar.zst.sha256
tar --zstd -xf L0xRE-BeeLLama-Low-R6-sm86-linux.tar.zst
cd L0xRE-BeeLLama-Low-R6-sm86-linux
sha256sum -c SHA256SUMS
mkdir -p models
curl -fL --retry 3 -o models/L0xRE-27b-Low.gguf \
  https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/9b74c81c19f8372888c2968b5334f42b0354d8b1/L0xRE-27b-Low.gguf
curl -fL --retry 3 -o models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/9b74c81c19f8372888c2968b5334f42b0354d8b1/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
sha256sum -c MODEL-SHA256SUMS
./l0xre serve --profile 12gb -m models/L0xRE-27b-Low.gguf \
  -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --host 127.0.0.1 --port 8080
```

## Update / upgrade guide

- **From r5-fast / r4 / r3 → R6:** stop the old server, download the R6 archive above into a fresh directory, verify checksums, and launch with `--profile 12gb`. The model files are unchanged (same SHA-256 identities below), so reuse them or re-download from the pinned HF revision.
- **Rollback:** extract an earlier release into a separate directory and reuse the unchanged model files; no service install or model conversion is performed.
- **Windows:** the certified numbers are Linux SM86 only. The Windows SM86 package is a candidate — verify its checksum, run `--qualification-probe`, and treat throughput/quality as unmeasured until you test it on your hardware.
- **SM89 / SM120:** common source and kernel compilation are prepared (see [PARITY.md](PARITY.md)), but complete fast runtimes and hardware qualification are pending; they are not certified in R6.

## Downloads

**[R6 release](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/beellama-v0.4.7-universal-r6)** · [Archive checksums](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r6/L0xRE-BeeLLama-Low-R6-sm86-linux.tar.zst.sha256)

| Platform | Download | Status |
| --- | --- | --- |
| **Linux x86-64 / WSL2 (SM86)** | [R6 SM86 `.tar.zst`](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r6/L0xRE-BeeLLama-Low-R6-sm86-linux.tar.zst) | **Certified** — 37 / 65 t/s, 80K context |
| Windows x64 (SM86) | [R6 SM86 candidate `.zip`](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r6/L0xRE-BeeLLama-Low-R6-sm86-windows-candidate.zip) | Candidate — offline checks passed; GPU throughput/quality pending |
| Linux (SM89) | [R6 SM89 candidate `.tar.zst`](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r6/L0xRE-BeeLLama-Low-R6-sm89-linux-candidate.tar.zst) | Candidate — awaiting SM89 hardware certification |
| Windows (SM89) | [R6 SM89 candidate `.zip`](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r6/L0xRE-BeeLLama-Low-R6-sm89-windows-candidate.zip) | Candidate — awaiting SM89 hardware certification |
| Model + drafter | [L0xRE-27b-Low on Hugging Face](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low) | `L0xRE-27b-Low.gguf` + Q4_K_M drafter; SHA-256 identities below |

Only the Linux SM86 package carries the certified 37 / 65 t/s numbers. The Windows and SM89 packages are installable test candidates; use `--qualification-probe` and do not treat them as certified. Older releases (r5-fast, r4, r3) remain available as rollback options.

## Requirements

- An AVX2-capable x86-64 CPU and an RTX 30 / 40 / 50 series GPU with compute capability 8.6 / 8.9 / 12.0 and at least 12 GB VRAM; check with `nvidia-smi`.
- Linux/WSL: Ubuntu 24.04 or an ABI-compatible x86-64 distribution. The fast launcher requires Python3.12+ and `numactl`, with at least8 available CPU workers. The SM120 payload needs glibc 2.38+ and `GLIBCXX_3.4.32`. WSL2 needs an NVIDIA Windows driver with WSL CUDA support.
- Windows: x64 Windows with PowerShell and an NVIDIA driver compatible with the bundled CUDA 13.3 runtime. CUDA runtime and MSVC runtime DLLs are included; a CUDA toolkit is not required to run it.
- Linux bundles CUDA 12.8 for SM86/SM89 and CUDA 13.0 for SM120. The host supplies the NVIDIA driver. Reserve about 20 GB of disk space for the runtime archive, extraction, and the two model files.

The target is a custom GGUF supported by this L0xRE runtime. Use this runtime for these weights; see the model card for model-specific compatibility and licensing.

## Linux / WSL installation

The quick start above covers the certified SM86 R6 path. If you need the fast-profile launcher instead, see [FAST-RELEASE.md](FAST-RELEASE.md). For the certified R6 path, extract the R6 Linux archive (the filename and checksum are in the Downloads table above), then verify, download the model, and launch with `--profile 12gb`. Install `zstd` and `curl` if needed (`sudo apt-get install zstd curl`).

The launcher detects the first visible GPU. On a machine with multiple GPUs, set `CUDA_VISIBLE_DEVICES` to the intended GPU index or UUID before launch. For example, `CUDA_VISIBLE_DEVICES=1 ./l0xre serve ...` selects GPU 1. `L0XRE_ARCH=sm86|sm89|sm120` overrides routing; it does not change which GPU CUDA uses.

## Windows installation

The certified 37 / 65 t/s numbers are Linux SM86 only; the Windows SM86 package is a candidate. Download the R6 SM86 Windows ZIP and its `.sha256` (in the Downloads table above) into one directory. Open PowerShell there, compare the ZIP's `Get-FileHash` result with its `.sha256` line, and extract:

```powershell
Get-FileHash .\L0xRE-BeeLLama-Low-R6-sm86-windows-candidate.zip -Algorithm SHA256
Expand-Archive .\L0xRE-BeeLLama-Low-R6-sm86-windows-candidate.zip -DestinationPath .
Set-Location .\L0xRE-BeeLLama-Low-R6-sm86-windows-candidate
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -fL --retry 3 -o models\L0xRE-27b-Low.gguf https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/9b74c81c19f8372888c2968b5334f42b0354d8b1/L0xRE-27b-Low.gguf
curl.exe -fL --retry 3 -o models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/9b74c81c19f8372888c2968b5334f42b0354d8b1/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
Get-FileHash .\models\*.gguf -Algorithm SHA256
.\l0xre.cmd serve --profile 12gb -m models\L0xRE-27b-Low.gguf -md models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf --host 127.0.0.1 --port 8080
```

Compare both model hashes with the table below before running. On a multi-GPU Windows machine, set `$env:CUDA_VISIBLE_DEVICES="1"` (or the GPU UUID) before launch. Stop the server with Ctrl+C. The API is at `http://127.0.0.1:8080/v1`; check `http://127.0.0.1:8080/health` for readiness.

## Models and integrity

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `L0xRE-27b-Low.gguf` | 8,619,127,680 | `b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543` |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1,143,006,816 | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |

Downloads are pinned to HF revision `9b74c81c19f8372888c2968b5334f42b0354d8b1`, verified against the repository's LFS identities on September 29, 2026. The model card explains the current recommended setup; the pinned model bytes are unchanged. The [MTP variant](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low-MTP) is a separate target file and is not interchangeable in benchmark attribution.

## Profiles

**Updated SM86 12 GB default:** `12gb`, `12gb-quality`, and `12gb-b84` select **80K context, target KVarN4/4, Q4_0/Q4_0 DFlash2 caches**, Q4_K_M drafter weights and medium reasoning. INT8 prefill and gated FP16 long attention are enabled. SM89/SM120 retain the r3 80K KVarN3/3 preset.

SM86 uses one slot, batch1024 / target ubatch512, draft ubatch32, DFlash2 depth3, GPU target/draft layer limit99, CPU token embeddings, no-op-offload, FlashAttention on, exact KV tail128, window16384, cache RAM0, fit off / target768, CPU32 workers and an8192-token reasoning budget. SM89/SM120 retain targetUB256/draftUB128 and their original placements. Sampling remains temperature0.7/top-p0.95/top-k20. A supplied Q8_0 projector runs on CPU with1024 image tokens; vision requires `--mmproj FILE` and a separate download.

| Platform / GPU | `12gb` selection | Other options |
| --- | --- | --- |
| Linux / SM86 | 80K prefill: 81,920 context, batch/ubatch 1024/512, KVarN4/4, INT8 + gated FP16, exact tail 128, Q4_K_M DFlash2 N3, draft KV Q4_0/Q4_0 | `12gb-quality`; `12gb-b84` compatibility alias now selects 80K; append `-c 65536` for less context |
| Linux / SM89 | Retained r3 80K KVarN3/3 / UB256 defaults, architecture-specific binaries/bridges | `16gb`, `full32k` retain their prior settings; new 12 GB preset inference/headroom pending |
| Linux / SM120 | Retained r3 80K KVarN3/3 / UB256 defaults with DFlash2 N3 when a drafter is supplied | Explicit `dflash-8k`, `ordinary-8k`, `ordinary-32k` retain r9 settings; new 80K preset inference/headroom pending |
| Windows / SM86 | SM86 KVarN4/4 / UB512, native INT8 + gated FP16 port, head128 | `12gb-quality` / `12gb-b84`; Windows GPU headroom and throughput unmeasured |
| Windows / SM89 / SM120 | Retained r3 KVarN3/3 / UB256 defaults with architecture-specific bridge selection | `16gb`, `full32k` retain prior settings; all Windows profiles await GPU qualification |

A profile name is a memory/configuration target, not a guarantee that every card of that capacity will fit it. Only the SM86 route enables the 128-thread head and K3 vector cubin. The 80K profile is hardware-tested on Linux SM86; changing defaults on other routes does not create a performance or memory qualification. Override `-t/-tb` for your CPU, and reduce context if available VRAM is insufficient.

## Vision: start with the projector on CPU

For a 12 GB SM86 setup, start with **CPU vision offload**: keep the Low language model and DFlash2 drafter on the GPU, and run the separate image encoder/projector in system RAM with `--no-mmproj-offload`. On our RTX 3060, an 80K GPU-vision configuration loaded the weights but failed when processing an image because its working buffers did not fit. The projector file size alone does not determine GPU fit.

The recommended projector for this CPU recipe is **Q8_0**. On the tested Z840, the current 80K Q8_0/32-worker setting processed the image plus prompt in **16.87 seconds** in the cache test and **17.10 seconds** in the live deployment check (20.98 seconds for the full reply). Earlier 96K format comparisons measured 16.94–17.02 seconds for image plus prompt. Q5_K-MIX is smaller on disk, but was slower on this CPU. These are measurements on one machine, not promised latency on other CPUs.

### Download and verify the optional projector

After downloading the target and drafter as above, run this from the extracted runtime directory:

```bash
curl -fL --retry 3 -o models/mmproj-Qwen3.8-27B-Q8_0.gguf \
  https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/71bc7b627595dc8a91039addd9c791ae548d6747/mmproj-Qwen3.8-27B-Q8_0.gguf
printf '%s\n' '2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb  models/mmproj-Qwen3.8-27B-Q8_0.gguf' | sha256sum -c -
```

On Windows, use `curl.exe` with the same pinned URL and output file, then compare `Get-FileHash .\models\mmproj-Qwen3.8-27B-Q8_0.gguf -Algorithm SHA256` with the hash above. The projector is **629,247,008 bytes (about 600 MiB)**. Allow additional system RAM for image-processing buffers; the measured Q8/32-worker test service peaked at about 1.8 GiB of RAM, which is a whole-service measurement, not the projector's incremental requirement.

### Opt-in SM86 configuration

Stop your existing server before starting a replacement on the same GPU and port. This command reproduces the current tested SM86 80K / KVarN4/4 / Q4-draft-cache / 1,024-image-token / 32-worker setting:

```bash
./l0xre serve --profile 12gb-quality \
  -m models/L0xRE-27b-Low.gguf \
  -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen3.8-27B-Q8_0.gguf --no-mmproj-offload \
  --image-min-tokens 1024 --image-max-tokens 1024 \
  -t 32 -tb 32 --reasoning-effort medium --host 127.0.0.1 --port 8080
```

For Windows, the matching command is below. Its arguments mirror Linux, but Windows vision inference and performance remain untested:

```powershell
.\l0xre.cmd serve --profile 12gb-quality `
  -m models\L0xRE-27b-Low.gguf `
  -md models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf `
  --mmproj models\mmproj-Qwen3.8-27B-Q8_0.gguf --no-mmproj-offload `
  --image-min-tokens 1024 --image-max-tokens 1024 `
  -t 32 -tb 32 --reasoning-effort medium --host 127.0.0.1 --port 8080
```

**Worker count is hardware-specific.** The tested Z840 has two Xeon E5-2620 v4 CPUs: 16 physical cores / 32 logical threads. On smaller machines, start with the physical-core count and compare timings; 32 workers is not a universal default. Here, 16 workers with Q5_K-MIX took 20.82 seconds, 32 took about 18.15 seconds, and Q8_0/32 took about 16.98 seconds. NUMA tuning gave little benefit; confining work to one socket was slower. The 1,024-token setting preserves the tested image resolution; reducing it to 512 was faster but needs separate OCR/grounding quality checks.

Reusable configs live in this repository: [Linux / WSL launcher](examples/vision/serve-cpu-sm86.sh), [Windows launcher](examples/vision/serve-cpu-sm86.ps1), and [Hermes config fragment](examples/vision/hermes-native.yaml). The r4 archives include these examples under `examples/vision/`. With a repository checkout or the bundled examples and an extracted runtime, use:

```bash
VISION_RUNTIME_DIR=/path/to/extracted/runtime VISION_WORKERS=32 \
  bash /path/to/L0xRE-BeeLLama-Low/examples/vision/serve-cpu-sm86.sh
```

```powershell
& C:\path\to\L0xRE-BeeLLama-Low\examples\vision\serve-cpu-sm86.ps1 `
  -RuntimeDirectory C:\path\to\extracted\runtime -Workers 32
```

### Hermes native vision and verification

Merge [hermes-native.yaml](examples/vision/hermes-native.yaml) into your existing Hermes config. It sets `agent.image_input_mode: native`, declares the local model's vision capability, and routes both the main model and `auxiliary.vision` to `http://127.0.0.1:8080/v1`. Use the server's published `L0xRE-27b-Low` alias; change both `base_url` values if the server is at another address. Restart Hermes after editing its config. This sends native image parts to your local runtime rather than asking a cloud vision model to describe them first.

Wait for `curl -fsS http://127.0.0.1:8080/health` to return `{"status":"ok"}`, then run the Python-standard-library smoke client from your repository checkout:

```bash
python3 /path/to/L0xRE-BeeLLama-Low/examples/vision/check-vision.py
```

It sends the included [smoke image](examples/vision/smoke.png) through `/v1/chat/completions` and checks `VISION 742` plus the red box, blue circle and green triangle. It prints image-plus-prompt and complete-response timings separately. For an actual Hermes check, attach the same image in a chat; the current CLI also supports `hermes chat --image /path/to/smoke.png -q "Read the text and describe the shapes." --oneshot`.

[Measured CPU-vision results and qualification limits](evidence/vision/SM86-CPU-VISION.json) cover Linux SM86 only. The current 80K cache preset passed code, vision and a 29,762-token retrieval canary, with 243 MiB minimum free VRAM; full-context memory qualification has not been performed. See [current 80K cache evidence](evidence/quality/SM86-80K-QUALITY.json). A separate thinking-off full 8-pack result is now available below. The text-only launch commands above remain unchanged. To return to text-only operation, stop the vision server and launch without `--mmproj` and the vision-specific flags; switch Hermes back to its previous image routing if needed.

## Test results and measured performance

| Build / hardware | Result | Scope and receipt |
| --- | --- | --- |
| Previous r2 Linux SM86 / Z840 RTX 3060 12 GB | Code **39.046 tok/s**; narrative **29.422 tok/s** | B84, temperature 0, seed 0, exactly 800 output tokens, two measured runs per prompt; [component hashes and method](evidence/universal/SM86-B84.json), raw receipts in the Linux archive |
| Previous r2 Linux SM86 / long context | 92,879 input + 128 greedy output tokens; prefill **272.60 tok/s**, decode **21.06 tok/s**, minimum free VRAM **215 MiB** | Output SHA matched the B74 reference; a narrow correctness canary, not a broad quality test |
| Historical Linux SM89 / RTX 4090 r1 settings | 12gb profile code **115.4 ± 2.4 tok/s** | Inherited result; original command, model and workload receipts retained under `architectures/sm89/receipts/` |
| Linux SM120 / RTX 5090 | Current L0xRE-27b-Low common-CLI path awaits hardware qualification | Included GPU runtime payload; no measured throughput is claimed for this model/path combination |
| Linux SM86 / Z840 RTX 3060, L0xRE-27b-Low (E3) | **118/150 pass@1 (78.7%)**; 119/150 pass@3 | Club-3090 full 8-pack, one external run, thinking forced off. Six more pass@1 scenarios than each of two saved LowGPU IQ3_XXS runs; those controls used BeeLlama 0.4.4-dev and different hardware/profile settings, so this is directional rather than a model-only comparison. [Scores, protocol and limits](evidence/quality/SM86-3060-FULL8-THINKING-OFF-2026-10-01.json) |
| Windows universal | **235 CUDA cubin entries per architecture** (SM86, SM89, SM120) | CUDA 13.3 / MSVC; new head128 backend and B74 bridge included; no GPU tokens/s result |
| Release source and launchers | **29 Linux stub cases pass locally**; r3 release validation recorded 25 Linux / 15 Windows cases and 4/4 inherited compiled CPU tests | Stub checks do not execute CUDA. The four added Linux cases cover empty and non-empty `LD_LIBRARY_PATH`; CI is configured to run them. [Release validation](evidence/universal/VALIDATION.json) and [current launcher tests](tools/universal/test-linux-launchers.py). Archive checksums and clean-extract PE/dependency checks are in [archive validation](evidence/universal/ARCHIVE-VALIDATION.json). |

The retained server/CUDA backend and model hashes match the Linux B84 baseline; r4 adds the qualified INT8 and attention libraries. Historical r3 evidence below describes its former configuration. The rows explicitly labeled previous/historical do not measure the new 80K default. Historical r3 Linux SM86 evidence is in [SM86-80K-QUALITY.json](evidence/quality/SM86-80K-QUALITY.json): code 40.32 tok/s on a short binary-search canary, correct code/vision/retrieval outputs, and 243 MiB minimum free VRAM. Those are not the historical 800-token prompts or a broad quality benchmark. The historical October 1 8-pack result under r3 adds broad task coverage for one thinking-off run: 15/15 ToolCall, 15/15 StructOutput, 14/20 HermesAgent and 26/40 CLI. It supports capability across the tested task types, not a universal intelligence guarantee or run-to-run variance estimate. The full 8-pack test did not qualify full-context 80K memory behavior. Interactive defaults use temperature 0.7; benchmark sampling differs. Cross-model quality comparisons remain directional where runtime, GPU or profile differs.

## Source, build and rollback

[Build and source provenance](BUILD-UNIVERSAL.md) · [Release changes](CHANGELOG-UNIVERSAL.md) · [Feedback](FEEDBACK.md)

The recent SM86 changes are committed on `release/sm86-b84-universal`: opt-in head128, B74 bridge, vector PTX reconstruction inputs, the Windows port/export definition, and universal build-script support. Packaged manifests pin component hashes and toolchains. The Linux archive retains qualified payload bytes instead of silently replacing the established SM89/SM120 kernels. Windows rebuilds the changed CUDA object/backend from verified matching source and object cache, plus the new bridge from source.

Each archive includes `SHA256SUMS`, a manifest, model checksums, source references, and applicable BeeLLama/CUDA/kernel license notices. Rebuilds with other toolchains may differ in bytes and need their own qualification.

To roll back, stop the server, extract the earlier release into another directory, and reuse the unchanged model files. No service installation or model conversion is performed by these commands.

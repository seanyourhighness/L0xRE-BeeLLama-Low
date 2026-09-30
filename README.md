# L0xRE BeeLLama

Run **L0xRE-27b-Low** with the BeeLLama runtime on NVIDIA RTX 30, 40, and 50 series GPUs. Download **one Linux / WSL archive** or **one Windows ZIP**; each contains the matching SM86, SM89, and SM120 runtime paths. Model weights are a separate download.

**Image input:** see the [opt-in CPU vision setup](#vision-start-with-the-projector-on-cpu), including Q8_0, worker settings and a native Hermes config.

The intended minimum is **12 GB of GPU VRAM**, not 12 GB of system RAM. Cards with less VRAM are outside this release target. Available VRAM, context length, display use, and driver overhead still matter; the measured RTX 3060 long-context run had only 215 MiB free at its lowest point.

## Downloads

**[Universal release candidate](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/beellama-v0.4.7-universal-r2)** · [Archive checksums](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r2/CHECKSUMS.txt)

| Platform | Download | GPU code included | Status |
| --- | --- | --- | --- |
| Linux x86-64 / WSL2 | [Universal `.tar.zst`](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r2/l0xre-beellama-low-v0.4.7-wsl-linux-x86_64-universal-r2.tar.zst) | Separate SM86 / SM89 / SM120 payloads | SM86 champion components tested on RTX 3060; inherited SM89 and SM120 receipts included; common launcher validated separately |
| Windows x64 | [Universal `.zip`](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r2/l0xre-win-universal-r2.zip) | One rebuilt CUDA backend with SM86 / SM89 / SM120 code, architecture-specific bridge cubins | Build, PE dependency, architecture and launcher checks passed; GPU execution and throughput pending |
| Model + drafter | [L0xRE-27b-Low on Hugging Face](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low) | `L0xRE-27b-Low.gguf` + `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | Existing weights unchanged; SHA-256 identities below |

This is a release candidate because the Windows GPU paths and the new SM120 common-CLI path with the non-MTP Low target have not had hardware execution tests. Architecture coverage is distinct from performance qualification. Older architecture-specific releases remain available as rollback options.

## Requirements

- An AVX2-capable x86-64 CPU and an RTX 30 / 40 / 50 series GPU with compute capability 8.6 / 8.9 / 12.0 and at least 12 GB VRAM; check with `nvidia-smi`.
- Linux/WSL: Ubuntu 24.04 or an ABI-compatible x86-64 distribution. The SM120 payload needs glibc 2.38+ and `GLIBCXX_3.4.32`. WSL2 needs an NVIDIA Windows driver with WSL CUDA support.
- Windows: x64 Windows with PowerShell and an NVIDIA driver compatible with the bundled CUDA 13.3 runtime. CUDA runtime and MSVC runtime DLLs are included; a CUDA toolkit is not required to run it.
- Linux bundles CUDA 12.8 for SM86/SM89 and CUDA 13.0 for SM120. The host supplies the NVIDIA driver. Reserve about 20 GB of disk space for the runtime archive, extraction, and the two model files.

The target is a custom GGUF supported by this L0xRE runtime. Use this runtime for these weights; see the model card for model-specific compatibility and licensing.

## Linux / WSL installation

Download the Linux archive and `CHECKSUMS.txt` above into one directory. Install `zstd` and `curl` if needed (`sudo apt-get install zstd curl`). Verify and extract:

```bash
sha256sum --ignore-missing -c CHECKSUMS.txt
tar --zstd -xf l0xre-beellama-low-v0.4.7-wsl-linux-x86_64-universal-r2.tar.zst
cd l0xre-beellama-low-v0.4.7-wsl-linux-x86_64-universal-r2
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

The launcher detects the first visible GPU. On a machine with multiple GPUs, set `CUDA_VISIBLE_DEVICES` to the intended GPU index or UUID before launch. For example, `CUDA_VISIBLE_DEVICES=1 ./l0xre serve ...` selects GPU 1. `L0XRE_ARCH=sm86|sm89|sm120` overrides routing; it does not change which GPU CUDA uses.

## Windows installation

Download the Windows ZIP and `CHECKSUMS.txt` above into one directory. Open PowerShell there, compare the ZIP's `Get-FileHash` result with its line in `CHECKSUMS.txt`, and extract:

```powershell
Get-FileHash .\l0xre-win-universal-r2.zip -Algorithm SHA256
Expand-Archive .\l0xre-win-universal-r2.zip -DestinationPath .
Set-Location .\l0xre-win-universal-r2
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

| Platform / GPU | `12gb` selection | Other options |
| --- | --- | --- |
| Linux / SM86 | B84: 98,304 context, batch/ubatch 1024/256, one slot, KVarN3/2, exact tail 128, Q4_K_M DFlash2 depth 3, draft KV q2_0/q2_0 | `12gb-b84` explicit alias; append `-c 32768` if less context is needed |
| Linux / SM89 | Retained r1 81,920-context profile | `16gb` (262,144 context), `full32k`; original receipts included |
| Linux / SM120 | Retained r9 8,192-context DFlash2 depth 4 with drafter; ordinary 8K without drafter | `dflash-8k`, `ordinary-8k`, `ordinary-32k`; historical r9 measurements used a separate MTP-containing checkpoint |
| Windows / SM86 | New B84 settings, head128 and B74 vector bridge | `12gb-b84` alias; GPU headroom and throughput unmeasured |
| Windows / SM89 / SM120 | Existing 81,920-context preset with rebuilt universal CUDA backend | `16gb`, `full32k`; all Windows profiles await GPU qualification |

A profile name is a memory/configuration target, not a guarantee that every card of that capacity will fit it. Only the SM86 route enables the new 128-thread head and K3 vector cubin. Reduce context if available VRAM is insufficient.

## Vision: start with the projector on CPU

For a 12 GB SM86 setup, start with **CPU vision offload**: keep the Low language model and DFlash2 drafter on the GPU, and run the separate image encoder/projector in system RAM with `--no-mmproj-offload`. On our RTX 3060, an 80K GPU-vision configuration loaded the weights but failed when processing an image because its working buffers did not fit. The projector file size alone does not determine GPU fit.

The recommended projector for this CPU recipe is **Q8_0**. On the tested Z840, Q8_0 with 32 workers processed an image plus prompt in **16.94–17.02 seconds**, with complete replies taking **20.46–20.55 seconds**. Q5_K-MIX is smaller on disk, but was slower on this CPU. These are measurements on one machine, not promised latency on other CPUs.

### Download and verify the optional projector

After downloading the target and drafter as above, run this from the extracted runtime directory:

```bash
curl -fL --retry 3 -o models/mmproj-Qwen3.8-27B-Q8_0.gguf \
  https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/71bc7b627595dc8a91039addd9c791ae548d6747/mmproj-Qwen3.8-27B-Q8_0.gguf
printf '%s\n' '2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb  models/mmproj-Qwen3.8-27B-Q8_0.gguf' | sha256sum -c -
```

On Windows, use `curl.exe` with the same pinned URL and output file, then compare `Get-FileHash .\models\mmproj-Qwen3.8-27B-Q8_0.gguf -Algorithm SHA256` with the hash above. The projector is **629,247,008 bytes (about 600 MiB)**. Allow additional system RAM for image-processing buffers; the measured Q8/32-worker test service peaked at about 1.8 GiB of RAM, which is a whole-service measurement, not the projector's incremental requirement.

### Opt-in SM86 configuration

Stop your existing server before starting a replacement on the same GPU and port. This command reproduces the tested 96K / 1,024-image-token / 32-worker setting:

```bash
./l0xre serve --profile 12gb-b84 \
  -m models/L0xRE-27b-Low.gguf \
  -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen3.8-27B-Q8_0.gguf --no-mmproj-offload \
  --image-min-tokens 1024 --image-max-tokens 1024 \
  -t 32 -tb 32 --reasoning-effort medium --host 127.0.0.1 --port 8080
```

For Windows, the matching command is below. Its arguments mirror Linux, but Windows vision inference and performance remain untested:

```powershell
.\l0xre.cmd serve --profile 12gb-b84 `
  -m models\L0xRE-27b-Low.gguf `
  -md models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf `
  --mmproj models\mmproj-Qwen3.8-27B-Q8_0.gguf --no-mmproj-offload `
  --image-min-tokens 1024 --image-max-tokens 1024 `
  -t 32 -tb 32 --reasoning-effort medium --host 127.0.0.1 --port 8080
```

**Worker count is hardware-specific.** The tested Z840 has two Xeon E5-2620 v4 CPUs: 16 physical cores / 32 logical threads. On smaller machines, start with the physical-core count and compare timings; 32 workers is not a universal default. Here, 16 workers with Q5_K-MIX took 20.82 seconds, 32 took about 18.15 seconds, and Q8_0/32 took about 16.98 seconds. NUMA tuning gave little benefit; confining work to one socket was slower. The 1,024-token setting preserves the tested image resolution; reducing it to 512 was faster but needs separate OCR/grounding quality checks.

Reusable configs live in this repository: [Linux / WSL launcher](examples/vision/serve-cpu-sm86.sh), [Windows launcher](examples/vision/serve-cpu-sm86.ps1), and [Hermes config fragment](examples/vision/hermes-native.yaml). They are optional examples and are **not inside the already-published universal-r2 archives**. With a repository checkout and an extracted runtime, use:

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

[Measured CPU-vision results and qualification limits](evidence/vision/SM86-CPU-VISION.json) cover Linux SM86 only. CPU vision with a short image prompt worked at the configured 96K context; a full-context vision memory test has not been performed. The text-only launch commands above remain unchanged. To return to text-only operation, stop the vision server and launch without `--mmproj` and the vision-specific flags; switch Hermes back to its previous image routing if needed.

## Test results and measured performance

| Build / hardware | Result | Scope and receipt |
| --- | --- | --- |
| Linux SM86 / Z840 RTX 3060 12 GB | Code **39.046 tok/s**; narrative **29.422 tok/s** | B84, temperature 0, seed 0, exactly 800 output tokens, two measured runs per prompt; [component hashes and method](evidence/universal/SM86-B84.json), raw receipts in the Linux archive |
| Linux SM86 / long context | 92,879 input + 128 greedy output tokens; prefill **272.60 tok/s**, decode **21.06 tok/s**, minimum free VRAM **215 MiB** | Output SHA matched the B74 reference; a narrow correctness canary, not a broad quality test |
| Linux SM89 / RTX 4090, retained r1 | 12gb profile code **115.4 ± 2.4 tok/s** | Inherited result; original command, model and workload receipts retained under `architectures/sm89/receipts/` |
| Linux SM120 / RTX 5090 | Current L0xRE-27b-Low common-CLI path awaits hardware qualification | Included GPU runtime payload; no measured throughput is claimed for this model/path combination |
| Windows universal | **235 CUDA cubin entries per architecture** (SM86, SM89, SM120) | CUDA 13.3 / MSVC; new head128 backend and B74 bridge included; no GPU tokens/s result |
| Release source and launchers | **4/4 CPU tests**, **11 Linux launcher checks passed**; nine earlier Windows launcher checks remain recorded | Stub launcher checks verify routing, arguments and environment isolation; they do not execute CUDA. The Windows model-name alias refresh was reviewed statically; the managed workspace cannot execute its PowerShell integration test. PE and package checks are in [validation](evidence/universal/VALIDATION.json); [archive round trips](evidence/universal/ARCHIVE-VALIDATION.json) verified 238 Linux and 131 Windows regular files and repeated the Windows PE check (70 files, 4,835 local imports) |

The L0xRE-27b-Low target, executable, CUDA backend, bridge, and vector-cubin hashes match the original B84 measurements. The universal launcher changes portable paths and incorporates preset overrides. Interactive defaults use temperature 0.7; benchmark sampling differs. Do not compare rows as a controlled GPU comparison: workloads, models and protocols differ.

## Source, build and rollback

[Build and source provenance](BUILD-UNIVERSAL.md) · [Release changes](CHANGELOG-UNIVERSAL.md) · [Feedback](FEEDBACK.md)

The recent SM86 changes are committed on `release/sm86-b84-universal`: opt-in head128, B74 bridge, vector PTX reconstruction inputs, the Windows port/export definition, and universal build-script support. Packaged manifests pin component hashes and toolchains. The Linux archive retains qualified payload bytes instead of silently replacing the established SM89/SM120 kernels. Windows rebuilds the changed CUDA object/backend from verified matching source and object cache, plus the new bridge from source.

Each archive includes `SHA256SUMS`, a manifest, model checksums, source references, and applicable BeeLLama/CUDA/kernel license notices. Rebuilds with other toolchains may differ in bytes and need their own qualification.

To roll back, stop the server, extract the earlier release into another directory, and reuse the unchanged model files. No service installation or model conversion is performed by these commands.

# L0xRE BeeLLama

Run **L0xRE-27b-Low** with the BeeLLama runtime on NVIDIA RTX 30, 40, and 50 series GPUs. Download **one Linux / WSL archive** or **one Windows ZIP**; each contains the matching SM86, SM89, and SM120 runtime paths. Model weights are a separate download.

The intended minimum is **12 GB of GPU VRAM**, not 12 GB of system RAM. Cards with less VRAM are outside this release target. Available VRAM, context length, display use, and driver overhead still matter; the measured RTX 3060 long-context run had only 215 MiB free at its lowest point.

## Downloads

**[Universal B84 release candidate](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/beellama-v0.4.7-sm86-b84-universal-r1)** · [Archive checksums](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-sm86-b84-universal-r1/CHECKSUMS.txt)

| Platform | Download | GPU code included | Status |
| --- | --- | --- | --- |
| Linux x86-64 / WSL2 | [Universal `.tar.zst`](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-sm86-b84-universal-r1/l0xre-beellama-low-v0.4.7-sm86-b84-wsl-linux-x86_64-universal.tar.zst) | Separate SM86 / SM89 / SM120 payloads | SM86 champion components tested on RTX 3060; inherited SM89 and SM120 receipts included; common launcher validated separately |
| Windows x64 | [Universal `.zip`](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-sm86-b84-universal-r1/l0xre-win-universal-b84-r1.zip) | One rebuilt CUDA backend with SM86 / SM89 / SM120 code, architecture-specific bridge cubins | Build, PE dependency, architecture and launcher checks passed; GPU execution and throughput pending |
| Model + drafter | [L0xRE-27b-Low on Hugging Face](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/tree/9b74c81c19f8372888c2968b5334f42b0354d8b1) | `L0xRE-27b-Low.gguf` + `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | Existing weights unchanged; SHA-256 identities below |

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
tar --zstd -xf l0xre-beellama-low-v0.4.7-sm86-b84-wsl-linux-x86_64-universal.tar.zst
cd l0xre-beellama-low-v0.4.7-sm86-b84-wsl-linux-x86_64-universal
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
Get-FileHash .\l0xre-win-universal-b84-r1.zip -Algorithm SHA256
Expand-Archive .\l0xre-win-universal-b84-r1.zip -DestinationPath .
Set-Location .\l0xre-win-universal-b84-r1
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

Downloads are pinned to HF revision `9b74c81c19f8372888c2968b5334f42b0354d8b1`, verified against the repository's LFS identities on September 29, 2026. The revision changed since the earlier publication; these model bytes did not. The [MTP variant](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low-MTP) is a separate target file and is not interchangeable in benchmark attribution.

## Profiles

| Platform / GPU | `12gb` selection | Other options |
| --- | --- | --- |
| Linux / SM86 | B84: 98,304 context, batch/ubatch 1024/256, one slot, KVarN3/2, exact tail 128, Q4_K_M DFlash2 depth 3, draft KV q2_0/q2_0 | `12gb-b84` explicit alias; append `-c 32768` if less context is needed |
| Linux / SM89 | Retained r1 81,920-context profile | `16gb` (262,144 context), `full32k`; original receipts included |
| Linux / SM120 | Retained r9 8,192-context DFlash2 depth 4 with drafter; ordinary 8K without drafter | `dflash-8k`, `ordinary-8k`, `ordinary-32k`; historical r9 measurements use the original MTP-containing E3 target |
| Windows / SM86 | New B84 settings, head128 and B74 vector bridge | `12gb-b84` alias; GPU headroom and throughput unmeasured |
| Windows / SM89 / SM120 | Existing 81,920-context preset with rebuilt universal CUDA backend | `16gb`, `full32k`; all Windows profiles await GPU qualification |

A profile name is a memory/configuration target, not a guarantee that every card of that capacity will fit it. Only the SM86 route enables the new 128-thread head and K3 vector cubin. Reduce context if available VRAM is insufficient.

## Test results and measured performance

| Build / hardware | Result | Scope and receipt |
| --- | --- | --- |
| Linux SM86 / Z840 RTX 3060 12 GB | Code **39.046 tok/s**; narrative **29.422 tok/s** | B84, temperature 0, seed 0, exactly 800 output tokens, two measured runs per prompt; [component hashes and method](evidence/universal/SM86-B84.json), raw receipts in the Linux archive |
| Linux SM86 / long context | 92,879 input + 128 greedy output tokens; prefill **272.60 tok/s**, decode **21.06 tok/s**, minimum free VRAM **215 MiB** | Output SHA matched the B74 reference; a narrow correctness canary, not a broad quality test |
| Linux SM89 / RTX 4090, retained r1 | 12gb profile code **115.4 ± 2.4 tok/s** | Inherited result; original command, model and workload receipts retained under `architectures/sm89/receipts/` |
| Linux SM120 / RTX 5090, retained r9 | E3 DFlash2 code **173.29**, prose **124.68 tok/s** | Pooled interleaved means on the original MTP-containing E3 file; **not a measurement of the new non-MTP Low common-CLI path**. See [original r9 parity](PARITY.md) |
| Windows universal | **235 CUDA cubin entries per architecture** (SM86, SM89, SM120) | CUDA 13.3 / MSVC; new head128 backend and B74 bridge included; no GPU tokens/s result |
| Release source and launchers | **4/4 CPU tests**, **11 Linux + 9 Windows launcher integration checks passed** | Stub launcher checks verify routing, arguments and environment isolation; they do not execute CUDA. PE and package checks are in [validation](evidence/universal/VALIDATION.json); [archive round trips](evidence/universal/ARCHIVE-VALIDATION.json) verified 238 Linux and 130 Windows regular files and repeated the Windows PE check (70 files, 4,835 local imports) |

The SM86 target, executable, CUDA backend, bridge, and vector-cubin hashes match the original B84 measurements. The universal launcher changes portable paths and incorporates preset overrides. Interactive defaults use temperature 0.7; benchmark sampling differs. Do not compare rows as a controlled GPU comparison: workloads, models and protocols differ.

## Source, build and rollback

[Build and source provenance](BUILD-UNIVERSAL.md) · [Release changes](CHANGELOG-UNIVERSAL.md) · [Feedback](FEEDBACK.md)

The recent SM86 changes are committed on `release/sm86-b84-universal`: opt-in head128, B74 bridge, vector PTX reconstruction inputs, the Windows port/export definition, and universal build-script support. Packaged manifests pin component hashes and toolchains. The Linux archive retains qualified payload bytes instead of silently replacing the established SM89/SM120 kernels. Windows rebuilds the changed CUDA object/backend from verified matching source and object cache, plus the new bridge from source.

Each archive includes `SHA256SUMS`, a manifest, model checksums, source references, and applicable BeeLLama/CUDA/kernel license notices. Rebuilds with other toolchains may differ in bytes and need their own qualification.

To roll back, stop the server, extract the earlier release into another directory, and reuse the unchanged model files. No service installation or model conversion is performed by these commands.

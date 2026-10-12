# L0xRE — Fast. Smart. Small.

**An agentic 27B coding assistant running on your machine. Now.**

L0xRE BeeLLama Low R6 runs **L0xRE-27b-Low** — a 27B model in 8.6 GB — at real working speed on any NVIDIA RTX 30-, 40-, or 50-series GPU with **12 GB or more**. 80K-token context, DFlash2 speculative decoding, INT8 prefill, and CPU vision are built into the certified profiles. The server speaks the OpenAI-compatible API, so it drops straight into coding agents, assistants, and editors. No cloud, no per-token cost, no tiny model.

## Measured speed (frozen Bench.sh protocol, sustained)

| Your GPU | Prose | Code | Cold prefill | Context |
| --- | ---: | ---: | ---: | ---: |
| **RTX 3060 12 GB** (Linux) | **37 t/s** | **65 t/s** | 566 t/s (10K) | 80K |
| **RTX 4090 24 GB** (Linux) | **104 t/s** | **167 t/s** | 2,034 t/s (10K) | 80K |
| **RTX 5090 32 GB** (Windows) | **137 t/s** | **230 t/s** | 3,611 t/s | 80K |
| **RTX 5090 32 GB** (Linux) | **131 t/s** | **229 t/s** | 3,527 t/s | 80K |

Every number is a certified measurement with published receipts — same prompts, same seeds, same frozen protocol, reproducible on your hardware.

## Get started

**Linux / WSL** (Ubuntu 24.04+, Python 3.12+, working NVIDIA driver):

```bash
curl -fL --retry 3 -o install-l0xre.sh \
  https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --dry-run   # preview the plan
bash install-l0xre.sh             # install
```

**Windows** (PowerShell):

```powershell
Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.ps1' -OutFile install-l0xre.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1
```

Setup detects your GPU, selects the right package and profile, verifies every SHA-256 (runtime, model, drafter), downloads about 10 GB of weights from Hugging Face, and prints a ready-to-run start command. The server is an OpenAI-compatible API on `http://127.0.0.1:8080/v1`.

## Supported hardware

One release, one installer, every RTX 30/40/50 card with at least 12 GB:

| GPU | Linux / WSL | Windows |
| --- | --- | --- |
| RTX 30 series (≥12 GB) | **Certified** — 80K profile, measured on RTX 3060 | **Starter** — 80K profile |
| RTX 40 series (≥12 GB) | **Certified** on RTX 4090 (80K) · **Starter** on other cards (80K) | **Starter** — 80K profile, measured 68/116 t/s on RTX 4070 Ti |
| RTX 5090 (32 GB) | **Certified** — 80K profile | **Certified** — 80K profile, CUDA 13.0 |
| Other RTX 50 series (≥12 GB) | **Starter** — 32K profile | **Starter** — 32K profile |

- **Certified** means the exact card, profile, and package passed the full qualification gates: correctness equivalence, greedy oracle, capacity, soak, and a consumer probe on the published archive. Evidence lives in [`receipts/`](receipts/) and the [certification docs](docs/).
- **Starter** means the best tuned configuration we have for that card, shipped ready to run and honestly labeled; full certification is in progress. Starters run the same model, drafter, and launcher.

## Downloads

**[Official R6 release](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/beellama-v0.4.7-r6)** — every archive, checksum, the installer catalog, and the machine-readable qualification receipts live on this one release page. You normally never download these by hand; the installer picks the right one.

| Package | For | Status |
| --- | --- | --- |
| `L0xRE-BeeLLama-Low-R6-sm86-linux.tar.zst` | RTX 30 series, Linux/WSL | Certified |
| `L0xRE-BeeLLama-Low-R6-sm89-linux-rtx4090-certified.tar.zst` | RTX 40 series, Linux/WSL | Certified (4090) + 12 GB starter profile |
| `L0xRE-SM120-certified-20261008.tar.zst` | RTX 5090, Linux/WSL | Certified |
| `L0xRE-BeeLLama-Low-R6-sm120-12gb-linux-starter.tar.zst` | RTX 50 series ≥12 GB, Linux/WSL | Starter |
| `l0xre-beellama-low-v0.4.7-r6-windows-x86_64-sm120-cuda130.zip` | RTX 5090, Windows | Certified |
| `L0xRE-BeeLLama-Low-R6-sm86-windows-starter.zip` | RTX 30 series, Windows | Starter |
| `L0xRE-BeeLLama-Low-R6-sm89-windows-starter.zip` | RTX 40 series, Windows | Starter |
| `L0xRE-BeeLLama-Low-R6-sm120-12gb-windows-starter.zip` | RTX 50 series ≥12 GB, Windows | Starter |

## Model files

Shared by every package, pinned by SHA-256, on [Hugging Face](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low):

| File | Size | Role |
| --- | ---: | --- |
| `L0xRE-27b-Low.gguf` | 8.62 GB | the model |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1.14 GB | speculative drafter (DFlash2, N7) |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.63 GB | optional CPU vision |

The installer downloads and verifies these automatically. Checksums: [`MODEL-SHA256SUMS`](MODEL-SHA256SUMS).

## What's inside

- **DFlash2 speculative decoding (N7)** — a small drafter accelerates the 27B target; acceptance is measured, not assumed.
- **KVarN4/4 quantized KV cache** — 80K context fits in 12 GB of VRAM.
- **INT8 prefill fast paths** — long-document prompts ingest at hundreds to thousands of t/s.
- **CPU vision** — image understanding without spending VRAM.
- **Reasoning control** — `--reasoning-effort` budgets thinking per request.
- **Frozen profiles** — every launch is byte-identical to a measured, receipted configuration.

## Requirements

- NVIDIA RTX 30/40/50 series GPU with **≥12 GB VRAM** (RTX 5090 recommended for the 80K SM120 profile; 12 GB Blackwell cards use the 32K starter profile).
- Linux / WSL2: Ubuntu 24.04 or compatible, Python 3.12+, `curl`, `zstd`, `numactl`, working NVIDIA driver.
- Windows 10/11 x64: nothing to install — CUDA and MSVC runtimes are bundled.
- ~10 GB free disk for model weights, plus 5–10 GB for the runtime.
- 8 accessible logical CPUs (CPUs 0–7) for the certified launch profiles.

## Evidence and methodology

Nothing here is a vibes benchmark. Each certified card has a sealed profile, frozen prompt set, seeded runs, greedy output hashes, soak runs, and a consumer probe that boots the published archive exactly as the installer does:

- [RTX 4090 / SM89 certification](docs/SM89-LINUX-4090-STATUS.md) · [RTX 5090 / SM120](docs/SM120-QUALIFICATION.md) · [RTX 4070 Ti / SM89 starter](docs/SM89-WINDOWS-STATUS.md)
- [Architecture status](docs/PARITY.md) · [machine-readable receipts](receipts/) · [release changes](docs/CHANGELOG-UNIVERSAL.md)

## Feedback

Issues and hardware reports are welcome — [open an issue](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues) or see [FEEDBACK.md](docs/FEEDBACK.md) for what makes a report actionable. If L0xRE runs well on your card, tell us the GPU, OS, and the numbers you see.

## License

L0xRE BeeLLama Low is distributed under the [BeeLlama license](licenses/BeeLLaMA-LICENSE) with third-party notices in [`licenses/`](licenses/).

# L0xRE — a 27B coding agent that sees, reads 128K tokens, and flies on a 12 GB GPU

**Vision. Speed. Long context. Fast prefill.** All four at once, on the card you already own.

L0xRE-27b-Low runs on your RTX 30/40/50 card with 12 GB or more. No cloud, no API bill, no data leaving your machine. Point any OpenAI-compatible agent or editor at it and go.

<!-- DEMO: replace this comment with the demo video/GIF of something L0xRE built on an RTX 3060 / 4070 Ti, e.g.
[![L0xRE demo](docs/demo/thumbnail.png)](docs/demo/demo.mp4)
-->

## TL;DR results — v0.1.3

On a **12 GB RTX 3060**, the cheapest card we support:

| | RTX 3060 12 GB | What it means |
| --- | ---: | --- |
| 🧠 **Context** | **128K tokens** | Whole repos, long chats, big docs (256K on 16 GB cards) |
| 👁️ **Vision** | **On** | Screenshots, diagrams and UI mockups go straight in, within the same 12 GB |
| ⚡ **Code generation** | **~72 tok/s** | +21% over v0.1.2 |
| 📝 **Prose generation** | **~40 tok/s** | +17% over v0.1.2 |
| 🚀 **Prefill** | **~690 tok/s** | A 10K-token prompt is read in about 15 seconds |
| ⏱️ **First token** | **~0.5 s** | Half of v0.1.2; follow-up turns 45% faster |
| ✅ **Quality** | **120 / 150** | Club 8-pack coding/agent suite, same score as v0.1.2 |

Bigger cards scale up (12 GB profile unless noted):

| Card | Code tok/s | Prose tok/s | 10K prefill tok/s | First token | Max context |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 3060 12 GB (Linux) | 72 | 40 | 686 | 0.49 s | 128K |
| RTX 4090 (Linux) | 152 | 89 | 2,350 | 0.29 s | 256K* |
| RTX 5090 (Windows) | 209 | 124 | 3,425 | 0.16 s | 256K* |
| RTX 5090 (WSL2) | 199 | 115 | 3,230 | 0.38 s | 256K* |

\*256K uses the 16 GB profile. The 4090 and 5090 are stand-ins that set memory envelopes for the 12/16 GB targets (RTX 4070 Ti, 5060 Ti, 5070). Your numbers depend on your exact card and what else is using the GPU.

**Community results:** run the selftest on your card and [submit a hardware report](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues/new/choose). Reviewed reports are added here.

| Card | Reported by | Status |
| --- | --- | --- |
| RTX 4070 Ti 12 GB (Windows) | community | report pending |

Full tables, every platform, accepted exceptions and known limits: **[v0.1.3 release notes](docs/V013-RELEASE.md)** · **[full release details](docs/RELEASE-DETAILS.md)**.

## 🤖 Let your AI install it

Paste this into Claude Code, Codex, Cursor or any coding agent with shell access:

> Read https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/AGENT-INSTALL.md and follow it to install L0xRE on this machine. Ask me before anything that needs admin rights.

[AGENT-INSTALL.md](AGENT-INSTALL.md) is written for agents. It checks your GPU, picks the right package, downloads and verifies the runtime, model, drafter and vision file, starts the server, and confirms the API answers.

## Install it yourself

**Requirements:** an NVIDIA RTX 30/40/50 GPU with 12 GB+ VRAM, a current NVIDIA driver, about 16 GB free disk space (runtime + models).

**Linux / WSL2** (Ubuntu 24.04 or compatible; needs Python 3.12+, `curl`, `zstd`, `numactl`):

```bash
curl -fL --retry 3 -o install-l0xre.sh https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --dry-run --vision   # shows exactly what it will do
bash install-l0xre.sh --vision             # downloads + verifies everything
~/L0xRE/start.sh                           # starts the server
```

**Windows 10/11** (PowerShell; runtime DLLs and Python are bundled):

```powershell
Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.ps1' -OutFile install-l0xre.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -DryRun -Vision
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -Vision
& "$env:LOCALAPPDATA\L0xRE\start.ps1"
```

The installer detects your GPU, picks the 12 GB or 16 GB profile, downloads the matching runtime plus the three model files below, and checks every SHA256. When the server is up, point your tools at:

```
http://127.0.0.1:8080/v1      (OpenAI-compatible; any API key)
```

More options (reuse models you already have, choose a GPU, update, roll back): [install guide](install/README.md).

## Where to get the files

The installer fetches all of these for you. If you'd rather download them yourself, use these pinned links and pass the folder with `--models-dir` / `-ModelsDir`:

| File | Role | Size | Download |
| --- | --- | ---: | --- |
| `L0xRE-27b-Low.gguf` | **Main model** (27B) | 8.6 GB | [Hugging Face](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/9b74c81c19f8372888c2968b5334f42b0354d8b1/L0xRE-27b-Low.gguf) |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | **Drafter**: speculative decoding, the source of the speed | 1.1 GB | [Hugging Face](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/9b74c81c19f8372888c2968b5334f42b0354d8b1/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) |
| `mmproj-Qwen3.8-27B-Q5_K-MIX.gguf` | **Vision projector**: lets it see images | 0.35 GB | [Hugging Face](https://huggingface.co/HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF/resolve/dbf78ac52891f19cf175abb8813997267adb0e92/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf) |
| Runtime package | Server + GPU kernels for your card | 0.65–0.75 GB | [v0.1.3 release](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/latest) (Linux `.tar.zst` / Windows `.zip`, per SM86 / SM89 / SM120) |

Checksums: [MODEL-SHA256SUMS](MODEL-SHA256SUMS) and the release's `SHA256SUMS`. Model card: [huggingface.co/YourHighnessLA/L0xRE-27b-Low](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low).

**Which runtime package?** RTX 30 series → `sm86`, RTX 40 → `sm89`, RTX 50 → `sm120`. Native Windows SM86/SM89 packages are build-verified; hardware reports for them are very welcome.

## Good to know

- **12 GB card that also runs your display?** Supported. L0xRE uses a 2048 MiB desktop arena; `--arena-mib 1920` or `1856` frees more VRAM for other apps.
- **Not enough free VRAM?** The launcher never silently downgrades. It offers Retry, an explicit slower mode where one is validated, or Quit.
- **Nothing is uploaded.** Selftest receipts stay on your machine unless you choose to share them.

## More

[v0.1.3 release notes](docs/V013-RELEASE.md) · [full release details & qualification](docs/RELEASE-DETAILS.md) · [Windows quickstart](docs/WINDOWS-QUICKSTART.md) · [model files](docs/MODELS.md) · [rollback](docs/V011-ROLLBACK.md) · [source archives](docs/SOURCE-ARCHIVES.md) · [third-party notices](THIRD-PARTY-NOTICES.md) · [licenses](licenses/)

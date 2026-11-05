<div align="center">

```text
.------------------------.
| L0xRE // BEELLAMA R6   |
| 27B / 128K / vision    |
| 12 GB / NO CPU OFFLOAD |
| > decode --truth       |
| ACCESS::GRANTED        |
'------------------------'
```

# 72 tok/s code · 128K context · vision

**On a 12 GB RTX 3060.** L0xRE keeps 128K context, vision, fast prefill and fast decode running at the same time — every model layer on the GPU, no CPU offload.

<sub>Headline numbers: RTX 3060 12 GB, native Linux, headless. Every table below names the card it was measured on.</sub>

**[⬇ Download the latest release](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/latest)** · **[Install](#install)** · **[Point your AI agent at it](#let-your-ai-agent-install-it)**

</div>

### Faster than Bonsai on both cards

Both runtimes, same card, same benchmark. The 3060 rows are where the 72 tok/s headline comes from.

**RTX 3060 12 GB · Linux, headless**

| | L0xRE | Bonsai |
|---|---:|---:|
| Code decode, tok/s · **1.3× faster** | **72** | 54 |
| First token, seconds · **2.5× faster** | **0.23** | 0.57 |

**RTX 4070 Ti 12 GB · Windows 11, with a display**

| | L0xRE | Bonsai |
|---|---:|---:|
| Decode at 128K, tok/s · **1.8× faster** | **54** | 30 |
| Read 128K prompt, tok/s · **1.2× faster** | **883** | 714 |
| Code decode, tok/s · **1.4× faster** | **118** | 86 |

**Quality** — Club-3090 8-pack, /150, thinking off, pass@1: **L0xRE 120** vs **Bonsai 108**.

<sub>Same benchmark for both runtimes on each card. Bonsai is shown in its fastest measured configuration. Quality: L0xRE on the RTX 3060, Bonsai's 108 from Club-3090 on an RTX 5090. Sources and method: [benchmarks](docs/BENCHMARKS.md).</sub>

## What to expect on your card

| Card (12 GB) | Story / chat | Code | 128K prompt in | Decode at 128K |
|---|---:|---:|---:|---:|
| **RTX 3060**, Linux | ~40 tok/s | ~72 tok/s | ~400 tok/s | ~22 tok/s |
| **RTX 4070 Ti**, Windows 11 with a display, browser and messaging apps open | ~67 tok/s | ~100–118 tok/s | ~920 tok/s | ~53 tok/s |

<sub>A 12 GB card that drives your monitor automatically leaves room for your desktop apps. Faster cards are faster still. Details: [benchmarks](docs/BENCHMARKS.md).</sub>

## Made with L0xRE

<!-- Example coming soon: something built with L0xRE. -->
*Example coming soon.*

## Install

You need an NVIDIA GPU with 12 GB or more (RTX 30, 40 or 50 series), a recent NVIDIA driver and about 20 GB of free disk. The download is about 11 GB.

**Windows 10/11.** Open a terminal in any folder and run:

```bat
curl.exe -LO https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.cmd
install.cmd -Vision
```

Then double-click **`start.cmd`** in the install folder. Setup prints where that is.

**Linux or WSL2 (Ubuntu 24.04).** Needs Python 3.12+, `curl`, `zstd` and `numactl`. Run:

```bash
curl -fL -o install-l0xre.sh https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --vision
~/L0xRE/start.sh
```

**Use it.** L0xRE serves an OpenAI-compatible API at **`http://127.0.0.1:8080/v1`** with the model name **`L0xRE-27b-Low`**. Point any app or agent that accepts a custom OpenAI endpoint at it.

**Update** with `update.cmd` on Windows or `bash install-l0xre.sh --update` on Linux. **Slow?** Run `doctor` and it tells you why ([how](docs/V0132-RELEASE.md)).

## Let your AI agent install it

Paste this to your AI agent (Claude Code, Codex, Cursor, Hermes or any agent that can run commands):

> Install L0xRE on this machine by following https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/AGENT-INSTALL.md, then connect yourself to it.

## More

- [Benchmarks and method](docs/BENCHMARKS.md)
- [Release notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases) and [all release details, qualification records and known limits](docs/DETAILS.md)
- [Windows quickstart](docs/WINDOWS-QUICKSTART.md) and [installer options](install/README.md)
- [Report your card's results](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues/new/choose)
- [Model checksums](MODEL-SHA256SUMS), [third-party notices](THIRD-PARTY-NOTICES.md) and [licenses](licenses/)

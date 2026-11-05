<div align="center">

```text
     .-======================================================================-.
   (O)                                                                      (O)
  |  0x01 // KNOWLEDGE                                  SOURCE // 0xFF   |
  |                                                                      |
  |   ████           ██████                  ████████     ██████████     |
  |   ████▒▒       ████▒▒████   ████  ████   ████▒▒████   ████▒▒▒▒▒▒▒▒   |
  |   ████▒▒       ████▒▒████▒▒   ██████▒▒▒▒ ████▒▒████▒▒ ████▒▒         |
  |   ████▒▒       ████▒▒████▒▒     ██▒▒▒▒   ████████▒▒▒▒ ████████       |
  |   ████▒▒       ████▒▒████▒▒   ██████     ██████▒▒▒▒   ████▒▒▒▒▒▒     |
  |   ████▒▒       ████▒▒████▒▒ ████▒▒████   ████▒▒████   ████▒▒         |
  |   ██████████     ██████▒▒▒▒   ▒▒▒▒  ▒▒▒▒ ████▒▒████▒▒ ██████████     |
  |     ▒▒▒▒▒▒▒▒▒▒     ▒▒▒▒▒▒                  ▒▒▒▒  ▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒   |
  |                                                                      |
  |          < INTELLIGENCE >                 < UNRESTRICTED >           |
  |           root@l0xre:~$ decode --truth     ACCESS::GRANTED           |
   (O)______________________________________________________________________(O)
    '-======================================================================-'
```

# 72 tok/s code · 128K context · vision · 12 GB

**L0xRE enables 128K context, vision, fast decode and fast prefill, all at the same time, all on 12 GB, and ALL ON GPU!**

<sub>Every model layer runs on the GPU. There is no CPU layer offload.</sub>

**[⬇ Download the latest release](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/latest)** · **[Install](#install)** · **[Point your AI agent at it](#let-your-ai-agent-install-it)**

</div>

| | **L0xRE** | Ternary Bonsai 2 27B | |
|---|---:|---:|---|
| Code decode | **72 tok/s** | 54 tok/s | **1.3× faster** |
| First token, short prompt | **0.23 s** | 0.57 s | **2.5× faster** |
| Prefill, 90K-token prompt | **444 tok/s** | 370 tok/s | **1.2× faster** |
| Quality (Club-3090 8-pack, /150) | **120** | 108 | **+12** |

<sub>Speed rows: the same RTX 3060 12 GB and the same benchmark script, with Bonsai in its fastest configuration for each row. Quality: thinking off, pass@1. Sources and method: [benchmarks](docs/BENCHMARKS.md).</sub>

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

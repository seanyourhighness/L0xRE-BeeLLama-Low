<div align="center">

<pre>
.=================================.
|██     ███          ████   █████ |
|██▒   ██▒██  ██ ██  ██▒██  ██▒▒▒▒|
|██▒   ██▒██▒  ███▒▒ ██▒██▒ ██▒   |
|██▒   ██▒██▒   █▒▒  ████▒▒ ████  |
|██▒   ██▒██▒  ███   ███▒▒  ██▒▒▒ |
|██▒   ██▒██▒ ██▒██  ██▒██  ██▒   |
|█████  ███▒▒  ▒▒ ▒▒ ██▒██▒ █████ |
| ▒▒▒▒▒  ▒▒▒          ▒▒ ▒▒  ▒▒▒▒▒|
|                                 |
|&lt; INTELLIGENCE &gt; &lt; UNRESTRICTED &gt;|
|   root@l0xre:~$ decode --truth  |
'================================='
</pre>

# 256K context · Up to 55 tok/s narrative · 98 tok/s code<br>on one RTX 3060 12 GB

**L0xRE v0.1.6.1 (RC5d r2) updates v0.1.6 Sparse on Linux and native Windows.**

<sub>Short-prompt decode on native Linux using the transformer essay and Python quicksort prompts. [Benchmark method](docs/BENCHMARKS.md#v016-sparse)</sub>

**[⬇ Download the latest release](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/latest)** · **[Install](#install)** · **[Point your AI agent at it](#let-your-ai-agent-install-it)**

**New in v0.1.6.1:** RC5d r2 runtime optimizations (padded Q3 output head, stable 32-bit D-LSA radix top-k across SM86/SM89/SM120, indexed record-copy DFlash2 hook) plus the loop-stopping verification-consistent server chat template. [v0.1.6.1 release notes](docs/V0161-RELEASE.md) · [v0.1.6 release notes](docs/V016-RELEASE.md)

</div>

## v0.1.6 decode benchmarks

### Short-prompt decode by card

| GPU and platform | Narrative (tok/s) | Code (tok/s) | Workload |
|---|---:|---:|---|
| **RTX 3060 12 GB · Linux** | **54.7** | **98.4** | Transformer essay and Python quicksort |
| RTX 3060 12 GB · Linux | 48.5 | 68.5 | Agent story and Python intervals |
| RTX 4090 · Linux | 105.2 | 132.2 | Agent story and Python intervals |
| RTX 5090 · WSL2 | 126.7 | 178.4 | Agent story and Python intervals |
| RTX 5090 · Windows | 181.9 | 318.7 | Transformer essay and Python quicksort |

The essay and quicksort rows share the same prompts. The three agent rows share a separate fixture with thinking enabled. [Prompt lengths and sample counts](docs/BENCHMARKS.md#peak-decode-by-card)

### RTX 5090 decode at depth

The native Windows benchmark added context before the transformer essay and Python quicksort instructions. Rates are in tokens/s.

| Prompt tokens | Narrative decode | Code decode |
|---:|---:|---:|
| 32,768 | 120.1 | 235.0 |
| 65,536 | 128.2 | 205.2 |
| 131,072 | 122.6 | 233.8 |
| 261,120 | 126.5 | 224.5 |

Each long-prompt row shows one measured sample per workload. [Method and WSL2 comparison](docs/BENCHMARKS.md#v016-sparse)

## v0.1.6 throughput by card

| GPU and platform | Prompt tokens | Workload | Prefill (tok/s) | Decode (tok/s) |
|---|---:|---|---:|---:|
| RTX 3060 12 GB · Linux | 131,072 | Greedy continuation | 803 | 54.8 |
| RTX 4090 · Linux | 131,700 | Needle retrieval | 2,472 | 136.6 |
| RTX 5090 · WSL2 | 131,700 | Needle retrieval | 3,662 | 139.9 |
| RTX 5090 · Windows | 131,072 | Quicksort | 3,809 | 233.8 |

Decode varies with the workload. The RTX 4090 and WSL2 RTX 5090 rows use the same needle fixture. [Workloads and sample counts](docs/BENCHMARKS.md#throughput-by-card)

<!-- Add this section back when the example is ready:
## Made with L0xRE
-->

## v0.1.5: L0xRE-V2-27b-Low

v0.1.5 makes **[L0xRE-V2-27b-Low](https://huggingface.co/YourHighnessLA/L0xRE-V2-27b-Low)** the default model. The size and speed are the same as v1, and it is 24% closer to BF16 overall on a closed holdout, 29% closer on agentic text and 34% closer on code. The runtime fast-paths four more projection shapes for it, with no kernel changes.
- **Coverage:** Linux sm86 is the RTX 3060 production build. Linux sm89 and sm120 and Windows sm120 were smoke-tested on RTX 4090 and 5090. Windows sm86/sm89 are build-verified.
- **Updating:** `update.sh` / `update.cmd` downloads V2 next to v1 and keeps v1 for rollback.

See the [v0.1.5 release notes](docs/V015-RELEASE.md). The earlier [v0.1.4 notes](docs/V014-RELEASE.md) still describe the runtime speed work.

## Install

You need an NVIDIA GPU with 12 GB or more (RTX 30, 40 or 50 series), a recent NVIDIA driver and about 20 GB of free disk. The download is about 11 GB.

**Windows 10/11.** Open a terminal in any folder and run:

```bat
curl.exe -LO https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.cmd
.\install.cmd -Vision
```

Then double-click **`start.cmd`** in the install folder. Setup prints where that is.

**Linux or WSL2 (Ubuntu 24.04).** Needs Python 3.12+, `curl`, `zstd` and `numactl`. Run:

```bash
curl -fL -o install-l0xre.sh https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --vision
~/L0xRE/start.sh
```

**Use it.** L0xRE serves an OpenAI-compatible API at **`http://127.0.0.1:8080/v1`** with the model name **`L0xRE-27b-Low`**. Point any app or agent that accepts a custom OpenAI endpoint at it.

**Update** with `update.cmd` on Windows or `bash install-l0xre.sh --update` on Linux. **Slow on Linux?** Run `doctor` and it tells you why ([how](docs/V0132-RELEASE.md)).

## Let your AI agent install it

Paste this to your AI agent (Claude Code, Codex, Cursor, Hermes or any agent that can run commands):

> Install L0xRE on this machine by following https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/AGENT-INSTALL.md, then connect yourself to it.

## More

- [Benchmarks and method](docs/BENCHMARKS.md)
- [Release notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases) and [all release details, qualification records and known limits](docs/DETAILS.md)
- [Windows quickstart](docs/WINDOWS-QUICKSTART.md) and [installer options](install/README.md)
- [Report your card's results](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues/new/choose)
- [Model checksums](MODEL-SHA256SUMS), [third-party notices](THIRD-PARTY-NOTICES.md) and [licenses](licenses/)

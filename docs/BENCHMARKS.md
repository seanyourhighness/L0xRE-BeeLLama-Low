# L0xRE benchmarks

This page holds the numbers behind the [README](../README.md) tables: where each one comes from and what it does and doesn't show. Unless a row says otherwise, numbers are from the v0.1.3 runtime. Later patch releases changed only the launcher, so the same numbers apply to them.

## L0xRE vs Ternary Bonsai 2 27B on an RTX 4070 Ti 12 GB

Both runtimes ran on the same RTX 4070 Ti 12 GB under native Windows 11, on 2026-11-05. L0xRE was v0.1.3 at the 2048 MiB arena. Each figure is one run, verified by the maintainer.

**Full context:** a 130,048-token prompt, 1,023 tokens generated, in a 131,072-token context.

| Metric | L0xRE | Bonsai MTP |
|---|---:|---:|
| Reading the prompt (prefill) | 883 tok/s | 714 tok/s |
| Decode | 54.1 tok/s | 29.8 tok/s |
| Total time | 166 s | 217 s |

Bonsai's token IDs were incomplete in this run, so its output could not be checked token by token. Its timings are the server's own.

**Short prompts** (`bench.sh`):

| Metric | L0xRE | Bonsai |
|---|---:|---:|
| Code decode | 118 tok/s | 86 tok/s |
| Narrative decode | 68.4 tok/s | 72.0 tok/s |

Bonsai is ahead on short narrative decode on this card. With the v0.1.3.1 desktop default (1856 MiB arena), the same card measured 921 tok/s prefill and 53.4 tok/s decode on a 130,048-token prompt.

## L0xRE vs Ternary Bonsai 2 27B on an RTX 3060 12 GB

Both ran on the same RTX 3060 12 GB under native Linux, with the same `bench.sh` script (the Club-3090 prompt set) and thinking off. Bonsai ran on PrismML's llama.cpp fork (club pin b12128-d7b2896). Each Bonsai figure is its fastest configuration for that metric:

| Metric | L0xRE | Bonsai (best config) | Bonsai configs measured |
|---|---:|---:|---|
| Code decode | ~72 tok/s | 54.4 tok/s (MTP, q8 KV) | MTP q4 54.2 · MTP q8 54.4 · pq2 q8 38.4 |
| Narrative decode | ~40 tok/s | 39.4 tok/s (MTP, q4 KV) | MTP q4 39.4 · MTP q8 39.2 · pq2 q8 38.4 |
| First token, short prompt | ~0.23 s | 0.57 s (MTP, q4 KV) | MTP q4 0.568 · pq2 q8 0.578 · MTP q8 0.670 |
| Prefill, ~90K-token prompt | 444 tok/s | 370 tok/s (pq2, q8 KV) | pq2 q8 370.1 · MTP q4 359.8 |

- **L0xRE code, narrative and first-token numbers** come from repeated `bench.sh` runs on the final v0.1.3 runtime on the 3060 (narrative 40–41, code 69–75, short first token about 230 ms). Code decode at 72 tok/s or above repeated across paired blocks.
- **The L0xRE 90K prefill number** was measured on v0.1.2 in the same session as the Bonsai runs. v0.1.3 did not change cold prefill (−0.15% vs v0.1.2 in the release gates).
- Narrative decode is a tie, which is why the README leaves it out.
- **Quality** is the Club-3090 8-pack, /150, thinking off, pass@1. L0xRE scores 120 (v0.1.3 on the RTX 3060, zero case flips against the v0.1 baseline). Bonsai pq2 scores 108 and Bonsai MTP 102, from Club-3090 discussion #1700, measured on an RTX 5090. With thinking on (two full passes, RTX 5090), L0xRE scores 125/127 and Bonsai 123/118.

## What to expect on your card

| Card | Story / chat | Code | 128K prompt in | Decode at 128K | Source |
|---|---:|---:|---:|---:|---|
| RTX 3060 12 GB, native Linux, headless | ~40 tok/s | ~72 tok/s | ~400 tok/s | ~22 tok/s | `bench.sh` as above; 128K figures from a 123,860-token fill on v0.1.2 (v0.1.3 decodes faster at depth: +17% at 32K) |
| RTX 4070 Ti 12 GB, native Windows 11, drives the display, with Hermes, a browser and WhatsApp open | ~67 tok/s | ~100–118 tok/s | ~920 tok/s | ~53 tok/s | v0.1.3.1 selftest at the 1856 MiB desktop arena (192-token samples: 66.9 / 102.2; 130,048-token prompt: 921 tok/s in, 53.4 tok/s out); 117.9 tok/s on a 2,048-token code sample ([report](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues/13)) |

The 4070 Ti numbers are single runs on a working desktop, not a clean benchmark. The 3060 has no display attached.

## Why 12 GB cards with a display use a smaller arena

On a 12 GB card that also drives your monitor, L0xRE picks a smaller working memory area (1856 MiB) so the desktop, a browser and messaging apps fit next to it. Short decode does not change. Decode with a full 128K context costs about 2% on the RTX 3060. To trade headroom back for deep-context speed, start with `--arena-mib 2048` or `--arena-mib 1920`.

## Full records

Per-target release gates, paired v0.1.2 comparisons and accepted exceptions are in the [v0.1.3 release notes](V013-RELEASE.md). Older qualification records are in [details](DETAILS.md).

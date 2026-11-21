# L0xRE benchmarks

## v0.1.6 Sparse

These runs used the v0.1.6 RC5d runtime with L0xRE-V2-27b-Low, DFlash2 and the 12 GB profile. Prefill and decode rates come from the server's prompt and generation timings. [v0.1.6 release notes](V016-RELEASE.md)

### Peak decode by card

The transformer attention essay and Python quicksort are the short `bench.sh` prompts. Both are 25 tokens here and run with thinking off, seed 17, temperature 0.6, top-p 0.95 and top-k 20. The native Linux RTX 3060 was measured on the running RC5d service. One measured request per prompt yielded 54.71 tok/s for the essay (935 output tokens) and 98.36 tok/s for quicksort (631 output tokens). Quicksort had one warmup. Both requests started with an empty prompt cache. The headline rounds those rates to whole tokens per second. The RTX 5090 Windows peaks are the fastest of five measured requests after three warmups per workload. The essay generated 1,000 tokens per run and quicksort generated 438. The medians were 181.3 tok/s for the essay and 318.5 tok/s for quicksort.

The RTX 3060, Linux RTX 4090 and WSL2 RTX 5090 also ran agent qualification requests for a librarian story and a Python function that merges overlapping intervals. The RTX 3060 peak is from six 192-token responses per prompt (three seeds, two RC5d boots). The RTX 4090 and WSL2 RTX 5090 peaks are from two 192-token responses per prompt (seeds 1000 and 1001), with 2,341 story and 2,345 code prompt tokens. These requests had thinking enabled, and decode includes reasoning tokens.

### RTX 5090 decode at depth

Deep runs inserted filler before the same essay and quicksort instructions and disabled prompt caching. The 261,120-token rows left 1,024 positions for generation within the 262,144-token context.

The depth comparison below uses the first measured result for each case on the same RTX 5090. Native Windows had prior warmups through 128K. The full-depth Windows cases and the WSL2 cases ran with zero warmups. Both platforms produced identical output token IDs in the 14 paired cases.

| Prompt tokens | Windows narrative | WSL2 narrative | Windows code | WSL2 code |
|---:|---:|---:|---:|---:|
| 4,096 | 168.1 | 157.5 | 315.7 | 290.1 |
| 8,192 | 139.8 | 130.8 | 268.2 | 248.3 |
| 32,768 | 120.1 | 113.8 | 235.0 | 222.0 |
| 65,536 | 128.2 | 121.8 | 205.2 | 196.3 |
| 131,072 | 122.6 | 118.1 | 233.8 | 226.9 |
| 261,120 | 126.5 | 122.4 | 224.5 | 225.7 |

### Throughput by card

- **RTX 3060, native Linux.** The two RC5d arms of the release qualification averaged 803 tok/s prefill and 54.8 tok/s decode with a 131,072-token prompt and 64 greedy output tokens. At 250,000 prompt tokens, the same sweep measured 795 tok/s prefill and 40.8 tok/s decode.
- **RTX 4090, native Linux, and RTX 5090, WSL2.** Each ran the release installer's 131,700-token needle smoke test once and generated 372 tokens. The 4090 measured 2,472 tok/s prefill and 136.6 tok/s decode. The 5090 measured 3,662 tok/s prefill and 139.9 tok/s decode.
- **RTX 5090, native Windows.** The first measured 131,072-token quicksort run generated 480 tokens at 233.8 tok/s after prefill at 3,809 tok/s. At 261,120 prompt tokens it generated 353 tokens at 224.5 tok/s after prefill at 3,697 tok/s. This is the same quicksort workload as in the depth table.

## Historical benchmarks through v0.1.5

These results record earlier runtimes, benchmarks and memory profiles. Each section identifies the version used.

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
| Code decode | ~77 tok/s | 54.4 tok/s (MTP, q8 KV) | MTP q4 54.2 · MTP q8 54.4 · pq2 q8 38.4 |
| Narrative decode | ~44 tok/s | 39.4 tok/s (MTP, q4 KV) | MTP q4 39.4 · MTP q8 39.2 · pq2 q8 38.4 |
| First token, short prompt | ~0.19 s | 0.57 s (MTP, q4 KV) | MTP q4 0.568 · pq2 q8 0.578 · MTP q8 0.670 |
| Prefill, ~90K-token prompt | ~480 tok/s | 370 tok/s (pq2, q8 KV) | pq2 q8 370.1 · MTP q4 359.8 |

- **L0xRE code, narrative, first-token and 90K prefill numbers** come from two `bench.sh` runs of v0.1.5 with the L0xRE-V2-27b-Low model on the 3060, on 2026-11-18:
  - code 78.1 and 79.5 tok/s
  - narrative 43.9 and 45.2 tok/s
  - short first token 182–200 ms
  - 10K prefill 718–721 tok/s
  - 90K prefill 479–482 tok/s

  The v0.1.3 runtime with the v1 model measured code 69–75, narrative 40–41 and first token about 230 ms.
- **The Bonsai numbers** were measured on the same RTX 3060 in an earlier session, alongside L0xRE v0.1.2, which prefilled 444 tok/s at 90K.
- Narrative decode is close: L0xRE is about 12% ahead, in this comparison.
- **Quality** is the Club-3090 8-pack, /150, thinking off, pass@1. L0xRE scores 121 (v0.1.5 with L0xRE-V2-27b-Low on the RTX 3060; v1 scored 120). Bonsai pq2 scores 108 and Bonsai MTP 102, from Club-3090 discussion #1700, measured on an RTX 5090. With thinking on (two full passes, RTX 5090), v1 scored 125/127 and Bonsai 123/118. V2 scored 127 and 125 pass@1 at medium effort with an 8K thinking budget, on an RTX 4090.

## What to expect on your card

| Card | Story / chat | Code | 128K prompt in | Decode at 128K | Source |
|---|---:|---:|---:|---:|---|
| RTX 3060 12 GB, native Linux, headless | ~44 tok/s | ~77 tok/s | ~400 tok/s | 35.9 tok/s | `bench.sh` as above (v0.1.5, V2); 128K figures from the shipped v0.1.3 deep selftest with v1, not re-measured (130,048-token prompt, 1,023 tokens generated: 394 tok/s in, 35.9 tok/s out; one run). The v0.1.1 release gate, with a different depth workload, measured 395.6 tok/s in and 32.06 tok/s out (10 runs) |
| RTX 4070 Ti 12 GB, native Windows 11, drives the display, with Hermes, a browser and WhatsApp open | ~67 tok/s | ~100–118 tok/s | ~920 tok/s | ~53 tok/s | v0.1.3.1 selftest at the 1856 MiB desktop arena (192-token samples: 66.9 / 102.2; 130,048-token prompt: 921 tok/s in, 53.4 tok/s out); 117.9 tok/s on a 2,048-token code sample ([report](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues/13)) |

The 4070 Ti numbers are single runs on a working desktop, not a clean benchmark. The 3060 has no display attached.

## Desktop arena in earlier releases

On a 12 GB card that also drives your monitor, L0xRE picks a smaller working memory area (1856 MiB) so the desktop, a browser and messaging apps fit next to it. Short decode does not change. Decode with a full 128K context costs about 2% on the RTX 3060. To trade headroom back for deep-context speed, start with `--arena-mib 2048` or `--arena-mib 1920`.

## Full records

Per-target release gates, paired v0.1.2 comparisons and accepted exceptions are in the [v0.1.3 release notes](V013-RELEASE.md). Older qualification records are in [details](DETAILS.md).

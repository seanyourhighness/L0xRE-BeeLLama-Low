# L0xRE v0.1.3

Current release: **[v0.1.3](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/l0xre-v0.1.3)**: faster decode, much faster prompt reuse (TTFT, follow-up turns, cached system prompts), and 12 GB cards that also drive a display now load (2048 MiB desktop arena; `--arena-mib 1920` or `1856` frees more VRAM).

Start with the [guided installer](install/README.md), [v0.1.3 release notes](docs/V013-RELEASE.md), [review summary](receipts/v0.1.1/REVIEW-SUMMARY.md), [provenance](docs/V011-PROVENANCE.md) and [rollback instructions](docs/V011-ROLLBACK.md). The [public install catalog](install/catalog.json) supplies current install URLs.

For complete runtime source and current release tooling, use the named [L0xRE-v0.1.3-source.tar.gz asset](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/l0xre-v0.1.3/L0xRE-v0.1.3-source.tar.gz), SHA256 `ba662bbb6d31d998fde2bc05054a4b90e4efb03dd5c75f013895dfdc6dae1982`, exact source commit `471761584b3da01547d019149eae9c8f5cf1b7fe`. GitHub's automatic Source code (zip/tar.gz) archives contain this public installer/documentation tree, not the complete runtime source.

## Inherited v0.1.1 qualification record

The hardware qualification matrix and gate results below are the v0.1.1 record, preserved unchanged. Current v0.1.3 speeds are in [Runtime expectations](#runtime-expectations-in-v013).

| Package | Profiles | Actual qualification |
| --- | --- | --- |
| Linux SM86 |12GB/128K | RTX3060 native Linux |
| Linux SM89 |12GB/128K;16GB/256K | RTX4090 native Linux |
| Linux SM120 |12GB/128K;16GB/256K | RTX5090 WSL2; no separate bare-metal Linux throughput claim |
| Windows SM120 |12GB/128K;16GB/256K | RTX5090 native Windows |
| Windows SM86 |12GB/128K | Build-verified, hardware validation in progress |
| Windows SM89 |12GB/128K;16GB/256K | Build-verified, hardware validation in progress |

SM86 has no16GB profile. Other architectures and cards below12GB are unsupported. The4090/5090 tests establish memory envelopes on24GB/32GB cards, not certification of every physical12GB/16GB card. Architecture compatibility alone does not establish another card's throughput, quality or VRAM peak.

Measured candidate rates below are generated from the final gates, in tok/s. Canonical and depth workloads differ; these are not a common-prompt throughput comparison. Each gated statistic has at least10 measured samples and CV at most5%; all72 effective speed floors pass. The depth column is the recorded depth_decode metric; separate full256K SM8916GB results are in the notes.

| Platform/preset | Narrative | Code |10K prefill | Depth decode |
| --- | ---: | ---: | ---: | ---: |
| linux/sm86/12gb | 35.45 | 60.35 | 693.83 | 32.06 |
| linux/sm89/12gb | 93.53 | 147.51 | 2269.07 | 78.89 |
| linux/sm89/16gb | 93.04 | 146.35 | 2273.87 | 82.89 |
| windows/sm120/12gb | 123.37 | 197.37 | 3398.90 | 105.32 |
| windows/sm120/16gb | 122.71 | 196.38 | 3407.09 | 112.52 |
| wsl/sm120/12gb | 112.20 | 178.46 | 3235.03 | 90.07 |
| wsl/sm120/16gb | 111.73 | 176.69 | 3226.41 | 103.57 |

The old SM8916GB10K-prefill1152.09 figure is historical and is not the current v0.1.1 result. Original v0.1 measurements remain in the unchanged [v0.1 notes](docs/V01-RELEASE.md). Full means/CVs, historical floors and source identities are in the [qualification summary](receipts/v0.1.1/qualification-summary.json) and [exact current notes](docs/V011-RELEASE.md).

Actual free VRAM must cover the measured requirement plus64MiB. If it does not fit, the launcher offers Retry, explicit Start slower where supported, or Quit; no silent fallback. Services/selftests require explicit acceptance of a slower rung. Close competing GPU applications or use an iGPU display when necessary. Keep enough disk/RAM for models, cache, extraction and the retained previous runtime. Submit selftest receipts through the [hardware report](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues/new/choose); nothing is uploaded automatically.

Known limits: WSL16GB short prefill versus12GB is -2.028086%, directly accepted only for that comparison. Mirai original full-suite losses remain recorded alongside isolated case pass rates; Windows CLI40 is not consistent regression, not both mixed. Native Windows16GB allocation recovery at5120MiB/256K/ub1024 remains an accepted inherited un-reinjected limitation, not execution PASS. Native graph re-injection was not performed. Decode-at-10K stays report-only; first-request/context-change warmup behavior and other LEAN deferrals remain documented. See the [review summary](receipts/v0.1.1/REVIEW-SUMMARY.md) for precise scopes.

Model files remain SHA-pinned in [MODEL-SHA256SUMS](MODEL-SHA256SUMS). Optional Mirai vectors are user-provided. See [third-party notices](THIRD-PARTY-NOTICES.md) and [licenses](licenses/); complete source/notice provenance ships in the named source asset.

The current installer uses the published v0.1.3 catalog. Historical release records and checksums remain unchanged. Older development branches are preserved as [source archive tags](docs/SOURCE-ARCHIVES.md); use those tags for historical builds.

<!-- l0xre-v013-expectations:start -->
## Runtime expectations in v0.1.3

Rounded v0.1.3 measurements on the named cards, 12GB profile, measured against v0.1.2 on the same GPU in the same session. They are workload examples, not forecasts for every compatible GPU. Full per-target tables, quality results and accepted exceptions are in the [v0.1.3 release notes](docs/V013-RELEASE.md).

| Measured card / platform | Narrative decode, tok/s | Code decode, tok/s | Short TTFT, ms | Follow-up TTFT, ms |
| --- | ---: | ---: | ---: | ---: |
| RTX 3060 / native Linux (SM86) | about 40 (+17%) | about 72 (+21%) | 494 (-50%) | 873 (-45%) |
| RTX 4090 / native Linux (SM89) | about 89 (±0%) | about 152 (-1%) | 293 (-47%) | 548 (-45%) |
| RTX 5090 / WSL2 (SM120) | about 115 (+10%) | about 199 (+9%) | 376 (-49%) | 399 (-57%) |
| RTX 5090 / native Windows (SM120) | about 124 (+10%) | about 209 (+7%) | 159 (-45%) | 264 (-42%) |

Decode speedups were developed on the RTX 3060; large cards gain less on decode, while TTFT and prompt-cache gains transfer everywhere. A GPU that also drives a display uses the 2048 MiB desktop arena; smaller arenas lower deep-context (about 128K) decode on fast cards. Free VRAM must cover the preset's measured requirement plus 64MiB; other GPU applications and desktop use reduce available memory.

When preflight cannot fit the default, Retry means free memory and check again; Start slower explicitly accepts a validated smaller rung and its speed cost where one exists; Quit stops. Services and selftests also require explicit slower-speed consent. The launcher never silently downgrades or treats an unvalidated setting as qualified. Native Windows SM86/SM89 remain build-only until matching hardware receipts are reviewed.

Use the [two-minute Windows quickstart](docs/WINDOWS-QUICKSTART.md) for setup. Accepted exception: WSL SM120 16GB cold 10K prefill is -2.43% versus v0.1.2; cached and follow-up turns are much faster on the same profile.
<!-- l0xre-v013-expectations:end -->

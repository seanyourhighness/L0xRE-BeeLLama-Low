# L0xRE v0.1.1

Run L0xRE-27b-Low with DFlash2 speculative decoding, GPU vision, Q4 KV streaming and explicit128K/256K memory profiles. Thinking is off by default. Target, drafter and qualified Q5 projector pins are unchanged from v0.1.

Start with the [guided installer](install/README.md), [release notes](docs/V011-RELEASE.md), [review summary](receipts/v0.1.1/REVIEW-SUMMARY.md), [provenance](docs/V011-PROVENANCE.md) and [rollback instructions](docs/V011-ROLLBACK.md). The [manifest](MANIFEST.json) indexes the eleven immutable assets; the [public install catalog](install/catalog.json) supplies current install URLs.

For complete runtime source use the named [L0xRE-v0.1.1-source.tar.gz asset](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/l0xre-v0.1.1/L0xRE-v0.1.1-source.tar.gz), SHA256 `ddd7db47f7f17b884f4e8ab75ad66dd649382f8931b1fb11c5a0ab58c66e99d9`, exact source commit `5da9c28f7283815f627931376de96fe9a2a4fb0e`. GitHub's automatic Source code(zip/tar.gz) archives contain this public installer/documentation tree, not the complete runtime source. The public distribution tag is distinct from the exact source commit. The named asset has a completed five-stage clean extraction build and17 exact kernel outputs.

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

The frozen catalog-v0.1.1.json release asset retains private URLs as historical qualification provenance and is not the installer catalog. Use install/catalog.json. Sean publishes the draft and marks it latest, then merges the metadata PR. After both steps, run `python3 tools/verify-public.py --verify-public --check-rollback` for anonymous download, public main and rollback checks. No agent publishes or merges this preparation.

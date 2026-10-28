# L0xRE v0.1

Run L0xRE-27b-Low with DFlash2 speculative decoding, GPU vision, Q4 KV streaming, RAM/checkpoint restore, and explicit 128K or 256K memory profiles. Thinking is off by default. The target and drafter are unchanged; v0.1 uses the pinned Q5 vision projector.

## Before you run

The launcher uses the fastest validated arena when **actual free VRAM covers the measured requirement plus 64 MiB**. The desktop reserve is a warning, not a second subtraction from free memory.

If it does not fit, loading stops. Close browsers, overlays, GPU wallpapers and game launchers. If your board has an iGPU, connect the display to the motherboard to free the discrete GPU. The launcher reports available memory and holding processes where the driver can identify them, then offers:

- **[R] Retry** after closing GPU applications; queries free memory again.
- **[S] Start slower** on a measured fitting fallback, with your explicit acceptance of its speed cost.
- **[Q] Quit** without loading the model.

When no rung fits, only Retry and Quit are offered. Services and selftests refuse a slower rung unless `--accept-slower` or `L0XRE_ACCEPT_SLOWER=1` is set. A validated explicit `--arena-mib` selection still checks memory and counts as consent. The selected arena, ubatch, memory figures and consent are recorded.

| Platform / preset | Arena / ubatch | Required free VRAM including margin | Near128K decode | Selection |
| --- | --- | ---: | ---: | --- |
| Linux SM86 12GB | 2240 / 1024 | 11691 MiB | 31.86 tok/s | Default; refuse if it does not fit |
| Linux SM89 12GB | 2312 / 1024 | 12082 MiB | 77.40 tok/s | Default; refuse if it does not fit |
| Linux SM89 16GB | 5120 / 1024 | 14898 MiB | 81.78 tok/s | Default for 16GB/256K |
| WSL2 SM120 12GB | 2144 / 1024 | 12058 MiB | 86.34 tok/s | Default; refuse if it does not fit |
| WSL2 SM120 16GB | 5120 / 1024 | 15036 MiB | Depth speed probe omitted | Default for 16GB/256K |
| Windows SM120 12GB | 2240 / 1024 | 12090 MiB | 100.19 tok/s | Full-speed default |
| Windows SM120 12GB fallback | 1280 / 512 | 11130 MiB | 65.22 tok/s | Explicit consent; about33% slower than retained97.71 full speed |
| Windows SM120 16GB | 5120 / 1024 | 14972 MiB | 104.76 tok/s | Default for 16GB/256K |

The optional Windows2144 rung is omitted. Windows SM86/SM89 bundles are **build-verified, hardware validation in progress**: native Windows throughput and VRAM peaks are not measured. Their provisional profiles use Linux reference envelopes, which may not predict Windows overhead; run selftest and submit the receipt before hardware certification.

On a12 GiB card driving a desktop, the fast Windows rung requires12090 MiB free. You may need an iGPU display/headless card, fewer GPU applications, or the consented fallback. A12GB/16GB preset is a tested memory envelope, not a throughput certificate for every physical12GB/16GB card. The4090 and5090 envelope tests used24GB and32GB cards.

## Get started

Linux/WSL2: Ubuntu24.04 or compatible, Python3.12+, `curl`, `zstd`, `numactl`, and a working NVIDIA driver.

```bash
curl -fL --retry 3 -o install-l0xre.sh \
  https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --dry-run
bash install-l0xre.sh
```

Windows10/11 x64, PowerShell; Python, CUDA runtime and Microsoft runtime DLLs are bundled.

```powershell
Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.ps1' -OutFile install-l0xre.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1
```

Setup detects the architecture, verifies archive and model hashes, preserves the previous installation on update, and prints the start command. Use that command to launch the OpenAI-compatible API. Model paths, API host/port and memory preset remain configurable. Keep enough free system-drive space for extraction, model mappings, checkpoints and the previous runtime; the profiles use4–8GiB RAM cache in addition to model/runtime memory.

## Profiles and validation

| Package | Presets / context | Actual test platform | Status |
| --- | --- | --- | --- |
| Linux SM86 | 12GB /131072 | RTX 3060 12GB, native Linux | Quality, correctness, speed, selftest and clean install/rollback validated |
| Linux SM89 | 12GB /131072;16GB /262144 | RTX 4090 24GB, native Linux | Quality, correctness, speed and selftests validated |
| Linux SM120 | 12GB /131072;16GB /262144 | RTX 5090 32GB, WSL2 | WSL2 validated; no separate bare-metal Linux throughput claim |
| Windows SM120 | 12GB /131072;16GB /262144 | RTX 5090 32GB, native Windows | Quality, correctness, speed, selftests and clean install/rollback validated |
| Windows SM86 | 12GB /131072 | No native test card available | Build-verified, hardware validation in progress |
| Windows SM89 | 12GB /131072;16GB /262144 | No native test card available | Build-verified, hardware validation in progress |

SM86 has no16GB profile. Other GPU architectures and cards below12GB are unsupported. Other cards within SM86/SM89/SM120 need their own hardware receipt; architecture compatibility does not establish their speed. WSL was tested on SM120 only.

## Measured speeds

Named canonical receipts use frozen Bench.sh with3 warmups,5 narrative,5 code and3 nominal10K-prefill runs. Depth uses3 repeats and1023 generated tokens. Prose, code and deterministic depth are different workloads; their acceptance rates differ.

| Profile / receipt | Narrative tok/s | Code tok/s | Nominal10K prefill tok/s | Near128K decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Linux SM86 /373+347 | 34.70 | 59.96 | 665.31 | 31.86 |
| Linux SM89 12GB /429 | 88.04 | 151.83 | 2194.37 | 77.40 |
| Linux SM89 16GB /372 | 88.80 | 149.07 | 1152.09 | 81.78 |
| WSL2 SM120 12GB /364+358 | 104.51 | 177.79 | 2760.26 | 86.34 |
| WSL2 SM120 16GB /386 | 105.06 | 188.14 | 3014.54 | Not measured |
| Windows SM120 12GB /407 | 107.90 | 187.98 | 3197.91 | 100.19 |
| Windows SM120 16GB /260+258 | 109.82 | 182.93 | 3042.05 | 104.76 |

**Sean's SM89 decision, October28,2026:** accept the paired451 comparison: old/new narrative88.84068/88.28072(−0.63%) andcode148.86917/146.28114(−1.74%), both within the unchanged 2% floor. The historical90.89421 narrative baseline175(I8cap4096) remains recorded; the old build today reaches88.84068. This explicit comparison decision changes no other gate or baseline.

The transferred host paths improved matched128K decode31.62→31.86(+0.75%) onSM86 and83.24→86.34(+3.72%) onWSL SM120, with identical full outputs and draft acceptance. The original SM89 gains were not assumed to transfer unchanged.

`decode_at_10k` is report-only by Sean's decision because it times only16 tokens. Windows407 measured119.43 against retained147.85 tok/s. Paired417 old/new/new/old measured151.28/144.66/102.76/134.85; combinedold143.06/new123.71(−13.53%). The unchangedold build itself varied151vs135. The nominal10K **prefill** and1023-token deep-decode gates remain active. No more GPU time was spent on this16-token metric.

Full CVs, quality scores, source hashes, headroom and known limitations are in [release notes](docs/V01-RELEASE.md), [provenance](docs/V01-PROVENANCE.md) and the release evidence archive. Historical80K R6 speeds are not substituted for these128K/256K profiles.

## Download and selftest

The [v0.1 release](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/l0xre-v0.1) contains six architecture/platform runtime archives, the source and evidence bundles, catalog and `SHA256SUMS`. Runtime archives contain no model weights or optional control vectors.

After extraction, run `l0xre selftest` on Linux/WSL or `l0xre.cmd selftest` on Windows with the same model, projector, GPU and `--memory-profile` options used for serving. The four main variants are12GB/16GB × vectorsOFF/Mirai where supported. Mirai is user-provided. Submit `receipt.json` through the [hardware validation report](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues/new/choose); nothing is uploaded automatically. Windows SM86/SM89 selftests explicitly use Linux-derived expected outputs until native receipts establish equivalence.

## Models and rollback

| Model file | Bytes | Role |
| --- | ---: | --- |
| `L0xRE-27b-Low.gguf` | 8619127680 | Target |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1143006816 | DFlash2 drafter |
| `mmproj-Qwen3.8-27B-Q5_K-MIX.gguf` | 347213792 | Required qualified vision projector |

Target/drafter use the pinned YourHighnessLA source; the Q5 projector uses pinned HermiHg revision`dbf78ac52891f19cf175abb8813997267adb0e92`. Setup verifies all three hashes in [MODEL-SHA256SUMS](MODEL-SHA256SUMS). The previous Q8 projector is retained separately for R6 rollback.

Updates preserve the previous runtime and launcher. Restore them with `python3 install/install.py --rollback --dir PATH` or `powershell -File install/install.ps1 -Rollback -InstallDir PATH`, then start the restored launcher. [Rollback instructions](docs/V01-ROLLBACK.md) retain the publishedR6 reference and checksums; no DLLs or cubins should be mixed between releases.

## Known limits and licenses

NativeWindows allocation recovery passed2240/ub1024(receipt442). Graph-instantiation recovery is covered by Linux/WSL SM120 receipt355 and was **not re-injected on nativeWindows for this rung**; Sean stopped further graph-fault shims, hooks and in-process patching. Generic`cudaMalloc` failure recovery is unproven. HistoricalR7 replay and return-to-short limitations remain documented. Memory envelopes on larger cards are not physicalSKU certification.

See [third-party notices](THIRD-PARTY-NOTICES.md), [licenses](licenses/), and [source/build provenance](docs/V01-PROVENANCE.md). Upstream source and generated-kernel notices remain with the source bundle. Optional control-vector rights remain the user's responsibility.

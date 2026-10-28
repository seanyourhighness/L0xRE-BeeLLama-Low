# L0xRE v0.1 release notes

This release uses explicit128K/256K presets, consent-based VRAM preflight and pinned runtime/model hashes. R6 remains the rollback reference. The release evidence records final archive and download verification.

V0.1 runs L0xRE-27b-Low with DFlash2 speculative decode, GPU vision, current cold/live/RAM/checkpoint exactness, and explicit memory profiles. The qualified default has thinking off. The model and drafter are unchanged from R6; V0.1 uses the separately pinned Q5 vision projector below.

## Before you run

The launcher starts the fastest validated setting automatically when measured free VRAM covers its measured requirement plus the 64 MiB safety margin. The desktop reserve is a warning level; it does not reduce the free-memory figure a second time.

If the full-speed setting does not fit, the launcher stops before model loading, reports free/required VRAM and processes where available, and offers `[R] Retry`, `[S] Start slower`, or `[Q] Quit`. Close browsers, overlays, game launchers and GPU wallpapers, then retry. If your board has integrated graphics, plug the monitor into the motherboard so the GPU is free. Retry queries memory again. Start slower selects the fastest fitting measured rung and records your consent. If no rung fits, only Retry and Quit are offered.

Services and `selftest` never prompt. They refuse a slower rung unless you provide `--accept-slower` or `L0XRE_ACCEPT_SLOWER=1`. An explicit validated `--arena-mib` also counts as consent. Every choice still checks free memory. `--desktop-reserve-mib` controls the warning level only. The launcher changes `-ub` with the chosen arena and records both, the free/required figures, consent source and measured speed cost.

Windows SM120 12GB measured ladder. The optional2144 rung is omitted using the first authorized scope cut. The full-speed default passes the speed floor; archive and release checks are recorded in the release evidence.

| Rung | Arena / ubatch | 128K decode tok/s | Cost against full speed | Measured peak + safety margin | Selection |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | 2240 / 1024 | 97.71 retained;100.19 fresh407 | 0% | 12026 +64 =12090 MiB | Full-speed default; speed gate passed |
| 1 | 1280 / 512 | 65.22 | about33% vs retained full speed | 11066 +64 =11130 MiB | Explicit consent |

Sources:302depth,271both-mode current/full128K restore,407fresh performance;315/382for the fallback. The fallback measured about15% lower10K prefill and19% lower short prefill against the best retained Windows12GB receipts. All measured shipping presets pass the recorded gates; archive identities are listed in the release catalog and SHA256SUMS.

On a12 GiB card driving a desktop, the full-speed Windows rung needs about12.1 GB free (12090 MiB including margin). Most users will need to free the GPU, use an iGPU display/headless card, or explicitly accept a measured slower rung. A12GB/16GB preset describes a tested VRAM envelope. Tests on a24GB or32GB card do not certify a physical12GB or16GB SKU's throughput.

## Profiles and coverage

| Platform | Architecture | Presets | Context | Physical test card | Current release status |
| --- | --- | --- | --- | --- | --- |
| Native Linux | SM86 | 12GB | 131,072 | RTX 3060 12GB | Core gates and private clean archive/selftests/rollback pass; published-archive checks are recorded in the closing audit |
| Native Linux | SM89 | 12GB / 16GB | 131,072 / 262,144 | RTX 4090 24GB | Quality/lifecycle/recovery, all speed gates and four selftests pass; final clean startup/restore and R6 rollback pass |
| Linux/WSL2 | SM120 | 12GB / 16GB | 131,072 / 262,144 | RTX 5090 32GB | Core gates, four WSL selftests and private clean archive/rollback pass; published-archive checks are recorded in the closing audit |
| Native Windows | SM120 | 12GB / 16GB | 131,072 / 262,144 | RTX 5090 32GB | Quality/lifecycle, both speed gates and private clean archive/selftests/rollback pass; native graph recovery limitation below; published-archive checks are recorded in the closing audit |
| Native Windows | SM86 | 12GB | 131,072 | None available | Build verified; hardware validation in progress; archive extraction and loader checks are recorded in the release evidence |
| Native Windows | SM89 | 12GB / 16GB | 131,072 / 262,144 | None available | Build verified; hardware validation in progress; archive extraction and loader checks are recorded in the release evidence |

SM86 has no 16GB preset. Other cards within an architecture need their own hardware-validation receipts. WSL was tested on SM120 only. The measured Linux SM1202144 MiB arena needs12058 MiB including its safety margin. Its launch decision uses actual free memory; the desktop reserve is a warning. Native Windows SM1201280/ub512 remains a measured slower rung, with its own quality receipts. Windows2240/ub1024 is rung0 and passes the speed floor. These Windows settings are not applied to Linux. The new consent launcher passed actual selftests on Windows SM120(416), Linux SM86(415), Linux SM89(414) and WSL SM120(419). Final archive startup and integrity checks remain required.

## Measured performance

These are measurements of the named candidate profiles, not final-archive certificates. Canonical results use unchanged Bench.sh with three warmups, five narrative runs, five code runs and three nominal 10K-prefill runs. Matched depth uses three repeats and 1023 generated tokens per sample. Canonical prose/code and the deterministic depth prompt have different acceptance rates and must not be compared as if they were one workload.

| Candidate / platform | Canonical narrative tok/s | Canonical code tok/s | Canonical 10K prefill tok/s | 128K depth tok/s |
| --- | ---: | ---: | ---: | ---: |
| SM86 324 / Linux / 12GB | 34.70 | 59.96 | 665.31 | 31.86 |
| SM89 304 / Linux / 12GB / 429 | 88.04 | 151.83 | 2194.37 | 77.40 |
| SM89 304 / Linux / 16GB / 372 | 88.80 | 149.07 | 1152.09 | 81.78 |
| SM120 320 / WSL / 12GB | 104.51 | 177.79 | 2760.26 | 86.34 |
| SM120 320 / WSL / 16GB / 386 | 105.06 | 188.14 | 3014.54 | Optional depth speed test omitted |
| SM120 406 / Windows / 12GB / 407 | 107.90 | 187.98 | 3197.91 | 100.19 |
| SM120 252 / Windows / 16GB | 109.82 | 182.93 | 3042.05 | 104.76 |

SM89 16GB measured 76.22 tok/s at near256K, with 1136.79 tok/s prefill; its near128K prefill was1718.17. The lower16GB canonical prefill result remains reported as measured. Linux SM120 results were measured under WSL2; they are not a separate bare-metal Linux throughput measurement.

Canonical CVs: SM86 narrative 1.95%, code 2.68%, prefill 0.68%; SM120 WSL narrative 3.11%, code 7.87%, prefill 4.51%; Windows SM120 16GB narrative 2.12%, code 2.96%, prefill 1.43%. SM89 12GB429 CVs were3.02% narrative,3.29% code and0.75% prefill. Previous80K R6 measurements are not substituted for the new profiles.

The fresh host-port comparisons measured SM86 128K decode at 31.62 → 31.86 tok/s (+0.75%) and SM120 WSL at 83.24 → 86.34 (+3.72%), with full output hashes and draft acceptance identical. The SM89 percentages that motivated the transfer were not assumed to apply to other architectures.

A further SM120 deterministic short-workload check measured narrative 88.87 → 87.99 tok/s (-0.99%, six samples per library across both test orders), code 187.98 → 188.35 (+0.20%) and exact-10K prefill 3194.05 → 3194.30 (+0.01%). The initial -2.62% narrative result did not reproduce in reverse order (+0.67%); the combined difference was within observed variation. This does not prove zero performance effect. The exact-10K test uses server prompt timings and identical token IDs; it is a separate protocol from canonical Bench.sh.

The full-speed Windows12GB run407 measured107.90 narrative,187.98 code,3197.91 nominal10K prefill,125.28 short decode,2452.19 short prefill,100.19 near128K decode and2068.26 near128K prefill. These seven headline comparisons pass the2% speed floor.

By Sean's explicit decision, `decode_at_10k` is report-only because this canonical probe times only16 output tokens, with3 samples. It remains fully reported:407measured119.43 versus the retained147.85 tok/s;417pairedold/new/new/old measured151.28,144.66,102.76 and134.85 tok/s. Combinedold143.06 versusnew123.71(-13.53%). The unchangedoldbuild itself varied151vs135. Allfourarms used identical numericalconfiguration, actualDLL hashes and affinity. No furtherGPUtime is allocated to this metric. Only this metric is report-only; no other2%limit was relaxed. Windows SM12012GB2240/ub1024 passes427/windows-speed-floor-results.json, with archive/startup/rollback evidence in the release packet.

**Sean's release decision, October28,2026:** accept Linux SM89 12GB using the same-session both-order451 comparison: narrative88.84068→88.28072 tok/s(−0.63%) andcode148.86917→146.28114(−1.74%), both within the unchanged 2% floor. Historical receipt175 reached90.89421 narrative tok/s with I8cap4096; the old build today reaches88.84068. The historical baseline and the original429 results(88.04 narrative/151.83code) remain retained. Sean explicitly accepts451 for this narrative comparison despite the original miss slightly exceeding the individual runCVs. Allothergates remain unchanged. All nine measured Linux/WSL/Windows presets pass the speed ledger477; Windows SM86/SM89 remain build-only, with no native throughput claim.

## Quality and correctness

| Final numerical path | Thinking-OFF N150, vectors OFF | Thinking-OFF N150, Mirai |
| --- | ---: | ---: |
| Linux SM86 324 | 120 / 150 | 117 / 150 |
| Linux SM89 12GB225 | 120 / 150 | 117 / 150 |
| Linux SM89 304 16GB | 120 / 150 | 118 / 150 |
| Linux/WSL SM120 320 | 120 / 150 | 118 / 150 |
| Native Windows SM120 full-speed401/406 12GB | 120 / 150 | 117 / 150 |
| Native Windows SM120 slower315 1280/ub512 | 121 / 150 | 118 / 150 |
| Native Windows SM120 252 16GB | 120 / 150 | 118 / 150 |

The floors remain 120 OFF and 115 Mirai. Scores apply only to the corresponding hashed numerical path; no threshold or exactness tolerance was lowered. Current cold/live/RAM/checkpoint checks include populated top10 parity, vision fixtures, image identity and speculative resumption. The old R7 frozen-replay fixture still fails and is retained as a known stale-fixture limitation; it is not relabeled as a passing test.

Both modes pass arena-lease refusal and graph-instantiation recovery on the new Linux host ports. This is not a claim about arbitrary `cudaMalloc` failures. The earlier interrupted SM120 16GB Mirai run was attributed by Sean to system-drive exhaustion; the unchanged-artifact retry passed with disk-space guards. Native Windows allocation recovery passed on the 2240/ub1024 rung (receipt442): actual arena-lease refusal, same-process survival, exact text/image outputs, resumed speculation and clean reconstruction. **Known limitation:** graph-instantiation recovery is covered by the Linux/WSL SM120 receipt355 and was not re-injected on native Windows for this rung. Sean stopped further Windows graph-fault shim, hook and in-process patching work. Generic cudaMalloc failure recovery remains unproven; arena-lease coverage does not establish it.

## Model files

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `L0xRE-27b-Low.gguf` | 8619127680 | `b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543` |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1143006816 | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |
| `mmproj-Qwen3.8-27B-Q5_K-MIX.gguf` | 347213792 | `023c9b484ba6c2c22dd64623cfe286dd4a888170e68bfad6745957e4bd045c8f` |

The target and drafter are available from the existing pinned model repository. The exact Q5 projector is available from the pinned HermiHg repository revision dbf78ac52891f19cf175abb8813997267adb0e92. Its published LFS hash matches qualification; the installer downloads it directly from that source. The old Q8 projector is not interchangeable with its qualified hash. Keep old model files when upgrading so rollback can use the previous projector.

## Start and validate

After extracting the appropriate completed archive, run `l0xre serve` on Linux/WSL or `l0xre.cmd serve` on Windows, with:

```text
--memory-profile 12gb -m PATH/TO/L0xRE-27b-Low.gguf -md PATH/TO/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --mmproj PATH/TO/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf --host 127.0.0.1 --port 8080
```

Select `16gb` only where listed. The launcher detects the GPU architecture and applies the sealed preset. The installer chooses 16GB on supported cards with at least 16GB, otherwise 12GB; SM86 always selects its supported 12GB preset. Model paths remain configurable.

Replace `serve` with `selftest` and keep the same model/preset options to collect GPU, driver, OS, hashes, preflight, exact reference/cache/vision checks, short timing and peak-VRAM receipts. `--deep` adds the optional 128K smoke. Submit `receipt.json` through the **Hardware validation report** issue form. Nothing is uploaded automatically. Windows SM86/SM89 must remain labeled **build-verified, hardware validation in progress** after full build verification, until real tester receipts are reviewed.

For the qualified Mirai configuration, add a user-provided vector with SHA-256 `3f3432d09822864e0bcb4aff27f52ef58424fa92734e352a6abe67c2b570d879` and:

```text
--control-vector PATH/TO/VECTOR.gguf --control-vector-mode project --control-vector-scale 1 --control-vector-layer-range 1 63
```

Other vectors or settings are not covered by the shipped Mirai reference. Prefer vectors OFF when producing a basic hardware report. The 192-token selftest timing is a smoke test, not canonical performance certification.

## WSL and host memory

Use a working NVIDIA driver and enough host RAM for the chosen model/cache settings. WSL emits a warning if the host cache plus model-file sizes exceed about 70% of the VM memory limit. The launcher preserves qualified cache sizes and removes `expandable_segments:True` from inherited PyTorch allocator settings on WSL. If mirrored localhost networking does not connect, check the Windows/WSL binding and firewall settings before changing the model profile. Bind only to a trusted interface when remote access is needed.

## Publication boundary

Windows SM86/SM89 build audits pass (439/456/458/460/466/467/476), including final archive extraction, bundled DLL loading and wrong-architecture refusal. Available-platform archives pass clean installation, normal startup without a qualification bypass, streaming, restore, selftests and actual R6 rollback. The closing audit records publication and downloaded-file verification. Windows SM86/SM89 remain build-only; the graph/allocation, return-to-short, historical R7 and physical-SKU limitations above are retained. See [provenance](V01-PROVENANCE.md) and [rollback](V01-ROLLBACK.md).

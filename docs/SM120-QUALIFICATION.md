# SM120 certification and release details

The R6 SM120 profiles use the same L0xRE-27b-Low target and Q4_K_M DFlash2 drafter on a single RTX 5090 with 32 GB. Linux / WSL and native Windows have separate payloads, measurements, and qualification records. The [main README](../README.md) presents rounded speed cards; this page retains the exact results and scope.

## Linux / WSL qualification

The Linux / WSL build was locally certified on October 8, 2026, with NVIDIA driver 617.42. The exact sealed archive is available in the [universal R6 release](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r6/L0xRE-SM120-certified-20261008.tar.zst). Older universal SM120 payloads do not inherit this certification.

| Measurement | Confirmed result |
| --- | --- |
| Prose decode | 131.32153613 t/s; five measured runs after three warmups |
| Code decode | 229.12536969 t/s; five measured runs after three warmups |
| Target p2048 prefill | 3526.978457 t/s; five measured runs |
| Stability | 1829.76-second soak; 258 requests; 249 repeat comparisons |
| Capacity | Two fresh 77,824-prompt + 4,092-output = 81,916-token runs, matching outputs |
| Quality at 32K | Candidate 129/150 pass@1 and 133/150 pass@3; R9 baseline 124/150 and 132/150; zero runaways |
| Other gates | Cold greedy/seeded correctness, restart, CPU vision, source/artifact verification, and clean consumer package checks passed |

The initial decode screen averaged 233.49 t/s; confirmation averaged 229.13 t/s and did **not** meet the original greater-than-233 target. Certification of the same soaked build proceeded with a disclosed, user-authorized decode-performance exception. The confirmed result is the score shown in the README. This exception belongs to this exact Linux build; Windows independently meets its release speed gates.

The frozen profile uses DFlash2 N7, context 81,920, batch 1,024 / ubatch 512, KVarN4/4, exact tail 128, graphs ON, asynchronous launches, CPU affinity 0–7, down16, head-next1/RT2, K256, INT8 sync8, and the immutable down-weight cache. The p2048 benchmark is a separate native target-prefill workload. Quality used the original 32K paired-case-review protocol; the counts above are observed scores, not fixed count floors. Mixed case losses and statistical uncertainty remain retained. This is local release qualification, not vendor certification or proof of a GPU-driver root-cause fix.

The [Linux certification summary](../receipts/sm120-linux/CERTIFICATION-SUMMARY.json) records exact scores, gates, component hashes, the sealed certificate hash, and archive identity. It is a public summary, not a replacement for the sealed certificate.

- Archive: `L0xRE-SM120-certified-20261008.tar.zst` (1,111,927,641 bytes)
- Archive SHA-256: `9af12274ef1b7c41923fc35bebea2097a37a20afdba6a6552ea1eb22eccb87ff`
- CUDA backend SHA-256: `30442d5b9c45cc40d17360e811b7b03441e0e7195d581b60530cb9716d073264`
- Packed bridge SHA-256: `d2c46b30b7135099909341d462ee0f3e1267cc759e625ed92bacb909e3e650a9`

## Native Windows qualification

This separate package is qualified on **RTX 5090, Windows 11, NVIDIA driver 617.42**, with the exact [model hashes in the README](../README.md#models-and-integrity). It bundles CUDA runtime13.0.96 and cuBLAS13.1.1.3; nvcc13.0.88 was used to build it. A CUDA compiler is not required to run the archive. The claim applies to one GPU and one slot, not to all RTX50-series configurations.

| Measurement | Result |
| --- | --- |
| Code decode | **229.68038347 t/s**, n15, sample SD8.22226, three fresh starts |
| Target p2048 prefill | **3611.28296150 t/s**, n10, sample SD13.36149, two fresh starts |
| Stability / capacity | 1841-second soak; fresh restart; two fresh77824+4092=81916-token runs |
| Quality, original32K protocol | Windows125/150 pass@1,133/150 pass@3; fresh Linux126/150,133/150; zero runaways |
| Other gates | Exact Linux greedy/seeded cold-correctness outputs, CPU vision fixture,15 compiled regressions, active kernel/DLL audit, real consumer launcher |

The frozen default is DFlash N7, context81920, batch1024/ubatch512, KVarN4/4, tail128, graphs ON, asynchronous launches, threads8/affinity255, down16, head-next1/RT2, K256, INT8 sync8 and down-weight caching. Profiling and diagnostic runs are excluded from performance means. The earlier CUDA13.3/13.0 experiment changed compiler and runtime jointly; no compiler-only improvement is claimed.

Download the [Windows SM120 release](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/beellama-v0.4.7-r6-sm120-windows-cuda130), its runtime ZIP and `WINDOWS-SM120-SHA256SUMS.txt`. Verify the ZIP's SHA-256, extract to a fresh directory, enter `package`, and run:

```powershell
.\l0xre.ps1 -Model 'D:\models\L0xRE-27b-Low.gguf' -Draft 'D:\models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf' -Port 31990
```

Use your actual model paths. Add `-DryRun` to verify files and print the resolved configuration first, or `-Mmproj 'D:\models\mmproj-Qwen3.8-27B-Q8_0.gguf'` for the tested CPU-vision path. The API is `http://127.0.0.1:31990/v1`; stop the foreground server with Ctrl+C. `QUICKSTART.txt`, the certificate, file manifest and source-input archive accompany the release. Model weights are separate.

**Quality scope and retained limitations:** the original Linux qualification used a paired case review at32K, not a fixed129/133 count floor. An initial Windows adapter mistakenly treated those historical scores as minimums; the error and earlier verdicts are retained in `REQUIREMENTS-AUDIT.json`. Mixed case losses remain documented, and one unseeded sample per arm does not prove distributional equivalence. An additional81920-context quality trial scored121/150 and129/150; it is retained separately and does not establish quality across every81K prompt. The explicit decode229.12537 and prefill3526.978457 requirements remain met without a performance exception.

The archived binary hashes are the qualification identities. Rebuilt binaries must be independently verified; source availability alone does not certify a new build. Existing universal Windows/SM89 assets retain their prior candidate status.

## Windows evidence

- [Release readiness, performance, and publication status](../receipts/windows-sm120-release-readiness.json)
- [Exact performance runs, variation, and gates](../receipts/windows-sm120-cuda130/PERFORMANCE.json)
- [Certificate](../receipts/windows-sm120-cuda130/WINDOWS-SM120-CERTIFICATE.json)
- [Quality case review](../receipts/windows-sm120-cuda130/QUALITY-CASE-REVIEW.json)
- [Requirements audit and historical-score-floor correction](../receipts/windows-sm120-cuda130/REQUIREMENTS-AUDIT.json)
- [Code and active-route audit](../receipts/windows-sm120-cuda130/CODE-AND-ROUTE-AUDIT.json)
- [Archive consumer verification](../receipts/windows-sm120-cuda130/ARCHIVE-CONSUMER-VERIFICATION.json)
- [Published assets and verified checksums](../receipts/windows-sm120-cuda130/GITHUB-PUBLICATION.json)

The confirmed Windows narrative mean is **136.58077960 t/s**, rounded to **137 t/s** in the README. Code and narrative used 15 measured runs per shape across three fresh server starts; p2048 prefill used 10 measured runs across two fresh starts. These starts are process restarts, not a claim of OS reboot parity.

## Unified project and next architecture

Both platforms stay in this repository and use the same model and drafter hashes. Each platform keeps its own archive, launcher, and certification evidence. Consolidating the documentation does not replace or repack the certified binaries. SM89 / RTX 4070 Ti Windows is the next hardware target; SM89 Linux and Windows packages remain candidates until their own correctness, speed, quality, capacity, and stability gates pass. See [architecture status](../PARITY.md).

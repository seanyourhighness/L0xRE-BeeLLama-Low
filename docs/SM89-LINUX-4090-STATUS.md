# R6 Linux RTX4090 optimization and qualification

The newest published Linux SM89 payload was installed on Z840, then updated from the current complete R6 native source inputs. The selected candidate preserves81920 context, KVarN4/4, one slot, DFlash2 Q4 N7, medium reasoning, eight CPU threads and NUMA0 at the host's persistent300W power limit. It enables the optional INT8 down-weight cache.

The named `r6-sm89-4090` profile is now **certified locally on RTX4090 Linux at300W**. After Sean removed the newly installed suspect RAM andRTX3060 and rebooted, the unchanged candidate passed fresh correctness, CPUvision, a complete32-minute mixed-depth soak, and the final archive consumer probe. The certified installation is separate from the frozen candidate. The earlier failed run is retained below. Universal catalog finalization and the first full stable publication remain pending.

## Optimization screens

These are screening observations, not final certification claims. Seed42 is pinned and effective sampling is observed. The initial unseeded exploratory arms are excluded from the comparison.

| Screen | Prose decode t/s | Code decode t/s | Cold prefill |
|---|---:|---:|---:|
| S00-published-seed42 | 101.73 | 168.78 | 10,240 tokens: 1859.66 t/s |
| M03-cache0-warm-repeat | 100.96 | 166.07 | 10,240 tokens: 1873.94 t/s |
| M04-cache1-warm-repeat | 100.76 | 166.24 | 10,240 tokens: 1942.68 t/s |
| L00-long-cache0 | — | — | 77,824 tokens: 1613.58 t/s |
| L01-long-cache1 | — | — | 77,824 tokens: 1673.43 t/s |
| R00-current-native-cache1 | 103.32 | 168.41 | 10,240 tokens: 1943.26 t/s |
| R16-current-native-splits16 | 94.14 | 160.30 | 10,240 tokens: 1940.09 t/s |
| R77-current-native-long | — | — | 77,824 tokens: 1674.28 t/s |

The warm cache comparison used three warmups and five measurements per prompt. The cache improved10K and77K prefill while warm decode remained essentially unchanged. The complete current native candidate reproduced the historical128-token greedy code/prose hashes and the long-prefill token. Split16 is supported by the newer backend but was slower; the older published backend rejects that selector. N9 is outside the existing compact-GDN rollback contract. Those failures and slower arms remain in the local evidence.

## Source and package identity

- Current GitHub source revision: `bb1675d0364a6e92e3565d0e711c630d72d6f53b`.
- Complete current native source-input archive SHA256: `398fa0d234f16bbfc56ad2fe5e9217837deeaa535a694a0a68fe2f8f19e9dfeb`.
- Matching current masked-GDN modules come from Windows SM89 archive SHA256 `5fb57e1525c1bdbe9c6cf3f0f4d40a83bef37c764fe9922b73cb9d21eab8343b`.
- Linux native build completed591 targets, plus packed/GDN/QK16 companions. CUDA libraries and unchanged fallback components are retained from the published Linux SM89 package.
- The4090 profile requires the named24GB RTX4090 and is separately sealed. Generic SM89 keeps the down-weight cache off and remains unqualified.

The earlier Z840 memory diagnostic reported corrected DRAM read errors. A clean workload observation window does not resolve the platform's broader memory issue. Qualification observes raw machine-check events and stops on a new event.

Machine-readable screen results and the frozen candidate are in `receipts/r6-linux-sm89-4090/`. The public first full release remains unfinished until the required gates and final artifacts pass.

## Completed performance/capacity gate

Three warmups and five measured requests per prompt per seed, four fresh starts in baseline42/candidate42/candidate1234/baseline1234 order. All64 effective sampler observations matched the frozen protocol.

| Metric | Published R6 baseline | Current optimized R6 | Change |
|---|---:|---:|---:|
| Prose decode t/s | 99.61 | 101.67 | +2.07% |
| Code decode t/s | 160.65 | 162.88 | +1.39% |
| 10K prefill t/s | 1870.87 | 1938.47 | +3.61% |
| 77K prefill t/s | 1612.95 | 1673.40 | +3.75% |

Candidate42 completed4091 continuation tokens after77825 input tokens, total81916, with no truncation and the full prefix retained. Other arms completed256 long continuation tokens. These gates alone do not complete the certificate.

## Paired quality and remaining qualification

Both full150 cohorts used production medium reasoning, unseeded temperature0.7/top-p0.95/top-k20/min-p0.05 and up to3attempts, with identical raw case definitions. Baseline scored126/150first attempts and136/150within three; candidate128/150 and132/150. All five new within-three losses remain failures and are individually reviewed in `QUALITY-REVIEW.json`. The paired p-values(.7744first,.21875within-three) prove neither equivalence nor improvement. Guard closures were16baseline/14candidate; no CUDA errors were recorded. Known model refusal/discipline failures remain; no generally safe-agent or perfect-model claim.

Greedy128code/prose and10K-prefill tokens match across serial/spec/restart on the tested cases. Sampled64outputs repeat within each mode and across speculative restart, but sampled serial/spec token sequences differ and are not claimed identical. Native CPU OCR/color/shape passed the fixed VISION742fixture. Broader vision accuracy and system-wide RAM health are outside those checks.

## Earlier stability gate abort

At2026-10-10 09:59:57UTC the raw machine-check trace recordedCPU0/socket0/MCA bank9 status0x8c000047000800c0. KernelEDAC identified a corrected DRAM scrubbing error atCPU_SrcID#0_Ha#0_Chan#0_DIMM#0; the kernel retired another page and HardwareCorrupted increased4→8KiB. The watchdog stopped the soak at1703.22seconds after controller start, before completing the minimum30-minute gate.335request records are retained. The old running-status file is not a pass, and systemd successful cleanup is not successful qualification.

A libc general-protection-fault record followed during the abort/cleanup interval; its causal relation is unproven. No GPUXid was observed in that captured interval. This is an unresolved platform-memory condition and does not establish that every earlier notice had the same cause. At that point no retry-until-pass, shortened gate, hardware exception, certified package or stable release was created. Resumption required a hardware condition change and a fresh complete stability run. Exact physical silkscreened DIMM mapping is not established by the EDAC label alone.


## Qualification after the hardware change

Z840 rebooted at2026-10-10 14:07UTC. The new inventory reports48GB installed RAM and only the expectedRTX4090. Both EDAC controllers started atCE0/UE0, with HardwareCorrupted0KiB. The boot helper was updated for the remaining4090 and its300W limit verified; the removed3060 backend and proxy socket were disabled. This workload qualification does not identify the physical cause of the earlier memory events or certify system-wide RAM health.

The unchanged native files and sealed profile passed fresh serial/spec/restart greedy checks, per-mode seeded repeat checks, and the CPU OCR/color/shape fixture. Results match the pre-removal correctness oracle. Sampled serial/spec sequences remain different. The earlier paired150 quality review and balanced performance comparison apply to the same frozen runtime and are explicitly retained as measurements before the hardware change.

The new soak completed1925.20seconds,15full cycles,375requests and366matching repeat comparisons. Independent audit verified all depths, counts, token hashes, telemetry and a clean server exit0. Peak GPU temperature was64C and peak allocation17284MiB. The final current-boot health audit foundCE0/UE0,0KiB quarantined memory, zero raw machine-check events and no kernel GPUXid or process faults.

The archive `L0xRE-BeeLLama-Low-R6-sm89-linux-rtx4090-certified.tar.zst` is986319840bytes, SHA256 `5bb052aa07c6158d2ab71a80d77aa1c0f547f76ef2c71a5f161d49ddb83633c2`. Fresh extraction verified160inventory files and55native files. Target, draft and CPUvision model hashes were checked again. Only the named4090 profile is qualified; genericSM89 remains unqualified.

A separate persistent installation at `/home/sean/work/l0xre-r6-4090-20261010/package-certified` matches that extraction. Its actual launcher ran without the qualification bypass flag, matched the greedy oracle, completed a cold10K request and77825input+4091output tokens(total81916) without truncation, retainedCPU0-7 affinity and the expected libraries, passedCPU OCR, and exited0. The private raw-event watcher was then archived and removed; no GPU compute process remains.

The shutdown logs retain the existing CUDA_Host compute-buffer accounting warning(3.2041MiB versus0.8448MiB); it appears in both the pre-removal and fresh logs. Clean exit and zero CUDA-error observations do not imply zero warning lines.

Final scope, artifact and installation receipts are in `receipts/r6-linux-sm89-4090/CERTIFICATION-FINAL.json`. The full universal stable release has not been published.


## Later hardware event during the full-power comparison

The later Bench.sh comparison stopped on another corrected DRAM patrol-scrub event at October 10, 2026, 15:25:27 UTC. The raw trace and offline decoder identify socket 0, memory-controller channel 0, MCA bank 9, status `0x8c000047000800c0`, address `0x197cc000`. This occurred after returning to 300 W for the second matched seed; it does not establish that 450 W caused the event. Exported EDAC counters and HardwareCorrupted stayed at zero, while the raw observer caught the event. The physical DIMM or memory-path component remains unisolated after the RAM removal.

The earlier 32-minute and clean-package gates remain measured historical observations. The current host's continuing stability and further qualification are blocked. No vanilla/native inference or headroom profiling has run after the new event. The 4090 is idle at the user-requested 450 W factory limit.

Three complete pinned Bench.sh arms were preserved. Both 450 W seeds average 102.95 t/s prose and 165.59 t/s code, with 2,086.16 t/s cold 10K prefill and 1,813.92 t/s cold 77K prefill. The completed seed-42 pair gives +1.19% code and +7.91% 77K prefill at 450 W versus 300 W. The second 300 W arm was interrupted, so no complete balanced power comparison or maximum-speed claim is made.

The latest requested baseline is an official unmodified BeeLlama v0.4.7 package and the similarly sized LowGPU NoMTP IQ3XXXS model (9,570,663,040 bytes versus L0xRE E3's 8,619,127,680). The official package and model are prepared and hash-pinned locally. Ordinary native decoding, shared DFlash2 N7, and a same-LowGPU runtime control are planned at matched settings once the hardware condition is addressed. Partial audited results and the blocked plan are in `receipts/r6-linux-sm89-4090/PARTIAL-COMPARISON-20261010.json` and `VANILLA-LOWGPU-PLAN.json`.


## Full-power re-qualification on the repaired host — 2026-10-11

The Strata service on the RTX4090 was stopped and disabled (`strata-qwen-flash-next-4090`, `strata-cpu-performance`), freeing the card. Z840 rebooted 2026-10-11 17:27 PDT (boot `cba21c13-68c8-4b48-914f-25be95667c51`) with **78 GB RAM** and both the RTX4090 and the RTX3060 present; the RTX3060 L0xRE service remained running and idle throughout. The RTX4090 stayed at its **450 W** persistent limit, which is also this card's **maximum supported power limit** (`Max Power Limit 450 W`), so full power is the hardware ceiling.

Per the operator's instruction, no RAM/memory gate was applied to this run: exported EDAC counters on this boot read mc1 CE=39 / mc3 CE=13 corrected events with UE=0, and the raw machine-check watcher used on October 10 was removed from the controller. Corrected-ECC activity is therefore **disclosed, not gated**, and this run does not certify system-wide RAM health. GPU-side guards (CUDA error scan, 82 C guard, clean-exit check, affinity check, frozen-sampler attestation) remained active.

### Frozen Bench.sh decode/prefill arms at 450 W

Same frozen package (`inventory_sha256 7e98996c…9166c4`) and sealed `r6-sm89-4090` profile (`89499a2a…181b57`); three warmups and five measured requests per prompt per seed; 16 effective sampler observations matched the frozen protocol; greedy code/prose hashes matched the October 10 oracle exactly.

| Arm | Prose decode t/s | Code decode t/s | 77K prefill t/s | 10K prefill t/s | 2K prefill t/s |
|---|---:|---:|---:|---:|---:|
| Q45 seed42 | 105.33 | 173.25 | 1780.62 | 2034.04 | 1520.52 |
| Q46 seed1234 | 103.37 | 160.99 | 1780.88 | 2033.84 | 1533.35 |
| **Two-seed mean** | **104.35** | **167.12** | **1780.75** | **2033.94** | — |

Against the October 10 450 W observations (48 GB configuration): prose +1.36%, code +0.92%, 10K prefill −2.50%, 77K prefill −1.83%. The host memory population and the presence of the second GPU differ, so this is reported as an observation, not a paired power or hardware claim.

Cross-card anchors from the same frozen Bench.sh protocol: RTX 4070 Ti Windows R6 measured 115.76 t/s code, so the 4090 is **+44.4%** on code decode. The certified RTX 5090 SM120 package reports 229 t/s code, so the 4090 reaches ~73% of it. The RTX 5090 Strata figures quoted elsewhere are a different runtime and are not comparable to these R6 numbers.

### Optimization screen at 450 W (full-headroom check)

A repeat of the seed-42 arm established the noise floor at +0.17% prose / +0.28% code / +0.15% prefill. Every screened variant landed inside or below that floor:

| Screen | Prose | Code | 77K prefill | Verdict |
|---|---:|---:|---:|---|
| O10 repeat (noise floor) | 105.50 | 173.73 | 1783.25 | baseline |
| O11 threads 12 (CPU0-11) | 105.16 | 173.59 | 1788.27 | neutral |
| O12 ubatch 1024 | 105.36 | 172.59 | 1608.04 | rejected (−9.8% prefill) |
| O13 down_splits 24 | — | — | — | `ggml_abort`, outside the K3 cubin contract |
| O20 KVarN window chunk 32768 | 105.26 | 172.38 | 1792.60 | within noise |
| O21 KVarN window chunk 8192 | 105.50 | 174.19 | 1771.50 | within noise |
| O22 ESCHA_W2_I8_HEAD=1 | 105.52 | 173.20 | 1779.20 | within noise |

Historic rejections remain in force: SM89 Q4 MMQ is slower (100.56/165.21 at 300 W), draft depth 9 exits the runtime, threads 2/4 are slower, down_splits 16 aborts, and the INT8 down-weight cache is retained because it improved prefill. Under load the card held 2715–2730 MHz of a 3105 MHz maximum with SW Power Cap intermittently active and no HW slowdown, thermal slowdown or power-brake event — i.e. the profile is running at the platform's power ceiling.

Fast-path attestation for the certified arms: packed bridge and QK16 context gate mapped in the server process, 16/16 sampler observations matching the frozen protocol, CPU0-7 affinity and NUMA0 placement verified. This is process/library and request-level attestation, not a per-kernel trace.

**Conclusion:** the frozen `r6-sm89-4090` profile is the best configuration found on this host at full power; no profile change was adopted, so the sealed profile and the existing certified archive remain byte-identical. Machine-readable results: `receipts/r6-linux-sm89-4090/OPTIMIZATION-450W-20261011.json` and `receipts/r6-linux-sm89-4090/20261011-fullpower/`.

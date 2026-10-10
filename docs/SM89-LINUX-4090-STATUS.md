# R6 Linux RTX4090 optimization and qualification

The newest published Linux SM89 payload was installed on Z840, then updated from the current complete R6 native source inputs. The selected candidate preserves81920 context, KVarN4/4, one slot, DFlash2 Q4 N7, medium reasoning, eight CPU threads and NUMA0 at the host's persistent300W power limit. It enables the optional INT8 down-weight cache.

The candidate is **not certified yet**. Matched two-seed performance/capacity qualification passed; Paired150 quality and its failure review are complete. Scoped serial/spec/restart correctness and CPUvision passed. The30-minute mixed-depth stability gate aborted on a new corrected DRAM scrub event at2026-10-10 09:59:57UTC. Final certification, clean-package qualification and full publication remain incomplete. The3060 and its previous backend/proxy remain stopped.

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

## Stability gate abort

At2026-10-10 09:59:57UTC the raw machine-check trace recordedCPU0/socket0/MCA bank9 status0x8c000047000800c0. KernelEDAC identified a corrected DRAM scrubbing error atCPU_SrcID#0_Ha#0_Chan#0_DIMM#0; the kernel retired another page and HardwareCorrupted increased4→8KiB. The watchdog stopped the soak at1703.22seconds after controller start, before completing the minimum30-minute gate.335request records are retained. The old running-status file is not a pass, and systemd successful cleanup is not successful qualification.

A libc general-protection-fault record followed during the abort/cleanup interval; its causal relation is unproven. No GPUXid was observed in that captured interval. This is an unresolved platform-memory condition and does not establish that every earlier notice had the same cause. No retry-until-pass, shortened gate, hardware exception, certified package or stable release was created. Memory-path maintenance/isolation and a fresh complete stability run are required. Exact physical silkscreened DIMM mapping is not established by the EDAC label alone.

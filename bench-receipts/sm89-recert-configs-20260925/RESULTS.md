# SM89 certified-config recert on v0.4.7 + ported kernels (2026-09-25)

Lead model: escha-e3-firstclass-v2 (non-MTP, L0xRE-27b-Low). Runtime: build-sm89
@ 492b1ee (champion kernels ported). club-3090 bench.sh, ONLY=code, n=5, exclusive GPU.

| Config | Drafter | Ctx | decode_TPS | wall | TTFT | VRAM |
|---|---|---|---|---|---|---|
| 12 GiB (b1024 ub64 N3 kvarn3/2, draftKV q4_0) | DFlash2 Q4_K_M | 80K | 115.4 ± 2.4 | 109.7 | 289ms | 11.7 GiB |
| 16 GiB (b1024 ub512 N3 kvarn3/3, draftKV q2_0, window16k) | DFlash2 Q2_K | 256K | 107.4 ± 3.5 | 103.3 | 295ms | 13.9 GiB |

Pre-port rc046-era baselines: 12GB ~78.6, 16GB ~78.2 (club-decode-r5).
Port yields +47% / +37% on the certified geometries.

# L0xRE v0.1.3

v0.1.3: faster decode and much faster prompt reuse (TTFT, follow-up turns, cached system prompts), and L0xRE now loads on 12 GB cards that also drive a display: a GPU with a display uses the 2048 MiB desktop arena; --arena-mib 1920 or 1856 frees more VRAM for other apps.

Speed is measured against l0xre-v0.1.2 on the same GPU in the same session (alternating runs). Every metric must also stay within two percent of the best speed measured for this release on that GPU. TTFT values are milliseconds (lower is better); the others are tokens/s.

| Target | Metric | v0.1.2 | v0.1.3 | Change | Best measured |
|---|---|---:|---:|---:|---:|
| linux/sm86/12gb | cached_prompt_ttft_ms | 908.88 | 749.40 | -17.55% | 749.40 |
| linux/sm86/12gb | code_decode | 59.24 | 71.64 | +20.93% | 71.64 |
| linux/sm86/12gb | depth_10k_decode | 34.53 | 41.06 | +18.90% | 41.06 |
| linux/sm86/12gb | depth_32k_decode | 30.44 | 35.44 | +16.43% | 35.44 |
| linux/sm86/12gb | followup_ttft_ms | 1579.55 | 872.69 | -44.75% | 872.69 |
| linux/sm86/12gb | narrative_decode | 34.09 | 39.98 | +17.28% | 39.98 |
| linux/sm86/12gb | prefill_10k | 686.67 | 685.63 | -0.15% | 685.63 |
| linux/sm86/12gb | short_ttft_ms | 980.06 | 493.80 | -49.61% | 493.80 |
| linux/sm89/12gb | cached_prompt_ttft_ms | 689.30 | 520.53 | -24.48% | 520.53 |
| linux/sm89/12gb | code_decode | 153.55 | 152.21 | -0.87% | 152.21 |
| linux/sm89/12gb | depth_10k_decode | 79.43 | 79.82 | +0.48% | 79.82 |
| linux/sm89/12gb | depth_32k_decode | 74.37 | 74.08 | -0.39% | 74.08 |
| linux/sm89/12gb | followup_ttft_ms | 999.66 | 548.27 | -45.15% | 548.27 |
| linux/sm89/12gb | narrative_decode | 88.93 | 88.97 | +0.05% | 88.97 |
| linux/sm89/12gb | prefill_10k | 2352.53 | 2350.18 | -0.10% | 2350.18 |
| linux/sm89/12gb | short_ttft_ms | 549.03 | 293.41 | -46.56% | 293.41 |
| linux/sm89/16gb | cached_prompt_ttft_ms | 666.95 | 553.38 | -17.03% | 553.38 |
| linux/sm89/16gb | code_decode | 152.50 | 150.39 | -1.38% | 150.39 |
| linux/sm89/16gb | depth_10k_decode | 79.77 | 80.44 | +0.85% | 80.44 |
| linux/sm89/16gb | depth_32k_decode | 71.52 | 72.66 | +1.59% | 72.66 |
| linux/sm89/16gb | followup_ttft_ms | 973.13 | 580.45 | -40.35% | 580.45 |
| linux/sm89/16gb | narrative_decode | 85.70 | 87.55 | +2.16% | 87.55 |
| linux/sm89/16gb | prefill_10k | 2342.73 | 2300.79 | -1.79% | 2300.79 |
| linux/sm89/16gb | short_ttft_ms | 546.53 | 328.09 | -39.97% | 328.09 |
| windows/sm120/12gb | cached_prompt_ttft_ms | 336.89 | 253.46 | -24.77% | 253.46 |
| windows/sm120/12gb | code_decode | 195.94 | 208.93 | +6.63% | 208.93 |
| windows/sm120/12gb | depth_10k_decode | 102.65 | 111.33 | +8.45% | 111.33 |
| windows/sm120/12gb | depth_32k_decode | 96.83 | 100.71 | +4.01% | 100.71 |
| windows/sm120/12gb | followup_ttft_ms | 459.47 | 264.25 | -42.49% | 264.25 |
| windows/sm120/12gb | narrative_decode | 113.26 | 124.22 | +9.68% | 124.22 |
| windows/sm120/12gb | prefill_10k | 3430.97 | 3425.48 | -0.16% | 3425.48 |
| windows/sm120/12gb | short_ttft_ms | 287.46 | 159.32 | -44.58% | 159.32 |
| windows/sm120/16gb | cached_prompt_ttft_ms | 364.91 | 251.56 | -31.06% | 251.56 |
| windows/sm120/16gb | code_decode | 194.27 | 219.09 | +12.77% | 219.09 |
| windows/sm120/16gb | depth_10k_decode | 97.29 | 109.80 | +12.87% | 109.80 |
| windows/sm120/16gb | depth_32k_decode | 89.74 | 98.90 | +10.21% | 98.90 |
| windows/sm120/16gb | followup_ttft_ms | 484.66 | 270.54 | -44.18% | 270.54 |
| windows/sm120/16gb | narrative_decode | 113.46 | 121.79 | +7.34% | 121.79 |
| windows/sm120/16gb | prefill_10k | 3382.22 | 3395.35 | +0.39% | 3395.35 |
| windows/sm120/16gb | short_ttft_ms | 294.66 | 203.83 | -30.83% | 203.83 |
| windows/sm86/12gb | Build only; no hardware speed claim | — | — | — | — |
| windows/sm89/12gb | Build only; no hardware speed claim | — | — | — | — |
| windows/sm89/16gb | Build only; no hardware speed claim | — | — | — | — |
| wsl/sm120/12gb | cached_prompt_ttft_ms | 677.75 | 413.28 | -39.02% | 413.28 |
| wsl/sm120/12gb | code_decode | 182.21 | 198.66 | +9.03% | 198.66 |
| wsl/sm120/12gb | depth_10k_decode | 91.04 | 95.72 | +5.14% | 95.72 |
| wsl/sm120/12gb | depth_32k_decode | 85.02 | 91.05 | +7.09% | 91.05 |
| wsl/sm120/12gb | followup_ttft_ms | 937.33 | 399.42 | -57.39% | 399.42 |
| wsl/sm120/12gb | narrative_decode | 104.25 | 114.61 | +9.94% | 114.61 |
| wsl/sm120/12gb | prefill_10k | 3244.66 | 3230.36 | -0.44% | 3230.36 |
| wsl/sm120/12gb | short_ttft_ms | 744.18 | 376.39 | -49.42% | 376.39 |
| wsl/sm120/16gb | cached_prompt_ttft_ms | 633.47 | 407.40 | -35.69% | 407.40 |
| wsl/sm120/16gb | code_decode | 186.13 | 197.03 | +5.86% | 197.03 |
| wsl/sm120/16gb | depth_10k_decode | 93.79 | 98.29 | +4.80% | 98.29 |
| wsl/sm120/16gb | depth_32k_decode | 84.38 | 90.37 | +7.10% | 90.37 |
| wsl/sm120/16gb | followup_ttft_ms | 875.27 | 386.11 | -55.89% | 386.11 |
| wsl/sm120/16gb | narrative_decode | 109.45 | 113.28 | +3.50% | 113.28 |
| wsl/sm120/16gb | prefill_10k | 3330.64 | 3249.78 | -2.43% | 3249.78 |
| wsl/sm120/16gb | short_ttft_ms | 703.45 | 385.15 | -45.25% | 385.15 |

Quality (Club 8-pack, 150 cases, temperature 0):

| Target | Mode | Previous | v0.1.3 |
|---|---|---:|---:|
| linux/sm86/12gb | mirai | 119/150 | 116/150 |
| linux/sm86/12gb | off | 120/150 | 120/150 |
| linux/sm89/12gb | mirai | 117/150 | 117/150 |
| linux/sm89/12gb | off | 120/150 | 120/150 |
| linux/sm89/16gb | mirai | 118/150 | 117/150 |
| linux/sm89/16gb | off | 120/150 | 120/150 |
| windows/sm120/12gb | mirai | 119/150 | 119/150 |
| windows/sm120/12gb | off | 120/150 | 120/150 |
| windows/sm120/16gb | mirai | 118/150 | 117/150 |
| windows/sm120/16gb | off | 119/150 | 119/150 |
| wsl/sm120/12gb | mirai | 118/150 | 119/150 |
| wsl/sm120/12gb | off | 119/150 | 119/150 |
| wsl/sm120/16gb | mirai | 117/150 | 120/150 |
| wsl/sm120/16gb | off | 120/150 | 119/150 |

Owner-accepted exception: wsl/sm120/16gb prefill_10k -2.43% versus l0xre-v0.1.2 (1. accept).

Speed gains that differ from the reference target, with the owner ruling:
- linux/sm89/12gb code_decode: lets keep it, 4090 isn't the target card its a 4070ti or similar 12gb sm89 so i will test 1.2 and 1.3 on that machine on windows myself. if all the code reaches the build and no regressions are present lets continue the release
- linux/sm89/12gb depth_10k_decode: lets keep it, 4090 isn't the target card its a 4070ti or similar 12gb sm89 so i will test 1.2 and 1.3 on that machine on windows myself. if all the code reaches the build and no regressions are present lets continue the release
- linux/sm89/12gb depth_32k_decode: lets keep it, 4090 isn't the target card its a 4070ti or similar 12gb sm89 so i will test 1.2 and 1.3 on that machine on windows myself. if all the code reaches the build and no regressions are present lets continue the release
- linux/sm89/12gb narrative_decode: lets keep it, 4090 isn't the target card its a 4070ti or similar 12gb sm89 so i will test 1.2 and 1.3 on that machine on windows myself. if all the code reaches the build and no regressions are present lets continue the release
- linux/sm89/16gb code_decode: lets keep it, 4090 isn't the target card its a 4070ti or similar 12gb sm89 so i will test 1.2 and 1.3 on that machine on windows myself. if all the code reaches the build and no regressions are present lets continue the release
- linux/sm89/16gb depth_10k_decode: lets keep it, 4090 isn't the target card its a 4070ti or similar 12gb sm89 so i will test 1.2 and 1.3 on that machine on windows myself. if all the code reaches the build and no regressions are present lets continue the release
- linux/sm89/16gb depth_32k_decode: lets keep it, 4090 isn't the target card its a 4070ti or similar 12gb sm89 so i will test 1.2 and 1.3 on that machine on windows myself. if all the code reaches the build and no regressions are present lets continue the release
- linux/sm89/16gb narrative_decode: lets keep it, 4090 isn't the target card its a 4070ti or similar 12gb sm89 so i will test 1.2 and 1.3 on that machine on windows myself. if all the code reaches the build and no regressions are present lets continue the release
- windows/sm120/12gb code_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- windows/sm120/12gb depth_10k_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- windows/sm120/12gb depth_32k_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- windows/sm120/12gb narrative_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- windows/sm120/16gb code_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- windows/sm120/16gb depth_10k_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- windows/sm120/16gb depth_32k_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- windows/sm120/16gb narrative_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- wsl/sm120/12gb code_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- wsl/sm120/12gb depth_10k_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- wsl/sm120/12gb depth_32k_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- wsl/sm120/12gb narrative_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- wsl/sm120/16gb code_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- wsl/sm120/16gb depth_10k_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- wsl/sm120/16gb depth_32k_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89
- wsl/sm120/16gb narrative_decode: nope the 5090 is also not the target card its a 5060 ti or 5070 that were targeting 12gb and 16gb cards so that gain is reasonable and better than zero on sm89

Known limitations:
- Decode speedups were developed on the RTX 3060 (+17-22%). Measured stand-ins: RTX 5090 +9-10%, RTX 4090 about 0% (decode not kernel-bound on large cards); TTFT and cache gains transfer everywhere.
- Windows SM86/SM89 packages are build-only in this release: their launcher, presets and references are fixed and verified by inspection and CPU tests, but no native Windows SM86/SM89 hardware run is claimed.
- Smaller arenas lower deep-context (about 128K) decode speed on fast cards; short decode is unchanged.
- The 128K selftest depth smoke with the Mirai control vector reports incomplete streamed token IDs on SM86; v0.1.2 behaves identically.
- Native Windows allocation/graph recovery is covered by the same-architecture WSL result, not a native fault injection.
- WSL SM120 16 GB: cold 10K-token prefill -2.43% versus v0.1.2 (two paired blocks), accepted; cached and follow-up turns are much faster on the same profile.

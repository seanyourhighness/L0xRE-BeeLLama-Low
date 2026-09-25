# SM89 champion-kernel port — bench receipts (2026-09-25)

Runtime: build-sm89 rebuilt from commit 107970f (champion decode kernels ported
from beellama-escha-rc046-sm89: lowgpu MROW/RT/ROWS + mmvf MTP-0071 small-N).
Bench: club-3090 scripts/bench.sh, ONLY=code, RUNS=5, WARMUPS=2, exclusive GPU.
Drafter: Qwen3.8-27B-DFlash2-Q4_K_M, draft-n-max 5. Profile env sourced
(escha-sprint-rc-sm89-fast).

| Model | decode_TPS mean | wall_TPS | TTFT | acceptance |
|---|---|---|---|---|
| escha-e3-with-mtp + DFlash2 | 142.97 ± 5.65 (max 150.8) | 136.25 | 277ms | 0.72, len 4.6 |
| escha-e3-firstclass-v2 (no-MTP) + DFlash2 | 139.52 ± 7.50 (max 149.9) | 130.46 | 292ms | 0.71, len 4.5 |

Pre-port baseline (same bench, same day): MTP 95.9, no-MTP 94.1.
Control (rc046-era binary, MTP): 141.9. Port restores parity; peak runs hit 150+.

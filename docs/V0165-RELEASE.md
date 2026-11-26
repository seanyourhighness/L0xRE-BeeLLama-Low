# L0xRE v0.1.6.5 (Sparse Defaults, Pooled Prefill & CLOCK Gather)

`L0xRE v0.1.6.5` updates `v0.1.6.4` across **Linux (`sm86`, `sm89`, `sm120`)** and **native Windows (`sm86`, `sm89`, `sm120`)** with the new sparse attention defaults, CLOCK unit-cache gather v3, short-append checkpoint retention, image-safe rollback, selective cache invalidation, and mask v2.

## What Changed Since v0.1.6.4

1. **New Sparse Attention Defaults & Gather v3 (CLOCK Cache)**
   - **Defaults:** `L0XRE_SPARSE_TILE=256`, `L0XRE_SPARSE_TOPK=16384`, `L0XRE_SPARSE_SINK=11648` (resident dense-phase threshold `28800` tokens; device cache unchanged at `32768`), `L0XRE_SPARSE_PREFILL_POOLED=1`, `L0XRE_SPARSE_GEN_ANCHOR=1` (`L0XRE_SPARSE_GEN_TILE=0`), and gather mode `3` (fully associative CLOCK unit cache over non-reserved slots).
   - **Lab quality (F1 + DFlash2 + s2 template):** KL vs dense on real Hermes trajectories dropped from `0.040` (RC5d release) to `0.0107` at 42–78K, `0.0140` at 128–170K, and `0.0263` at 180–200K; recall `72/72` (matching dense `72/72`).

2. **Stability, Vision & Checkpoint Fixes**
   - **`n - 4` short-append checkpoint retention:** Keeps the near-end checkpoint on short appends instead of restoring `n - 1024` every step, saving ~1,020 tokens per step in multi-turn agent loops.
   - **Image-safe sparse rollback:** Fixes sparse KV rollback and refreshes the near-end checkpoint after `mtmd` image chunks.
   - **Selective cache invalidation on cuts:** Keeps cached units below the cut instead of flushing the entire sparse unit cache.
   - **Mask v2:** Uses a template row per tile for the sparse attention mask.
   - **Generation tuning knobs:** Exposes `L0XRE_SPARSE_GEN_TOPK`, `L0XRE_SPARSE_GEN_SINK`, `L0XRE_SPARSE_GEN_WIDTH`, `L0XRE_SPARSE_GEN_WINDOW`, and `L0XRE_SPARSE_GEN_SINK_AUTO` as runtime switches (verified via `l0xre/v016/flags.py --check`).

3. **RTX 3060 12 GB Check & Speed-Floor Waiver**
   - **RTX 3060 12 GB (`sm86` Linux release kit, 262,144 sparse context + Q8_0 vision):** Booted cleanly at **11,175 MiB / 12,288 MiB** VRAM. `bench.sh` measured **55.73 tok/s** narrative decode (wall 54.94 tok/s, CV 0.1%), **100.17 tok/s** code decode (wall 96.52 tok/s, CV 0.0%), **839.41 tok/s** prefill at 10K (`76.64 tok/s` decode), and **687.86 tok/s** prefill at 90K (`78.46 tok/s` decode).
   - **Speed-floor waiver:** Cold prefill at 90K is −13% on the z840 host (`2,199` vs `2,535` t/s); 10K prefill and decode are unchanged, and the `n - 4` checkpoint fix saves ~1,020 tokens per agent step. Explicitly waived for v0.1.6.5.

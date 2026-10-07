# R6 C07 source

R6 restores the DFlash2 drafter to Q4_K_M. It adds compact FP32 recurrent history through native GGML/cache/checkpoint APIs, repairs hybrid history-input initialization, selects native Q4 matrix multiplication for six to eight draft rows on SM86, and uses one tensor-core operation per useful projection tile with a matching packed activation layout.

SM86 R6 passed the hardware performance and paired-quality gates. The frozen seed42 screen measured 37.27 t/s prose and 65.21 t/s code; the sustained seed42 arm measured 37.16/65.06. Balanced seeds42/1234 measured 34.95/62.69 versus R5's 29.21/54.09. Cold prefill measured 566.56 t/s at 10K and 469.39 at 77K, within 0.1% of R5. The 81,916-token capacity test and matching 256-token long-context greedy continuation passed. On the paired 150-scenario set, R6 scored 127/150 pass@1 and 132/150 pass@3 versus R5's 123/150 and 130/150; R6 had no runaway requests or CUDA errors, while R5 had one runaway and no CUDA errors. The paired differences are directionally positive but not statistically decisive, and the case-level gains and losses are retained in the evidence.

The projection change passed 56 shape/width comparisons over 38,191,104 output values, with poisoned destinations and checked redzones. Compact state and checkpoint tests passed 18 model-level bitwise comparisons. The original reasoning guard, target weights, one-slot 81,920-token context, KVarN4/4 and CPU vision remain part of the candidate profile.

Use the published R5 base source archive, SHA-256 `a9dd23bfa3ed38c040ffed6447e7085e083cb6abbb640a5b0cb3cf48f7de7263`:

```bash
python3 source/r6/prepare-source.py \
  --archive l0xre-fast-runtime-source-v0.4.7.tar.xz \
  --destination /path/to/r6-source

python3 source/r6/build-parity.py --arch sm86 --runtime \
  --source /path/to/r6-source --build /path/to/build-sm86 \
  --cuda /path/to/cuda --output /path/to/r6-sm86 --jobs 4
```

The prepared source requires the matching packed projection bridge and cubin together, with `L0XRE_PACKED_DECODE_INPUT=1`. The bridge must link the shared CUDA runtime. The QK16 long-context direct-attention gate is built for SM86, SM89 and SM120. An ordinary build or cross-compilation does not establish hardware qualification; each architecture needs its own performance, long-context, quality and package gates.

SM89 has a separate `overlay-sm89` dispatcher change for the six-to-eight-row Q4 MMQ path. It is opt-in through `L0XRE_SM89_Q4_MMQ=1` (the universal launcher exposes `--sm89-q4-mmq`) and defaults off until SM89 testing is complete. The shared SM86 overlay and frozen SM86 runtime payload are unchanged by the SM89 portability work. SM120 uses the widened QK16 gate, its native Q4 MMQ dispatcher, and a separate `overlay-sm120` GDN dispatch gate so the R6 masked-prefix optimization can be exercised on CC1200. SM120 and SM89 require separate target-hardware certification; neither qualifies through a cross-architecture build.

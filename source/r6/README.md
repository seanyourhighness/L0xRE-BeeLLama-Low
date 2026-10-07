# R6 C07 source

R6 restores the DFlash2 drafter to Q4_K_M. It adds compact FP32 recurrent history through native GGML/cache/checkpoint APIs, repairs hybrid history-input initialization, selects native Q4 matrix multiplication for six to eight draft rows on SM86, and uses one tensor-core operation per useful projection tile with a matching packed activation layout.

The frozen SM86 screen measured 37.27 t/s prose and 65.21 t/s code. Sustained seed42 qualification measured 37.16/65.06; balanced seeds42/1234 measured 34.95/62.69 versus R5's 29.21/54.09. Cold prefill measured 566.56 t/s at10K and469.39 at77K, within0.1% of R5. The81,916-token capacity test and matching256-token long-context greedy continuation passed. Full paired quality and clean package qualification remain in progress. SM120 hardware certification remains separate.

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

The prepared source requires the matching packed projection bridge and cubin together, with `L0XRE_PACKED_DECODE_INPUT=1`. The bridge must link the shared CUDA runtime. An ordinary build or cross-compilation does not establish hardware qualification. New architecture packages must pass their own performance, long-context, quality and package gates before serving through the R6 profile.

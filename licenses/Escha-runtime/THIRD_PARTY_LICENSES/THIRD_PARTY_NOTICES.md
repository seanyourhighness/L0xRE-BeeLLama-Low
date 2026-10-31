# Third-Party Notices — Escha Runtime (`escha` 1.2.0)

This repository distributes the **runtime** only (the `escha` wheel: quantization/serving
kernels + a modified SGLang fork). SGLang is the only engine in this repository. The quantized **model weights** are distributed
separately and carry their own license — see the model repository
(**`EschaLabs/Qwen3.8-27B-Escha-W2`**).

All bundled third-party code is under **permissive licenses (Apache-2.0, MIT, BSD-3-Clause)** —
an audit found **no copyleft (GPL / LGPL / AGPL / MPL / SSPL) code**. The obligations below are
satisfied by the files in this `THIRD_PARTY_LICENSES/` directory together with the copyright
notices retained verbatim in the redistributed source (inside the wheel). The same texts are also
embedded in the wheel metadata (`escha-1.2.0+qwen3dense.dist-info/licenses/LICENSE`).

The `escha` runtime itself is released under the Apache License, Version 2.0 — see the top-level
`LICENSE` file of this repository.

---

## 1. Acknowledgements — formats referenced, no code bundled

The compiled CUDA runtime (`escha._C`) is **original work**. Its kernels implement, as
independent (clean-room) reimplementations, numeric *formats* referenced from the two upstream
projects below. **No upstream source is copied into the binary** and neither project's licence
obliges us to reproduce it here — the attribution is a courtesy, and we would rather over-credit
than under-credit work we learned from.

| Upstream | Relationship | License | Text |
|---|---|---|---|
| **exllamav3** (Turboderp) | quantization codec *format* referenced by the escha decode kernels | MIT, © 2025 Turboderp | `exllamav3-LICENSE.txt` |
| **AQLM** (Vahe1994/AQLM) | additive-codebook *format* referenced by `escha_aqlm_*` kernels | Apache-2.0 | `AQLM-NOTICE.txt` (text in `SGLang-LICENSE.txt`) |

`bitblas` (Microsoft, MIT) is an **optional runtime import** only — not bundled or redistributed here.

**Also gratefully acknowledged — the serving framework this engine is built on.**
Unlike the formats above, its code is genuinely **redistributed** here, so it is covered by real
licence obligations rather than courtesy: **[SGLang](https://github.com/sgl-project/sglang)**,
bundled as a modified fork — see §2. This runtime would not exist in this form without it.

## 2. Bundled serving engine — `sglang/` (modified fork)

The wheel bundles a **modified fork of SGLang** as source.

- **License:** Apache-2.0 — © 2023–2026 SGLang Team. Full text: `SGLang-LICENSE.txt`.
- **Statement of changes (Apache-2.0 §4(b)):** the fork adds quantization-format support
  (the `escha` dense and `eschamoe` mixture-of-experts methods, plus the int8 dense-layer
  path) and kernel dispatch.
- **NOTICE (Apache-2.0 §4(d)):** upstream sgl-project/sglang publishes **no NOTICE file**
  (verified against the fork's pristine upstream-mirror at the merge base and across the full
  tree); therefore none is reproduced.

SGLang itself vendors code from other permissive projects; those files are redistributed
**unmodified with their copyright headers intact**, which preserves their notices:

| Upstream (as vendored in SGLang) | License | Reference text |
|---|---|---|
| vLLM project | Apache-2.0 | `SGLang-LICENSE.txt` |
| IST-DASLab / Marlin (Elias Frantar) | Apache-2.0 | `SGLang-LICENSE.txt` |
| EleutherAI + HuggingFace (transformers) | Apache-2.0 | `SGLang-LICENSE.txt` |
| NVIDIA (CUTLASS / kernels) | BSD-3-Clause / Apache-2.0 | `BSD-3-Clause.txt` |
| FlashAttention, Mamba (Tri Dao, Albert Gu) | BSD-3-Clause | `BSD-3-Clause.txt` |
| flash-linear-attention (Songlin Yang, Yu Zhang) | MIT | `MIT.txt` |
| Model configs (Qwen, DeepSeek, Moonshot, LG AI, Black Forest Labs, PaddlePaddle, Antgroup, …) | Apache-2.0 | `SGLang-LICENSE.txt` |

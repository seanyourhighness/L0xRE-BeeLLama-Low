# Escha v0.4.6 RC candidate

This branch ports the first-class Escha model support onto BeeLLaMA v0.4.6 and
keeps the retained CUDA paths behind narrow, same-binary rollback gates. It is
an RC candidate for qualification, not a production promotion.

## Source and build

- Base: BeeLLaMA v0.4.6 tag `78af8326522d94fb5fc24b60cfd6f26e29f12490`.
- RC branch: `escha-rc-v046`.
- RC tip: see `git rev-parse HEAD`; the remote 4090 qualification build is
  built from the same commit series.
- Target: Linux Z840, RTX 4090 (SM89), CUDA 12.8, Release, CUDA architecture
  89, `GGML_CUDA_FA=ON`, `GGML_CUDA_KVARN=ON`.
- Existing Escha W2/E3 GGUFs are used unchanged; no model reformat or
  quantization is part of this candidate.

## Retained paths

- The list below is functional grouping, not commit order: the R260 route and
  its safety closure landed before the later R258 state-view commit.
- R258 direct recurrent-state view: `ESCHA_DIRECT_RECURRENT_STATE_R258=1`.
  It is limited to one token, one sequence, one resident recurrent slot, and
  otherwise uses the normal gather path.
- R260 fused recurrent convolution: `ESCHA_DIRECT_RECURRENT_CONV_R260=1`.
  It requires the exact six-node cache-update topology and rejects a bias
  source.
- Decode alpha gate epilogue fusion is enabled by default; rollback is
  `ESCHA_NO_FUSE_ALPHA_GATE=1`.
- Residual `ADD -> RMS_NORM -> MUL` fusion is enabled by default; rollback is
  `ESCHA_NO_FUSE_ADD_RMS_NORM=1`.
- The retained warp finalizer and Escha projection paths remain shape- and
  layout-gated. Non-matching graphs fall back to the normal implementation.

## Qualification snapshot

Measured with `llama-bench`, F16 KV, FA on, `-b 2048 -ub 2048 -t 8`, separate
processes on the RTX 4090. R258/R260 were enabled for the candidate rows.

| Model | Decode d256 (tok/s) | Prefill p2048 (tok/s) |
|---|---:|---:|
| Escha W2 | 54.814 | 1835.843 |
| Escha E3 | 58.891 | 1714.163 |

The focused CUDA/Escha/KVarN suite is green: 12/12 tests passed, including the
runtime KVarN test. Same-binary CLI checks produced identical generated token
text for W2 and E3 with the R258/R260 gates enabled; only timing lines differed.
The W2 and E3 A/B rollback checks also produced identical token text with both
default-on fusions disabled (`ESCHA_NO_FUSE_ALPHA_GATE=1` and
`ESCHA_NO_FUSE_ADD_RMS_NORM=1`). The measured W2 decode rollback was 54.679
tok/s versus 54.814 tok/s with the fusions enabled; this is a correctness gate,
not a claimed standalone speed win.

The R204 K2 indexed-load cubin remains a lab artifact and is intentionally not
embedded in this source RC. The SM120 preview package below supplies the
official code-GEMM/F32-decode bridge replacement; bit-exact model-output parity
is still a release gate.

## SM120 preview bridge handoff

The CUDA-only Sprint bridge replacements are packaged outside the source tree
under `/home/sean/kernel-lab5090/preview-bridge-sm120` and
`/home/sean/kernel-lab5090/preview-bridge-sm89`. They contain the official
pretransformed code-GEMM cubins extracted from the retained Escha wheel, the
F32-input decode cubin entry points, and the seven-cubin GDN chunk bridge. The
matching wrapper libraries have no PyTorch, SGLang, or native Escha prefill
dependency.

Use the raw selector only; `ESCHA_OFFICIAL_BRIDGE` must remain unset:

```sh
export ESCHA_OFFICIAL_RAW_BRIDGE=1
export ESCHA_OFFICIAL_RAW_SWIGLU_FUSION=1
export ESCHA_OFFICIAL_RAW_DECODE_BRIDGE=1
export ESCHA_OFFICIAL_RAW_DECODE_F32_INPUT=1
export ESCHA_OFFICIAL_RAW_DECODE_SWIGLU_FUSION=1
export ESCHA_OFFICIAL_RAW_DECODE_K3_SPLIT_DIV2=0
export ESCHA_OFFICIAL_BRIDGE_LIBRARY=/home/sean/kernel-lab5090/preview-bridge-sm120/libescha_official_bridge_cuda_sm120.so
export ESCHA_OFFICIAL_CODE_GEMM_CUBIN=/home/sean/kernel-lab5090/preview-bridge-sm120/code-gemm-sm120.cubin
export ESCHA_OFFICIAL_F32_DECODE_CUBIN=/home/sean/kernel-lab5090/preview-bridge-sm120/f32_input.sm120.cubin
export ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32
export ESCHA_GDN_CHUNK_BRIDGE=1
export ESCHA_GDN_CHUNK_BRIDGE_LIBRARY=/home/sean/kernel-lab5090/preview-bridge-sm120/libescha_gdn_chunk_bridge_sm120_ported.so
export ESCHA_GDN_CHUNK_CUBIN_ROOT=/home/sean/kernel-lab5090/preview-bridge-sm120/gdn-cubins-sm120-exact
```

The direct-F32 decode profile passes the runtime's contiguous F32 activation
unchanged; use the `ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32` selector only with
`f32_input.sm120.cubin`. The parity-safe code-GEMM profile omits that selector
and keeps the wrapper's cached F32-to-F16 conversion. Native Escha GEMM prefill
remains disabled. The package manifests record SHA-256 hashes and the
corresponding SM89 selector paths. The exact remaining execution sequence is recorded in
[`escha-rc-v046-gate-runbook.md`](escha-rc-v046-gate-runbook.md).

The RC source also now dispatches `ESCHA_GDN_CHUNK_BRIDGE` from the fused GDN
CUDA operator for the qualified Escha shape (`S_v=128`, `H=48`, `2048` tokens,
one sequence, `K=1`, non-KDA). The port is fail-closed for all other shapes and
keeps the existing BeeLlama GDN kernel as the fallback. The GDN adapter accepts
BeeLlama's `[D,H,T,B]` row strides (`sq1=sv1=1`); its cubins remain separate
from native Escha GEMM prefill.

For one-token Escha decode, the RC now keeps Q/K in compact grouped-head form
and lets the fused GDN CUDA kernel perform the Sprint head expansion and
per-head L2 normalization. Set `ESCHA_NO_FUSE_GDN_QK_NORM_REPEAT=1` to retain
the older materialized-repeat control path while qualifying parity.

The opt-in `ESCHA_FUSE_GDN_DECODE_PREP_R262=1` lane also moves the scalar
alpha/beta preparation into the one-token CUDA GDN kernel. It is restricted to
one sequence with no rollback snapshots; leave it unset until a fixed-prompt
GPU A/B proves parity against the ordinary preparation graph.

## Release gates still open

1. Complete bit-exact model-output parity against the Sprint reference and
   attach the SM120/SM89 bridge artifact receipts; the current `PARITY_OK`
   control proves the route but is not a logits-equivalence certificate.
2. Run the full long-context, batch, rollback, and relocation matrix on this
   v0.4.6 binary.
3. Decide whether R258/R260 become defaults after the full matrix; omission is
   currently the safe rollback.
4. Run the final Hermes review against the exact commit, build manifest, test
   receipts, and benchmark receipts before tagging a public RC.

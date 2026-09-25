# Escha v0.4.6 RC 120 preview gate runbook

This runbook is for the clean RC binary in `build-sm120-preview`. It keeps
native Escha GEMM prefill disabled and selects the CUDA-only Sprint bridge.
Run each model in a fresh server process on the RTX 5090.

## Common bridge environment

```sh
unset ESCHA_OFFICIAL_BRIDGE
export ESCHA_OFFICIAL_RAW_BRIDGE=1
export ESCHA_OFFICIAL_RAW_SWIGLU_FUSION=1
export ESCHA_OFFICIAL_RAW_DECODE_BRIDGE=1
export ESCHA_OFFICIAL_RAW_DECODE_F32_INPUT=1
export ESCHA_OFFICIAL_RAW_DECODE_SWIGLU_FUSION=1
export ESCHA_OFFICIAL_RAW_DECODE_K3_SPLIT_DIV2=0
export ESCHA_OFFICIAL_BRIDGE_ACCUMULATION=mixed
export ESCHA_OFFICIAL_BRIDGE_LIBRARY=/home/sean/kernel-lab5090/preview-bridge-sm120/libescha_official_bridge_cuda_sm120.so
export ESCHA_OFFICIAL_CODE_GEMM_CUBIN=/home/sean/kernel-lab5090/preview-bridge-sm120/code-gemm-sm120.cubin
# The code-GEMM cubin is the parity-safe decode lane. The faster direct-F32
# lane uses f32_input.sm120.cubin plus the explicit ABI selector below.
export ESCHA_OFFICIAL_F32_DECODE_CUBIN=/home/sean/kernel-lab5090/preview-bridge-sm120/code-gemm-sm120.cubin
# export ESCHA_OFFICIAL_F32_DECODE_CUBIN=/home/sean/kernel-lab5090/preview-bridge-sm120/f32_input.sm120.cubin
# export ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32
export ESCHA_GDN_CHUNK_BRIDGE=1
export ESCHA_GDN_CHUNK_BRIDGE_LIBRARY=/home/sean/kernel-lab5090/preview-bridge-sm120/libescha_gdn_chunk_bridge_sm120_ported.so
export ESCHA_GDN_CHUNK_CUBIN_ROOT=/home/sean/kernel-lab5090/preview-bridge-sm120/gdn-cubins-sm120-exact

# Retained SM89 decode stack, guarded to M=1 decode shapes in source.  These
# selectors are no-ops for prefill and are the complete SM120 parity profile.
export ESCHA_DIRECT_RECURRENT_STATE_R258=1
export ESCHA_DIRECT_RECURRENT_CONV_R260=1
export ESCHA_FUSE_GDN_DECODE_PREP_R262=1
export ESCHA_FUSE_GDN_NORM_GATE_R264=1
export ESCHA_OFFICIAL_RAW_DECODE_QKV_Z_OVERLAP=1
export ESCHA_OFFICIAL_RAW_DECODE_ATTN_QKV_OVERLAP=1
```

The corresponding SM89 package substitutes the SM89 wrapper, cubin, and GDN
root. Do not set both official bridge selectors.

The two QKV overlap selectors are retained in the Preview profile only as a
decode performance experiment. They require the same fixed-prompt output and
graph-stability A/B receipt as the rest of the R-stack before promotion; they
are not evidence of Sprint-identical execution by themselves.

For the apples-to-apples Sprint oracle investigation, add only the INT8-head
selector below after the conservative smoke has passed. The two QKV overlap
selectors are already part of the safe SM120 performance profile above; the
INT8 head remains separate until fixed-prompt parity and logits checks qualify
it:

```sh
export ESCHA_W2_I8_HEAD=1
```

`ESCHA_W2_I8_HEAD=1` routes the W2 I8/F16-row-scale head through the explicit
CUDA LowGPU operator. Its direct kernel supports one output token only; any
actual batched head invocation is rejected rather than launching one vocabulary
GEMV grid per prompt token. Run p2048 with it unset, then use a fresh,
decode-only process for this experimental head.

Do not set `ESCHA_OFFICIAL_RAW_DECODE_GATE_UP_OVERLAP` for the current Qwen
W2 graph. CUDA graph scheduling visits multiple independent gate nodes before
their matching up nodes, while that prototype requires strict K2-gate/K3-up
adjacency and fails closed. `ESCHA_ALLOW_CUDA_GRAPHS` is not a runtime selector;
graphs are enabled by leaving `GGML_CUDA_DISABLE_GRAPHS` unset.

The retained decode-preparation fusion is already enabled in the performance
profile. If doing an A/B correctness qualification, isolate it with:

```sh
export ESCHA_FUSE_GDN_DECODE_PREP_R262=1
```

This is a one-token, one-sequence, no-rollback CUDA lane. Keep promotion of
the selector into an unconditional source default gated on its A/B output
parity receipt.

## SM120 decode receipt

The pre-fix F32-input receipt of **81.934606 +/- 1.067854 tok/s** is
superseded: its fixed control returned `PAR///////////////` because the
wrapper used the wrong activation ABI. With the corrected selector, the same
bounded command measures **82.37 +/- 2.01 tok/s** on W2 and **84.63 +/- 3.38
tok/s** on E3. Both fixed controls return `PARITY_OK`; top-20 logit probes are
20/20 identical at all four generated positions. The process exits 0 and the
GPU returns to 7 MiB used / 32,181 MiB free. This receipt excludes
`ESCHA_W2_I8_HEAD` and
`ESCHA_OFFICIAL_RAW_DECODE_GATE_UP_OVERLAP`.

The GDN selector is active in this RC source only for the exact fused-prefill
shape (`S_v=128`, `H=48`, `n_tokens=2048`, `n_seqs=1`, `K=1`, non-KDA). Any
other shape must remain on the native BeeLlama GDN fallback. Native Escha GEMM
prefill is not part of this route.

This is a performance smoke receipt, not a Preview release gate: functional and
MTP gates must precede promotion. A release receipt also records the source
manifest hash, CUDA/driver, graph state, power/clock state, Xid-clean pre/post
checks, and path-specific bridge-load evidence.

## Per-model functional gates

Start a target server with the exact command below (use `-c 4096` for the
bounded functional pass; use `-c 262144` only for the separately authorized
NIAH process), then:

```sh
build-sm120-preview/bin/llama-server -m /home/sean/kernel-lab5090/beellama-escha/escha-w2-firstclass.gguf \
  -c 4096 -np 1 -ub 512 -b 2048 -t 8 -ngl 99 -fa on -ctk kvarn3 -ctv kvarn2 \
  --jinja --host 127.0.0.1 --port 8090
```

The E3 functional pass substitutes `escha-e3-firstclass-v2.gguf`. The merged MTP
artifact is not assumed: `--spec-type draft-mtp --spec-draft-n-max 2` is run only
when the model metadata reports an embedded MTP head; otherwise the row is
`NOT_APPLICABLE`, not a failed MTP claim.

1. Request the deterministic control prompt: `Return exactly the string
   PARITY_OK and nothing else.` with temperature zero.
2. Send a long repeated prefix twice with a changed suffix; record
   `timings.cache_n`, `cache_reason`, and `cache_source` to prove KVarN reuse.
3. For the merged MTP artifact, use `--spec-type draft-mtp
   --spec-draft-n-max 2` and record `draft_n` and `draft_n_accepted`.
4. Compare fixed-prompt generated text/logits against the Sprint reference;
   the control string alone is not a bit-exact parity certificate.

## NIAH

Use a fresh nonce in every filler prompt so no prefix checkpoint is reusable.
Run at least 32k, 128k, and 260k target tokens at 10%, 50%, and 90% needle
depths. The needle is `The secret passphrase is: BLUE-FLAMINGO-42`; request
the passphrase only with `/no_think`. Record actual token count, depth,
prompt tok/s, answer, and pass/fail for every case.

## Throughput and quality

Run the canonical endpoint benchmark against each live model:

```sh
URL=http://127.0.0.1:8090 MODEL=escha-w2-sm120-rc CONTAINER=none \
  ENGINE_KIND=llamacpp RUNS=5 WARMUPS=3 ENABLE_THINKING=0 \
  bash /home/sean/club-3090/scripts/bench.sh
```

Repeat with the E3 model name. Then run the complete 8-pack, 150-scenario
behavioral suite twice per model with thinking forced off:

```sh
URL=http://127.0.0.1:8090 MODEL=escha-w2-sm120-rc CONTAINER=none \
  NO_THINKING=1 bash /home/sean/club-3090/scripts/quality-test.sh \
  --full --no-thinking --repeat 2 --progress \
  --save-json /home/sean/kernel-lab5090/quality-escha-w2-sm120-rc.json
```

Repeat with the E3 model name and a separate result path. A Preview release
requires both models to complete all functional, NIAH, benchmark, and quality
receipts; interrupted or partial runs remain open gates.

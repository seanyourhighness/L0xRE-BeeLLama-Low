# L0xRE v0.1.5: L0xRE-V2-27b-Low

v0.1.5 ships the new default model, **[L0xRE-V2-27b-Low](https://huggingface.co/YourHighnessLA/L0xRE-V2-27b-Low)**, and the small runtime change it needs. The file size (8.62 GB) and the speed are the same as v1, and every layer still runs on the GPU. The model is measurably closer to BF16 Qwen3.8-27B, especially on code, tool calls and agent traces.

## What changed

- **Model:** the catalog target is now `L0xRE-V2-27b-Low.gguf` (sha256 `8eb326fd…`). It is a refit with better calibration and a re-balanced mix of 2-bit and 3-bit codes. The DFlash2 drafter and the vision projector are unchanged.
- **Runtime:** the int8 bridge fast-paths four more projection shapes (roles 8–11), and fused FFN prefill accepts every K2/K3 gate/up pair. V2 uses them, so 57 projections that v0.1.4 would run on the slower FP16 path stay on the int8 path.
  - There are no kernel changes and no host-library changes. The v1 role table, the v1 fused path and its caller are unchanged.
  - Each preset sets `L0XRE_INT8_PROJ_MASK=4095` (it was 255).
  - Source: `L0xRE-v0.1.5-runtime.patch` applies to `L0xRE-v0.1.4-source.tar.gz` and changes 5 files: three bridge/backend sources plus the Windows backend and its export list.
- **Installers:** the start scripts take the model filename from the catalog. On update, V2 downloads next to the existing `L0xRE-27b-Low.gguf`, which is kept, so `--rollback` to v0.1.4 keeps working. Existing model files are never overwritten.

## V2 quality

KLD against BF16 Qwen3.8-27B on gen-masked positions of the public quant-challenge golden set; lower is better. The holdout is 200 documents frozen before fitting and scored once.

|Split|v1|V2|Change|
|---|---:|---:|---:|
|Holdout overall|0.1223|0.0928|−24.2% [−20.4, −26.9]|
|Holdout agentic mean (code, tools, reasoning, math)|0.0701|0.0497|−29.1%|
|Holdout code|0.0752|0.0499|−33.7%|
|Holdout agent and SWE traces|0.0596|0.0383|−35.8%|

club-3090 N150 on an RTX 3060:
- Thinking off: **121**. The previous production weights scored 119.
- Thinking on (medium effort, 8K reasoning budget), two runs: **127/133** and **125/135** pass@1/pass@3.

**Known limit:** a few thinking-off prompts that still carry the xhigh reasoning instruction (an empty think block under an "xhigh" system line) score worse on V2. The server's thinking-off mode does not render that prompt.

## Packages and coverage

|Package|Coverage|
|---|---|
|Linux sm86|The exact bridge and backend bytes that have served RTX 3060 production since 2026-11-15. V2 was qualified there: holdout, N150, `bench.sh` headline metrics within 2% of the previous weights, and a paired 90K depth decode (61.3 vs 50.2 tok/s).|
|Linux sm89|RTX 4090 smoke: V2 12 GB and 16 GB, and v1, all load with mask 4095 and 0 fallthroughs; tool call and code requests correct.|
|Linux sm120 / WSL2|RTX 5090 smoke: V2 12 GB and 16 GB, and v1, all load with mask 4095 and 0 fallthroughs; chat, tool call and code correct.|
|Windows sm120|Native Windows RTX 5090 smoke: V2 12 GB loads, mask 4095, all int8 roles 0–11 enqueued, tool call and code correct.|
|Windows sm86 / sm89|Build-verified: exports and archive integrity checked; no matching hardware run, as in v0.1.4.|

For every Linux target, an unpatched control rebuild produced code, GPU-payload and constant sections identical to the v0.1.4 libraries. Speed claims carry over from v0.1.4. Measured V2 speeds on the RTX 3060 match the previous weights.

**Not regenerated:** the `selftest` references still describe the v0.1.4 runtime with the v1 model. On v0.1.5, `selftest` reports that no shipped reference matches V2; it does not test V2. That doesn't affect serving.

# L0xRE v0.1.4 — P2 runtime and host timer correction

Release candidate: all six packages are staged and verified in the existing draft. Model weights are unchanged.

## Packages and coverage

|Package|Coverage|
|---|---|
|Linux sm86|Existing RTX 3060 deep/quality/recovery coverage carried; final host library is promoted on production with matching greedy outputs and a real Hermes tool/cache check.|
|Linux sm89|RTX 4090 smoke-only coverage, both preset deep checks carried; no full N150 claim.|
|Linux sm120 / WSL2|Final fixed library accepted in 12gb and 16gb envelopes on RTX 5090.|
|Windows sm86 / sm89|Build-verified, archive/import/load/installer lifecycle checks; no matching hardware claim.|
|Windows sm120|Final DLL accepted in native 12gb and 16gb envelopes on RTX 5090.|

The catalog routes all six targets to v0.1.4. The RTX 5090 results are recorded envelope tests; no separate bare-metal Linux sm120 throughput claim is made.

## First-token latency and the checkpoint pause

Follow-up and cached-prompt requests now return their first token sooner (roughly half the previous time on an RTX 5090). The prompt checkpoint that used to be saved before the first token is now saved just after it, leaving a short pause of about 0.1–0.15 s between the first and subsequent tokens. The timer correction does not change total request time; reported prompt time includes the save, as in v0.1.3.2.

|Lane / preset|Fresh short prompt (ms)|Cached prompt (ms)|Follow-up (ms)|Short-TTFT disposition|
|---|---|---|---|---|
|WSL 12gb|320.8 → 332.5|383.2 → 211.7|394.8 → 164.1|Owner waiver (short TTFT, v0.1.4)|
|WSL 16gb|323.2 → 327.7|363.0 → 199.1|386.0 → 159.5|Pass|
|Windows 12gb|150.8 → 166.7|235.7 → 160.3|250.3 → 144.7|Owner waiver (short TTFT, v0.1.4)|
|Windows 16gb|156.9 → 166.9|244.1 → 165.3|257.9 → 149.0|Owner waiver (short TTFT, v0.1.4)|

First-token time on a short fresh prompt measures about 10–16 ms slower on the waived RTX 50-series presets; cached and follow-up requests are substantially faster, as listed above. Short-prompt TTFT is planned for v0.1.5. The accepted bound is the larger of 5% or 20 ms at the median. Raw Fail/Unresolved results remain unchanged. WSL 16gb short TTFT passes the ordinary 2% floor.

Fresh 10K prompt processing on WSL 12gb measures 1.72% slower than v0.1.3.2 and passes outright; further prefill work is planned for v0.1.5. No preset needed the separate prefill waiver.

## Final headline changes

Medians with 95% bootstrap intervals against published v0.1.3.2. Rates are higher-is-better; TTFT is lower-is-better. WSL 12gb used four ABBA blocks; the other presets used two. All samples and A/A warnings are retained. Every non-waived headline median satisfies the 2% floor.

|Metric|WSL 12gb|WSL 16gb|Windows 12gb|Windows 16gb|
|---|---|---|---|---|
|narrative_decode|+0.35% [-1.54, +1.88]|+0.99% [-1.27, +4.37]|-0.45% [-5.19, +2.18]|-1.10% [-3.48, +2.31]|
|code_decode|+1.10% [-1.01, +2.55]|+2.43% [-1.08, +8.64]|-0.41% [-3.45, +3.02]|-1.67% [-4.84, +1.54]|
|prefill_10k|-1.72% [-2.67, -0.99]|+0.11% [-1.36, +1.52]|-0.19% [-0.46, +0.25]|-0.18% [-0.43, +0.33]|
|depth_10k_decode|+0.56% [-2.42, +4.01]|+1.62% [-3.88, +7.25]|+0.90% [-4.94, +6.30]|+1.34% [-5.01, +6.11]|
|depth_32k_decode|+0.52% [-4.94, +8.39]|+1.85% [-7.52, +10.88]|+0.66% [-7.07, +9.53]|+1.83% [-6.85, +11.82]|
|short_ttft_ms|+3.65% [-2.82, +11.73]|+1.39% [-4.07, +10.34]|+10.60% [+6.77, +16.90]|+6.34% [-4.28, +15.80]|
|cached_prompt_ttft_ms|-44.77% [-49.83, -38.18]|-45.14% [-51.17, -37.38]|-31.96% [-33.21, -30.40]|-32.26% [-33.96, -30.68]|
|followup_ttft_ms|-58.43% [-60.61, -56.73]|-58.68% [-60.20, -56.56]|-42.17% [-42.70, -41.51]|-42.25% [-44.31, -41.43]|

Depth diagnostics also report draft acceptance, cycles per 100 tokens, cycle cost, end-to-end time and aligned decode-start where available. V2 is report-only in v0.1.4 and becomes blocking work for v0.1.5; missing native clock alignment is not treated as a measured pass. No extra deep/recovery/cache/quality sweeps were run for this timer-only change.

## Build and source composition

Frozen base source: `e7c3a18defbfadb96a877d923e4b1c9c3157d530`. The source archive remains this base. Binary host libraries additionally include [L0xRE-v0.1.4-host-timer-fix.patch](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/l0xre-v0.1.4/L0xRE-v0.1.4-host-timer-fix.patch), SHA256 `fd0bed18fadd373bd8e5055c76e2806fd88fd2c1fb8585d0f8a28b605550d3c8`.

Apply the patch normally for the Linux host. The Windows host base is byte-identical and takes the same hunk:

```sh
git apply L0xRE-v0.1.4-host-timer-fix.patch
git apply -p3 --directory=windows-host/base L0xRE-v0.1.4-host-timer-fix.patch
```

Exactly one host object/library changes. CUDA, bridge, model weights, presets and launchers are unchanged from the pre-repack v0.1.4 assets. All six archives have full per-file proofs, import/symbol/load checks and real install/update/rollback checks. The Linux control rebuild is byte-identical. Windows control is not claimed byte-reproducible; import/export interfaces are identical and the final DLL passed both native preset gates.

## Recommended request settings

The server starts with thinking off. Request reasoning explicitly when wanted; the supported budget field is `reasoning_budget_tokens`. Existing preset bytes were preserved by the lib-only update. Recommendations belong in the client request:

```json
{"model":"L0xRE-27b-Low","messages":[{"role":"user","content":"Explain binary search."}],"temperature":0.6,"presence_penalty":0.5,"min_p":0.0,"reasoning_effort":"low","reasoning_budget_tokens":2048,"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}}
```

## Install and rollback

Use the existing installer/update command. The prior verified runtime remains available through `--rollback` on Linux/WSL or `-Rollback` on Windows. This stays a draft until Sean publishes and merges PR #23.

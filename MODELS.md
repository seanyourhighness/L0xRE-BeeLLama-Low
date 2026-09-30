# L0xRE-27b-Low model files

[Download the model](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low). The recommended setup uses these two files:

| File | Role | Bytes |
| --- | --- | ---: |
| `L0xRE-27b-Low.gguf` | Main 27-billion-parameter model | 8,619,127,680 |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | Optional drafter, recommended for token-generation speed | 1,143,006,816 |

The drafter is a helper file; it is not another name for the main model. The recommended pair is 9.76 GB of downloads, separate from the runtime.

Verify against [MODEL-SHA256SUMS](MODEL-SHA256SUMS) and follow [installation](README.md). A card with at least12GB GPU VRAM is the intended minimum; available context depends on the selected GPU/profile.

[L0xRE-27b-Low-MTP](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low-MTP) is a separate checkpoint with its own file hash and qualification. Historical measurements from that checkpoint do not establish performance of the current model path.

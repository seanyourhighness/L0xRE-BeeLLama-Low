# Runtime feedback

Report issues in [L0xRE-BeeLLama-Low](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues). Include the archive name, OS/version, GPU model and VRAM, NVIDIA driver version, selected profile, context length, model hashes, exact launch command, expected behavior, and relevant startup error/output.

First verify the archive against `CHECKSUMS.txt` and the extracted files:

```bash
# Linux / WSL, from the extracted package directory
sha256sum -c SHA256SUMS
nvidia-smi
```

```powershell
# Windows, from the extracted package directory
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1
nvidia-smi
```

Record any `CUDA_VISIBLE_DEVICES` or `L0XRE_ARCH` selection, especially on multi-GPU machines. `--help` shows the supported profiles. The server should become ready at `/health`; the default API base is `http://127.0.0.1:8080/v1`.

For a performance comparison, include input/output token counts, sampling settings, warmup and measured-run counts, acceptance statistics, memory headroom and the exact comparison model/runtime hashes. Separate prefill and decode measurements. The published SM86 numbers are deterministic, workload-specific receipts; Windows GPU performance and the new SM120 non-MTP Low common-CLI path remain unmeasured.

Share prompts or generated output only if you choose to disclose them. The historical SM120 diagnostic tool validates its original checkpoint hashes; use the universal verification steps above for the current model and Windows package.

To roll back, stop the server with Ctrl+C and run the earlier immutable release from its own directory with the unchanged model files. These installation instructions do not replace services or convert model weights.

# Replicate the measured Windows SM89 configuration

The configuration is ready to integrate into the full Windows launcher on another build machine. The native EXE, DLL and cubin files were unchanged during tuning. This commit supplies configuration and launcher source; producing a new full Windows archive is the next step on the build machine.

The authoritative settings are in [r6-windows-sm89-4070ti-profile.json](../tools/universal/r6-windows-sm89-4070ti-profile.json). Consume its `server_args` and every entry of `env` directly to avoid losing refresh-specific flags. The matching implementation is [r6-launch.ps1](../tools/universal/r6-launch.ps1); the existing `l0xre-r6.cmd` wrapper delegates to it.

## Tested host and payload

- Windows 11 Pro, Ryzen 5 5600X (6 cores / 12 logical processors), approximately 48 GB system RAM.
- RTX 4070 Ti, 12 GB, compute capability 8.9 (SM89), NVIDIA driver 616.64.
- R6 refresh: 0.4.7-dev, build 6, runtime revision `47708f863027c0f3162ad59b7a2c9d7f2018a758-dirty`, MSVC 19.44, CUDA 13.0.
- Original archive: `L0xRE-BeeLLama-Low-R6-sm89-windows-refresh-candidate.zip`; SHA256 `5fb57e1525c1bdbe9c6cf3f0f4d40a83bef37c764fe9922b73cb9d21eab8343b`.

`required_runtime_sha256` pins the measured native payload, and `model_sha256` pins the target, draft and optional vision projector. Rebuilding native code produces different bytes: retain new provenance and repeat qualification before attaching these measured results to that rebuilt payload.

## Settings to preserve

| Setting | Measured configuration |
| --- | --- |
| Target CPU threads / batch threads | `-t 2 -tb 6` |
| Draft CPU threads / batch threads | `--spec-draft-threads 2 --spec-draft-threads-batch 6` |
| Process affinity | All 12 tested logical processors: decimal `4095`, hexadecimal `0xFFF` |
| Context / slots | `-c 81920 -np 1` |
| GPU offload / Flash Attention | `-ngl 99 -fa on`; draft GPU layers `99` |
| Batch / microbatch | `-b 1024 -ub 512`; draft microbatch `32` |
| Target KV / tail | `-ctk kvarn4 -ctv kvarn4 --kv-tail-tokens 128` |
| Speculative decoding | `draft-dflash`, maximum draft length `7`, draft KV `q4_0 / q4_0` |
| Target embeddings | `-ot token_embd.lowgpu_.*=CPU` |
| Reasoning defaults | On, medium effort, budget `8192`, Jinja enabled |
| Sampling defaults | Temperature `0.7`, top_p `0.95`, top_k `20`, min_p `0.05` |
| Memory / fitting | `--no-op-offload --cache-ram 0 --fit off --fit-target 768` |
| Vision | CPU projector, image min/max tokens both `1024` |
| Listener | `127.0.0.1:8080` |

The complete argument vector and environment are in the JSON, including INT8 prefill, masked GDN, packed decode, head and bridge settings. They are part of the tested refresh configuration. Do not infer defaults for omitted flags from an older package.

## Full Windows package integration

1. Start from the pinned Windows SM89 refresh package. Preserve its `bin/` and `bridge/` trees and model identities.
2. Copy the updated `tools/universal/r6-launch.ps1` and new `tools/universal/r6-windows-sm89-4070ti-profile.json` into the full package. Keep the refresh package's existing generic `r6-windows-profile.json` unless deliberately regenerating its manifest too. The generic repository profile and refresh package profile can have different hashes.
3. Select `--profile r6-sm89-4070ti` in the full launcher. For a UI integration, route the SM89 measured preset to this profile. Require SM89 and select the actual GPU UUID with `nvidia-smi`; do not hard-code this host's UUID or drive letters.
4. Resolve `@PACKAGE_ROOT@` to the package root and `@ARCH_ROOT@` to `bridge/sm89`. Clear inherited `L0XRE_*`, `ESCHA_*`, and `GGML_*` values before applying the profile; prepend `bin` and `bridge` to PATH. Apply all JSON environment values, and set `CUDA_VISIBLE_DEVICES` to the selected GPU UUID. Use the packaged CUDA DLLs; machine-specific CUDA installation paths are not in the portable profile.
5. Apply affinity before creating the server process so it inherits the intended mask. The supplied launcher derives the all-logical-processors mask and restores its own original affinity afterward. On the tested 12-thread host this is `4095`. Other CPU counts need their own measurements; the supplied implementation supports up to 63 logical processors in one processor group.
6. Refresh `architectures/sm89/R6-MANIFEST.json` entries in `runtime_sha256` for the changed launcher and new profile, along with package checksum inventories. Keep `hardware_qualified=false`. Do not copy the measured native hash pins onto a different native build.
7. Check the sealed dry run before starting the rebuilt package. Then perform the pending consumer-launcher start/stop, quiet-prefill and remaining quality closeout checks before promoting hardware qualification.

```powershell
.\l0xre-r6.cmd serve --profile r6-sm89-4070ti --qualification-probe `
  -m models\L0xRE-27b-Low.gguf `
  -md models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf --dry-run
```

Remove `--dry-run` to start the server on the build/test machine. Add `--mmproj` with the pinned projector for vision. The package still requires `--qualification-probe` while hardware qualification is pending. Profile-controlled options are sealed by the launcher.

## Reproduce the measurements

For decode speed, use two fresh starts; each start has three warmups and five measured requests for each workload. Requests disable thinking and use temperature `0.6`, top_p `0.95`, top_k `20`, min_p `0`. The mean over both starts is 67.88 prose and 115.76 code tokens/s; wall throughput is 67.15 and 113.13 respectively. These request settings differ from the server's reasoning and sampling defaults.

Completed quality packs used medium reasoning, temperature `0.7`, top_p `0.95`, top_k `20`, min_p `0.05`, a per-case timeout of 300 seconds, and up to three attempts. The local 75-case runner requested a thinking maximum of 16,384 tokens; preserve request-level settings separately from the server's 8,192-token default when comparing runs. See the [status and completed-pack score table](SM89-WINDOWS-STATUS.md) and [machine-readable receipt](../receipts/r6-windows-sm89-4070ti-status.json).

The target speeds remain 75 prose / 125 code tokens/s, with 1,500 prefill tokens/s as a stretch target. L0xRE was shut down after these measurements; publishing this configuration did not restart inference.

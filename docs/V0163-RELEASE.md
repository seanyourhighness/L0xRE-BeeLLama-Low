# L0xRE v0.1.6.3 (On-Demand Vision + 12GB Desktop Headroom)

`L0xRE v0.1.6.3` updates `v0.1.6.2` on Linux and Windows to give 12 GB display-attached cards safe VRAM headroom while keeping `262,144` sparse context, embedded WebUI, and RC5d r2 decode speed.

## What Changed Since v0.1.6.2

1. **On-Demand Vision Enabled by Default (`L0XRE_VISION_ONDEMAND=1`)**
   - When `--mmproj` is passed in Sparse mode, projector weights (`331 MiB`) and the CLIP compute workspace (`248 MiB`) are now loaded **on demand only during image encoding**, then unloaded and trimmed immediately afterward — freeing **~582 MiB** of VRAM at rest and during text turns (`L0XRE_VISION_ONDEMAND=0` restores eager resident loading).
   - Verified on RTX 5090 with real image requests: startup drops from `11,888 MiB` to `11,306 MiB`, and VRAM returns cleanly to baseline after each image request.

2. **Fixed `--no-mmproj-offload` in Sparse Mode**
   - Fixed a leftover KV-streaming guard in `common_streamed_dflash_config_error()` (`common/common.cpp`) that wrongly rejected `--no-mmproj-offload` with `"on-demand DFlash vision requires GPU projector placement"` when KV streaming was disabled.
   - CPU projector encoding now works cleanly alongside DFlash2 with **0 MiB** GPU vision overhead even during image encoding.

3. **Windows 12 GB Display Fallback (`start.ps1` + `start-12gb-lowvram.ps1` / `-LowVram`)**
   - `start.ps1` now queries `nvidia-smi.exe` at startup; on `<= 13,000 MiB` GPUs with an active display (`display_active=Enabled`) or `< 11,600 MiB` free VRAM, it automatically selects `-ub 512` and `L0XRE_SPARSE_CACHE=16384` (keeping `-c 262144` and `8,192`-slot resident attention unchanged, and respecting any explicit user `-ub` or `L0XRE_SPARSE_CACHE` override).
   - Added explicit `.\start-12gb-lowvram.ps1` (and `-LowVram` on `start.ps1`) to force the balanced 12 GB desktop preset (`~10,706 MiB` load / `~11,068 MiB` active text, **~1.24 GiB** below v0.1.6.2 peak).
   - GPUs with >12 GB VRAM and headless 12 GB cards remain at `-ub 1024` and `L0XRE_SPARSE_CACHE=32768`.

4. **All v0.1.6.2 Embedded WebUI & RC5d r2 Speedups Preserved**
   - Built-in `llama-ui` (`LLAMA_UI_HAS_ASSETS=1`), `BT_HEAD8=staged-padded`, `BT_RADIX_TOPK=1` (`stable32`), `L0XRE_RECORD_COPY=1`, and `tool-verification-consistent.jinja`.

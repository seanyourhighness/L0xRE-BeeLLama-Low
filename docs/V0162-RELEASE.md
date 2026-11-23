# L0xRE v0.1.6.2 (RC5d r2 + Embedded WebUI)

`L0xRE v0.1.6.2` builds on `v0.1.6.1` (`RC5d r2`) by embedding the `llama-ui` web assets directly into `llama-server` (`LLAMA_UI_HAS_ASSETS=1`) and enabling the built-in WebUI by default on both Linux and Windows.

## What Changed Since v0.1.6.1

1. **Embedded `llama-ui` WebUI Assets (`LLAMA_UI_HAS_ASSETS=1`)**
   - Restored the complete `tools/ui/` configuration and static asset tree and prebuilt `tools/ui/dist/` so CMake embeds and gzips the full `llama-ui` bundle into `llama-server` on both Linux and Windows (`GET /` returns `200 OK` HTML instead of `404 Not Found`).
   - Verified out of the box with the bundled `tool-verification-consistent.jinja` chat template.

2. **WebUI Enabled by Default (`--no-webui` Removed from Default Profiles)**
   - Removed `--no-webui` from the default `sm86`, `sm89`, and `sm120` launcher profiles on Linux and Windows.
   - Pass `-- --no-webui` (Linux `launch.py`) or `-ExtraArgs --no-webui` (Windows `start.ps1`) if you want headless API-only serving.

3. **All RC5d r2 Runtime Speedups & Fixes Preserved**
   - Padded Staged Q3 Output Head (`BT_HEAD8=staged-padded`)
   - Stable 32-bit D-LSA Radix Top-K (`BT_RADIX_TOPK=1`, `BT_RADIX_TOPK_TUNE=stable32`) generalized across `sm86`, `sm89`, and `sm120`
   - Portable Indexed Record-Copy DFlash2 Proposal Hook (`L0XRE_RECORD_COPY=1`)
   - Bundled verification-consistent chat template (`tool-verification-consistent.jinja`, SHA-256 `8600019b72ecff2b7f39701e3772873fd265cb2b0908b850d0b5d7fbe17d90d0`)
   - Preserved Windows `l0xre_sparse_gather_k1` / `k2` gather patch

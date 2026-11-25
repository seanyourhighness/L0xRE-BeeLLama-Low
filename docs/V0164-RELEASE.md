# L0xRE v0.1.6.4 (Q8 Vision, Branded WebUI & Live Uncensored Toggle)

`L0xRE v0.1.6.4` upgrades `v0.1.6.3` across **Linux (`sm86`, `sm89`, `sm120`)** and **native Windows (`sm86`, `sm89`, `sm120`)** with full Q8_0 vision quality + VRAM reclaim, an agent-windowed template, dark-terminal WebUI branding, and one-click / one-command **Uncensored** steering.

[Chat template hotfix for v0.1.6.4](V0164-CHAT-TEMPLATE-HOTFIX.md)

## What Changed Since v0.1.6.3

1. **Q8_0 Vision Projector (`256..2048` tokens) + Lever 1+2 Vision VRAM Reclaim**
   - Upgraded default multimodal projector from `mmproj-Qwen3.8-27B-Q5_K-MIX.gguf` (`347 MB`) to `mmproj-Qwen3.8-27B-Q8_0.gguf` (`629 MB`) with `--image-min-tokens 256 --image-max-tokens 2048`.
   - Added runtime **Lever 1** (sparse KV buffer suspend/resume) + **Lever 2** (target compute scheduler suspend/resume) during on-demand image encoding (`L0XRE_VISION_RECLAIM=3` default), reclaiming ~1.15 GiB during encoding so full `Q8_0` at `2048` image tokens fits comfortably on 12 GB RTX 3060 cards at `262,144` sparse context.

2. **Windowed Agent Chat Template (`l0xre-agent-windowed.jinja`)**
   - Replaces `tool-verification-consistent.jinja` as the default bundled chat template to improve long multi-step tool-calling agent sessions and preserve clean reasoning boundaries.

3. **Dark-Terminal Branded WebUI + `/props` `l0xre` Telemetry**
   - Embedded WebUI updated with dark-terminal L0xRE branding, preset chips (`Coding / Agent`, `Fast Chat`, `Deep Reasoning`), and live `/props` telemetry status bar (`256K D-LSA SPARSE`, `DFLASH2`, `VISION`, `REASONING`).

4. **Uncensored Mode Launchers + Live WebUI On/Off Toggle (`GET/POST /l0xre/cvec`)**
   - Added **`UNCENSORED:ON` / `UNCENSORED:OFF`** clickable toggle pill in the WebUI status bar backed by `GET/POST /l0xre/cvec` — switches the loaded steering vector at zero reload cost by zeroing/restoring GPU direction tensors while keeping graph topology intact and flushing slot + RAM KV caches so steered and unsteered turns never mix.
   - Added `--uncensored` flag, Linux **`./l0xre-uncensored`**, and Windows **`start-uncensored.cmd`** / **`start-uncensored.ps1`** / **`l0xre-uncensored.cmd`**, pre-configured with `L0XRE_CVEC_MODE=project`, `--control-vector-layer-range 20 47`, `--control-vector-scale 1.0`, and auto-discovery of `Qwen3.8-27B-S-mirai-refusal-direction.gguf` in the model directory (or via `--control-vector` / `-ControlVector`).

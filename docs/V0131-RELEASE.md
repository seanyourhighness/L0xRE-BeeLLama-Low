# L0xRE v0.1.3.1

v0.1.3.1 is a launcher-only patch to v0.1.3. On a 12 GB card that also drives a display, L0xRE now defaults to the smallest validated arena, **1856 MiB**, instead of 2048 MiB. That leaves room for an agent, a browser and a messaging app such as WhatsApp or Telegram to stay open next to the model. This was confirmed on an RTX 4070 Ti desktop, where 1856 MiB is the setting that holds with all three running.

## What changed

| GPU | v0.1.3 | v0.1.3.1 |
|---|---|---|
| 12 GB, drives a display | 2048 MiB arena | **1856 MiB arena** |
| 12 GB, headless (no display, or display on an iGPU) | full arena | full arena (unchanged) |
| 16 GB | full arena | full arena (unchanged) |

- `--arena-mib 2048` or `--arena-mib 1920` still selects a larger validated arena if you prefer deep-context speed over headroom.
- Only `tools/universal/vram_preflight.py` changed, together with the package manifest and checksum list that record it. Binaries, presets, models and selftest references are identical to v0.1.3, and selftest references for 1856 MiB already shipped in v0.1.3.
- The speed results in the [v0.1.3 notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/l0xre-v0.1.3) still apply. They were measured headless at the full arena, which is unchanged.

## Cost of the smaller arena

Short-context decode does not change with the arena size. Measured deep-context (about 127K tokens) decode at 1856 MiB compared with the full arena:

- RTX 3060: -2.2%.
- RTX 5090: -4.4% to -10.5%.
- RTX 4090: -31%.

The 4090 and 5090 are stand-ins for the 12 GB target cards. They reserve memory in proportion to their SM count, so their cost is larger than on a real 12 GB card.

## Upgrade

Linux/WSL: run `install.py --update`. Windows: run `update.cmd` in your install folder; on an install made before `update.cmd` existed, download [`install.cmd`](https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.cmd) and run `.\install.cmd -Update -InstallDir <your install folder>`. No `-ExecutionPolicy` flag is needed. `--rollback` / `-Rollback` restores v0.1.3.

# L0xRE v0.1.3.2

v0.1.3.2 is a launcher-only patch to v0.1.3.1. When L0xRE feels slow, it is now easy to find out why, and whether GPU memory is **paging**.

## Why this matters

On Windows and WSL, the GPU's memory is shared with every other program: the desktop, browsers, games, video and AI apps. When they need more VRAM than is left, Windows quietly moves part of L0xRE's GPU memory into system RAM and back. Nothing fails; L0xRE just gets much slower. On an RTX 5090, decode fell from about 107 to 43 tok/s (WSL) and from about 115 to 48 tok/s (native Windows) while another program held most of the GPU's memory. Linux drivers never do this.

## What's new

**`doctor`, one command that says why it is slow.** Run it while L0xRE is slow:

- Windows: double-click `doctor.cmd` in the L0xRE runtime folder (the folder that holds `l0xre.cmd`).
- Linux / WSL: run `doctor` in the runtime folder, or `start.sh --doctor`.

It checks three things and answers in plain words:

```
[1] GPU memory paging: YES - the GPU is out of memory: up to 2.1 GB of L0xRE's GPU memory was in system RAM; Windows is copying GPU memory back and forth (13% of the copy engine).
    Other programs using GPU memory: other WSL programs 17.7 GB, Windows desktop (dwm) 0.6 GB, chrome 0.1 GB.
    Fix: close some of those programs (browsers, games, video and AI apps). Full speed comes back by itself once GPU memory is free.
[2] Another program using the GPU: NO.
[3] GPU health: OK (NVIDIA GeForce RTX 5090, 51 C, 2872 of 3090 MHz, 99% busy).

Result: PAGING - other programs took GPU memory, so Windows is moving L0xRE's memory into system RAM.
```

1. **Paging.** Uses Windows' own per-program GPU memory counters, which read the same on every Windows language.
2. **Another program using the GPU at the same time.** The GPU takes turns between programs, which can slow L0xRE as much as paging does.
3. **Heat or a hardware slowdown** reported by the NVIDIA driver.

**A warning in the server window.** On Windows and WSL the launcher now watches for paging in the background. If GPU memory pages while L0xRE is answering, the server window shows one clear warning, with the programs using GPU memory and the doctor's path. A second message says when full speed is back; no restart is needed once the memory is freed. It only warns: it never changes the arena, restarts the server, or touches a setting. It has no measurable speed cost (107.4 vs 107.0 tok/s with it off). `--no-gpu-watch` turns it off; `L0XRE_GPU_WATCH_DEBUG=1` prints one line per check for bug reports.

Two things that look like paging but are not, and do not trigger warnings:

- A few GB of "shared GPU memory" in Task Manager is L0xRE's normal pinned system memory.
- Windows parks the memory of an idle program. It comes back on the next request at full speed, so the watch only counts what happens while L0xRE is answering.

## What did not change

Binaries, presets, arenas, models and selftest references are identical to v0.1.3.1. Changed: `tools/universal/v01-launch.py` and `tools/universal/v01-windows-launch.py`. Added: `tools/universal/gpu_doctor.py` and `doctor` / `doctor.cmd`. The package manifest and checksum list record them. The speed results in the [v0.1.3 notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/l0xre-v0.1.3) still apply.

## Upgrade

Linux/WSL: run `install.py --update`. Windows: run `update.cmd` in your install folder; on an install made before `update.cmd` existed, download [`install.cmd`](https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.cmd) and run `install.cmd -Update -InstallDir <your install folder>`. No `-ExecutionPolicy` flag is needed. `--rollback` / `-Rollback` restores v0.1.3.1.

If `start` fails with "hardware qualification is pending; --qualification-probe applies only to a staged candidate", your start script was written by an older installer for a GPU model that is not on the measured list (for example an RTX 4070 Ti, 5060 Ti or 5070). Run the update above once; it rewrites the start script without that flag.

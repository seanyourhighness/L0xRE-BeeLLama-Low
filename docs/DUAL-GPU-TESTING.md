# Experimental dual GPU testing

**Experimental / untested / uncertified. No physical dual-card inference, correctness, performance, capacity, or stability results are available for this release.** The objective is to retain R6's quality and fast runtime behavior on two matched cards. Matching settings and compiling successfully do not establish parity or a speedup.

## What ships

Use two of the same RTX 30-, 40-, or 50-series GPU model with the same VRAM capacity and **at least 12 GB per card**. The installer rejects mixed models, duplicate devices, and smaller cards. Examples include 2× RTX 3060 12 GB, 3090 24 GB, 4060 Ti 16 GB, 4070 Ti 12 GB, 5060 Ti 16 GB, or 5090 32 GB. Two 8 GB cards are outside this release's target.

The selected single-card R6 runtime and profile supply the target model, Q4_K_M DFlash2 drafter, KVarN4/4 caches, N7 speculation, sampler, context, and other settings. The add-on gives the packed bridge and masked GDN bridge separate device state, including CUDA modules, scratch buffers, cuBLAS handles and stream ownership. Kernel arithmetic is unchanged. Certified single-card files remain intact.

The launcher appends BeeLlama's native `--split-mode layer --tensor-split 1,1 --device CUDA0,CUDA1 --spec-draft-device CUDA0`. Layer allocation is equal by weight; drafter and runtime overhead can make VRAM usage unequal. The default context is the selected single-card profile's context, not the sum of the cards' capacity. Small SM120 candidates retain their separate 32K profile; other selected R6 profiles use 80K.

This uses the **L0xRE-aware runtime** for the custom Low GGUF. BeeLlama's native split flags control placement; they do not imply that an unmodified upstream executable can load these weights. The experimental path retains the external fast bridges. Some retained backends restrict the long QK16 optimization to logical CUDA0, so CUDA1 can use the existing native attention fallback. Closest possible parity remains the goal; identical routes and performance are not promised.

[BeeLlama server documentation](https://github.com/Anbeeld/beellama.cpp/blob/main/tools/server/README.md) documents layer, row, and experimental tensor splitting. [Club 3090's PCIe/P2P notes](https://github.com/noonghunna/club-3090/blob/master/docs/PCIE_P2P.md) support layer splitting as a practical starting point without requiring NCCL. Those recipes use different model/runtime combinations and do not validate this L0xRE build.

## Install and preview

Follow the [guided installer setup](../install/README.md), reuse your model directory, and select both physical indexes or UUIDs:

```bash
bash install-l0xre.sh --gpus 0,1 --allow-candidate --models-dir /path/to/models --dry-run
bash install-l0xre.sh --gpus 0,1 --allow-candidate --models-dir /path/to/models
~/L0xRE/start.sh --dry-run
~/L0xRE/start.sh
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -Gpus 0,1 -AllowCandidate -ModelsDir 'D:\models' -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -Gpus 0,1 -AllowCandidate -ModelsDir 'D:\models'
& "$env:LOCALAPPDATA\L0xRE\start.ps1" -DryRun
& "$env:LOCALAPPDATA\L0xRE\start.ps1"
```

Use separate installation directories for single- and dual-card comparisons, sharing the model directory. Stop one server before running the other. Preserve the dry-run JSON, `INSTALLATION.json`, package/add-on hashes, and complete server logs. The initialization log should include `L0XRE_DUAL_FAST_INIT devices=2 experimental=1`; missing initialization or a fallback must be reported, not counted as successful fast-path parity.

## Measurements we need

- Exact cards and VRAM, OS, driver, CPU/RAM, PCIe link widths/generations, NVLink if present, release hashes, and unchanged launcher/profile settings.
- A single-card control and dual-card run with the **same target, drafter, context, sampler, prompts, seeds, warmups, repeats, and CPU affinity**. Report code and narrative decode, target prefill separately, startup time, and peak VRAM for each GPU. Include individual runs and spread, not only the best result. Do not combine different context/profile results.
- Cold greedy and seeded output checks, the established paired quality protocol, and all changed/failed cases. Native placement support alone does not establish model correctness. Use the original paired review rules rather than converting historical pass counts into new fixed floors.
- Staged context/capacity checks, a 30-minute soak, and a fresh restart. Include CPU vision if tested; preserve errors, OOMs, hangs, and route fallback logs.
- A clear verdict for each gate: pass, fail, or not tested. Certification stays pending until the full evidence is reviewed. Slower dual decode is a valid result; two cards do not guarantee twice the speed.

Offline compilation, architecture inspection, DLL/shared-library dependency checks, CPU ABI guard tests, archive integrity checks, and mocked installer/launcher tests are engineering checks only. They do not substitute for the physical pair tests above.

## Club 3090 post to copy

> **Help test L0xRE R6 on matched dual GPUs — experimental / untested / uncertified**
>
> We have single-card certifications on RTX 3060 Linux and RTX 5090 Linux/Windows. We are preparing RTX 4070 Ti Windows certification and need volunteers with matched pairs of RTX 30-, 40-, or 50-series cards, with at least 12 GB on each card.
>
> The R6 dual candidate uses BeeLlama's native layer split and a per-device version of L0xRE's fast bridges. It keeps the selected R6 profile, model, drafter and N7 settings. We want to measure correctness and quality parity, retained speed, and stability. We have not tested physical pairs and are not claiming a speedup or certification.
>
> Particularly useful: 2× 3060 12 GB, 3090, 4060 Ti 16 GB, 4070 Ti, 5060 Ti 16 GB, or other matched eligible pairs on Linux or Windows. An 8 GB pair is outside this release's scope.
>
> Setup, exact scope, comparison requirements and reporting checklist: https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/DUAL-GPU-TESTING.md
>
> Please share your hardware/driver details, archive hashes, single versus dual results with unchanged settings, per-GPU VRAM and full logs. Failures and regressions are as useful as successful runs.

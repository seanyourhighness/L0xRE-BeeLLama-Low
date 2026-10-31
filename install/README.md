# Guided L0xRE v0.1.1 install

The [public catalog](catalog.json) pins six platform/architecture archives and unchanged model hashes. Use this maintained catalog, not the frozen catalog-v0.1.1.json release asset, whose private URLs and workflow metadata are retained only as qualification provenance.

Linux SM86 has12GB/128K; Linux SM89 and Linux/WSL SM120 have12GB/128K and16GB/256K. Native Windows SM120 has both profiles. Windows SM86/SM89 are build-verified, hardware validation in progress; no native throughput/quality/VRAM claim is made. WSL was measured on RTX5090, not separately on bare-metal Linux. See the [qualification matrix](../README.md).

Linux/WSL requires Python3.12+, curl, zstd, numactl and a working NVIDIA driver on Ubuntu24.04 or compatible:

```bash
curl -fL --retry 3 -o install-l0xre.sh https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --dry-run --vision
bash install-l0xre.sh --vision
~/L0xRE/start.sh
```

Native Windows requires x64 Windows10/11, PowerShell and a working NVIDIA driver; runtime DLLs and Python are bundled:

```powershell
Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.ps1' -OutFile install-l0xre.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -DryRun -Vision
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -Vision
```

Run the start command printed by setup. Vision selects the pinned Q5 projector. Reuse models with --models-dir / -ModelsDir; choose a GPU with --gpu / -Gpu. The installer selects16gb where supported and detected memory is at least16000MiB, otherwise12gb. The launcher still performs actual-free-VRAM preflight and offers Retry, explicitly consented Start slower where a validated fitting rung exists, or Quit. No silent fallback is allowed. An envelope does not certify another GPU merely because its capacity matches.

Use --runtime-archive / -RuntimeArchive for an already downloaded exact catalog archive, --runtime-only / -RuntimeOnly to supply models later, and --catalog / -CatalogPath for an explicit local catalog. Setup checks archive/model hashes and extracted package checksums and prints a launcher; it does not start an inference service or alter drivers/clocks.

Stop the server before --update / -Update. Keep previous runtime, receipt, launcher and models. See [rollback instructions](../docs/V011-ROLLBACK.md), [release notes](../docs/V011-RELEASE.md), [source provenance](../docs/V011-PROVENANCE.md) and [model checksums](../MODEL-SHA256SUMS).

Sean publishes the public draft and marks it latest, then merges the metadata PR. After both steps, run `python3 tools/verify-public.py --verify-public --check-rollback` from the merged checkout. Publishing first makes the assets available before the live catalog points at them. Neither publication nor merge is performed by this documentation task.

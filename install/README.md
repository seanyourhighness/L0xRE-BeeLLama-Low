# Guided L0xRE install

The [public catalog](catalog.json) pins the current release's six platform/architecture archives and the model hashes.

v0.1.6 uses a 12 GB profile with 256K context on all supported cards. Windows SM120 has been tested on a native RTX 5090. Windows SM86 and SM89 are build-verified.

Linux/WSL requires Python3.12+, curl, zstd, numactl and a working NVIDIA driver on Ubuntu24.04 or compatible:

```bash
curl -fL --retry 3 -o install-l0xre.sh https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --dry-run --vision
bash install-l0xre.sh --vision
~/L0xRE/start.sh
```

Native Windows requires x64 Windows10/11, PowerShell and a working NVIDIA driver; the runtime DLLs are bundled:

```bat
curl.exe -LO https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.cmd
.\install.cmd -DryRun -Vision
.\install.cmd -Vision
```

`install.cmd` runs `install.ps1` for you, so PowerShell's script execution policy never blocks it. It takes the same options as `install.ps1`. Start the server with `start.cmd` in the install folder (or double-click it); update with `update.cmd`. Vision selects the pinned Q5 projector. Reuse models with --models-dir / -ModelsDir; choose a GPU with --gpu / -Gpu. Setup also verifies the pinned sparse map.

Use --runtime-archive / -RuntimeArchive for an already downloaded exact catalog archive, --runtime-only / -RuntimeOnly to supply models later, and --catalog / -CatalogPath for an explicit local catalog. Setup checks archive/model hashes and extracted package checksums and prints a launcher; it does not start an inference service or alter drivers/clocks.

Stop the server before --update / -Update. Keep previous runtime, receipt, launcher and models. See [rollback instructions](../docs/V011-ROLLBACK.md), [release notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases), [source provenance](../docs/V011-PROVENANCE.md) and [model checksums](../MODEL-SHA256SUMS).

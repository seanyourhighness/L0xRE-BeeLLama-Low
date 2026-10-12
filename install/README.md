# Guided L0xRE install

The Linux / WSL and native Windows installers share [one package and model catalog](catalog.json). They select an architecture from `nvidia-smi`, verify archive and model SHA-256 identities, verify the extracted package, and create a start script using the package's existing launcher and frozen R6 profile. Certified single-card profiles stay intact.

## Package selection

One R6 release contains the architecture builds; one installer per operating system downloads the selected build. Models are shared across all builds.

| GPU / platform | Installer choice | Status |
| --- | --- | --- |
| RTX 30 series, >=12 GB, Linux / WSL | SM86 R6, 80K | **Certified** on RTX 3060 |
| RTX 30 series, >=12 GB, Windows | SM86 R6, 80K | Starter |
| RTX 40 series, >=12 GB, Linux / WSL | SM89 R6, 80K | **Certified** on RTX 4090; starter on other cards |
| RTX 40 series, >=12 GB, Windows | SM89 CUDA 13.0 R6, 80K | Starter; measured 68/116 t/s on RTX 4070 Ti |
| RTX 5090, 32 GB, either platform | Platform-specific SM120 R6, 80K | **Certified** on one RTX 5090 |
| Other RTX 50 series, >=12 GB | SM120 32K profile, down-weight cache disabled | Starter |

Certified routes name the exact measured card; every other 12 GB-plus card runs the best-known starter configuration out of the box, clearly labeled as such in the installer output. Architecture compatibility does not transfer certification to another card or configuration. All profiles retain KVarN4/4 and DFlash2 N7. The 32K SM120 profile is a separate memory configuration; the certified 5090 profile remains unchanged. A profile target is not a measured capacity guarantee.

GPU selection defaults to the first `CUDA_VISIBLE_DEVICES` entry, or physical GPU 0. Use `--gpu` / `-Gpu` for an index or UUID; the start script pins the selected UUID. The certified Windows SM120 package is preserved byte for byte, with a separately hashed launcher copy adapting its physical-GPU selection. The packaged profile requires eight accessible logical CPUs, including CPUs 0–7.

## Linux / WSL

Use Ubuntu 24.04 or a compatible system meeting the [runtime requirements](../README.md#requirements), with Python 3.12+, a working NVIDIA driver, `curl`, `zstd`, and `numactl`. On Ubuntu / Debian, install missing dependencies with:

```bash
sudo apt-get install python3 curl zstd numactl
```

Download and preview, then install:

```bash
curl -fL --retry 3 -o install-l0xre.sh \
  https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --dry-run
bash install-l0xre.sh
```

With a repository checkout, use `bash install/install.sh` or `python3 install/install.py`. The default destination is `~/L0xRE`; model downloads go in `~/L0xRE/models`. Start after setup:

```bash
~/L0xRE/start.sh --dry-run
~/L0xRE/start.sh
```

Reuse models, choose a GPU, and include the optional Q8_0 CPU projector:

```bash
bash install-l0xre.sh --models-dir /path/to/models --gpu 1 --vision
```

An already downloaded archive can be supplied with `--runtime-archive PATH`; it must match the catalog's exact size and SHA-256.

## Native Windows

Use Windows PowerShell 5.1 or later on x64 Windows, with a working NVIDIA driver and `curl.exe`. Python and the CUDA compiler are not required. The CUDA/MSVC runtime DLLs are included in the package.

```powershell
Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.ps1' -OutFile install-l0xre.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1
```

With a repository checkout, invoke `install\install.ps1`. The default destination is `$env:LOCALAPPDATA\L0xRE`. Start after setup:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\L0xRE\start.ps1" -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\L0xRE\start.ps1"
```

Reuse models, choose a GPU, and include CPU vision:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -ModelsDir 'D:\models' -Gpu 1 -Vision
```

## Options

| Purpose | Linux | Windows |
| --- | --- | --- |
| Preview without writing or downloading artifacts | `--dry-run` | `-DryRun` |
| Choose installation directory | `--dir PATH` | `-InstallDir PATH` |
| Reuse an existing model directory | `--models-dir PATH` | `-ModelsDir PATH` |
| Select physical GPU index or UUID | `--gpu VALUE` | `-Gpu VALUE` |
| Update an existing installation | `--update` | `-Update` |
| Restore its previous verified installation | `--rollback` | `-Rollback` |
| Add the SHA-pinned CPU vision projector | `--vision` | `-Vision` |
| Use an already downloaded runtime archive | `--runtime-archive PATH` | `-RuntimeArchive PATH` |
| Install runtime only; supply models later | `--runtime-only` | `-RuntimeOnly` |
| Accept planned downloads without prompting | `--yes` | `-Yes` |
| Select localhost API port (default 8080) | `--port NUMBER` | `-Port NUMBER` |
| Use a local catalog | `--catalog PATH` | `-CatalogPath PATH` |

For an RTX 4070 Ti, reuse the existing model folder and preview the Windows starter:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -ModelsDir 'D:\models' -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -ModelsDir 'D:\models'
```

Installation verifies files and leaves the server stopped. Run the generated `start.ps1 -DryRun`, then start without that flag. An earlier r3/r4 run on the same card does not certify this R6 build.

## Downloads, reuse, and rollback

The target and drafter total about 9.8 GB; CPU vision adds about 0.6 GB. Setup also reserves 5 GB for the runtime download and extraction. Free space is checked on the installation and model drives. Model URLs are pinned to specific Hugging Face revisions, and archive hashes are pinned to specific releases.

Interrupted downloads retain a `.partial` file for `curl` to resume. Complete matching files are reused. A file with the wrong size or hash is rejected without overwriting it. If a failed download cannot resume, remove only its `.partial` file and rerun setup. Extracted package files are checked against `SHA256SUMS` before installation and on reuse. Extraction happens in a temporary directory, then the verified runtime moves into a directory named for its architecture and archive hash. ZIP/tar paths that escape the extraction root are rejected.

Setup creates `INSTALLATION.json` and `start.sh` / `start.ps1`, then prints the start command. It does not launch an inference server, install a service, require an administrator account, alter the NVIDIA driver, or change clock/power settings. On launch, the package performs its own runtime/model/profile checks. Stop the foreground server with Ctrl+C. The API defaults to `http://127.0.0.1:8080/v1`.

Stop the running server before updating. The generated updater downloads the current installer/catalog, preserves GPU selection, models, vision and port, and verifies the selected package. Matching runtime and model files are reused. A changed runtime, add-on, or launcher retains the previous start script and receipt.

```bash
~/L0xRE/update.sh --dry-run
~/L0xRE/update.sh
# Restore the previous verified runtime and launcher:
~/L0xRE/update.sh --rollback
```

```powershell
& "$env:LOCALAPPDATA\L0xRE\update.ps1" -DryRun
& "$env:LOCALAPPDATA\L0xRE\update.ps1"
& "$env:LOCALAPPDATA\L0xRE\update.ps1" -Rollback
```

The updater itself needs network access. For offline rollback, invoke an already downloaded installer with `--rollback --dir PATH` or `-Rollback -InstallDir PATH`. Rollback verifies the previous runtime, experimental add-on if used, and retained launcher files. It restores configuration but does not start a server. Old archives and model files remain available. Migration from r3/r4 uses a new guided installation pointing at the existing model directory.

## Validation

Offline tests exercise selection and starter routing, GPU visibility, checksum failures, complete-partial reuse, clean extraction, path traversal rejection, model paths with spaces/apostrophes, generated start scripts, and tampered-package rejection. The Windows suite runs in native Windows PowerShell 5.1. These checks include update preferences, rollback, and the Windows SM120 GPU adapter. They do not establish GPU correctness, capacity, performance or certification.

```bash
python3 -m unittest discover -s install -p 'test_install.py'
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\test-windows.ps1
```

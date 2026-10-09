# Guided L0xRE install

The Linux / WSL and native Windows installers share [one package and model catalog](catalog.json). They select an architecture from `nvidia-smi`, verify archive and model SHA-256 identities, verify the extracted package, and create a start script using the package's existing launcher and frozen R6 profile. They do not rebuild the runtime or change the packaged serving configuration.

## Package selection

| Selected GPU / platform | Installer choice |
| --- | --- |
| Linux / WSL, SM86 with 12 GB or more | Published certified SM86 R6 package; measured on RTX 3060 |
| Windows, RTX 5090 / SM120 with 32 GB | Published certified CUDA 13.0 SM120 package |
| Linux / WSL, RTX 5090 / SM120 with 32 GB | Locally certified package; supply `--runtime-archive` until it is published |
| Linux / Windows, RTX 4090 / SM89 with 24 GB | R6 prefill-port2 candidate, only with explicit candidate opt-in |
| Windows, SM86 with 12 GB or more | R6 prefill-port2 candidate, only with explicit candidate opt-in |

SM86 certification measurements were taken on the RTX 3060; selecting the same architecture on another card does not promise the same speed. This catalog conservatively limits SM89 setup to the RTX 4090 and SM120 setup to the RTX 5090. Smaller RTX 40/50-series cards do not automatically receive a profile tested on a larger card. Candidate opt-in does not certify that build or guarantee that it will fit.

GPU selection defaults to the first `CUDA_VISIBLE_DEVICES` entry, or physical GPU 0 when that variable is absent. Use `--gpu` / `-Gpu` to select another physical index or UUID. The generated start script pins that GPU UUID. The packaged profile requires eight accessible logical CPUs, including CPUs 0–7. Runtime binaries, model files, and profile defaults remain unchanged.

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

For the certified local Linux SM120 archive:

```bash
bash install-l0xre.sh --runtime-archive /path/to/L0xRE-SM120-certified-20261008.tar.zst
```

The supplied archive must match the catalog's exact SHA-256 and size. There is no fallback to an older unqualified SM120 download.

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
| Add the SHA-pinned CPU vision projector | `--vision` | `-Vision` |
| Explicitly allow an unqualified candidate | `--allow-candidate` | `-AllowCandidate` |
| Use an already downloaded runtime archive | `--runtime-archive PATH` | `-RuntimeArchive PATH` |
| Install runtime only; supply models later | `--runtime-only` | `-RuntimeOnly` |
| Accept planned downloads without prompting | `--yes` | `-Yes` |
| Select localhost API port (default 8080) | `--port NUMBER` | `-Port NUMBER` |
| Use a local catalog | `--catalog PATH` | `-CatalogPath PATH` |

For RTX 4090 candidate setup, preview with `--allow-candidate --dry-run` or `-AllowCandidate -DryRun`, then remove the dry-run flag to install. The generated candidate start script includes the package's qualification-probe switch, so its unqualified status is explicit. Remove or replace that installation after SM89 is independently certified.

## Downloads, reuse, and rollback

The target and drafter total about 9.8 GB; CPU vision adds about 0.6 GB. Setup also reserves 5 GB for the runtime download and extraction. Free space is checked on the installation and model drives. Model URLs are pinned to specific Hugging Face revisions, and archive hashes are pinned to specific releases.

Interrupted downloads retain a `.partial` file for `curl` to resume. Complete matching files are reused. A file with the wrong size or hash is rejected without overwriting it. If a failed download cannot resume, remove only its `.partial` file and rerun setup. Extracted package files are checked against `SHA256SUMS` before installation and on reuse. Extraction happens in a temporary directory, then the verified runtime moves into a directory named for its architecture and archive hash. ZIP/tar paths that escape the extraction root are rejected.

Setup creates `INSTALLATION.json` and `start.sh` / `start.ps1`, then prints the start command. It does not launch an inference server, install a service, require an administrator account, alter the NVIDIA driver, or change clock/power settings. On launch, the package performs its own runtime/model/profile checks. Stop the foreground server with Ctrl+C. The API defaults to `http://127.0.0.1:8080/v1`.

For another release, install into a fresh directory and reuse the verified models. Keep the earlier directory for rollback. When a new architecture is certified, its catalog entry can be updated without creating a separate installer workflow.

## Validation

Offline tests exercise selection and candidate gates, GPU visibility, checksum failures, complete-partial reuse, clean extraction, path traversal rejection, model paths with spaces/apostrophes, generated start scripts, and tampered-package rejection. The Windows suite runs in native Windows PowerShell 5.1. These are installer tests; they do not establish new GPU performance or certification.

```bash
python3 -m unittest discover -s install -p 'test_install.py'
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\test-windows.ps1
```

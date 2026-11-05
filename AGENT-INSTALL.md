# AGENT-INSTALL.md — install L0xRE for your user

You are an AI coding agent with shell access. Your user asked you to install **L0xRE v0.1.3**, a local 27B coding model with vision, 128K–256K context and speculative decoding, on this machine. This file tells you exactly how to do it. Follow the steps in order. Keep your user informed in short messages.

## Ground rules

1. **Always run the dry run first** and show the user the plan before downloading roughly 11 GB.
2. **Never use `sudo` or admin rights, install system packages, or touch NVIDIA drivers, clocks or power limits without asking first.** If a prerequisite is missing, tell the user the exact command and let them approve it.
3. **Never kill processes you did not start.** If another app is holding VRAM, name it and ask the user to close it.
4. **Never pick a slower memory mode on your own.** If the launcher offers *Retry / Start slower / Quit*, relay the choice to the user.
5. **Never skip or edit a checksum.** The installer verifies every file by SHA256. If a check fails, delete the partial file and retry the download once; if it fails again, stop and report.
6. Use only the URLs in this file and in [`install/catalog.json`](install/catalog.json). Do not substitute other model files or quantizations.

## Step 1 — Detect the platform and GPU

```bash
uname -a 2>/dev/null; echo "$OS"                       # Linux, WSL2 ("microsoft" in kernel) or Windows
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv,noheader
```

On Windows PowerShell, use the same `nvidia-smi` command, plus `$PSVersionTable.PSVersion`.

Map the result:

| `compute_cap` | GPU family | Package arch |
| --- | --- | --- |
| 8.6 | RTX 30 series | `sm86` |
| 8.9 | RTX 40 series | `sm89` |
| 12.0 | RTX 50 series | `sm120` |
| anything else | — | **unsupported: stop and tell the user** |

- `memory.total` must be **at least 12000 MiB**. Below that, stop: L0xRE does not support it.
- 16 GB+ cards get the 16 GB / 256K profile on SM89 and SM120. SM86 is always 12 GB / 128K.
- If `nvidia-smi` is missing or fails, stop. The user needs a working NVIDIA driver; on WSL2 that is the Windows driver, not a Linux one.
- Several GPUs: ask the user which one to use, then pass `--gpu <index>` / `-Gpu <index>` in later steps.
- Note `memory.used`. If it is more than about 1.5 GB, other apps are using the GPU. Warn the user now, because startup may need that memory.

## Step 2 — Check prerequisites

**Linux / WSL2:**

```bash
python3 --version          # needs 3.12 or newer
command -v curl zstd numactl
nproc                      # needs at least 4 cores
df -h ~                    # needs about 16 GB free
```

If something is missing, ask the user to run (Ubuntu/Debian): `sudo apt-get install -y curl zstd numactl python3`

**Windows 10/11 (native):** PowerShell 5.1+ is enough; Python and runtime DLLs are bundled. Check free space on the drive holding `%LOCALAPPDATA%` (about 16 GB).

**Already have the model files?** Search before downloading 11 GB again:

```bash
find ~ /mnt -maxdepth 5 \( -name 'L0xRE-27b-Low.gguf' -o -name 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf' -o -name 'mmproj-Qwen3.8-27B-Q5_K-MIX.gguf' \) 2>/dev/null
```

If all are in one folder, pass it with `--models-dir` / `-ModelsDir`. The installer re-verifies their hashes and reuses them.

## Step 3 — Download the installer and dry-run it

**Linux / WSL2:**

```bash
mkdir -p ~/l0xre-setup && cd ~/l0xre-setup
curl -fL --retry 3 -o install-l0xre.sh https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --dry-run --vision
```

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\l0xre-setup" | Out-Null; Set-Location "$env:USERPROFILE\l0xre-setup"
Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.ps1' -OutFile install-l0xre.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -DryRun -Vision
```

Always include `--vision` / `-Vision`: the v0.1.3 launcher requires the vision projector. Add `--gpu`, `--models-dir` or `--port` (default `8080`) if needed.

The dry run prints a JSON plan (Linux) or a summary (Windows). Summarize it for the user: GPU, package (for example `L0xRE-v0.1.3-linux-sm86.tar.zst`), memory profile (`12gb`/`16gb`), install folder, models folder, and whether the package is **certified** or **build-verified** on their card. Ask the user to confirm.

## Step 4 — Install

Same command without the dry-run flag. Add `--yes` / `-Yes` only after the user has confirmed the plan:

```bash
bash install-l0xre.sh --vision --yes
```
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1 -Vision -Yes
```

This downloads and verifies:

| File | Role | Size |
| --- | --- | ---: |
| `L0xRE-v0.1.3-<os>-<arch>` | Runtime package (server + GPU kernels) | 0.65–0.75 GB |
| `L0xRE-27b-Low.gguf` | Main model | 8.6 GB |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | Drafter for speculative decoding (speed) | 1.1 GB |
| `mmproj-Qwen3.8-27B-Q5_K-MIX.gguf` | Vision projector | 0.35 GB |

Downloads can take a while. Run it in the background if your harness supports that, and report progress. On success it prints a start command. Defaults: `~/L0xRE/start.sh` (Linux/WSL2) or `%LOCALAPPDATA%\L0xRE\start.ps1` (Windows).

## Step 5 — Check, then start the server

Check the launch plan first (it starts nothing):

```bash
~/L0xRE/start.sh --dry-run
```
```powershell
& "$env:LOCALAPPDATA\L0xRE\start.ps1" -DryRun
```

Then start it, keeping a log. It runs in the foreground, so use a background task, `nohup`, `tmux` or a separate terminal:

```bash
nohup ~/L0xRE/start.sh > ~/L0xRE/server.log 2>&1 &
```
```powershell
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$env:LOCALAPPDATA\L0xRE\start.ps1" -RedirectStandardOutput "$env:LOCALAPPDATA\L0xRE\server.log" -RedirectStandardError "$env:LOCALAPPDATA\L0xRE\server.err.log"
```

Loading takes about 1–3 minutes. Poll every 10 s, for up to 6 minutes:

```bash
curl -sf http://127.0.0.1:8080/v1/models
```

If the log shows a free-VRAM preflight refusal or a *Retry / Start slower / Quit* prompt, stop and tell the user how much memory is free versus needed, and which apps hold VRAM (`nvidia-smi`). Let them choose (see ground rule 4). On a 12 GB card that also drives the display, the 2048 MiB desktop arena is used automatically; `--arena-mib 1920` or `1856` frees more VRAM, at some deep-context speed.

## Step 6 — Smoke test

```bash
MODEL=$(curl -s http://127.0.0.1:8080/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python one-liner that reverses a string.\"}],\"max_tokens\":512}"
```

A sensible answer means it works. The model may think before answering, so give it enough `max_tokens`. Optional vision test: send an `image_url` content part with a small base64 PNG and ask what it shows.

## Step 7 — Connect the user's tools

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8080/v1` (OpenAI-compatible) |
| API key | any non-empty string (none is checked) |
| Model | the `id` from `/v1/models` |
| Context window | 128K (12 GB profile) or 256K (16 GB profile) |
| Vision | yes (OpenAI `image_url` content parts) |

Offer to set these in the user's agent or editor config. Many tools read `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Leave some context headroom for the reply: do not configure the agent to fill the whole window with prompt.

## Step 8 — Report back

Tell the user, briefly: GPU and profile, install and model folders, the start command, the API URL, smoke-test result, and anything they need to do (for example "close Chrome before starting to free VRAM").

**Optional — share a hardware report.** Results from cards we have not measured yet (RTX 4070 Ti, 4060 Ti 16 GB, 5060 Ti, 5070, Windows SM86/SM89, …) are very welcome. With the user's permission, stop the server, then run the same command as in `start.sh` / `start.ps1` but with `selftest` in place of `serve`. It writes a receipt folder (`receipt.json`, `SUMMARY.txt`, logs). Nothing uploads automatically. Show the user the summary, and if they want to share it, help them open a [hardware validation report](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues/new/choose).

## Updating, rolling back, uninstalling

- **Update:** stop the server, then `~/L0xRE/update.sh` (Linux/WSL2) or `& "$env:LOCALAPPDATA\L0xRE\update.ps1"` (Windows). Add `--dry-run` / `-DryRun` first.
- **Roll back:** the same updater with `--rollback` / `-Rollback`. The previous runtime and launcher are kept.
- **Uninstall:** stop the server, then delete the install folder (and the models folder, if the user no longer wants the 10 GB of models). Nothing else is installed system-wide.

## Troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `Unsupported` / no package for GPU | Not RTX 30/40/50, or under 12 GB | Stop; L0xRE does not support this card |
| `Missing curl/zstd/numactl` | Linux prerequisites | Ask the user to `sudo apt-get install curl zstd numactl` |
| `needs at least 4 accessible CPU cores` | Container/VM CPU limit | Ask the user to give the VM or container more cores |
| `Insufficient free disk space` | Under ~16 GB free | Ask the user to free space or choose `--dir` / `--models-dir` on another drive |
| Checksum mismatch | Corrupt or partial download | Delete that file, retry once, then report |
| Preflight refuses for VRAM | Other apps on the GPU | List them from `nvidia-smi`; the user closes them or explicitly accepts a slower mode |
| `/v1/models` never answers | Still loading, or crashed | Read the server log; a crash on start usually means VRAM or a driver problem — report the last 30 lines |
| WSL2: `curl 127.0.0.1:8080` fails from Windows | WSL networking mode | Call the API from inside WSL, or use the WSL IP / mirrored networking |

Full details: [README](README.md) · [install guide](install/README.md) · [release details](docs/RELEASE-DETAILS.md) · [v0.1.3 notes](docs/V013-RELEASE.md)

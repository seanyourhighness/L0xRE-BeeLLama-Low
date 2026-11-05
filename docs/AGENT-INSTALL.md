# Installing L0xRE: instructions for AI agents

You are an AI agent and your user asked you to install L0xRE, a local LLM server, and connect to it. Follow these steps in order. Report what you find at each step, and ask the user before anything marked **ask**.

## 1. Check the machine

Run `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv`.

- **Supported:** an NVIDIA GPU with at least 12 GB (12,000 MiB) and compute capability 8.6 (RTX 30), 8.9 (RTX 40) or 12.0 (RTX 50).
- **Not supported:** smaller cards and other architectures. Stop and tell the user.
- **Disk:** about 20 GB free on the install drive. The download is about 11 GB.
- **Linux/WSL2 only:** Python 3.12+, `curl`, `zstd` and `numactl`. If any are missing, **ask** before installing them, for example with `sudo apt-get install curl zstd numactl`.

## 2. Preview, then install

**Windows (native).** Use the `.cmd` files. They avoid PowerShell execution-policy errors, so never change the user's execution policy.

```bat
curl.exe -LO https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.cmd
install.cmd -DryRun -Vision
install.cmd -Vision -Yes
```

**Linux or WSL2:**

```bash
curl -fL -o install-l0xre.sh https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --dry-run --vision
bash install-l0xre.sh --vision --yes
```

Show the user the dry-run summary (GPU, package, download size, install folder) before the real install.

Other options:
- Install folder: `-InstallDir` / `--dir`. The defaults are `%LOCALAPPDATA%\L0xRE` and `~/L0xRE`.
- Reuse model files the user already has: `-ModelsDir` / `--models-dir`.
- Port: `-Port` / `--port`. The default is 8080.

The installer verifies every file hash. If it reports a mismatch, stop and report it. Do not work around it.

## 3. Start the server

- **Windows:** run `start.cmd` from the install folder. Setup prints the path.
- **Linux/WSL:** run `~/L0xRE/start.sh`.

The server runs in the foreground and must keep running. Start it in its own window, or as a background process you can monitor. For an unattended start, such as a service or a background job, set `L0XRE_NONINTERACTIVE=1` so the launcher never waits for input.

The first startup takes about 1–3 minutes. Poll until it answers:

```bash
curl -s http://127.0.0.1:8080/health
```

The answer `{"status":"ok"}` means it's ready.

## 4. Connect

L0xRE is an OpenAI-compatible server:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8080/v1` |
| Model | `L0xRE-27b-Low` |
| API key | any placeholder; none is required |
| Context window | 131,072 tokens on 12 GB cards (16 GB cards use a larger profile when supported) |

Configure yourself, or the app the user named, to use this endpoint, then send one short test message. Check the answer, and report `timings.predicted_per_second` (decode speed) to the user.

## 5. If something is wrong

| Symptom | What to do |
|---|---|
| Startup warns about free VRAM | Other programs are using GPU memory. **Ask** before closing any of them. The launcher already uses the smallest arena (1856 MiB) on a card that drives a display. |
| Slow answers | Run the doctor while the server is answering. Windows: `doctor.cmd` in the runtime folder (next to `l0xre.cmd`). Linux/WSL: `~/L0xRE/start.sh --doctor`. Report what it says. |
| `hardware qualification is pending; --qualification-probe applies only to a staged candidate` | The start script came from an old installer. Re-run the install command with `-Update` / `--update`. |
| `running scripts is disabled on this system` | Use `install.cmd` / `start.cmd` instead of the `.ps1` files. |

Do not add `--qualification-probe` or `--arena-mib` yourself, do not edit files in the install folder, and do not change GPU drivers or clocks.

## Updating and rolling back

- **Update:** Windows `update.cmd`; Linux `bash install-l0xre.sh --update`.
- **Rollback to the previous version:** add `-Rollback` / `--rollback` to the install command.

Stop the server before either one.

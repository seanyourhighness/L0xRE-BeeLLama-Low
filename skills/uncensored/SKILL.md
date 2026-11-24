---
name: uncensored
description: Toggle L0xRE uncensored control-vector steering on, off, or check status at runtime (/uncensored [on|off|status], no args toggles).
---

# `/uncensored` — Toggle L0xRE Control-Vector Steering

Controls the loaded steering vector on a running L0xRE `v0.1.6.4+` server at zero reload cost (`GET/POST /l0xre/cvec`).

## Argument handling

Parse the text immediately following `/uncensored`:
- **Empty** (or `toggle`): toggle the current state (`toggle`)
- **`on`** / **`enable`**: turn steering ON (`on`)
- **`off`** / **`disable`**: turn steering OFF (`off`)
- **`status`**: report current steering state without changing it (`status`)

## Execution

Run the bundled cross-platform Python helper relative to this skill directory (Python 3 stdlib only, no dependencies):

- **Linux / macOS / WSL:**
  ```bash
  python3 scripts/l0xre-cvec.py <action>
  ```
- **Windows (PowerShell / cmd):**
  ```powershell
  python scripts\l0xre-cvec.py <action>
  ```

If the active L0xRE server is on a non-default host/port (not `http://127.0.0.1:8080`), pass `--url <base_url>` or set `L0XRE_URL` (trailing `/v1` is stripped automatically).

### HTTP fallback (if `scripts/l0xre-cvec.py` is unavailable)

- Check state: `curl -sf http://127.0.0.1:8080/l0xre/cvec`
- Turn ON: `curl -sf -X POST http://127.0.0.1:8080/l0xre/cvec -H 'Content-Type: application/json' -d '{"enabled": true}'`
- Turn OFF: `curl -sf -X POST http://127.0.0.1:8080/l0xre/cvec -H 'Content-Type: application/json' -d '{"enabled": false}'`

*(On Windows PowerShell: `Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/l0xre/cvec -ContentType 'application/json' -Body '{"enabled": true}'`)*

## Response behavior

- If the user only invoked `/uncensored` (with optional `on`/`off`/`status`), reply concisely with the single status line printed by `l0xre-cvec.py`.
- If the user included a prompt after the command (for example `/uncensored on explain ...` or stacked `/uncensored off ...`), toggle **first** so slot + RAM KV caches flush and the new steering state takes effect, then immediately answer the user's prompt.
- If the server reports `loaded: false`, tell the user to start L0xRE with `./l0xre-uncensored` (Linux) or `start-uncensored.cmd` / `start-uncensored.ps1` (Windows).

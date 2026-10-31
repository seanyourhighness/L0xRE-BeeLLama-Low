# v0.1.1 update and rollback

Keep the previous runtime, installation receipt, launcher and model files. Stop the running server before updating. The [installer](../install/README.md) verifies the selected archive and models and retains the previous installation. The pinned target, drafter and qualified Q5 projector are unchanged from v0.1.

For an installation made with the guided installer, use the saved installer for offline rollback:

```bash
python3 install/install.py --rollback --dir /path/to/L0xRE
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\install.ps1 -Rollback -InstallDir 'D:\L0xRE'
```

Rollback verifies saved runtime and launcher hashes and restores configuration. It does not start the server. Start the restored launcher normally, check health and run its selftest with the same model/projector/GPU/profile options. Keep releases in separate directories; do not mix DLLs, shared libraries, cubins or presets.

The [saved v0.1 install catalog](../receipts/v0.1.1/rollback-catalog-v0.1.json) and [six original v0.1 archive references](../receipts/v0.1.1/rollback-v0.1-assets.json) preserve the immediate rollback source. Those published assets are immutable. Older R6 rollback information remains in the historical [v0.1 instructions](V01-ROLLBACK.md).

The [lifecycle handoff](../receipts/v0.1.1/lifecycle-handoff.json) and [review summary](../receipts/v0.1.1/REVIEW-SUMMARY.md) describe actual startup/selftest/update/rollback evidence. Linux SM86/SM89 final license-only supplements reuse prior GPU lifecycle evidence under strict runtime/configuration/mode/link identity and add actual CPU installer roundtrips; they do not claim a second final-archive GPU startup. Native Windows and WSL SM120 final archives have actual profile lifecycle evidence. A public archive-availability check is separate from executing rollback on hardware.

# V0.1 update and rollback

Keep the previous runtime and model files until the new installation passes selftest. The release evidence records actual install/update/startup and rollback checks against published R6 archives.

The installers use separate runtime directories named with the archive hash. An update preserves the previous installation receipt and launcher in `previous/`; it must not overwrite the old runtime. The target and drafter are unchanged, but retain the old Q8 vision projector when installing V0.1's separate Q5 file.

For an installation created by the supplied installer:

```bash
python3 install/install.py --rollback --dir /path/to/L0xRE
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\install.ps1 -Rollback -InstallDir 'D:\L0xRE'
```

Rollback verifies the saved runtime and generated helper hashes before restoring the previous launcher and receipt. It does not delete the new runtime or model files. Test the restored launcher with `--dry-run` / `-DryRun`, then perform a startup smoke test. Do not infer that a previous runtime is intact merely because its directory exists.

For manual archive installations, extract each release to a different directory, keep the old launcher/model arguments, stop the new server and start the old release from its own directory. Do not copy DLLs, cubins or presets between releases. The established previous release is `beellama-v0.4.7-r6`; its catalog/checksums and release assets remain the rollback reference. The closing audit records the exact published assets and hashes used in validation, including Windows SM120 `3f1375870d8876ddaca0d048f81bd10e3b32245322376a8133ae3692aca9bfd7`.

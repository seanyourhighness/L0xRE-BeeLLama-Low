# Windows quickstart

Allow about two minutes to start setup; model downloads take longer. Use x64 Windows 10/11, a working NVIDIA driver and at least 12GB VRAM. Native Windows RTX 5090 is measured. SM86/SM89 Windows packages remain build-only until matching hardware receipts are reviewed; a compatible 4070 Ti is not automatically certified.

1. Open a terminal (Command Prompt or PowerShell) in a writable folder, download the installer and preview:

   ```bat
   curl.exe -LO https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.cmd
   .\install.cmd -DryRun -Vision
   ```

2. Read the selected GPU, package, model locations and space requirements. Close other GPU applications, then install:

   ```bat
   .\install.cmd -Vision
   ```

   Add `-ModelsDir 'D:\Models'` to reuse existing pinned models. The installer checks hashes and prints the start command. It leaves the service stopped.

3. Double-click `start.cmd` in the install folder. Keep the window open while using the server. Stop it before running `update.cmd`. To restore your previous installation, run `update.cmd -Rollback`.

4. For a hardware report, include your GPU model, driver version, package filename, prompt length and server timing lines. The [v0.1.6 release notes](V016-RELEASE.md) list tested hardware and performance.

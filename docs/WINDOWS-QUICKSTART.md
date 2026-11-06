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

3. Run `start.cmd` in the install folder (or double-click it); setup prints its path. Update later with `update.cmd`. Keep the window open. If free VRAM is insufficient, Retry after closing GPU apps, explicitly accept Start slower when offered, or Quit. A smaller validated rung can reduce throughput; there is no silent fallback. The [README](../README.md) lists measured free-memory requirements. If L0xRE gets slow later, double-click `doctor.cmd` in the runtime folder (the server window prints its exact path): it says whether Windows is paging GPU memory, another program is using the GPU, or the GPU is too hot.

4. For a hardware report, run `l0xre.cmd selftest` from the installed package with the same model paths/profile you use for serving. Keep `receipt.json`, `SUMMARY.txt` and logs. Review the receipt before sharing; nothing uploads automatically. A reviewed PASS can establish hardware coverage for that exact card and profile.

For Ubuntu under WSL2 use the Linux installer, not the Windows runtime. WSL and native Windows have separate measurements. WSL 16GB short prefill was 2.028086% below 12GB in the recorded comparison, with one scoped owner acceptance; it does not change any other speed floor. See [what to expect](../README.md) for the source and exact scope.

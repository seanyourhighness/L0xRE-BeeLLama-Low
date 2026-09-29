# v0.4.7 B84 universal release candidate — September 29, 2026

- One Linux/WSL archive and one Windows ZIP cover NVIDIA SM86, SM89 and SM120.
- Linux SM86 includes the B74 vector bridge, opt-in 128-thread E3 head, and B84 96K Q4-drafter configuration. Model weights remain unchanged.
- Windows rebuild includes the head patch, a warning-free B74 bridge DLL, its SM86 vector cubin, and an SM86-only B84 preset.
- Portable launchers fix the Windows package-root bug and translate the common Linux CLI into the SM120 route. GPU selection respects `CUDA_VISIBLE_DEVICES`; invalid profiles fail clearly; SM86 environment choices are cleared for other architectures.
- Pin model downloads by HF revision and exact SHA-256. Add raw SM86 receipts, source references, compiler/dependency/architecture/test records, checksum verification and applicable third-party notices.
- Windows includes the previously missing OpenMP runtime (`vcomp140.dll`) needed for a clean installation.
- The canonical installation and test page is the runtime repository README; the L0xRE hub links to it.

Linux SM89 r1 and SM120 r9 payloads retain their existing receipts. Windows GPU execution, the new SM120 Low common-CLI path, and portable-wrapper SM86 clean-extract inference remain pending; the candidate label reflects those limits. Historical SM120 numbers use the separate MTP-containing target. Source and integrity checks do not replace hardware qualification.

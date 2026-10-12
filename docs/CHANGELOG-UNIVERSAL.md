# Universal r5-fast — October7,2026

- Package exact certified SM86 S71 GDN512/Q2_K/N7 bytes: Bench.sh code53.88/narrative29.13t/s; prefill10K567.58/77K469.42t/s; full150127pass@1/134pass@3 against baseline131/134.
- Add an explicit portable `--profile fast`, runtime hash checks, optional exact model-hash verification, and architecture qualification gates. Retain original guard512 and production sampler; exclude all guard-tuning experiments.
- Prepare one source/profile for SM86/89/120 and compile the transformed bridge, GDN512 bridge, fused-head templates and retargeted PTX kernels on all three. SM89/SM120 full-runtime integration and hardware qualification remain pending, with inherited payloads retained for compatibility profiles.
- Publish source snapshot identity, all quality losses/guard observations, compilation receipts and clean-archive validation. Windows stays at its immutable r4 release.

# Universal r4 — October 3, 2026

- SM86 12gb/12gb-quality/12gb-b84: 81920 targetKVarN4/4, B1024/UB512, draftUB32/ngl99, CPUembedding/no-op-offload, fit-target768.
- All-projection INT8 prefill replaces the B74 bridge for qualifying M256–512 shapes; small/decode requests retain B74 arithmetic.
- Qualified gated long-context FP16 QK attention; short contexts and decode stay on the original route.
- Linux ships exact qualified libraries. Windows adds a native backend entry and an explicit bridge initializer outside DLL loader lock; no Windows speed claim.
- DFlash2, CPU vision, caches, loop guard and GGUF weights preserved. Held GDN/materialization experiments excluded.
- Fresh77K429.05t/s, sustained decode28.31t/s on Linux RTX3060; fullcapacity+recovery passed. See evidence/prefill-r4/.

# v0.4.7 universal r3 — 80K quality-first 12 GB defaults

- `12gb` and `12gb-quality` select 81,920 context, target KVarN3/3, Q4_0/Q4_0 draft KV, DFlash2 N3, batch/ubatch 1024/256, exact tail 128, and window chunk 16384 on every universal route. The drafter weights remain Q4_K_M.
- Use 32 generation/batch CPU workers, medium reasoning with an 8,192-token budget, cache RAM disabled, and automatic memory fitting disabled. Users can explicitly override worker/context settings.
- A supplied vision projector runs on CPU with min/max 1024 image tokens; Q8_0 is recommended. Vision remains opt-in. Include the vision examples, Hermes80K native config, and smoke client in both packages.
- SM86 retains the B74 bridge and head128 choices. The `12gb-b84` name remains a compatibility alias on SM86, now selecting the80K preset. Explicit non-12GB legacy profiles keep their settings.
- Sean selected this less aggressive cache quantization after reporting better everyday results. Linux SM86 code/vision and 29,762-token retrieval passed; minimum free VRAM 243 MiB vs19MiB for 96K with the same upgraded caches. This is not a general quality score or full 80K qualification.
- Configurations are aligned on SM89/SM120 and Windows; new-profile inference, throughput and memory qualification remain pending on those routes. Compiled binaries, CUDA kernels, bridges and model weights are unchanged from r2. Older releases remain immutable rollback options.

# v0.4.7 universal r2 — model naming and documentation

- Public model documentation uses L0xRE-27b-Low consistently.
- The model card explains download sizes, the optional drafter, required runtime, measured configurations and qualification limits.
- Current Linux and Windows launchers advertise L0xRE-27b-Low as the API model name.
- Compiled runtime binaries, CUDA kernels and model weights are unchanged from the verified universal build.

# v0.4.7 B84 universal release candidate — September 29, 2026

- One Linux/WSL archive and one Windows ZIP cover NVIDIA SM86, SM89 and SM120.
- Linux SM86 includes the B74 vector bridge, opt-in 128-thread model head, and B84 96K Q4-drafter configuration. Model weights remain unchanged.
- Windows rebuild includes the head patch, a warning-free B74 bridge DLL, its SM86 vector cubin, and an SM86-only B84 preset.
- Portable launchers fix the Windows package-root bug and translate the common Linux CLI into the SM120 route. GPU selection respects `CUDA_VISIBLE_DEVICES`; invalid profiles fail clearly; SM86 environment choices are cleared for other architectures.
- Pin model downloads by HF revision and exact SHA-256. Add raw SM86 receipts, source references, compiler/dependency/architecture/test records, checksum verification and applicable third-party notices.
- Windows includes the previously missing OpenMP runtime (`vcomp140.dll`) needed for a clean installation.
- The canonical installation and test page is the runtime repository README; the L0xRE hub links to it.

Linux SM89 r1 and SM120 r9 payloads retain their existing receipts. Windows GPU execution, the new SM120 Low common-CLI path, and portable-wrapper SM86 clean-extract inference remain pending; the candidate label reflects those limits. Historical SM120 numbers use the separate MTP-containing target. Source and integrity checks do not replace hardware qualification.

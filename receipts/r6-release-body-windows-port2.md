## Certified scope

This release certifies the frozen C07 R6 profile on Linux SM86. The sustained seed-42 run measured **37.16 t/s prose and 65.06 t/s code**. Across seeds 42 and 1234, the balanced result was **34.95 t/s prose and 62.69 t/s code**, versus R5 at 29.21/54.09. Cold 10K and 77K prefill stayed within 0.1% of R5, and the 81,916-token capacity test passed.

On the paired 150-scenario quality set, R6 scored 127/150 pass@1 and 132/150 pass@3; R5 scored 123/150 and 130/150. R6 had no runaway requests or CUDA errors; R5 had one runaway and no CUDA errors. The paired quality differences are directionally positive but not statistically decisive. Full evidence and hashes are included in the package.

The Linux/WSL package excludes model weights. The adjacent `.sha256` asset is the archive digest. SM89 and SM120 are outside this certificate and require their own hardware qualification.


## Packages awaiting hardware certification

The `sm89-linux-candidate` archive and `sm86-windows-candidate` / `sm89-windows-candidate` ZIPs are installable test packages, with adjacent SHA256 checksums. SM89 and Windows results are outside the Linux SM86 certificate. Use `--qualification-probe` to run them; SM89 Q4 MMQ is opt-in through `--sm89-q4-mmq`.

The SM89 Linux package contains the rebuilt native R6 runtime, packed decode bridge, QK16 gate and matched SM89 fallback dependencies. Its masked GDN path defaults off because the current compact-GDN architecture gate does not admit SM89.

The original Windows candidate packages contain the native R6 runtime and packed decode bridge with SM86 and opt-in SM89 Q4 MMQ routing. DLL imports, bridge exports/device architectures and archive integrity passed offline checks. Windows INT8 prefill and masked GDN remain disabled pending their separate port/qualification; native QK16 is SM86 only. No Windows throughput or quality claim is made.

## Windows prefill port2 candidates

The new `L0xRE-BeeLLama-Low-R6-sm86-windows-prefill-port2-candidate.zip` and `L0xRE-BeeLLama-Low-R6-sm89-windows-prefill-port2-candidate.zip` packages enable native INT8 prefill, masked GDN with the native rollback tail, and long-context QK16 for SM86/SM89. Adjacent `.sha256` files are provided. These are the updated Windows test candidates; `--qualification-probe` remains required, and SM89 Q4 MMQ remains opt-in with `--sm89-q4-mmq`.

Native CUDA 13.3/MSVC builds, Windows CPU-only DLL ABI/output-guard checks, all fourteen cubin architecture/entry checks, DLL dependency checks, payload hashes and archive CRC/hash checks passed. Actual Windows GPU dispatch, throughput, quality and long-context capacity are still pending. The Linux SM86 certificate does not apply to these Windows candidates.

Source branch: `release/r6-windows-prefill-port2`; commit `d9bdf77f37f1cfb2c2616f74b2fc313234012a5f`; source tag: `beellama-v0.4.7-universal-r6-windows-prefill-port2`. Exact platform overrides, bridge recipes and offline receipts are included in both packages.

SM120 certification remains pending: short-context performance screens are promising, but long-context testing triggered Windows TDR resets. SM120 is not included as a certified package. All packages exclude model weights.


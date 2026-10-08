## Certified scope

This release certifies the frozen C07 R6 profile on SM86. The sustained seed-42 run measured **37.16 t/s prose and 65.06 t/s code**. Across seeds 42 and 1234, the balanced result was **34.95 t/s prose and 62.69 t/s code**, versus R5 at 29.21/54.09. Cold 10K and 77K prefill stayed within 0.1% of R5, and the 81,916-token capacity test passed.

On the paired 150-scenario quality set, R6 scored 127/150 pass@1 and 132/150 pass@3; R5 scored 123/150 and 130/150. R6 had no runaway requests or CUDA errors; R5 had one runaway and no CUDA errors. The paired quality differences are directionally positive but not statistically decisive. Full evidence and hashes are included in the package.

The Linux/WSL package excludes model weights. The adjacent `.sha256` asset is the archive digest. SM89 and SM120 are outside this certificate and require their own hardware qualification.


# L0xRE BeeLLama universal r5-fast

The SM86 fast profile packages the certified S71 GDN512 / DFlash2 Q2_K / N7 runtime. It preserves one-slot81,920 context, KVarN4/4 target cache,1024/512 batch/ubatch, medium reasoning with8192 budget, original guard settings, and optional CPU vision. Production sampling remains .7/.95/k20/minp.05. The fast profile uses8 CPU workers and the qualified NUMA0/CPU0-7 policy, so `numactl` is required.

| Frozen Bench.sh decode-only qualification | Baseline | Fast SM86 |
|---|---:|---:|
| Code decode |39.71 t/s|**53.88 t/s**|
| Narrative decode |28.90 t/s|**29.13 t/s**|
|10K prefill |561.19 t/s|567.58 t/s|
|77K prefill |464.45 t/s|469.42 t/s|
| Full150 pass@1 |131/150|127/150|
| Full150 pass@3 |134/150|**134/150**|

Two seeds, two fresh boots per arm, five measured requests after three warmups per prompt/boot. These are RTX3060 measurements, not cross-GPU forecasts. Original code and narrative limits were800 and1000 tokens. Package smoke separately passed10K prefill at571.29t/s, greedy token hashes, speculative-verification activation and CPU vision `VISION 742`.

## Run

Download the Linux/WSL archive and extract it. Target and draft models are separate downloads from `YourHighnessLA/L0xRE-27b-Low`, revision `a18987908d220da9c40cf2887913e727cdbf65a4`:

- `L0xRE-27b-Low.gguf`, SHA-256 `b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543`.
- **`Qwen3.8-27B-DFlash2-Q2_K.gguf`**, SHA-256 `e3eb7705404817cdbcdabe56049a1952b3b37bcc8df6e4d4efaec5d41563fb7e`.

```bash
./l0xre serve --profile fast --verify-models \
  -m /path/to/L0xRE-27b-Low.gguf \
  -md /path/to/Qwen3.8-27B-DFlash2-Q2_K.gguf \
  --host 127.0.0.1 --port 8080
```

`--verify-models` checks the exact qualified model hashes; it adds startup file-reading time. Runtime components are hash-checked on every fast launch. Set `CUDA_VISIBLE_DEVICES` before launch to select the GPU. `--mmproj /path/to/mmproj.gguf` enables the inherited CPU-vision path. `--dry-run` prints the resolved configuration without loading CUDA. Other server arguments pass through; overrides leave the exact certification scope.

The common launcher keeps existing `12gb` and legacy profiles available. The new fast profile is explicit. Default bind is localhost; choose a different address explicitly if needed. On Z840, the already-certified service remains unchanged while this portable package is published.

## Three-architecture parity

See [PARITY.md](PARITY.md) and [shared source/build preparation](source/fast/README.md). SM86/SM89/SM120 use one prepared S71 source overlay and one fast configuration. The transform/GDN512 bridges, fused-head template and retargeted kernels compiled for all three. Complete SM89/SM120 runtime integration and hardware qualification remain pending; the archive retains their existing payloads and rejects `fast` on those routes until a complete fast manifest is installed.

## Qualification limits

Pass@3 equals the paired baseline. Pass@1 is four lower; one unseeded draw per arm does not prove equivalence or noninferiority. DE-05 has two invalid-JSON attempts and a systematic failure; IF-10, DE-10 andCLI-17 also lose at pass@3 while four other scenarios improve. All original outcomes remain in `evidence/fast/`. Guard events remain documented:15 reasoning closures and4 visible stops in the fast quality leg. This is not a loop-free-output claim or a guard-tuning release.

The new head changes floating-point reduction; no global bitwise-logit equivalence is claimed. Minimum measured GPU headroom was37MiB, so display use, driver allocations and additional slots can exceed the tested capacity. Scope isLinux/WSL RTX3060 sm86, one slot and the stated models/configuration. Windows is unchanged at its prior release.

For rollback, extract the preceding immutable release into a separate directory and use its original model/profile settings. Z840's preserved service rollback is documented in `evidence/fast/CERTIFICATION.md`.

# L0xRE-27b-Low performance and validation

The current release supports the L0xRE-27b-Low file whose SHA-256 is `b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543`. Performance attribution follows the exact model file, runtime components, GPU and command.

| Configuration | Result | Scope |
| --- | --- | --- |
| RTX3060 12GB, Linux, recommended drafter,96K context | Code39.046 / narrative29.422 tokens/s | Temperature0, seed0, fixed800 generated tokens; two measured runs per prompt |
| RTX3060 long-context canary | 92,879 input +128 greedy output tokens; reference hash matched | Prefill272.60 / decode21.06 tokens/s; minimum free VRAM215MiB |
| RTX4090, Linux,12gb profile | Code115.4 ±2.4 tokens/s | Five measured runs;81,920-token allocated context |
| Windows and current SM120 common-CLI model path | Hardware qualification pending | Architecture, dependency and launcher checks are separate from GPU inference tests |

See [component and methodology records](evidence/universal/SM86-B84.json), [build checks](evidence/universal/VALIDATION.json), and [archive integrity](evidence/universal/ARCHIVE-VALIDATION.json). Different workloads and protocols prevent a controlled comparison between these GPU rows.

Historical runtime ratios from a separate MTP-containing checkpoint are not a parity claim for this model file. Broad quality evaluation and generic guarantees for all12GB GPUs are not established by these workload measurements.

# RTX 4070 Ti / SM89 Windows R6 status

**Certification mostly complete. Completed quality packs: 79/90 first attempts; 83/90 within three attempts. Remaining packs pending.**

The R6 refresh binaries are unchanged. This configuration uses two target/draft CPU threads, six batch threads, and all 12 logical processors on the tested Ryzen 5 5600X. It retains one 81,920-token slot, KVarN4/4, the pinned 27B target and Q4 DFlash2 N7 drafter.

| Measurement | Two-start mean |
| --- | ---: |
| Prose decode | 67.88 tokens/s |
| Code decode | 115.76 tokens/s |
| Prose wall throughput | 67.15 tokens/s |
| Code wall throughput | 113.13 tokens/s |

Canonical measurements use thinking disabled, temperature 0.6, top_p 0.95, top_k 20, min_p 0; each fresh start has three warmups and five measured requests per workload. These are measured speeds; 75 prose / 125 code and 1,500 prefill remain future targets.

Package/model integrity, restart token checks, real 81,916-token capacity, a 30-minute/500-request soak, varied-content prefill, and the CPU vision fixture passed.

Only completed, valid packs enter this quality comparison. These are raw verifier scores; pending adjudication and stricter checks do not silently change them.

| Completed pack | Candidate first attempt | Candidate within three | Baseline first attempt | Baseline within three |
| --- | ---: | ---: | ---: | ---: |
| Tool calling | 12/15 | 13/15 | 14/15 | 14/15 |
| Instruction following | 14/15 | 15/15 | 14/15 | 15/15 |
| Structured output | 15/15 | 15/15 | 15/15 | 15/15 |
| Data extraction | 13/15 | 13/15 | 12/15 | 12/15 |
| Reasoning/math | 11/15 | 12/15 | 12/15 | 13/15 |
| Bug finding | 14/15 | 15/15 | 14/15 | 15/15 |
| **Completed packs total** | **79/90** | **83/90** | **81/90** | **84/90** |

Corrected Hermes and CLI candidate packs remain pending and contribute no published score. Original invalid cohorts are excluded. Recorded task failures, verifier defects and reasoning-loop guards remain in the evidence. A full paired quality score remains pending.

The final consumer-launcher and quiet-prefill closeout checks also remain pending. Hardware qualification remains false. L0xRE and the benchmark environment were shut down at the user's request.

## Use the measured configuration

Integrate the profile and launcher source into the full Windows package on the build machine using [the replication instructions](SM89-WINDOWS-CONFIG.md), then run:

```powershell
.\l0xre-r6.cmd serve --profile r6-sm89-4070ti --qualification-probe -m models\L0xRE-27b-Low.gguf -md models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

The launcher checks the required refresh binary hashes and model hashes. The profile is opt-in and SM89-only. Other hardware has no speed guarantee from this measurement. The package archive SHA256 is `5fb57e1525c1bdbe9c6cf3f0f4d40a83bef37c764fe9922b73cb9d21eab8343b`.

Exact configuration: [r6-windows-sm89-4070ti-profile.json](../tools/universal/r6-windows-sm89-4070ti-profile.json). Machine-readable status: [r6-windows-sm89-4070ti-status.json](../receipts/r6-windows-sm89-4070ti-status.json).

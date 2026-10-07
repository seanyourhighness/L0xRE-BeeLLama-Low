# Fast L0xRE build certification

Certified and installed as the enabled Z840 default: **S71 GDN512 / DFlash2 Q2_K / N7**.
Endpoint: `192.168.1.159:8080`. Unit: `l0xre-sm86-3060-96k.service`.
Activation verified at 2026-10-07T03:11:33.457383+00:00; PID 595348; zero service restarts.

| Measurement | Baseline | Fast build | Change |
|---|---:|---:|---:|
| Bench.sh code decode | 39.71 t/s | **53.88 t/s** | +35.70% |
| Bench.sh narrative decode | 28.90 t/s | **29.13 t/s** | +0.79% |
| 10,240-token prefill | 561.19 t/s | 567.58 t/s | +1.14% |
| 77,824-token prefill | 464.45 t/s | 469.42 t/s | +1.07% |
| Full150 pass@1 | 131/150 | 127/150 | -4 cases |
| Full150 pass@3 | 134/150 | **134/150** | equal |

## What passed

- Frozen Bench.sh decode-only balanced qualification: two fresh boots per arm, seeds42/1234, five measured requests after three warmups per prompt and boot. All forty measured requests usable; zero benchmark errors.
- Both 10K and77K prefill gates;81,916 total-token capacity including4,091 continuation tokens without truncation.
- Identical150 raw scenario definitions and pack versions across the paired quality legs; production medium effort, unseeded .7/.95/k20/minp.05 settings attested. Candidate wrapper exited0; zero classified timeouts/token-limit/agent-runner timeouts or CUDA errors.
- Packaged launcher: greedy code/prose token hashes, fresh10K prefill571.29t/s, rejection-verification activation, CPU affinity0-7, CPU vision OCR `VISION 742`.
-17 runtime manifest entries and29 external dependency hashes verified, including the target andQ2 draft models.
- Installed default: exact frozen argv and environment, loaded candidate libraries, original guard minimum512, healthy native/public endpoints, structured JSON response through8080, enabled user service, zero restarts.

## Quality and guard limitations

Acceptance follows the user's pass@3 reference: both arms134/150. Candidate pass@1 is lower127/150 versus131/150; paired exact McNemar p=.4545. This one unseeded draw per arm does not establish equivalence or noninferiority. All losses and retries remain in the original receipts.

DE-05 has two invalid-JSON attempts and remains a candidate-only systematic failure, alongside IF-10, DE-10 andCLI-17. Four other scenarios improve at pass@3. The candidate journal has15 hidden-reasoning closures and4 visible-output stops. These are retained failure observations, not relabeled as fixes.

Actual historical journals demonstrate14 hidden-reasoning guard closures inR4 PID499411 and14 inR5 PID500226 during the earlier seed1234 quality runs, before the current decode sprint. Earlier word-level replay and short greedy hash parity were insufficient to establish global output-distribution equivalence. The current certificate applies to the fast build's original guard configuration; the million-token threshold, cue and1024-token threshold experiments are excluded.

Minimum GPU headroom in qualification was37MiB; scope isRTX3060 sm86, one slot and81,920 configured context. The changed head reduction is not bitwise identical globally. No claim of arbitrary workload speed,40t/s narrative or loop-free generation is made.

## Evidence and rollback

Machine-readable certificate: `certification-final.json`. Gates: `certification-gates.json`. Package probe: `package-probe.json`. Live verification: `activation-receipt.json`. Quality decision: `quality-review.json`. Frozen manifest: `candidate/runtime-sha256.json`; complete staged-package checksums: `candidate/SHA256SUMS`.

Rollback onZ840:

```bash
bash /home/sean/work/l0xre-decode-qualification-20261006/candidate/rollback.sh
```

The preceding user unit and its original launch settings are preserved. All abandoned quality jobs are stopped; no further baseline guard experiments are running.

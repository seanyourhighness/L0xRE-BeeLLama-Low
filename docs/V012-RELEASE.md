# L0xRE v0.1.2

Release tooling and documentation update. All six runtime archives reuse the exact published v0.1.1 bytes; their internal runtime version remains 0.1.1. Shipping runtime source, kernels, presets, launchers and binaries remain unchanged. Private CPU CI compiles the host library for validation; release archives use the original runtime bytes.

The release planner maps every tracked path to T0–T3 gates. Hash-based reuse, drift canaries, development CUDA object caching, recipe selection checks and automated draft staging are included in the source archive.

decode_at_10k is now gated on the versioned 256-token measurement with at least ten pooled samples per build and CV at most 5%. Fresh workers run A1/B1/B2/A2, each retaining one excluded warmup and five measurements. Its unchanged-v0.1.1 baselines are separate from the historical 16-token probe. The speed floor remains 2%, output hashes remain identical and consent behavior remains unchanged.

WSL SM120 16GB retains its original failed ten-sample candidate result (-2.604703%). Sean authorized exactly one additional paired session and an assessment of all original ten plus all new ten candidate observations against the original frozen baseline and unchanged 2% floor. The resulting twenty-sample mean is 105.260324 tok/s (CV 3.273023%, change -1.778637%). The repeat also passes its own paired floor. Both session records, all warmups and measured samples, and the original failure remain unchanged. The separate assessment protocol is `decode-owner-authorized-two-session-aggregate-v1`; it is not same-session pooling or a replacement baseline.

The Windows observer retains its raw failure. A separate terminal-only assessment addresses the known System COPY observation at completion; its provider interval was not recorded and may straddle completion. Its startup-expanded counter set cannot certify processes created later. Current measurement eligibility uses the original matching frozen-v3 producer protocol and hygiene, with these observer limits explicit; it is not a full dynamic Windows GPU quiet certificate.

Inherited coverage limitations:

- Windows SM86 and SM89 retain original build-only status; Sean declined the optional Windows SM89 hardware upgrade.
- Unchanged-runtime compilation evidence is inherited with its original Linux SM120/17-cubin clean-extract scope; no all-architecture fresh reproduction or shipping rebuild is claimed.
- Native Windows SM120 16GB allocation recovery remains inherited accepted nonexecution, not an allocation-recovery execution PASS.
- The original WSL 16GB short-prefill comparison exception and scoped CLI case variance remain limited to their original evidence; neither waives the new decode floor.
- Future native combined GPU jobs and native Windows process capture remain explicitly unsupported until actual producers/captures are registered; historical GPU executions are not backfilled.
- WSL SM120 16GB original valid candidate failure remains recorded. Exactly one owner-authorized additional ABBA and every original/new B sample form B20 against the fixed original A10 and unchanged 2% floor; this explicitly scoped two-session assessment does not replace the original baseline or relabel its original failure.
- The acquired Windows observer raw verdict and quiet/full-Windows/foreign-absence flags remain FALSE. Its startup-expanded instances cannot discover later PIDs; terminal System COPY provider intervals may straddle completion. Eligibility is under the original matching frozen-v3 producer hygiene and explicit owner repeat scope, not a full dynamic Windows GPU quiet certificate.
- Future WSL SM120 Windows GPU Engine guard uses fresh instance queries, verified CPU late-PID discovery and a read-only formatting probe. Its sampled intervals have observable gaps; it does not prove uninterrupted foreign-process absence, retroactive coverage, or native Windows frontend qualification.

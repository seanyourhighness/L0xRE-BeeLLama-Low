# SM89 Linux prefill port2 candidate

The SM89 overlay now admits ordinary and compact masked GDN alongside the existing opt-in Q4 MMQ route. Existing stride, shape, precision and native rollback-tail guards remain intact. The candidate profile enables INT8 prefill, masked GDN, long-context QK16 and packed decode. GPU correctness, throughput and capacity certification are pending.

Use `--qualification-probe` with the packaged `tools/universal/r6-launch.py`; add `--sm89-q4-mmq` to test that path. The source identity and three architecture overlay hashes are in the package. The updated builder accepts multiple overlays per architecture, verifies the common C07 input, applies the selected overlays and emits a build receipt.

Offline checks in `evidence/sm89-prefill-port2` cover the native/bridge build, payload hashes, loader links, dependencies, seven masked-GDN cubin architectures/entry names, unsupported-call ABI/output guards, and a clean archive extraction. The CLI version query exited0 with CUDA initialization unavailable on this host. No GPU work was launched and these checks do not extend the Linux SM86 certificate.

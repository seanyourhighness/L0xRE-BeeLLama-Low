# Architecture parity status

Parity is tracked at three levels: the shared source/profile, successful compilation, and hardware-qualified runtime behavior. A compilation receipt does not establish inference correctness or speed.

| Feature or gate | SM86 / RTX30 | SM89 / RTX40 | SM120 / RTX50 |
|---|---|---|---|
| Common frozen S71 sampler/server source | Prepared | Prepared | Prepared |
| Common81,920 context/KVarN4/4/Q2/N7 fast profile | Prepared | Prepared | Prepared |
| Transform bridge, GDN512 bridge and fused-head compilation | Passed | Passed | Passed |
| Projection PTX and seven masked-GDN kernels assembled | Passed | Passed | Passed |
| Complete fast runtime package | **Certified bytes packaged** | Full build/integration pending | Full build/integration pending |
| Greedy output, CPU vision and clean package smoke | **Passed** | Pending | Pending |
| Balanced Bench.sh decode, prefill and capacity | **Passed** | Pending | Pending |
| Full150 production-medium quality review | **127 pass@1 /134 pass@3** | Pending | Pending |
| Hardware qualification | **RTX3060 qualified** | RTX4090 currently absent | Hardware run pending |

The Linux universal r5-fast archive ships the original certified SM86 S71 bytes. It retains prior SM89/SM120 payloads for existing profiles; those are not relabeled as optimized or requalified. `--profile fast` is explicitly gated by each architecture's `FAST-MANIFEST.json` and cannot use an incomplete payload. Windows remains at the prior release.

The shared build preparation in [source/fast](source/fast/README.md) is the handoff for updating SM89, then SM120. The same source/configuration and qualification gates apply to all three. Hardware-specific launch geometry, numerical behavior and throughput must be measured on each GPU before claiming full runtime parity.

Reference SM86 results: code53.88t/s, narrative29.13t/s,10K prefill567.58t/s,77K469.42t/s,81,916-token capacity. Paired baseline code39.71/narrative28.90 and131/134 quality versus fast127/134. Guard events and all quality losses are retained in [fast qualification evidence](evidence/fast/CERTIFICATION.md).

Compilation receipts: [SM86](evidence/parity/sm86-compile.json), [SM89](evidence/parity/sm89-compile.json), [SM120](evidence/parity/sm120-compile.json). They contain exact artifact hashes and explicitly mark hardware qualification false for the cross-builds.

# Windows SM89 R6 refresh candidate

This native Windows CUDA 13.0 build starts from the exact source inputs published with the certified Windows SM120 runtime. It applies the final startup environment cache and INT8 source, then admits SM89 to the ordinary/compact masked-GDN gates. The SM89 down12, QK16 and 16K prefill chunk settings remain; the 5.7 GB down-weight cache is disabled for the 12 GB target. The SM120-specific INT8 synchronization fence remains gated to SM120.

**RTX 4070 Ti hardware certification is pending.** Previous r3/r4 results and the SM120 certificate do not qualify this build. The profile retains 80K context, KVarN4/4, DFlash2 N7, graphs, one slot and medium reasoning. The launcher pins the selected GPU UUID, the packaged backend, and CPUs 0–7, and clears stale runtime experiment variables. Q4 MMQ on SM89 remains a separate opt-in.

The adjacent build plan identifies the published source archive (`91bcffbcc55a396cbbfed41ffec257885324d2814e5cd5c54ba669b53443478a`), overlaid source file hashes and the two eligibility changes. `build.cmd` records the actual CUDA 13.0/MSVC portable CPU/shared-backend recipe. Preparation/packaging scripts record local build paths; adapt those paths when reproducing elsewhere. They never alter the original certified archives.

The universal R6 release includes `L0xRE-BeeLLama-Low-R6-sm89-windows-refresh-source-inputs.zip`, containing the complete prepared native source, packed bridge, masked-GDN companion, launch metadata and checksums. The source archive SHA-256 is `398fa0d234f16bbfc56ad2fe5e9217837deeaa535a694a0a68fe2f8f19e9dfeb`. The [release receipts](../../../receipts/r6-universal-refresh/) distinguish offline checks from hardware qualification.

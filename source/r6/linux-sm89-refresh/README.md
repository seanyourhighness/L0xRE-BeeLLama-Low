# Current R6 native source on Linux SM89

The build consumes the exact published Windows SM89 R6 source-input archive. Its native source is portable; Linux builds retain the native Linux backend and bridge entry points. The archive already contains the final startup environment cache, current INT8 bridge, SM89 masked-GDN eligibility and architecture guards. Its SHA256 is `398fa0d234f16bbfc56ad2fe5e9217837deeaa535a694a0a68fe2f8f19e9dfeb`.

```bash
python3 source/r6/linux-sm89-refresh/build.py --output /path/to/sm89-build --jobs 6
```

Use `--source-inputs /path/to/L0xRE-BeeLLama-Low-R6-sm89-windows-refresh-source-inputs.zip` to reuse an existing archive. CUDA13.0, CMake, Ninja and a compatible host compiler are required. The recipe preserves portable CPU code, CUDA graphs, KVarN and the SM89 target. Building and source verification do not certify inference.

The build outputs native server/CLI/bench libraries plus packed, masked-GDN and QK16 companions. A final package also requires the inherited SM89 CUDA libraries, fallback bridges, packed-input and K3 cubins from the published Linux SM89 prefill-port2 archive. Use the current matching masked-GDN GPU modules from the Windows SM89 refresh archive SHA256 `5fb57e1525c1bdbe9c6cf3f0f4d40a83bef37c764fe9922b73cb9d21eab8343b`; the assembled candidate verified these modules on Linux with the current companion. Relocate ELF runpaths to `$ORIGIN`, refresh package inventories, then qualify the assembled package and its exact launch profile on the target hardware.

RTX4090 optimization is separate from the 12GB SM89 profile. The optional INT8 down-weight cache allocates about5.7GB extra VRAM. It must remain off for the 12GB profile. The current 4090 candidate preserves80K context, KVarN4/4, one slot, N7, medium reasoning, fast bridges, eight CPU threads and NUMA0 placement at the host's persistent300W limit. Its newer native build and final certification are in progress.

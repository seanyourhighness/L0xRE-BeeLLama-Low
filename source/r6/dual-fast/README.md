# Experimental per-device fast bridges

These changes are **experimental / untested / uncertified on physical dual GPUs**. They retain the selected R6 runtime/profile and kernel arithmetic. They do not establish quality, throughput, capacity, or stability parity.

`dual-fast.patch` records the source changes: packed bridge INT8 state and code-GEMM modules/functions become per-device; masked GDN scratch, initialization and stream ownership become per-device. Initialization restores the caller's CUDA device. The launcher limits exposure to two matched GPUs with at least 12 GB each and uses native layer split `1,1` with the drafter on logical CUDA0.

The experimental Linux and Windows add-on archives contain the complete modified bridge sources, definitions, launch metadata, startup environment header, build commands and file hashes. The packed bridge base is the published Windows SM120 source input archive (`91bcffbcc55a396cbbfed41ffec257885324d2814e5cd5c54ba669b53443478a`); SM86/SM89 masked GDN retains the port2 launch metadata, and SM120 retains its separate 512-token metadata. Use the full SM89 refresh source archive for the common ggml headers. Build recipes record the actual local toolchain paths, which must be adapted on another host.

Linux SM86 uses CUDA 12.8 / cuBLAS 12.8.4.1; Linux SM89/SM120 use CUDA 13.0. Windows add-ons use CUDA 13.0. Each add-on links against its selected runtime's supplied CUDA/cuBLAS libraries. Do not interchange the Linux SM86 and SM89 shared libraries merely because the packed device code contains both architectures.

The build receipts, dependency reports, CUDA architecture inspection and [CPU ABI checks](../../../receipts/dual-fast/) are offline engineering evidence. The ABI checks exercise loadability, version/export identity and null/invalid-device rejection; they do not launch inference or initialize INT8 GPU workspaces. See [physical testing requirements](../../../docs/DUAL-GPU-TESTING.md).

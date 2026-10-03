# Qualified SM86 prefill paths

Target E3 / single slot / logical CUDA device0 / RTX3060 SM86. Actual GGUF files unchanged; INT8 weights are temporary scratch. INT8 CAP512 rolesmask255; M256–512 only. Gated QKFP16 mode2 for KV>16384, M128–1024, head256, KVarN4/4, ratio6 and qualified domain; test rollback flags preserve stock routes. Decode and shortcontexts fall through. Qualifying source is copied from the benchmark experiment, with only guarded Windows portability/init and native dispatch integration.

Linux packages preserve exact qualified binaries, original backend and LD_PRELOAD ordering. build-linux.sh can reconstruct from this full source and a matching backend; new builds need independent qualification.

Windows uses the same arithmetic in the backend and bridge-int8-allproj.dll. Explicit l0xre_int8_prefill_init_v1 initializes scratch/cuBLAS after LoadLibrary, at backend startup ahead of CUDA graphs. Never call CUDA from DllMain. Windows GPU performance remains unqualified. Full source branch and package manifests pin identity separately from retained executable metadata.

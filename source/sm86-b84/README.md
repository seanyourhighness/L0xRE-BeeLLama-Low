# SM86 B74 bridge and B84 configuration source

Pinned Linux runtime baseline: ab1698c739b3bc13b9e06fb8febdc774ff106d23. The release source branch also contains the existing Windows loader/build fixes through eed82efdd415ad8afc8032e193a598c3172f035a. The opt-in 128-thread model head patch is committed in the model-specific CUDA source. It changes reduction order and is enabled only by the packaged SM86 head selection; keep it disabled outside the measured SM86 path.

B74 routes K3 raw decode through the vector cubin, rounding FP32 input via FP16 first. K2 retains the official cubin route. Buffer caches are keyed by device, stream, and length. The C bridge ABI is unchanged. The retained PTX and kernels require the third-party notices included in the release archive.

Reconstruct the vector PTX with:

```bash
python3 b71_vectorize_f32.py
python3 b72_vectorize_half.py
ptxas -arch=sm_86 k2k3-sm86-k3-f32-v4-half-v4.ptx -o k3-vector-all.cubin
nvcc -shared -Xcompiler=-fPIC -O3 -std=c++17 -arch=sm_86 \
  official_bridge_b74_k3_vector.cu -o libbridge-b74-k3-vector.so -lcuda
```

The production cubin SHA256 is fe1069f84644825c37d92bb39a1295a063bb25f2ba53e99dba93c1d9c13a7059. Builds on another toolkit can differ in bytes; rerun output/kernel/endpoint gates before claiming qualification. Linux bridge was built with CUDA12.8. Windows instructions are in windows/ and use CUDA13.3/MSVC.

B84 is a serving configuration on this B74 runtime: published Low target plus Q4_K_M DFlash2 drafter, one98304-token context, target batch1024/ubatch256, draft ubatch128, target KVarN3/2 with128-token exact tail, draft q2_0/q2_0, DFlash2 N3, window chunk16384, cache-ram0, fit on/fit-target768. Target model b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543 and drafter1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd are unchanged.

The universal release contains measured Linux SM86 artifacts, not a claim of byte-reproducible fresh compilation of the entire runtime on every toolchain. The Linux server is e36639081643e7f3be37e1de0008823ff71a899352c05380e0a9072535bcbfee; patched CUDA backend b3f117d779d3059232dc09ecbdb26923564464d40d8a99e075f535de198de27d; bridge2b03be5f77ea7dd866a2da3180807e9df9746e5d1b685d2b5153355b56575ffc. Original benchmark launcher d7f4e2c882172ca3470a747d286ce00838f06f7ceec8c5f2f58c7df714f6ceec is a Bash script hash. Portable launcher changes are covered separately by package validation.

Roll back to the previous SM86 experimental release or omit the new head/vector selections; preserve model files. Existing SM89 and SM120 routes remain separate payloads.

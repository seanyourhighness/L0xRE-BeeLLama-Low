Strata Qwen Flash source review adjudication, October3:
1. Missing initializer conditional is addressed: escha-moe.cu loader calls optional export; ggml_backend_cuda_init invokes loader before graph creation; requested INT8 missing export aborts. No initializer from DllMain.
2. cuBLAS reset claim is false. NVIDIA cuBLAS13.4 docs section2.4.8 explicitly say cublasSetStream unconditionally resets workspace; keep reapplication. Source: https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace . Mutex serializes host access, and alternate streams fall through before scratch writes. Multi-slot is outside qualified scope.
3. Hard fail on init/OOM is deliberate fail-closed inherited qualified behavior, not a Windows-specific defect. Don't advertise a failed optimization as active.
4. Device0 is logical CUDA device selected using CUDA_VISIBLE_DEVICES; one-GPU/single-slot is explicit qualification scope. Nonzero-device calls fall through. Do not broaden arithmetic beyond qualified scope.
5. fattn-common includes common.cuh CUDA declarations; compile validates. The type/stdlib headers are present. No blanket change warranted.
6. Full bridge serializes map use with g_mutex in inherited ABI entry paths; partial snippet omission is not proof of missing locking.
All suggested changes that would alter qualified arithmetic/scope rejected. Actual Windows compilation/PE/export checks still required and cannot establish GPU performance.

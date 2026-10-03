#include "fattn-kvarn-dispatch.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-kvarn-case-decl.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include "fattn-qk16-prefill.cuh"
extern "C" bool l0xre_qk16_cached_direct_entry_v1(ggml_backend_cuda_context &ctx, ggml_tensor *dst, ggml_cuda_fattn_kvarn_entry_path) {
    return l0xre_qk16_dispatch(ctx,dst);
}

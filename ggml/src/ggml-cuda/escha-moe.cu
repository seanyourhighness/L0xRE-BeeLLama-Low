#include "common.cuh"
#include "escha-moe.cuh"
#include "mmid.cuh"
#include "mma.cuh"
#include "mmvf.cuh"
#include "unary.cuh"
#include <cuda_pipeline.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "escha-dl-shim.cuh"
#if !defined(_WIN32)
#include <unistd.h>
#endif

// Fused decode + routed matmul for Escha ESCHAM experts.
//
//   y = T128(T128(x * rin) @ decode(code)) * rout
//
// Weights decode independently -- weight[p] = lut[sum_j bit(payload, dep[p][j]) << j],
// no trellis state -- so the decode is a plain gather and the whole thing is one big
// embarrassingly parallel reduction.
//
// Split into three kernels because at batch 1 there is very little natural
// parallelism: 8 slots x OC/128 column groups is 32 blocks for gate/up, which leaves
// most of the GPU idle. Slicing the IC reduction across blocks and summing the
// partials afterwards is what fills it. The partials are summed in a fixed order
// rather than with atomics so results stay bit-reproducible run to run.
//
// There are two matmul kernels, picked by batch size:
//
//   escha_matmul_partial  one row per block, reduction sliced. Right when rows are
//                         scarce, but every row decodes the weights again.
//   escha_matmul_tiled    ESCHA_ROWS rows of ONE expert per block, so a decoded
//                         weight is reused across all of them. Needs the rows
//                         grouped by expert first, which is what mm_ids_helper
//                         already does for mul_mat_id.
//
// Both write the same partial layout and share escha_finalize.

#define ESCHA_TILE     16   // decode tile is 16x16
#define ESCHA_NT      128   // threads per block, one per output column
#define ESCHA_GROUPS  (ESCHA_NT/ESCHA_TILE)
#define ESCHA_MAX_W    24   // uint32 words per payload, 48 int16 at K=3
#define ESCHA_TARGET  512   // block count we try to reach by slicing the reduction
#define ESCHA_ROWS     16   // rows of one expert per block on the batched path
// dense reuses every decoded weight across this many rows. the decode, not bandwidth, is
// the limit (see the routed path's ablation), so this is the main prefill lever: 16 -> 64
// took a perplexity pass from 47.2 s to 18.9 s. kept separate from ESCHA_ROWS so the routed
// path stays exactly as measured.
//
// But acc[R] is per thread whatever the real row count is, so a big R costs occupancy at
// batch 1 and cripples generation (R=64 measured 1.84 t/s). Generation gets its own small
// instantiation instead -- the decode work per token is the same either way, so all that
// matters there is filling the device.
#define ESCHA_ROWS_DENSE      64
#define ESCHA_ROWS_DENSE_GEN   1
#define ESCHA_GEN_MAX_ROWS    16   // at or below this, use the generation instantiation
#define ESCHA_GEN_TARGET_MUL   4
// prefill tile for the register-tiled kernel. BM*BN = TM*TN*NT, so these four fix the
// thread count too: NT = (BM/TM)*(BN/TN). BN stays 128 -- activation traffic is
// rows*IC*(OC/BN), so a narrower BN buys reuse with global bandwidth, which is a losing trade.
#define ESCHA_BM 128
#define ESCHA_BN 128
#define ESCHA_MMA_BM 128   // tensor-core prefill tile. Accumulators per thread are
#define ESCHA_MMA_BN 128   //   BM*BN/256, so BM drives register pressure and occupancy.
#define ESCHA_TM   8
#define ESCHA_TN   8
                                   //     prefill: at batch 1 there are only n_ocb blocks
                                   //     before slicing (136 for the FFN), which leaves an
                                   //     82-SM device idle. Swept 1/2/4/8/16/32 -- flat from
                                   //     10.3 to 11.3 tok/s, 4 marginally best.

// Lab-only adapter for replaying the already packed GGUF tensors through the
// independently verified Escha kernel. This does not convert, cache, or alter
// model data: the bridge receives the existing code/rin/rout device pointers.
// It is deliberately opt-in and dynamically loaded so the normal runtime has no
// dependency on PyTorch or the SGLang extension.
using escha_official_code_gemm_fn = int (*)(
    cudaStream_t, int, float *, const float *, const int16_t *, const void *, const void *,
    int, int, int, int, int, int, int);
using escha_official_code_gemm_pretransformed_fn = int (*)(
    cudaStream_t, int, void *, const void *, const int16_t *, const void *,
    int, int, int, int, int);
using escha_official_decode_gemv_raw_fn = int (*)(
    cudaStream_t, int, float *, void *, const void *, const int16_t *, const void *, const void *,
    int, int, int, int, int);
using escha_official_decode_gemv_main_raw_fn = int (*)(
    cudaStream_t, int, float *, const void *, const int16_t *, const void *,
    int, int, int, int, int);
using escha_official_decode_gemv_main_f32_raw_fn = int (*)(
    cudaStream_t, int, float *, const float *, const int16_t *, const void *,
    int, int, int, int, int);
using escha_official_bridge_error_fn = const char * (*)();

struct escha_official_bridge_api {
    escha_official_code_gemm_fn gemm = nullptr;
    escha_official_code_gemm_pretransformed_fn pretransformed = nullptr;
    escha_official_decode_gemv_raw_fn decode = nullptr;
    escha_official_decode_gemv_main_raw_fn decode_main = nullptr;
    escha_official_decode_gemv_main_f32_raw_fn decode_main_f32 = nullptr;
    escha_official_bridge_error_fn error = nullptr;
};

static bool escha_official_bridge_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_BRIDGE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static bool escha_official_raw_bridge_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_BRIDGE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static bool escha_official_raw_swiglu_fusion_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_SWIGLU_FUSION");
    return escha_official_raw_bridge_requested() &&
           value != nullptr && std::strcmp(value, "1") == 0;
}

static bool escha_official_raw_decode_bridge_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_BRIDGE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static bool escha_official_raw_decode_swiglu_fusion_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_SWIGLU_FUSION");
    return escha_official_raw_decode_bridge_requested() &&
           value != nullptr && std::strcmp(value, "1") == 0;
}

static bool escha_official_raw_decode_gate_up_overlap_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_GATE_UP_OVERLAP");
    return escha_official_raw_decode_swiglu_fusion_requested() &&
           value != nullptr && std::strcmp(value, "1") == 0;
}

static bool escha_official_raw_decode_qkv_z_overlap_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_QKV_Z_OVERLAP");
    return escha_official_raw_decode_bridge_requested() &&
           value != nullptr && std::strcmp(value, "1") == 0;
}

static bool escha_official_raw_decode_attn_qkv_overlap_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_ATTN_QKV_OVERLAP");
    return escha_official_raw_decode_bridge_requested() &&
           value != nullptr && std::strcmp(value, "1") == 0;
}

// Lab-only state for the batch-1 gate/up overlap probe.  The graph visits the
// fused gate+SiLU before the fused up+mul.  Under the opt-in selector we retain
// the gate's immutable tensor pointers, then launch gate and up together when
// the up node arrives.  The auxiliary stream rejoins the main stream before the
// up finalizer consumes gate_silu, so no work escapes the ordinary graph order.
struct escha_decode_gate_pending {
    bool valid = false;
    int device = -1;
    cudaStream_t main_stream = nullptr;
    const int16_t * code = nullptr;
    const half * rin = nullptr;
    const half * rout = nullptr;
    const float * x = nullptr;
    float * gate_silu = nullptr;
    int64_t gate_nb1 = 0;
    int64_t gate_nb2 = 0;
};

struct escha_decode_overlap_resources {
    bool initialized = false;
    int device = -1;
    cudaStream_t stream = nullptr;
    cudaEvent_t fork = nullptr;
    cudaEvent_t done = nullptr;
};

static thread_local escha_decode_gate_pending g_escha_decode_gate_pending;
static thread_local escha_decode_overlap_resources g_escha_decode_overlap_resources;

static escha_decode_overlap_resources & escha_decode_overlap_resources_get(
        int device, cudaStream_t main_stream) {
    escha_decode_overlap_resources & resources = g_escha_decode_overlap_resources;
    if (resources.initialized) {
        if (resources.device != device) {
            GGML_ABORT("escha: gate/up overlap does not support switching CUDA devices");
        }
        return resources;
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(main_stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        GGML_ABORT("escha: gate/up overlap resources must be initialized before CUDA graph capture");
    }
    CUDA_CHECK(cudaSetDevice(device));
    const char * high_priority_value =
        std::getenv("ESCHA_OFFICIAL_RAW_DECODE_OVERLAP_AUX_HIGH_PRIORITY");
    if (high_priority_value != nullptr && std::strcmp(high_priority_value, "1") == 0) {
        int least_priority;
        int greatest_priority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
        CUDA_CHECK(cudaStreamCreateWithPriority(
            &resources.stream, cudaStreamNonBlocking, greatest_priority));
    } else {
        CUDA_CHECK(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&resources.fork, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&resources.done, cudaEventDisableTiming));
    resources.device = device;
    resources.initialized = true;
    return resources;
}

static bool escha_official_raw_decode_native_epilogue_enabled() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_LEGACY_EPILOGUE");
    return value == nullptr || std::strcmp(value, "1") != 0;
}

static bool escha_official_raw_decode_f32_input_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_F32_INPUT");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static bool escha_official_raw_decode_rotated_f16_input_requested() {
    const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_ROTATED_F16_INPUT");
    return escha_official_raw_decode_f32_input_requested() &&
           value != nullptr && std::strcmp(value, "1") == 0;
}

static const escha_official_bridge_api & escha_official_bridge_load() {
    static escha_official_bridge_api api;
    static std::once_flag once;
    std::call_once(once, [] {
        const char * path = std::getenv("ESCHA_OFFICIAL_BRIDGE_LIBRARY");
        if (path == nullptr || path[0] == '\0') {
            GGML_ABORT("escha: ESCHA_OFFICIAL_BRIDGE_LIBRARY must name the lab bridge library");
        }
        void * handle = escha_dl_open(path);
        if (handle == nullptr) {
            GGML_ABORT("escha: failed to load official bridge '%s': %s", path, escha_dl_error());
        }
        api.gemm = reinterpret_cast<escha_official_code_gemm_fn>(
            escha_dl_sym(handle, "escha_official_code_gemm"));
        api.pretransformed = reinterpret_cast<escha_official_code_gemm_pretransformed_fn>(
            escha_dl_sym(handle, "escha_official_code_gemm_pretransformed"));
        api.decode = reinterpret_cast<escha_official_decode_gemv_raw_fn>(
            escha_dl_sym(handle, "escha_official_decode_gemv_raw"));
        api.decode_main = reinterpret_cast<escha_official_decode_gemv_main_raw_fn>(
            escha_dl_sym(handle, "escha_official_decode_gemv_main_raw"));
        api.decode_main_f32 = reinterpret_cast<escha_official_decode_gemv_main_f32_raw_fn>(
            escha_dl_sym(handle, "escha_official_decode_gemv_main_f32_raw"));
        api.error = reinterpret_cast<escha_official_bridge_error_fn>(
            escha_dl_sym(handle, "escha_official_bridge_error"));
        if (api.gemm == nullptr || api.pretransformed == nullptr || api.decode == nullptr ||
            api.decode_main == nullptr || api.error == nullptr) {
            GGML_ABORT("escha: official bridge '%s' is missing its C ABI", path);
        }
    });
    return api;
}

static int escha_official_bridge_acc_mode(const ggml_tensor * code, int K, int IC) {
    int acc_mode = IC <= 6144 ? 1 : 0;
    if (const char * value = std::getenv("ESCHA_OFFICIAL_BRIDGE_ACCUMULATION")) {
        if (std::strcmp(value, "fp32") == 0) {
            acc_mode = 0;
        } else if (std::strcmp(value, "k2-fp32") == 0) {
            acc_mode = K == 2 ? 0 : acc_mode;
        } else if (std::strcmp(value, "mixed") != 0) {
            GGML_ABORT("escha: unsupported ESCHA_OFFICIAL_BRIDGE_ACCUMULATION policy '%s'", value);
        }
    }
    GGML_UNUSED(code);
    return acc_mode;
}

static __global__ void escha_copy_f16_f32(const half * src, float * dst, size_t n) {
    const size_t i = 2*((size_t) blockIdx.x*blockDim.x + threadIdx.x);
    if (i + 1 < n) {
        const half2 h = reinterpret_cast<const half2 *>(src)[i/2];
        reinterpret_cast<float2 *>(dst)[i/2] = __half22float2(h);
    } else if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

static __global__ void escha_copy_f32_f16(const float * src, half * dst, size_t n) {
    const size_t i = 2*((size_t) blockIdx.x*blockDim.x + threadIdx.x);
    if (i + 1 < n) {
        const float2 f = reinterpret_cast<const float2 *>(src)[i/2];
        reinterpret_cast<half2 *>(dst)[i/2] = __floats2half2_rn(f.x, f.y);
    } else if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

// Escha codebook A, the one this checkpoint uses (its config leaves "codebook" unset,
// which eschamoe.py defaults to cbA / codebook_id 1). It is computed, not stored -- the
// same QTIP-family trick as 3INST but with its own multiplier and no addend. Recovered
// from escham_reconstruct_kernel<1, K> and checked against all 65536 entries.
// The fp16 add must stay in fp16 to match the table bit for bit.
static __device__ __forceinline__ float escha_codebook(uint32_t idx) {
    // (x & 0x8fff8fff) ^ 0x3b603b60 is one 3-input logic op; spelling it as lop3 stops
    // ptxas from splitting it across the two 16-bit lanes. immLut 0x6a = (a & b) ^ c.
    uint32_t x = idx*0xcbac1fedu;
    asm("lop3.b32 %0, %1, %2, %3, 0x6a;"
        : "=r"(x) : "r"(x), "n"(0x8fff8fffu), "n"(0x3b603b60u));
    __half2 h;
    memcpy(&h, &x, sizeof(h));
    return __half2float(__hadd(__low2half(h), __high2half(h)));
}

// as escha_codebook, but stopping at the half. The codebook's last operation is already
// an fp16 add, so the fp16 weight is exact -- only the ACTIVATIONS lose precision below.
static __device__ __forceinline__ half escha_codebook_h(uint32_t idx) {
    uint32_t x = idx*0xcbac1fedu;
    asm("lop3.b32 %0, %1, %2, %3, 0x6a;"
        : "=r"(x) : "r"(x), "n"(0x8fff8fffu), "n"(0x3b603b60u));
    __half2 h;
    memcpy(&h, &x, sizeof(h));
    return __hadd(__low2half(h), __high2half(h));
}

// in-place normalized Sylvester-Hadamard over each block of 128
static __device__ __forceinline__ void escha_hadamard_128(float * v, int n, int tid, int nt) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int idx = tid; idx < (n/128)*64; idx += nt) {
            const int blk = idx / 64;
            const int j   = idx % 64;
            const int i   = (j / len)*(2*len) + (j % len);

            float * b = v + blk*128 + i;
            const float a0 = b[0];
            const float a1 = b[len];

            b[0]   = a0 + a1;
            b[len] = a0 - a1;
        }
        __syncthreads();
    }

    const float scale = rsqrtf(128.0f);
    for (int i = tid; i < n; i += nt) {
        v[i] *= scale;
    }
    __syncthreads();
}

// u[row] = T128(x[row] * rin[expert]) -- hoisted out of the matmul so the column
// blocks do not each redo it
static __global__ void escha_rotate_in(
        const half    * __restrict__ rin,
        const float   * __restrict__ x,
        const int32_t * __restrict__ ids,
        float         * __restrict__ u,
        const int IC, const int n_x, const int n_ids,
        const int64_t nb_x1, const int64_t nb_x2,
        const int64_t nb_i0, const int64_t nb_i1) {
    extern __shared__ float s_u[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int it  = row / n_ids;
    const int is  = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);

    const half  * rin_e = rin + (int64_t) e*IC;
    const float * x_row = (const float *)((const char *) x + (int64_t)(is % n_x)*nb_x1 + it*nb_x2);

    for (int i = tid; i < IC; i += blockDim.x) {
        s_u[i] = x_row[i]*__half2float(rin_e[i]);
    }
    __syncthreads();

    escha_hadamard_128(s_u, IC, tid, blockDim.x);

    float * dst = u + (int64_t) row*IC;
    for (int i = tid; i < IC; i += blockDim.x) {
        dst[i] = s_u[i];
    }
}

// partial[slice][row][c] = sum over this slice's input tiles of u . decode(code)
template <int K>
static __global__ void escha_matmul_partial(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        const int32_t * __restrict__ ids,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_ids, const int n_rows, const int n_slices,
        const int64_t nb_i0, const int64_t nb_i1) {
    extern __shared__ char s_raw[];

    // dep is stored transposed and packed two entries per word: [b/2][r][cc]. that makes
    // the 16 threads of a group read 16 consecutive words, which is conflict-free -- the
    // natural [p][b] layout puts them 32 bytes apart and costs an 8-way bank conflict
    uint32_t * s_dep = (uint32_t *) s_raw;                 // [8][16][16]
    uint32_t * s_pay = s_dep + 8*256;                      // [ESCHA_GROUPS][ESCHA_MAX_W]

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid   = threadIdx.x;
    const int row   = blockIdx.x;
    const int ocb   = blockIdx.y;
    const int slice = blockIdx.z;

    const int it = row / n_ids;
    const int is = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);
    const int16_t * code_e = code + (int64_t) e*nit*nct*(16*K);

    for (int j = tid; j < 8*256; j += ESCHA_NT) {
        const int b2 = j / 256;
        const int p  = j % 256;
        s_dep[j] = (uint32_t) (uint16_t) dep[p*16 + 2*b2]
                 | ((uint32_t) (uint16_t) dep[p*16 + 2*b2 + 1] << 16);
    }
    __syncthreads();

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = ocb*ESCHA_GROUPS + grp;

    const int per   = nit/n_slices;
    const int ti0   = slice*per;
    const float * u_row = u + (int64_t) row*IC;

    uint32_t * pay = s_pay + grp*ESCHA_MAX_W;
    float sum = 0.0f;

    for (int ti = ti0; ti < ti0 + per; ++ti) {
        const uint32_t * src = (const uint32_t *)(code_e + (int64_t)(ti*nct + tj)*(16*K));
        for (int w = cc; w < n_wd; w += ESCHA_TILE) {
            pay[w] = src[w];
        }
        __syncwarp();

        const float * uu = u_row + ti*ESCHA_TILE;

        #pragma unroll 4
        for (int r = 0; r < ESCHA_TILE; ++r) {
            const uint32_t * d = s_dep + r*ESCHA_TILE + cc;

            uint32_t idx = 0;
            #pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;

                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }

            sum += uu[r]*escha_codebook(idx);
        }
        __syncwarp();
    }

    partial[((int64_t) slice*n_rows + row)*OC + ocb*ESCHA_NT + tid] = sum;
}

// one work item per block of the tiled kernel: up to ESCHA_ROWS compact rows of expert e.
// order does not matter -- items write disjoint output rows -- so an atomic counter is enough
static __global__ void escha_build_work(
        const int32_t * __restrict__ bounds,
        int4          * __restrict__ work,
        int32_t       * __restrict__ n_work,
        const int n_expert) {
    for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < n_expert; e += gridDim.x*blockDim.x) {
        const int lo = bounds[e];
        const int hi = bounds[e + 1];
        for (int s = lo; s < hi; s += ESCHA_ROWS) {
            work[atomicAdd(n_work, 1)] = make_int4(e, s, min(ESCHA_ROWS, hi - s), 0);
        }
    }
}

// partial[row][c] = u[row] . decode(code), for R rows sharing one expert
template <int K, int R>
static __global__ void escha_matmul_tiled(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        const int32_t * __restrict__ ids_dst,
        const int4    * __restrict__ work,
        const int32_t * __restrict__ n_work,
        float         * __restrict__ partial,
        const int IC, const int OC) {
    extern __shared__ char s_raw[];

    uint32_t * s_dep = (uint32_t *) s_raw;                             // [8][16][16]
    uint32_t * s_pay = s_dep + 8*256;                                  // [ESCHA_GROUPS][ESCHA_MAX_W]
    float    * s_u   = (float *)(s_pay + ESCHA_GROUPS*ESCHA_MAX_W);    // [R][16]

    // the grid is sized to an upper bound, so the tail blocks have nothing to do.
    // uniform across the block, so the syncs below are still safe
    if (blockIdx.x >= (unsigned) *n_work) {
        return;
    }

    const int4 w = work[blockIdx.x];
    const int e     = w.x;
    const int start = w.y;
    const int nrow  = w.z;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid = threadIdx.x;

    for (int j = tid; j < 8*256; j += ESCHA_NT) {
        const int b2 = j / 256;
        const int p  = j % 256;
        s_dep[j] = (uint32_t) (uint16_t) dep[p*16 + 2*b2]
                 | ((uint32_t) (uint16_t) dep[p*16 + 2*b2 + 1] << 16);
    }

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = blockIdx.y*ESCHA_GROUPS + grp;

    const int16_t * code_e = code + (int64_t) e*nit*nct*(16*K);
    uint32_t * pay = s_pay + grp*ESCHA_MAX_W;

    float acc[R];
#pragma unroll
    for (int m = 0; m < R; ++m) {
        acc[m] = 0.0f;
    }
    __syncthreads();

    for (int ti = 0; ti < nit; ++ti) {
        // the whole block cooperates on one 16-wide slice of u per row, then every
        // thread reads all of it -- 16 threads of a group hit the same address, so
        // the reads broadcast instead of conflicting
        for (int j = tid; j < R*ESCHA_TILE; j += ESCHA_NT) {
            const int m = j / ESCHA_TILE;
            const int r = j % ESCHA_TILE;
            s_u[j] = m < nrow ? u[(int64_t) ids_dst[start + m]*IC + ti*ESCHA_TILE + r] : 0.0f;
        }

        const uint32_t * src = (const uint32_t *)(code_e + (int64_t)(ti*nct + tj)*(16*K));
        for (int wd = cc; wd < n_wd; wd += ESCHA_TILE) {
            pay[wd] = src[wd];
        }
        __syncthreads();

#pragma unroll 4
        for (int r = 0; r < ESCHA_TILE; ++r) {
            const uint32_t * d = s_dep + r*ESCHA_TILE + cc;

            uint32_t idx = 0;
#pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;

                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }

            const float wv = escha_codebook(idx);
#pragma unroll
            for (int m = 0; m < R; ++m) {
                acc[m] += s_u[m*ESCHA_TILE + r]*wv;
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < nrow; ++m) {
        partial[(int64_t) ids_dst[start + m]*OC + blockIdx.y*ESCHA_NT + tid] = acc[m];
    }
}

// sum the slices, rotate the 128-column group, scale by rout
static __global__ void escha_finalize(
        const half    * __restrict__ rout,
        const int32_t * __restrict__ ids,
        const float   * __restrict__ partial,
        float         * __restrict__ dst,
        const int OC, const int n_ids, const int n_rows, const int n_slices,
        const int64_t nb_i0, const int64_t nb_i1,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int ocb = blockIdx.y;

    const int it = row / n_ids;
    const int is = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);
    const int c = ocb*ESCHA_NT + tid;

    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + is*nb_d1 + it*nb_d2);
    dst_row[c] = s_acc[tid]*__half2float(rout[(int64_t) e*OC + c]);
}

void ggml_cuda_op_escha_moe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * lut  = dst->src[3];
    const ggml_tensor * dep  = dst->src[4];
    const ggml_tensor * x    = dst->src[5];
    const ggml_tensor * ids  = dst->src[6];

    GGML_ASSERT(code->type == GGML_TYPE_I16 && dep->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16 && lut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);

    const int K   = code->ne[0]/16;
    const int OC  = code->ne[1]*16;
    const int IC  = code->ne[2]*16;
    const int nit = IC/ESCHA_TILE;

    const int n_expert = code->ne[3];
    const int n_ids    = ids->ne[0];
    const int n_tokens = ids->ne[1];
    const int n_rows   = n_ids*n_tokens;
    const int n_ocb    = OC/ESCHA_NT;

    // reuse only pays once a block can expect a decent share of ESCHA_ROWS rows for
    // its expert. below that the sliced kernel wins, and at batch 1 it is the only
    // one that fills the device at all
    const bool tiled = n_rows >= 8*n_expert;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> u_buf(ctx.pool(), (size_t) n_rows*IC);

    escha_rotate_in<<<n_rows, 256, IC*sizeof(float), stream>>>(
        (const half *) rin->data, (const float *) x->data, (const int32_t *) ids->data,
        u_buf.get(), IC, (int) x->ne[1], n_ids,
        x->nb[1], x->nb[2], ids->nb[0], ids->nb[1]);

    const size_t smem_dep = 8*256*sizeof(uint32_t) + ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint32_t);

    if (tiled) {
        // mm_ids_helper assumes a token uses an expert at most once, which top-k routing
        // guarantees. duplicates would desync its offsets and write out of bounds
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));

        ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_expert + 1);

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            n_expert, n_tokens, n_ids, (int) x->ne[1], (int) (ids->nb[1]/ggml_element_size(ids)), 1,
            /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());

        // an expert can end with a part-full chunk, so the item count is bounded but not known
        const int n_work_max = (n_rows + ESCHA_ROWS - 1)/ESCHA_ROWS + n_expert;

        ggml_cuda_pool_alloc<int4>    work(ctx.pool(), n_work_max);
        ggml_cuda_pool_alloc<int32_t> n_work(ctx.pool(), 1);

        CUDA_CHECK(cudaMemsetAsync(n_work.get(), 0, sizeof(int32_t), stream));
        escha_build_work<<<1, 256, 0, stream>>>(bounds.get(), work.get(), n_work.get(), n_expert);

        ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_rows*OC);

        const size_t smem = smem_dep + ESCHA_ROWS*ESCHA_TILE*sizeof(float);

        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_work_max, n_ocb), ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                u_buf.get(), ids_dst.get(), work.get(), n_work.get(), p_buf.get(), IC, OC);
        };

        switch (K) {
            case 2: launch(escha_matmul_tiled<2, ESCHA_ROWS>); break;
            case 3: launch(escha_matmul_tiled<3, ESCHA_ROWS>); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }

        escha_finalize<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
            (const half *) rout->data, (const int32_t *) ids->data, p_buf.get(), (float *) dst->data,
            OC, n_ids, n_rows, 1, ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2]);
        return;
    }

    // slice the reduction until the launch is wide enough to fill the device, but only
    // by factors that divide nit evenly
    int n_slices = 1;
    while (n_rows*n_ocb*n_slices*2 <= ESCHA_TARGET && nit % (n_slices*2) == 0) {
        n_slices *= 2;
    }

    ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_slices*n_rows*OC);

    const dim3 grid(n_rows, n_ocb, n_slices);

    auto launch = [&](auto kernel) {
        kernel<<<grid, ESCHA_NT, smem_dep, stream>>>(
            (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
            u_buf.get(), (const int32_t *) ids->data, p_buf.get(),
            IC, OC, n_ids, n_rows, n_slices, ids->nb[0], ids->nb[1]);
    };

    switch (K) {
        case 2: launch(escha_matmul_partial<2>); break;
        case 3: launch(escha_matmul_partial<3>); break;
        default: GGML_ABORT("escha: unsupported K=%d", K);
    }

    escha_finalize<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
        (const half *) rout->data, (const int32_t *) ids->data, p_buf.get(), (float *) dst->data,
        OC, n_ids, n_rows, n_slices, ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2]);
}

// ===========================================================================
// dense escha (ggml_escha_mul_mat)
//
// Same codec and same rotations as the routed path above, minus the routing: one weight
// matrix, every row goes through it. That removes the ids indirection, mm_ids_helper and
// the work list -- a block owns R consecutive rows outright. The IC reduction is still
// sliced across blocks, because at batch 1 a single row would otherwise leave the device
// mostly idle.
// ===========================================================================

// u[row] = T128(x[row] * rin)
//
// Staged through shared memory in fixed chunks rather than all of IC at once: the dense
// projections go up to IC = 17408 (mlp.down), and 17408 floats is 68 KB, well past the
// 48 KB a block gets. The rotation is independent per 128-block, so any chunk that is a
// multiple of 128 splits it exactly.
#define ESCHA_ROT_CHUNK 2048   // 8 KB of shared memory

// U is float for the scalar paths and half for the tensor-core path. Emitting half here
// rather than converting during staging is bit-identical -- the same __float2half, moved
// earlier -- and it is what lets cp.async copy activations straight into shared, since
// cp.async moves bytes verbatim and cannot convert.
template <typename U>
static __global__ void escha_rotate_in_dense(
        const half  * __restrict__ rin,
        const float * __restrict__ x,
        U           * __restrict__ u,
        const int IC, const int ne1,
        const int64_t nb_x1, const int64_t nb_x2) {
    __shared__ float s_u[ESCHA_ROT_CHUNK];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const float * x_row = (const float *)((const char *) x + (int64_t)(row % ne1)*nb_x1
                                                           + (int64_t)(row / ne1)*nb_x2);
    U * dst = u + (int64_t) row*IC;

    for (int off = 0; off < IC; off += ESCHA_ROT_CHUNK) {
        const int n = min(ESCHA_ROT_CHUNK, IC - off);

        for (int i = tid; i < n; i += blockDim.x) {
            s_u[i] = x_row[off + i]*__half2float(rin[off + i]);
        }
        __syncthreads();

        escha_hadamard_128(s_u, n, tid, blockDim.x);

        for (int i = tid; i < n; i += blockDim.x) {
            if constexpr (sizeof(U) == sizeof(half)) {
                dst[off + i] = __float2half(s_u[i]);
            } else {
                dst[off + i] = s_u[i];
            }
        }
        __syncthreads();
    }
}

// Decode-only rotation variant. One warp owns one 128-element Hadamard block, keeping
// four values per lane in registers. The first five stages stay within the warp; the
// len=32 and len=64 stages exchange the four per-lane values. This preserves the
// shared-memory kernel's pair order and float arithmetic while removing its seven
// block-wide barriers per 2048-element chunk. The host selects this only for float
// decode, never for the half-activation MMA prefill path.
template <typename U>
static __global__ void escha_rotate_in_dense_warp(
        const half  * __restrict__ rin,
        const float * __restrict__ x,
        U           * __restrict__ u,
        const int IC, const int ne1,
        const int64_t nb_x1, const int64_t nb_x2) {
    constexpr int WARP = 32;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp = tid / WARP;
    const int nwarps = blockDim.x / WARP;
    const int row = blockIdx.x;

    const float * x_row = (const float *)((const char *) x + (int64_t)(row % ne1)*nb_x1
                                                           + (int64_t)(row / ne1)*nb_x2);
    U * dst = u + (int64_t) row*IC;

    for (int off = 0; off < IC; off += ESCHA_ROT_CHUNK) {
        const int n = min(ESCHA_ROT_CHUNK, IC - off);
        const int groups = n / 128;

        for (int group = warp; group < groups; group += nwarps) {
            float v[4];

#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int idx = off + group*128 + k*WARP + lane;
                v[k] = x_row[idx] * __half2float(rin[idx]);
            }

#pragma unroll
            for (int len = 1; len < WARP; len <<= 1) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float partner = __shfl_xor_sync(0xFFFFFFFFULL, v[k], len, WARP);
                    v[k] = (lane & len) == 0 ? v[k] + partner : partner - v[k];
                }
            }

            float t;
            t = v[0]; v[0] = t + v[1]; v[1] = t - v[1];
            t = v[2]; v[2] = t + v[3]; v[3] = t - v[3];
            t = v[0]; v[0] = t + v[2]; v[2] = t - v[2];
            t = v[1]; v[1] = t + v[3]; v[3] = t - v[3];

            const float scale = rsqrtf(128.0f);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int idx = off + group*128 + k*WARP + lane;
                const float value = v[k] * scale;
                if constexpr (sizeof(U) == sizeof(half)) {
                    dst[idx] = __float2half(value);
                } else {
                    dst[idx] = value;
                }
            }
        }
    }
}

// escha's dep table is computable, so the dense kernel never reads it.
//
// The 16 bits that form a weight's codebook index are always 16 cyclically-consecutive
// positions of a bit-stream over the tile payload, where the stream visits 32-bit word 0
// first and then walks the words downwards (0, NW-1, NW-2, ...). Only the start position
// varies, and it is affine in the tile row and column:
//
//   pi(r) = (r&1) | ((r>>3)&1)<<1 | ((r>>1)&3)<<3      // bit 2 is left free for c
//   t     = pi(r) + 32*c + 4*(c>>3)
//   s     = ((32-K) - K*t) mod 256K
//
// Verified exact against both shipped tables, all 4096 entries (dep3.py). This turns
// 8 shared dep reads + 16 payload reads + ~48 bit ops per weight into two reads and a
// funnel shift, and drops the 8 KB per-block dep table that made batch 1 setup-bound.
__device__ __forceinline__ int escha_dep_pi(int r) {
    return (r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3);
}

// Register-tiled variant of the dense kernel, for prefill.
//
// The column-per-thread kernel below couples decode reuse to one thread's registers: it
// accumulates R rows in acc[R], so reusing a decode more means more registers in the SAME
// thread. At R=64 that is 255 registers with spill, 2 blocks/SM, ~17% occupancy -- and it
// is why prefill decodes at ~60 G/s while the batch-1 path manages ~372 G/s.
//
// Here the decoded weights go to SHARED memory instead, so every row group in the block
// reuses them. Reuse becomes BM (rows per BLOCK) while registers stay TM*TN (the thread's
// own output tile), which decouples the two:
//
//     BM * BN = (TM * TN) * NT
//
// BN is held at 128 on purpose: total activation traffic is rows * IC * (OC/BN), so
// narrowing BN to buy reuse would multiply global reads instead (that mistake was caught
// on paper, not in silicon -- BN=16 would have cost 8x the activation bandwidth).
//
// TN must be > 1. With one column per thread every shared read feeds exactly one MAC and
// shared bandwidth becomes the new ceiling; a TMxTN tile does TM*TN MACs per TM+TN reads.
template <int K, int BM, int BN, int TM, int TN>
static __global__ void escha_matmul_dense_tiled(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    constexpr int NT  = (BM/TM)*(BN/TN);   // threads per block
    constexpr int NTJ = BN/ESCHA_TILE;     // output tiles covered
    constexpr int NCX = BN/TN;             // threads across the column axis

    extern __shared__ char s_raw[];
    uint32_t * s_pay = (uint32_t *) s_raw;                          // [NTJ][ESCHA_MAX_W]
    float    * s_w   = (float *)(s_pay + NTJ*ESCHA_MAX_W);          // [16][BN]
    float    * s_u   = s_w + ESCHA_TILE*BN;                         // [2][16][BM] transposed

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    const int NW = 8*K;
    const int NB = 32*NW;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid  = threadIdx.x;
    const int cx   = tid % NCX;            // this thread's column strip
    const int ry   = tid / NCX;            // this thread's row strip
    const int row0 = blockIdx.x*BM;
    const int oc0  = blockIdx.y*BN;

    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    float acc[TM*TN];
#pragma unroll
    for (int i = 0; i < TM*TN; ++i) {
        acc[i] = 0.0f;
    }

    // this thread's payload slot, fixed for the whole loop. One word per thread only:
    static_assert(NTJ*((16*K)/2) <= NT, "escha: payload needs more than one word per thread");
    const bool has_pay = tid < NTJ*n_wd;
    const int  pt = tid/n_wd, pw = tid % n_wd;
    uint32_t   ppre = 0;
    if (has_pay && lo < hi) {
        ppre = ((const uint32_t *)(code + (int64_t)(lo*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
    }

    // stage tile lo's activations into buffer 0 before the loop, so that inside the loop
    // the fetch for ti+1 can be issued into the OTHER buffer and overlap this tile's work
    if (lo < hi) {
        for (int j = tid; j < BM*ESCHA_TILE; j += NT) {
            const int m = j / ESCHA_TILE, r = j % ESCHA_TILE;
            const int row = row0 + m;
            s_u[r*BM + m] = row < n_rows ? u[(int64_t) row*IC + lo*ESCHA_TILE + r] : 0.0f;
        }
    }

    for (int ti = lo; ti < hi; ++ti) {
        float * su_cur = s_u + (((ti - lo) & 1)      )*(ESCHA_TILE*BM);
        float * su_nxt = s_u + (((ti - lo) & 1) ^ 1  )*(ESCHA_TILE*BM);
        // publish the payload fetched last round, then issue the next fetch before the
        // barrier, so the global latency overlaps the decode instead of stalling every warp
        if (has_pay) {
            s_pay[pt*ESCHA_MAX_W + pw] = ppre;
        }
        if (has_pay && ti + 1 < hi) {
            ppre = ((const uint32_t *)(code + (int64_t)((ti + 1)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
        }
        __syncthreads();

        // next tile's activations go to the other buffer: no barrier separates them from
        // the reads of su_cur below, and the barrier at the top of the next iteration is
        // what makes them visible
        if (ti + 1 < hi) {
            for (int j = tid; j < BM*ESCHA_TILE; j += NT) {
                const int m = j / ESCHA_TILE, r = j % ESCHA_TILE;
                const int row = row0 + m;
                su_nxt[r*BM + m] = row < n_rows ? u[(int64_t) row*IC + (ti + 1)*ESCHA_TILE + r] : 0.0f;
            }
        }

        // decode this input tile's 16 x BN weights once, for the whole block
        for (int j = tid; j < ESCHA_TILE*BN; j += NT) {
            const int r = j / BN, c = j % BN;
            const uint32_t * pay = s_pay + (c/ESCHA_TILE)*ESCHA_MAX_W;
            const int ccl = c % ESCHA_TILE;

            int sp = ((32 - K) - K*(escha_dep_pi(r) + 32*ccl + 4*(ccl >> 3))) % NB;
            if (sp < 0) {
                sp += NB;
            }
            const int g0 = sp >> 5;
            const int w0 = g0 ? (NW - g0) : 0;
            const int w1 = w0 ? (w0 - 1)  : (NW - 1);

            s_w[r*BN + c] = escha_codebook(__funnelshift_r(pay[w0], pay[w1], sp & 31) & 0xffffu);
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < ESCHA_TILE; ++r) {
            float a[TM], b[TN];
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                a[m] = su_cur[r*BM + ry*TM + m];
            }
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                b[n] = s_w[r*BN + cx*TN + n];
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m*TN + n] += a[m]*b[n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int row = row0 + ry*TM + m;
        if (row < n_rows) {
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                partial[((int64_t) sl*n_rows + row)*OC + oc0 + cx*TN + n] = acc[m*TN + n];
            }
        }
    }
}


// Prefill on tensor cores. Same decode as escha_matmul_dense_tiled, but the GEMM runs on
// m16n8k16 HMMA with fp32 accumulate, which on GA102 is 71 TFLOPS against 35.6 for FP32 FMA.
//
// Two layout changes fall out of the fragment shapes, and both are cheap:
//   s_u becomes [m][k] (was [k][m]) -- which makes staging a straight contiguous copy
//   s_w becomes [n][k] (was [k][n]) -- B is the ".col" operand of mma.row.col
//
// The weights stay exact (escha_codebook_h). The ACTIVATIONS are rounded to fp16, which is
// what escha's own runtime does, and costs rel_rms ~2.1e-4 against the fp32 reference.
template <int K, int BM, int BN, bool FUSE_FINALIZE = false, bool DIRECT_B = false, bool FP16_ACC = false>
static __global__ void __launch_bounds__(256, 1) escha_matmul_dense_tiled_mma(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const half    * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices,
        const half  * __restrict__ rout,
        float       * __restrict__ dst,
        const int ne1, const int64_t nb_d1, const int64_t nb_d2) {
#ifdef TURING_MMA_AVAILABLE
    constexpr int NT   = 256;
    constexpr int NW   = NT/32;          // warps
    constexpr int WN   = 2;              // warps across the column axis
    constexpr int WM   = NW/WN;          // warps down the row axis
    constexpr int MT   = BM/16/WM;       // 16-row accumulator tiles per warp
    constexpr int NTT  = BN/8/WN;        // 8-col accumulator tiles per warp
    constexpr int NTJ  = BN/ESCHA_TILE;  // output tiles whose payload this block holds

    extern __shared__ char s_raw[];
    uint2    * s_pay = (uint2 *) s_raw;                               // [NTJ][ESCHA_MAX_W] pairs
    half     * s_u   = (half *)(s_pay + NTJ*ESCHA_MAX_W);             // [2][BM][16]
    half     * s_w   = s_u + 2*BM*ESCHA_TILE;                         // [BN][16]

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    const int NWD = 8*K;
    const int NB  = 32*NWD;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int lane = threadIdx.x;          // must stay the lane: mma.cuh indexes on it
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;        // flat id, for the layout-agnostic staging loops
    const int row0 = blockIdx.x*BM;
    const int oc0  = blockIdx.y*BN;

    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    const int wm   = warp / WN;
    const int wn   = warp % WN;

    constexpr int DPT = (ESCHA_TILE*BN)/NT;   // weights this thread decodes per tile
    static_assert(NT % ESCHA_TILE == 0,        "escha: r would not be thread-invariant");
    static_assert((ESCHA_TILE*BN) % NT == 0,   "escha: ragged decode assignment");
    static_assert(NT/ESCHA_TILE <= ESCHA_TILE, "escha: ccl would not be thread-invariant");

    // cp.async moves 16 bytes = 8 halves per thread; BM*ESCHA_TILE halves is exactly
    // NT*8 at BM=128/NT=256, so every thread issues one copy and none loops.
    constexpr int CPB = 16;                              // bytes per thread per tile
    static_assert(BM*ESCHA_TILE*sizeof(half) == NT*CPB,  "escha: activation copy is ragged");
    const int cp_m  = tid / (ESCHA_TILE*sizeof(half)/CPB);   // row this thread copies into
    const int cp_h  = (tid % (ESCHA_TILE*sizeof(half)/CPB))*(CPB/sizeof(half));

    const int dr   = tid % ESCHA_TILE;         // this thread's r, every k, every tile
    const int dccl = tid / ESCHA_TILE;         // and its column within the 16-wide tile
    int dsp = ((32 - K) - K*(escha_dep_pi(dr) + 32*dccl + 4*(dccl >> 3))) % NB;
    if (dsp < 0) {
        dsp += NB;
    }
    const int dg0 = dsp >> 5;
    const int dw0 = dg0 ? (NWD - dg0) : 0;
    const int dw1 = dw0 ? (dw0 - 1)   : (NWD - 1);
    const int dsh = dsp & 31;

    typedef ggml_cuda_mma::tile<16, 8, float> tile_c;
    typedef ggml_cuda_mma::tile<16, 8, half2> tile_a;
    typedef ggml_cuda_mma::tile<8,  8, half2> tile_b;
    typedef ggml_cuda_mma::tile<16, 4, half2> tile_ah;

    tile_c acc[MT][NTT];
    tile_ah acc16[MT][NTT];

    // one payload word per thread, held a tile ahead
    static_assert(NTJ*((16*K)/2) <= NT, "escha: payload needs more than one word per thread");
    const bool has_pay = tid < NTJ*n_wd;
    const int  pt = tid/n_wd, pw = tid % n_wd;
    uint32_t   ppre = 0;
    if (has_pay && lo < hi) {
        ppre = ((const uint32_t *)(code + (int64_t)(lo*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
    }

    // activations for tile lo into buffer 0
    if (lo < hi) {
        {
            const int row = row0 + cp_m;
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(s_u + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + lo*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
        }
        __pipeline_commit();
    }

    for (int ti = lo; ti < hi; ++ti) {
        half * su_cur = s_u + (((ti - lo) & 1)     )*(BM*ESCHA_TILE);
        half * su_nxt = s_u + (((ti - lo) & 1) ^ 1 )*(BM*ESCHA_TILE);

        if (has_pay) {
            // word pw is the high half of pair pw and the low half of pair pw+1
            s_pay[pt*ESCHA_MAX_W + pw].y = ppre;
            s_pay[pt*ESCHA_MAX_W + (pw + 1 == NWD ? 0 : pw + 1)].x = ppre;
        }
        if (has_pay && ti + 1 < hi) {
            ppre = ((const uint32_t *)(code + (int64_t)((ti + 1)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
        }
        // the copy for THIS tile was committed last round; drain it before the barrier
        // that publishes s_pay, so su_cur is visible to every warp below
        __pipeline_wait_prior(0);
        __syncthreads();

        if (ti + 1 < hi) {
            const int row = row0 + cp_m;
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(su_nxt + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + (ti + 1)*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
            __pipeline_commit();
        }

        if constexpr (!DIRECT_B) {
            // Control: decode into shared [n][k], then load MMA B fragments.
#pragma unroll
            for (int k = 0; k < DPT; ++k) {
                const uint2 * pay = s_pay + k*ESCHA_MAX_W;
                const int c = dccl + ESCHA_TILE*k;
                s_w[c*ESCHA_TILE + dr] =
                    escha_codebook_h(__funnelshift_r(pay[dw0].y, pay[dw0].x, dsh) & 0xffffu);
            }
            __syncthreads();
        }

        {
            const half2 * su2 = (const half2 *) su_cur;   // [BM][8] half2
            const half2 * sw2 = (const half2 *) s_w;      // [BN][8] half2

            tile_a A[MT];
            tile_b B[NTT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                ggml_cuda_mma::load_ldmatrix(A[i], su2 + (size_t)(wm*(16*MT) + i*16)*8, 8);
            }
#pragma unroll
            for (int j = 0; j < NTT; ++j) {
                if constexpr (DIRECT_B) {
                    // The CUDA tile descriptor names the two fp16 values held by each
                    // physical half2 register. Decode those values directly from the
                    // packed payload, bypassing the shared-B round trip and its barrier.
#pragma unroll
                    for (int l = 0; l < tile_b::ne; ++l) {
                        const int n = wn*(8*NTT) + j*8 + tile_b::get_i(l);
                        const int rp = tile_b::get_j(l);
                        half values[2];
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            const int r = 2*rp + q;
                            const int ccl = n % ESCHA_TILE;
                            const uint2 * pay = s_pay + (n/ESCHA_TILE)*ESCHA_MAX_W;
                            int sp = ((32 - K) - K*(escha_dep_pi(r) + 32*ccl + 4*(ccl >> 3))) % NB;
                            if (sp < 0) {
                                sp += NB;
                            }
                            const int g0 = sp >> 5;
                            const int w0 = g0 ? (NWD - g0) : 0;
                            values[q] = escha_codebook_h(
                                __funnelshift_r(pay[w0].y, pay[w0].x, sp & 31) & 0xffffu);
                        }
                        B[j].x[l] = __halves2half2(values[0], values[1]);
                    }
                } else {
                    ggml_cuda_mma::load_ldmatrix(B[j], sw2 + (size_t)(wn*(8*NTT) + j*8)*8, 8);
                }
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j = 0; j < NTT; ++j) {
                    if constexpr (FP16_ACC) {
                        ggml_cuda_mma::mma(acc16[i][j], A[i], B[j]);
                    } else {
                        ggml_cuda_mma::mma(acc[i][j], A[i], B[j]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (FUSE_FINALIZE) {
        // n_slices == 1 is enforced by the host dispatch.  The two activation buffers
        // are dead after the final HMMA, so reuse the first one as a 16x128 float tile.
        // This is 2048 floats (8 KiB), exactly the size of one row group and does not
        // alter the decode, activation conversion, HMMA, or accumulation order above.
        static_assert(BN == ESCHA_NT && BM % ESCHA_TILE == 0,
                      "escha fused finalize requires 16x128 row groups");
        float * s_ep = reinterpret_cast<float *>(s_u);

        for (int group = 0; group < BM/ESCHA_TILE; ++group) {
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j = 0; j < NTT; ++j) {
#pragma unroll
                    for (int l = 0; l < tile_c::ne; ++l) {
                        const int m = wm*(16*MT) + i*16 + tile_c::get_i(l);
                        const int n = wn*(8*NTT) + j*8 + tile_c::get_j(l);
                        if (m/ESCHA_TILE == group) {
                            s_ep[(m - group*ESCHA_TILE)*BN + n] = acc[i][j].x[l];
                        }
                    }
                }
            }
            __syncthreads();

            // Publish the complete 16x128 tile, then apply the same normalized
            // Hadamard used by escha_finalize_dense.  The helper supplies a barrier
            // between butterfly stages and after normalization before any lane reads.
            // tid is flattened across the 32x8 MMA block; blockDim.x alone is
            // only the lane dimension and would make eight threads share each
            // butterfly index.
            escha_hadamard_128(s_ep, ESCHA_TILE*ESCHA_NT, tid,
                               blockDim.x*blockDim.y);

            if (tid < BN) {
                const int c = oc0 + tid;
                for (int r = 0; r < ESCHA_TILE; ++r) {
                    const int row = row0 + group*ESCHA_TILE + r;
                    if (row < n_rows) {
                        float * dst_row = (float *)((char *) dst
                                                   + (int64_t)(row % ne1)*nb_d1
                                                   + (int64_t)(row / ne1)*nb_d2);
                        dst_row[c] = s_ep[r*BN + tid]*__half2float(rout[c]);
                    }
                }
            }
            __syncthreads();
        }
    } else {
#pragma unroll
        for (int i = 0; i < MT; ++i) {
#pragma unroll
            for (int j = 0; j < NTT; ++j) {
                if constexpr (FP16_ACC) {
#pragma unroll
                    for (int l = 0; l < 2; ++l) {
                        const half2 v = acc16[i][j].x[l];
                        const int m0 = wm*(16*MT) + i*16 + tile_c::get_i(2*l + 0);
                        const int n0 = wn*(8*NTT)  + j*8  + tile_c::get_j(2*l + 0);
                        const int m1 = wm*(16*MT) + i*16 + tile_c::get_i(2*l + 1);
                        const int n1 = wn*(8*NTT)  + j*8  + tile_c::get_j(2*l + 1);
                        const int row_a = row0 + m0;
                        const int row_b = row0 + m1;
                        if (row_a < n_rows) {
                            partial[((int64_t) sl*n_rows + row_a)*OC + oc0 + n0] = __half2float(v.x);
                        }
                        if (row_b < n_rows) {
                            partial[((int64_t) sl*n_rows + row_b)*OC + oc0 + n1] = __half2float(v.y);
                        }
                    }
                } else {
#pragma unroll
                    for (int l = 0; l < tile_c::ne; ++l) {
                        const int m   = wm*(16*MT) + i*16 + tile_c::get_i(l);
                        const int n   = wn*(8*NTT) + j*8  + tile_c::get_j(l);
                        const int row = row0 + m;
                        if (row < n_rows) {
                            partial[((int64_t) sl*n_rows + row)*OC + oc0 + n] = acc[i][j].x[l];
                        }
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(code, lut, dep, u, partial, IC, OC, n_rows, n_slices,
                     rout, dst, ne1, nb_d1, nb_d2);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

template <int K, int BM, int BN>
static __global__ void __launch_bounds__(256, 1) escha_matmul_dense_tiled_mma_pipe(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const half    * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
#ifdef TURING_MMA_AVAILABLE
    constexpr int NT   = 256;
    constexpr int NW   = NT/32;          // warps
    constexpr int WN   = 2;              // warps across the column axis
    constexpr int WM   = NW/WN;          // warps down the row axis
    constexpr int MT   = BM/16/WM;       // 16-row accumulator tiles per warp
    constexpr int NTT  = BN/8/WN;        // 8-col accumulator tiles per warp
    constexpr int NTJ  = BN/ESCHA_TILE;  // output tiles whose payload this block holds

    extern __shared__ char s_raw[];
    // C1 pipelined layout: s_pay triple-buffered (stage/decode/consume pipeline),
    // s_u and s_w double-buffered so decode(t+1) overlaps HMMA(t) without hazards.
    uint2    * s_pay = (uint2 *) s_raw;                               // [3][NTJ][ESCHA_MAX_W] pairs
    half     * s_u   = (half *)(s_pay + 3*NTJ*ESCHA_MAX_W);           // [2][BM][16]
    half     * s_w   = s_u + 2*BM*ESCHA_TILE;                         // [2][BN][16]

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    const int NWD = 8*K;
    const int NB  = 32*NWD;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int lane = threadIdx.x;          // must stay the lane: mma.cuh indexes on it
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;        // flat id, for the layout-agnostic staging loops
    const int row0 = blockIdx.x*BM;
    const int oc0  = blockIdx.y*BN;

    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    const int wm   = warp / WN;
    const int wn   = warp % WN;

    constexpr int DPT = (ESCHA_TILE*BN)/NT;   // weights this thread decodes per tile
    static_assert(NT % ESCHA_TILE == 0,        "escha: r would not be thread-invariant");
    static_assert((ESCHA_TILE*BN) % NT == 0,   "escha: ragged decode assignment");
    static_assert(NT/ESCHA_TILE <= ESCHA_TILE, "escha: ccl would not be thread-invariant");

    // cp.async moves 16 bytes = 8 halves per thread; BM*ESCHA_TILE halves is exactly
    // NT*8 at BM=128/NT=256, so every thread issues one copy and none loops.
    constexpr int CPB = 16;                              // bytes per thread per tile
    static_assert(BM*ESCHA_TILE*sizeof(half) == NT*CPB,  "escha: activation copy is ragged");
    const int cp_m  = tid / (ESCHA_TILE*sizeof(half)/CPB);   // row this thread copies into
    const int cp_h  = (tid % (ESCHA_TILE*sizeof(half)/CPB))*(CPB/sizeof(half));

    const int dr   = tid % ESCHA_TILE;         // this thread's r, every k, every tile
    const int dccl = tid / ESCHA_TILE;         // and its column within the 16-wide tile
    int dsp = ((32 - K) - K*(escha_dep_pi(dr) + 32*dccl + 4*(dccl >> 3))) % NB;
    if (dsp < 0) {
        dsp += NB;
    }
    const int dg0 = dsp >> 5;
    const int dw0 = dg0 ? (NWD - dg0) : 0;
    const int dw1 = dw0 ? (dw0 - 1)   : (NWD - 1);
    const int dsh = dsp & 31;

    typedef ggml_cuda_mma::tile<16, 8, float> tile_c;
    typedef ggml_cuda_mma::tile<16, 8, half2> tile_a;
    typedef ggml_cuda_mma::tile<8,  8, half2> tile_b;

    tile_c acc[MT][NTT];

    // one payload word per thread, held a tile ahead
    static_assert(NTJ*((16*K)/2) <= NT, "escha: payload needs more than one word per thread");
    const bool has_pay = tid < NTJ*n_wd;
    const int  pt = tid/n_wd, pw = tid % n_wd;
    uint32_t   ppre = 0;
    if (has_pay && lo < hi) {
        ppre = ((const uint32_t *)(code + (int64_t)(lo*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
    }

    // C1 pipelined schedule: 2 barriers per K-tile (down from 3), decode(t+1) interleaved
    // after HMMA(t) issues so the tensor pipe drains while decode ALU issues.
    // Per iteration ti (buffer b = (ti-lo)&1):
    //   wait u[ti] -> barrier(1) [cross-thread u visibility + prev publish] ->
    //   issue cp.async u[ti+1] -> HMMA(ti, s_w[b]) -> decode(ti+1, s_w[b^1]) ->
    //   stage s_pay[(ti+2)%3] -> barrier(2) [publish decode+staging, release buffers].
    // Hazards avoided by buffer disjointness: s_w[b] read (HMMA) vs s_w[b^1] written (decode);
    // s_pay[(ti+1)%3] read (decode) vs s_pay[(ti+2)%3] written (stage).
    if (lo < hi) {
        // stage payload tiles lo (slot 0) and lo+1 (slot 1); ppre then holds tile lo+2
        if (has_pay) {
            const uint32_t p0 = ppre;
            const uint32_t p1 = (lo + 1 < hi)
                ? ((const uint32_t *)(code + (int64_t)((lo + 1)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw]
                : 0u;
            s_pay[0*NTJ*ESCHA_MAX_W + pt*ESCHA_MAX_W + pw].y = p0;
            s_pay[0*NTJ*ESCHA_MAX_W + pt*ESCHA_MAX_W + (pw + 1 == NWD ? 0 : pw + 1)].x = p0;
            s_pay[1*NTJ*ESCHA_MAX_W + pt*ESCHA_MAX_W + pw].y = p1;
            s_pay[1*NTJ*ESCHA_MAX_W + pt*ESCHA_MAX_W + (pw + 1 == NWD ? 0 : pw + 1)].x = p1;
            if (lo + 2 < hi) {
                ppre = ((const uint32_t *)(code + (int64_t)((lo + 2)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
            }
        }
        __syncthreads();   // publish s_pay[0..1] before the cooperative decode below reads them
        {
            const int row = row0 + cp_m;
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(s_u + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + lo*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
        }
        __pipeline_commit();
        // decode tile lo into s_w[0] now: the first loop iteration's HMMA consumes it and
        // there is no earlier decode step for it (the loop only decodes tile ti+1).
        {
#pragma unroll
            for (int k = 0; k < DPT; ++k) {
                const uint2 * pay = s_pay + 0*NTJ*ESCHA_MAX_W + k*ESCHA_MAX_W;
                const int c = dccl + ESCHA_TILE*k;
                s_w[c*ESCHA_TILE + dr] =
                    escha_codebook_h(__funnelshift_r(pay[dw0].y, pay[dw0].x, dsh) & 0xffffu);
            }
        }
        __syncthreads();   // publishes s_pay[0..1], s_u[lo], s_w[0] to all warps
    }

    for (int ti = lo; ti < hi; ++ti) {
        const int b  = (ti - lo) & 1;
        const int b1 = b ^ 1;
        half * su_cur = s_u + b *(BM*ESCHA_TILE);
        half * sw_cur = s_w + b *(BN*ESCHA_TILE);
        half * sw_nxt = s_w + b1*(BN*ESCHA_TILE);
        uint2 * pay_cur = s_pay + ((ti + 1 - lo) % 3)*NTJ*ESCHA_MAX_W;   // payload for tile ti+1
        uint2 * pay_nxt = s_pay + ((ti + 2 - lo) % 3)*NTJ*ESCHA_MAX_W;   // payload for tile ti+2

        // this thread's u[ti] cp.async group has landed; barrier makes every thread's
        // writes visible and publishes last iteration's decode + payload staging.
        __pipeline_wait_prior(0);
        __syncthreads();

        // issue the activation copy for ti+1 immediately after the barrier so it has a
        // full tile window to land before it is drained at the top of the next iteration.
        if (ti + 1 < hi) {
            const int row = row0 + cp_m;
            const int src_row = row < n_rows ? row : 0;
            half * su_nxt = s_u + b1*(BM*ESCHA_TILE);
            __pipeline_memcpy_async(su_nxt + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + (ti + 1)*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
            __pipeline_commit();
        }

        // tensor-core work for tile ti on s_w[b]
        {
            const half2 * su2 = (const half2 *) su_cur;
            const half2 * sw2 = (const half2 *) sw_cur;

            tile_a A[MT];
            tile_b B[NTT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                ggml_cuda_mma::load_ldmatrix(A[i], su2 + (size_t)(wm*(16*MT) + i*16)*8, 8);
            }
#pragma unroll
            for (int j = 0; j < NTT; ++j) {
                ggml_cuda_mma::load_ldmatrix(B[j], sw2 + (size_t)(wn*(8*NTT) + j*8)*8, 8);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j = 0; j < NTT; ++j) {
                    ggml_cuda_mma::mma(acc[i][j], A[i], B[j]);
                }
            }
        }

        // decode tile ti+1 into s_w[b^1]; issues after the HMMA stream so the tensor pipe
        // drains while the ALU/decode instructions issue. Disjoint buffer: no hazard.
        if (ti + 1 < hi) {
#pragma unroll
            for (int k = 0; k < DPT; ++k) {
                const uint2 * pay = pay_cur + k*ESCHA_MAX_W;
                const int c = dccl + ESCHA_TILE*k;
                sw_nxt[c*ESCHA_TILE + dr] =
                    escha_codebook_h(__funnelshift_r(pay[dw0].y, pay[dw0].x, dsh) & 0xffffu);
            }
        }

        // stage the payload word for tile ti+2 (ppre already holds it); then prefetch ti+3.
        if (has_pay && ti + 2 < hi) {
            pay_nxt[pt*ESCHA_MAX_W + pw].y = ppre;
            pay_nxt[pt*ESCHA_MAX_W + (pw + 1 == NWD ? 0 : pw + 1)].x = ppre;
            if (ti + 3 < hi) {
                ppre = ((const uint32_t *)(code + (int64_t)((ti + 3)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
            }
        }

        // publish: s_w[b^1] (decode) and s_pay[(ti+2)%3] (staging) visible to all warps;
        // releases s_u[b] and s_w[b] for reuse at iteration ti+2 / ti+1 respectively.
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NTT; ++j) {
#pragma unroll
            for (int l = 0; l < tile_c::ne; ++l) {
                const int m   = wm*(16*MT) + i*16 + tile_c::get_i(l);
                const int n   = wn*(8*NTT) + j*8  + tile_c::get_j(l);
                const int row = row0 + m;
                if (row < n_rows) {
                    partial[((int64_t) sl*n_rows + row)*OC + oc0 + n] = acc[i][j].x[l];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(code, lut, dep, u, partial, IC, OC, n_rows, n_slices);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

// partial[slice][row][c] = sum over this slice's input tiles of u . decode(code)
template <int K, int R>
static __global__ void escha_matmul_dense(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    extern __shared__ char s_raw[];

    // every decode wants the adjacent pair (pay[w0-1], pay[w0]), so hold the payload AS
    // overlapping pairs: one aligned LDS.64 then replaces the two LDS the funnel shift needed
    uint2 * s_pay = (uint2 *) s_raw;                                   // [ESCHA_GROUPS][ESCHA_MAX_W]
    float * s_u   = (float *)(s_pay + ESCHA_GROUPS*ESCHA_MAX_W);       // [R][16]

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    const int NW = 8*K;          // 32-bit words in a tile payload
    const int NB = 32*NW;        // bits

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid   = threadIdx.x;
    const int start = blockIdx.x*R;
    const int nrow  = min(R, n_rows - start);

    // this block's share of the input tiles
    const int sl  = blockIdx.z;
    const int lo  = (int) (((int64_t) nit*sl)/n_slices);
    const int hi  = (int) (((int64_t) nit*(sl + 1))/n_slices);

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = blockIdx.y*ESCHA_GROUPS + grp;

    uint2 * pay = s_pay + grp*ESCHA_MAX_W;

    // start position for this thread's column, before the per-row term
    int s0 = ((32 - K) - K*(32*cc + 4*(cc >> 3))) % NB;
    if (s0 < 0) {
        s0 += NB;
    }

    float acc[R];
#pragma unroll
    for (int m = 0; m < R; ++m) {
        acc[m] = 0.0f;
    }
    // At R == 1 the block reads one row, so its entire slice of u fits in shared and is
    // staged once here -- the per-tile staging below then disappears, and with it the
    // block-wide barrier that ordered it. This is what their 10,240-byte allocation is.
    if constexpr (R == 1) {
        const float * u_row = u + (int64_t) start*IC + (int64_t) lo*ESCHA_TILE;
        const int n_stage = (hi - lo)*ESCHA_TILE;
        for (int j = tid; j < n_stage; j += ESCHA_NT) {
            s_u[j] = u_row[j];
        }
    }
    // payload words this thread owns, and the tile fetched one iteration ahead
    constexpr int NPW = (8*K + ESCHA_TILE - 1)/ESCHA_TILE;
    uint32_t pre[NPW];
    if (lo < hi) {
        const uint32_t * s0p = (const uint32_t *)(code + (int64_t)(lo*nct + tj)*(16*K));
#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                pre[i] = s0p[wd];
            }
        }
    }
    __syncthreads();

    for (int ti = lo; ti < hi; ++ti) {
        // as in the routed kernel: the block cooperates on one 16-wide slice of u per row,
        // and the 16 threads of a group then read the same entry, so it broadcasts
        if constexpr (R != 1) {
            for (int j = tid; j < R*ESCHA_TILE; j += ESCHA_NT) {
                const int m = j / ESCHA_TILE;
                const int r = j % ESCHA_TILE;
                s_u[j] = m < nrow ? u[(int64_t)(start + m)*IC + ti*ESCHA_TILE + r] : 0.0f;
            }
        }

        // publish the tile fetched last round, then issue the next fetch immediately: the
        // LDG then overlaps this tile's decode instead of stalling in front of it, which is
        // what the LDG -> dependent STS pair at the top of the loop was costing
#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                // word wd is the high half of pair wd and the low half of pair wd+1
                pay[wd].y = pre[i];
                pay[wd + 1 == NW ? 0 : wd + 1].x = pre[i];
            }
        }
        if (ti + 1 < hi) {
            const uint32_t * nxt = (const uint32_t *)(code + (int64_t)((ti + 1)*nct + tj)*(16*K));
#pragma unroll
            for (int i = 0; i < NPW; ++i) {
                const int wd = cc + i*ESCHA_TILE;
                if (wd < n_wd) {
                    pre[i] = nxt[wd];
                }
            }
        }
        if constexpr (R == 1) { __syncwarp(); } else { __syncthreads(); }

        const float * uu = s_u + (ti - lo)*ESCHA_TILE;

        // generation unrolls fully so pi(r) folds to a compile-time constant; prefill keeps
        // a partial unroll, where acc[R] already claims the registers
#pragma unroll (R <= 8 ? 16 : 4)
        for (int r = 0; r < ESCHA_TILE; ++r) {
            // K*pi(r) <= 81 < NB, so one conditional add restores the range
            int sp = s0 - K*escha_dep_pi(r);
            if (sp < 0) {
                sp += NB;
            }

            const int g0 = sp >> 5;
            const int w0 = g0 ? (NW - g0) : 0;
            const int w1 = w0 ? (w0 - 1)  : (NW - 1);

            const uint2 p = pay[w0];
            GGML_UNUSED(w1);
            const uint32_t idx = __funnelshift_r(p.y, p.x, sp & 31) & 0xffffu;

            const float wv = escha_codebook(idx);
            if constexpr (R == 1) {
                acc[0] += uu[r]*wv;
            } else {
#pragma unroll
                for (int m = 0; m < R; ++m) {
                    acc[m] += s_u[m*ESCHA_TILE + r]*wv;
                }
            }
        }
        if constexpr (R == 1) { __syncwarp(); } else { __syncthreads(); }
    }

    for (int m = 0; m < nrow; ++m) {
        partial[((int64_t) sl*n_rows + start + m)*OC + blockIdx.y*ESCHA_NT + tid] = acc[m];
    }
}

template <int K, int R, bool CHUNKU, bool PFP, bool SCHED>
static __global__ void escha_matmul_dense_adv(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    extern __shared__ char s_raw[];

    // every decode wants the adjacent pair (pay[w0-1], pay[w0]), so hold the payload AS
    // overlapping pairs: one aligned LDS.64 then replaces the two LDS the funnel shift needed
    uint2 * s_pay = (uint2 *) s_raw;                                   // [ESCHA_GROUPS][ESCHA_MAX_W]
    float * s_u   = (float *)(s_pay + ESCHA_GROUPS*ESCHA_MAX_W);       // [R][16]

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    const int NW = 8*K;          // 32-bit words in a tile payload
    const int NB = 32*NW;        // bits

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid   = threadIdx.x;
    const int start = blockIdx.x*R;
    const int nrow  = min(R, n_rows - start);

    // this block's share of the input tiles
    const int sl  = blockIdx.z;
    const int lo  = (int) (((int64_t) nit*sl)/n_slices);
    const int hi  = (int) (((int64_t) nit*(sl + 1))/n_slices);

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = blockIdx.y*ESCHA_GROUPS + grp;

    uint2 * pay = s_pay + grp*ESCHA_MAX_W;

    // start position for this thread's column, before the per-row term
    int s0 = ((32 - K) - K*(32*cc + 4*(cc >> 3))) % NB;
    if (s0 < 0) {
        s0 += NB;
    }

    float acc[R];
#pragma unroll
    for (int m = 0; m < R; ++m) {
        acc[m] = 0.0f;
    }
    // At R == 1 the block reads one row, so its entire slice of u fits in shared and is
    // staged once here -- the per-tile staging below then disappears, and with it the
    // block-wide barrier that ordered it. This is what their 10,240-byte allocation is.
    if constexpr (R == 1) {
        const float * u_row = u + (int64_t) start*IC + (int64_t) lo*ESCHA_TILE;
        const int n_stage = (hi - lo)*ESCHA_TILE;
        for (int j = tid; j < n_stage; j += ESCHA_NT) {
            s_u[j] = u_row[j];
        }
    }
    // payload words this thread owns, and the tile fetched one iteration ahead
    constexpr int NPW = (8*K + ESCHA_TILE - 1)/ESCHA_TILE;
    uint32_t pre[NPW];
    if (lo < hi) {
        const uint32_t * s0p = (const uint32_t *)(code + (int64_t)(lo*nct + tj)*(16*K));
#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                pre[i] = s0p[wd];
            }
        }
    }
    // per-(thread, r) payload window schedule: [4:0] funnel shift, [9:5] pair index
    uint32_t sched[SCHED ? ESCHA_TILE : 1];
    if constexpr (SCHED) {
#pragma unroll
        for (int rs = 0; rs < ESCHA_TILE; ++rs) {
            int sp = s0 - K*escha_dep_pi(rs);
            if (sp < 0) {
                sp += NB;
            }
            const int g0 = sp >> 5;
            const int w0 = g0 ? (NW - g0) : 0;
            sched[rs] = ((uint32_t) w0 << 5) | (uint32_t) (sp & 31);
        }
    }

    __syncthreads();

    constexpr int UCHUNK = 16;

    for (int cbase = lo; cbase < hi; cbase += UCHUNK) {
    const int cend = MIN(hi, cbase + UCHUNK);
    if constexpr (R != 1 && CHUNKU) {
        // close the previous chunk: every warp must be done reading s_u before it is rewritten
        __syncthreads();
        // one chunk of u per row, laid out [chunk tile][row][ESCHA_TILE]
        const int nc = cend - cbase;
        for (int j = tid; j < nc*R*ESCHA_TILE; j += ESCHA_NT) {
            const int t   = j / (R*ESCHA_TILE);
            const int rem = j - t*(R*ESCHA_TILE);
            const int m   = rem / ESCHA_TILE;
            const int r   = rem - m*ESCHA_TILE;
            s_u[(size_t) t*R*ESCHA_TILE + m*ESCHA_TILE + r] =
                m < nrow ? u[(int64_t)(start + m)*IC + (int64_t)(cbase + t)*ESCHA_TILE + r] : 0.0f;
        }
        __syncthreads();
    }
    for (int ti = cbase; ti < cend; ++ti) {
        // as in the routed kernel: the block cooperates on one 16-wide slice of u per row,
        // and the 16 threads of a group then read the same entry, so it broadcasts
        if constexpr (R != 1 && !CHUNKU) {
            for (int j = tid; j < R*ESCHA_TILE; j += ESCHA_NT) {
                const int m = j / ESCHA_TILE;
                const int r = j % ESCHA_TILE;
                s_u[j] = m < nrow ? u[(int64_t)(start + m)*IC + ti*ESCHA_TILE + r] : 0.0f;
            }
        }

        // publish the tile fetched last round, then issue the next fetch immediately: the
        // LDG then overlaps this tile's decode instead of stalling in front of it, which is
        // what the LDG -> dependent STS pair at the top of the loop was costing
#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                // word wd is the high half of pair wd and the low half of pair wd+1
                pay[wd].y = pre[i];
                pay[wd + 1 == NW ? 0 : wd + 1].x = pre[i];
            }
        }
        if (ti + 1 < hi) {
            const uint32_t * nxt = (const uint32_t *)(code + (int64_t)((ti + 1)*nct + tj)*(16*K));
#pragma unroll
            for (int i = 0; i < NPW; ++i) {
                const int wd = cc + i*ESCHA_TILE;
                if (wd < n_wd) {
                    pre[i] = nxt[wd];
                }
            }
        }
        // CHUNKU leaves the tile loop with the (group-private) weight payload as its only
        // shared traffic, so a warp-level fence is the correct one -- the same the R == 1 path
        // has always used.  The chunk staging is ordered by the chunk-top block barrier.
        if constexpr (R == 1 || CHUNKU) { __syncwarp(); } else { __syncthreads(); }

        const float * uu = CHUNKU ? s_u + (size_t)(ti - cbase)*R*ESCHA_TILE
                        : (R == 1 ? s_u + (size_t)(ti - lo)*ESCHA_TILE : s_u);

        // put the whole tile's payload (16 dependent-address LDS.64) in flight up front
        uint2 pv[PFP ? ESCHA_TILE : 1];
        if constexpr (PFP) {
#pragma unroll
            for (int rp = 0; rp < ESCHA_TILE; ++rp) {
                if constexpr (SCHED) {
                    pv[rp] = pay[sched[rp] >> 5];
                } else {
                    int sp = s0 - K*escha_dep_pi(rp);
                    if (sp < 0) {
                        sp += NB;
                    }
                    const int g0 = sp >> 5;
                    pv[rp] = pay[g0 ? (NW - g0) : 0];
                }
            }
        }

        // generation unrolls fully so pi(r) folds to a compile-time constant; prefill keeps
        // a partial unroll, where acc[R] already claims the registers
#pragma unroll (R <= 8 ? 16 : 4)
        for (int r = 0; r < ESCHA_TILE; ++r) {
            // K*pi(r) <= 81 < NB, so one conditional add restores the range
            int sp;
            uint2 p;
            if constexpr (SCHED) {
                sp = (int) (sched[r] & 31u);
                if constexpr (PFP) { p = pv[r]; } else { p = pay[sched[r] >> 5]; }
            } else {
                sp = s0 - K*escha_dep_pi(r);
                if (sp < 0) {
                    sp += NB;
                }
                const int g0 = sp >> 5;
                const int w0 = g0 ? (NW - g0) : 0;
                if constexpr (PFP) { p = pv[r]; } else { p = pay[w0]; }
            }
            const uint32_t idx = __funnelshift_r(p.y, p.x, sp & 31) & 0xffffu;

            const float wv = escha_codebook(idx);
            if constexpr (R == 1) {
                acc[0] += uu[r]*wv;
            } else if constexpr (CHUNKU) {
#pragma unroll
                for (int m = 0; m < R; ++m) {
                    acc[m] += uu[m*ESCHA_TILE + r]*wv;
                }
            } else {
#pragma unroll
                for (int m = 0; m < R; ++m) {
                    acc[m] += s_u[m*ESCHA_TILE + r]*wv;
                }
            }
        }
        if constexpr (R == 1 || CHUNKU) { __syncwarp(); } else { __syncthreads(); }
    }
    }

    for (int m = 0; m < nrow; ++m) {
        partial[((int64_t) sl*n_rows + start + m)*OC + blockIdx.y*ESCHA_NT + tid] = acc[m];
    }
}

// sum the slices in a fixed order (so the result is reproducible), rotate the
// 128-column group, scale by rout
static __global__ void escha_finalize_dense(
        const half  * __restrict__ rout,
        const float * __restrict__ partial,
        float       * __restrict__ dst,
        const int OC, const int ne1, const int n_rows, const int n_slices,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int c   = blockIdx.y*ESCHA_NT + tid;

    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + (int64_t)(row % ne1)*nb_d1
                                             + (int64_t)(row / ne1)*nb_d2);
    dst_row[c] = s_acc[tid]*__half2float(rout[c]);
}

// Decode finalizer: one warp owns one 128-column Hadamard group.  Four values
// stay in each lane's registers, avoiding the shared-memory round trips and
// eight CTA-wide barriers in the general finalizer while preserving the same
// slice summation and Sylvester-Hadamard stage order.
static __global__ void escha_finalize_dense_warp(
        const half  * __restrict__ rout,
        const float * __restrict__ partial,
        float       * __restrict__ dst,
        const int OC, const int ne1, const int n_rows, const int n_slices,
        const int64_t nb_d1, const int64_t nb_d2) {
    constexpr int WARP = 32;
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int c0 = blockIdx.y*ESCHA_NT + lane;

    float v[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int s = 0; s < n_slices; ++s) {
        const float * p = partial + ((int64_t) s*n_rows + row)*OC + c0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] += p[k*WARP];
        }
    }

#pragma unroll
    for (int len = 1; len < WARP; len <<= 1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float partner = __shfl_xor_sync(0xffffffff, v[k], len);
            v[k] = (lane & len) == 0 ? v[k] + partner : partner - v[k];
        }
    }

    float t;
    t = v[0]; v[0] = t + v[1]; v[1] = t - v[1];
    t = v[2]; v[2] = t + v[3]; v[3] = t - v[3];
    t = v[0]; v[0] = t + v[2]; v[2] = t - v[2];
    t = v[1]; v[1] = t + v[3]; v[3] = t - v[3];

    const float scale = rsqrtf(128.0f);
    float * dst_row = (float *)((char *) dst + (int64_t)(row % ne1)*nb_d1
                                             + (int64_t)(row / ne1)*nb_d2);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = c0 + k*WARP;
        dst_row[c] = v[k]*scale*__half2float(rout[c]);
    }
}

// Decode-only FFN-down handoff: finish one 128-column Escha output group,
// add the live residual, and publish both the residual values and this group's
// sum of squares. A second single-CTA kernel completes RMSNorm*weight.
static __global__ void escha_finalize_dense_warp_add_ss(
        const half  * __restrict__ rout,
        const float * __restrict__ partial,
        const float * __restrict__ residual_src,
        float       * __restrict__ residual_dst,
        float       * __restrict__ group_ss,
        const int OC, const int n_slices) {
    constexpr int WARP = 32;
    const int lane = threadIdx.x;
    const int c0 = blockIdx.x*ESCHA_NT + lane;

    float v[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int s = 0; s < n_slices; ++s) {
        const float * p = partial + (int64_t) s*OC + c0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] += p[k*WARP];
        }
    }

#pragma unroll
    for (int len = 1; len < WARP; len <<= 1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float partner = __shfl_xor_sync(0xffffffff, v[k], len);
            v[k] = (lane & len) == 0 ? v[k] + partner : partner - v[k];
        }
    }

    float t;
    t = v[0]; v[0] = t + v[1]; v[1] = t - v[1];
    t = v[2]; v[2] = t + v[3]; v[3] = t - v[3];
    t = v[0]; v[0] = t + v[2]; v[2] = t - v[2];
    t = v[1]; v[1] = t + v[3]; v[3] = t - v[3];

    const float scale = rsqrtf(128.0f);
    float sum_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = c0 + k*WARP;
        const float value = v[k]*scale*__half2float(rout[c]) + residual_src[c];
        residual_dst[c] = value;
        sum_sq += value*value;
    }
#pragma unroll
    for (int offset = WARP/2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    if (lane == 0) {
        group_ss[blockIdx.x] = sum_sq;
    }
}

template<int BLOCK_SIZE>
static __global__ void escha_rms_mul_from_group_ss(
        const float * __restrict__ residual,
        const float * __restrict__ group_ss,
        const float * __restrict__ weight,
        float       * __restrict__ dst,
        const int ncols, const int ngroups, const float eps) {
    const int tid = threadIdx.x;
    float sum_sq = tid < ngroups ? group_ss[tid] : 0.0f;
    extern __shared__ float s_sum[];
    sum_sq = block_reduce<block_reduce_method::SUM, BLOCK_SIZE>(sum_sq, s_sum);
    const float scale = rsqrtf(sum_sq/ncols + eps);
    for (int c = tid; c < ncols; c += BLOCK_SIZE) {
        dst[c] = scale*residual[c]*weight[c];
    }
}

template<bool APPLY_SILU, bool APPLY_MUL>
static __global__ void escha_finalize_dense_warp_swiglu(
        const half  * __restrict__ rout,
        const float * __restrict__ partial,
        const float * __restrict__ gate_silu,
        float       * __restrict__ dst,
        const int OC, const int ne1, const int n_rows, const int n_slices,
        const int64_t nb_d1, const int64_t nb_d2) {
    constexpr int WARP = 32;
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int c0 = blockIdx.y*ESCHA_NT + lane;

    float v[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int s = 0; s < n_slices; ++s) {
        const float * p = partial + ((int64_t) s*n_rows + row)*OC + c0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] += p[k*WARP];
        }
    }

#pragma unroll
    for (int len = 1; len < WARP; len <<= 1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float partner = __shfl_xor_sync(0xffffffff, v[k], len);
            v[k] = (lane & len) == 0 ? v[k] + partner : partner - v[k];
        }
    }

    float t;
    t = v[0]; v[0] = t + v[1]; v[1] = t - v[1];
    t = v[2]; v[2] = t + v[3]; v[3] = t - v[3];
    t = v[0]; v[0] = t + v[2]; v[2] = t - v[2];
    t = v[1]; v[1] = t + v[3]; v[3] = t - v[3];

    const float scale = rsqrtf(128.0f);
    float * dst_row = (float *)((char *) dst + (int64_t)(row % ne1)*nb_d1
                                             + (int64_t)(row / ne1)*nb_d2);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = c0 + k*WARP;
        float value = v[k]*scale*__half2float(rout[c]);
        if constexpr (APPLY_SILU) {
            value = ggml_cuda_op_silu_single(value);
        }
        if constexpr (APPLY_MUL) {
            value *= gate_silu[(int64_t) row*OC + c];
        }
        dst_row[c] = value;
    }
}

// One-sided SwiGLU epilogue: the gate projection and its ordinary F32 SiLU
// remain separate; only the completed gate tensor is consumed here.
static __global__ void escha_finalize_dense_mul(
        const half  * __restrict__ rout,
        const float * __restrict__ partial,
        const float * __restrict__ gate_silu,
        float       * __restrict__ dst,
        const int OC, const int ne1, const int n_rows, const int n_slices,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int c   = blockIdx.y*ESCHA_NT + tid;

    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + (int64_t)(row % ne1)*nb_d1
                                             + (int64_t)(row / ne1)*nb_d2);
    const float up_value = s_acc[tid]*__half2float(rout[c]);
    dst_row[c] = gate_silu[(int64_t) row*OC + c]*up_value;
}

// Two-sided SwiGLU epilogue.  This is the same F32 SiLU helper used by the
// ordinary CUDA unary path, applied after the fixed Escha reduction, Hadamard,
// and output scaling.  The result is written directly to the unary node's
// buffer, so the standalone gate finalizer and SiLU launch are both removed.
static __global__ void escha_finalize_dense_silu(
        const half  * __restrict__ rout,
        const float * __restrict__ partial,
        float       * __restrict__ dst,
        const int OC, const int ne1, const int n_rows, const int n_slices,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int c   = blockIdx.y*ESCHA_NT + tid;

    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + (int64_t)(row % ne1)*nb_d1
                                             + (int64_t)(row / ne1)*nb_d2);
    dst_row[c] = ggml_cuda_op_silu_single(s_acc[tid]*__half2float(rout[c]));
}

static __global__ void escha_f16_silu_f32(
        const half * __restrict__ src,
        float      * __restrict__ dst,
        const size_t n) {
    const size_t i = 2*((size_t) blockIdx.x*blockDim.x + threadIdx.x);
    if (i + 1 < n) {
        const float2 f = __half22float2(reinterpret_cast<const half2 *>(src)[i/2]);
        reinterpret_cast<float2 *>(dst)[i/2] = make_float2(
            ggml_cuda_op_silu_single(f.x), ggml_cuda_op_silu_single(f.y));
    } else if (i < n) {
        dst[i] = ggml_cuda_op_silu_single(__half2float(src[i]));
    }
}

static __global__ void escha_f16_mul_f32(
        const half  * __restrict__ src,
        const float * __restrict__ gate,
        float       * __restrict__ dst,
        const size_t n) {
    const size_t i = 2*((size_t) blockIdx.x*blockDim.x + threadIdx.x);
    if (i + 1 < n) {
        const float2 f = __half22float2(reinterpret_cast<const half2 *>(src)[i/2]);
        const float2 g = reinterpret_cast<const float2 *>(gate)[i/2];
        reinterpret_cast<float2 *>(dst)[i/2] = make_float2(f.x*g.x, f.y*g.y);
    } else if (i < n) {
        dst[i] = __half2float(src[i])*gate[i];
    }
}

// E1 lab instrumentation only.  The normal path is untouched unless the
// capture directory is explicitly supplied by the lab harness.  Capturing
// before finalize preserves the producer's raw split-K layout for exact
// control/candidate comparison without making p_buf part of the public API.
static std::atomic<uint64_t> g_escha_e1_capture_invocation{0};

static bool escha_e1_filter_matches(const char * filter, const char * tensor_name) {
    if (filter == nullptr || *filter == '\0' || tensor_name == nullptr || *tensor_name == '\0') {
        return false;
    }
    std::string names(filter);
    size_t begin = 0;
    while (begin <= names.size()) {
        const size_t end = names.find(',', begin);
        const std::string item = names.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (item == tensor_name) return true;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return false;
}

static std::string escha_e1_env(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr ? value : "";
}

static std::string escha_e1_safe_name(const char * name) {
    std::string safe = name != nullptr ? name : "";
    for (char & c : safe) if (c == '/' || c == '\\') c = '_';
    return safe;
}

static void escha_e1_require_capture_only(bool candidate_selectors_disabled) {
    if (!candidate_selectors_disabled) {
        std::fprintf(stderr, "E1 capture requires candidate selectors disabled\n");
        std::abort();
    }
}

static void escha_e1_require_provenance() {
    static const char * const names[] = {
        "ESCHA_SOURCE_IDENTITY_SHA256",
        "ESCHA_RUNTIME_IDENTITY_SHA256",
        "ESCHA_RECEIPT_SHA256",
        "ESCHA_MODEL_SHA256",
        "ESCHA_GGUF_SHA256",
        "ESCHA_KERNEL_SOURCE_SHA256",
    };
    for (const char * name : names) {
        const char * value = std::getenv(name);
        if (value == nullptr || *value == '\0') {
            std::fprintf(stderr, "E1 capture requires nonempty %s\n", name);
            std::abort();
        }
    }
}

static std::string escha_e1_stem(const char * dir, const char * tensor_name,
                                 uint64_t invocation, int n_rows, int IC, int OC) {
#if defined(_WIN32)
    const uint64_t pid = static_cast<uint64_t>(_getpid());
#else
    const uint64_t pid = static_cast<uint64_t>(getpid());
#endif
    return std::string(dir) + "/" + escha_e1_safe_name(tensor_name) + ".pid" +
           std::to_string(pid) + ".inv" + std::to_string(invocation) + ".rows" +
           std::to_string(n_rows) + ".ic" + std::to_string(IC) + ".oc" + std::to_string(OC);
}

static void escha_e1_write_binding(std::ofstream & meta, const ggml_tensor * code,
                                   uint64_t invocation, int M, int n_rows, int IC, int OC,
                                   int K, int n_slices, const char * route_id,
                                   const char * actual_route, bool candidate_selectors_disabled,
                                   const char * kind, const char * output_boundary = nullptr) {
    escha_e1_require_provenance();
    meta << "schema=escha-e1-capture/2\n"
         << "capture_kind=" << kind << "\n";
    // Output captures live at a named producer boundary.  Activation/partial
    // captures pass nullptr so their metadata stays byte-identical.
    if (output_boundary != nullptr && *output_boundary != '\0') {
        meta << "output_boundary=" << output_boundary << "\n";
    }
    meta << "capture_only=true\n"
         << "timing_usable=false\n"
         << "candidate_selectors_disabled=" << (candidate_selectors_disabled ? "true" : "false") << "\n"
         << "source_tensor=" << code->name << "\n"
         << "invocation_id=" << invocation << "\n"
         << "M=" << M << "\n"
         << "n_rows=" << n_rows << "\n"
         << "IC=" << IC << "\n"
         << "OC=" << OC << "\n"
         << "K=" << K << "\n"
         << "n_slices=" << n_slices << "\n"
         << "route_id=" << (route_id != nullptr ? route_id : "") << "\n"
         << "actual_route=" << (actual_route != nullptr ? actual_route : "") << "\n"
         << "graph_mode=" << (std::getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr
                              ? "eager" : "graph-enabled-or-unsupported") << "\n"
         << "graph_mode_source=env-policy\n"
         << "source_identity_sha256=" << escha_e1_env("ESCHA_SOURCE_IDENTITY_SHA256") << "\n"
         << "receipt_sha256=" << escha_e1_env("ESCHA_RECEIPT_SHA256") << "\n"
         << "model_sha256=" << escha_e1_env("ESCHA_MODEL_SHA256") << "\n"
         << "gguf_sha256=" << escha_e1_env("ESCHA_GGUF_SHA256") << "\n"
         << "runtime_identity_sha256=" << escha_e1_env("ESCHA_RUNTIME_IDENTITY_SHA256") << "\n"
         << "kernel_source_sha256=" << escha_e1_env("ESCHA_KERNEL_SOURCE_SHA256") << "\n";
}

static void escha_e1_capture_partial(const ggml_tensor * code, const float * partial, size_t count,
                                     uint64_t invocation, int M, int IC, int K,
                                     int n_slices, int n_rows, int OC,
                                     const char * actual_route,
                                     bool candidate_selectors_disabled, cudaStream_t stream) {
    const char * dir = std::getenv("ESCHA_E1_PARTIAL_CAPTURE_DIR");
    const char * filter = std::getenv("ESCHA_E1_PARTIAL_CAPTURE_TENSORS");
    if (dir == nullptr || *dir == '\0' || !escha_e1_filter_matches(filter, code->name)) {
        return;
    }
    escha_e1_require_capture_only(candidate_selectors_disabled);
    const char * case_id = std::getenv("ESCHA_E1_CASE_ID");
    const char * route_id = std::getenv("ESCHA_E1_ROUTE_ID");
    if (case_id == nullptr || *case_id == '\0' || route_id == nullptr || *route_id == '\0') {
        std::fprintf(stderr, "E1 partial capture requires ESCHA_E1_CASE_ID and ESCHA_E1_ROUTE_ID\n");
        std::abort();
    }

    std::vector<float> host(count);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), partial, count*sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::string stem = escha_e1_stem(dir, code->name, invocation, n_rows, IC, OC) + "." + case_id + "." + route_id;
    const std::string data_path = stem + ".partial.f32";
    const std::string data_tmp = data_path + ".tmp";
    std::ofstream data(data_tmp, std::ios::binary | std::ios::trunc);
    if (!data.good()) {
        std::fprintf(stderr, "cannot create E1 partial capture: %s\n", data_tmp.c_str());
        std::abort();
    }
    data.write(reinterpret_cast<const char *>(host.data()), (std::streamsize) (count*sizeof(float)));
    data.close();
    if (!data.good() || std::rename(data_tmp.c_str(), data_path.c_str()) != 0) {
        std::fprintf(stderr, "cannot commit E1 partial capture: %s\n", data_path.c_str());
        std::abort();
    }

    const std::string meta_path = stem + ".partial.meta";
    const std::string meta_tmp = meta_path + ".tmp";
    std::ofstream meta(meta_tmp, std::ios::trunc);
    if (!meta.good()) {
        std::fprintf(stderr, "cannot create E1 partial metadata: %s\n", meta_tmp.c_str());
        std::abort();
    }
    escha_e1_write_binding(meta, code, invocation, M, n_rows, IC, OC, K, n_slices,
                           route_id, actual_route, candidate_selectors_disabled, "partial");
    meta
         << "elements=" << count << '\n'
         << "bytes=" << count*sizeof(float) << '\n';
    meta.close();
    if (!meta.good() || std::rename(meta_tmp.c_str(), meta_path.c_str()) != 0) {
        std::fprintf(stderr, "cannot commit E1 partial metadata: %s\n", meta_path.c_str());
        std::abort();
    }
}

// Default-off lab hook. It captures the pre-rotation F32 activation for exact
// hidden-contract provenance; it is inert unless both the directory and an
// exact code->name filter are supplied.
static void escha_e1_capture_activation(const ggml_tensor * code, const ggml_tensor * x,
                                        uint64_t invocation, int M, int n_rows, int IC, int OC, int K,
                                        int n_slices, bool candidate_selectors_disabled,
                                        const char * actual_route, cudaStream_t stream) {
    const char * dir = std::getenv("ESCHA_E1_ACTIVATION_CAPTURE_DIR");
    const char * filter = std::getenv("ESCHA_E1_ACTIVATION_CAPTURE_TENSORS");
    if (dir == nullptr || *dir == '\0' || filter == nullptr || *filter == '\0' || code->name[0] == '\0') {
        return;
    }
    if (!escha_e1_filter_matches(filter, code->name)) return;
    escha_e1_require_capture_only(candidate_selectors_disabled);

    const size_t row_bytes = (size_t) IC * sizeof(float);
    std::vector<float> host((size_t) n_rows * IC);
    for (int row = 0; row < n_rows; ++row) {
        const int64_t i1 = row % x->ne[1];
        const int64_t i2 = row / x->ne[1];
        const char * src = (const char *) x->data + i1*x->nb[1] + i2*x->nb[2];
        CUDA_CHECK(cudaMemcpyAsync(host.data() + (size_t) row*IC, src, row_bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::string stem = escha_e1_stem(dir, code->name, invocation, n_rows, IC, OC);
    const std::string tmp = stem + ".f32.tmp";
    const std::string data_path = stem + ".f32";
    std::ofstream data(tmp, std::ios::binary | std::ios::trunc);
    if (!data.good()) { std::fprintf(stderr, "cannot create activation capture: %s\n", tmp.c_str()); std::abort(); }
    data.write(reinterpret_cast<const char *>(host.data()), (std::streamsize) host.size()*sizeof(float));
    data.close();
    if (!data.good() || std::rename(tmp.c_str(), data_path.c_str()) != 0) {
        std::fprintf(stderr, "cannot commit activation capture: %s\n", data_path.c_str()); std::abort();
    }
    std::ofstream meta(data_path + ".meta", std::ios::trunc);
    if (!meta.good()) { std::fprintf(stderr, "cannot create activation metadata: %s\n", data_path.c_str()); std::abort(); }
    const char * route_id = std::getenv("ESCHA_E1_ROUTE_ID");
    escha_e1_write_binding(meta, code, invocation, M, n_rows, IC, OC, K, n_slices,
                           route_id, actual_route, candidate_selectors_disabled, "activation");
    meta << "rows=" << n_rows << "\n"
         << "bytes=" << host.size()*sizeof(float) << "\n"
         << "dtype=f32\nlayout=row-major\n"
         << "stride_x_nb1=" << x->nb[1] << "\nstride_x_nb2=" << x->nb[2] << "\n"
         << "hash=sha256-required-by-catalog\n";
    meta.close();
    if (!meta.good()) { std::fprintf(stderr, "cannot commit activation metadata: %s\n", data_path.c_str()); std::abort(); }
}

// Default-off lab hook for the control route's final output.  It is the third
// capture family (activation, partial, output) and is inert unless both the
// output directory and an exact code->name filter are supplied.  It runs after
// the normal escha_finalize_dense kernel has written dst, so the bytes here are
// the post-finalizer F32 result the ordinary (non-candidate) dispatch path
// produces.  The finalizer may write dst with nb[1]/nb[2] strides, so each row
// is copied with the same row % ne1 / row / ne1 addressing the kernel uses.
static void escha_e1_capture_output(const ggml_tensor * code, const ggml_tensor * x,
                                    const ggml_tensor * dst,
                                    uint64_t invocation, int M, int n_rows, int IC, int OC, int K,
                                    int n_slices, bool candidate_selectors_disabled,
                                    const char * actual_route, cudaStream_t stream) {
    const char * dir = std::getenv("ESCHA_E1_OUTPUT_CAPTURE_DIR");
    const char * filter = std::getenv("ESCHA_E1_OUTPUT_CAPTURE_TENSORS");
    if (dir == nullptr || *dir == '\0' || filter == nullptr || *filter == '\0' || code->name[0] == '\0') {
        return;
    }
    if (!escha_e1_filter_matches(filter, code->name)) return;
    escha_e1_require_capture_only(candidate_selectors_disabled);

    const char * case_id = std::getenv("ESCHA_E1_CASE_ID");
    const char * route_id = std::getenv("ESCHA_E1_ROUTE_ID");
    if (case_id == nullptr || *case_id == '\0' || route_id == nullptr || *route_id == '\0') {
        std::fprintf(stderr, "E1 output capture requires ESCHA_E1_CASE_ID and ESCHA_E1_ROUTE_ID\n");
        std::abort();
    }

    // escha_finalize_dense splits rows with x->ne[1] and writes dst with dst's
    // own strides; mirror that addressing exactly so the copied rows are the
    // rows the finalizer produced.
    const int64_t ne1 = x->ne[1];
    const size_t row_bytes = (size_t) OC * sizeof(float);
    std::vector<float> host((size_t) n_rows * OC);
    for (int row = 0; row < n_rows; ++row) {
        const int64_t i1 = row % ne1;
        const int64_t i2 = row / ne1;
        const char * src = (const char *) dst->data + i1*dst->nb[1] + i2*dst->nb[2];
        CUDA_CHECK(cudaMemcpyAsync(host.data() + (size_t) row*OC, src, row_bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::string stem = escha_e1_stem(dir, code->name, invocation, n_rows, IC, OC) + "." + case_id + "." + route_id;
    const std::string data_path = stem + ".output.f32";
    const std::string data_tmp = data_path + ".tmp";
    std::ofstream data(data_tmp, std::ios::binary | std::ios::trunc);
    if (!data.good()) {
        std::fprintf(stderr, "cannot create E1 output capture: %s\n", data_tmp.c_str());
        std::abort();
    }
    data.write(reinterpret_cast<const char *>(host.data()), (std::streamsize) (host.size()*sizeof(float)));
    data.close();
    if (!data.good() || std::rename(data_tmp.c_str(), data_path.c_str()) != 0) {
        std::fprintf(stderr, "cannot commit E1 output capture: %s\n", data_path.c_str());
        std::abort();
    }

    const std::string meta_path = stem + ".output.meta";
    const std::string meta_tmp = meta_path + ".tmp";
    std::ofstream meta(meta_tmp, std::ios::trunc);
    if (!meta.good()) {
        std::fprintf(stderr, "cannot create E1 output metadata: %s\n", meta_tmp.c_str());
        std::abort();
    }
    escha_e1_write_binding(meta, code, invocation, M, n_rows, IC, OC, K, n_slices,
                           route_id, actual_route, candidate_selectors_disabled, "output",
                           "post-finalizer");
    meta << "elements=" << host.size() << '\n'
         << "bytes=" << host.size()*sizeof(float) << '\n'
         << "dtype=f32\nlayout=row-major\n"
         << "stride_dst_nb1=" << dst->nb[1] << "\nstride_dst_nb2=" << dst->nb[2] << "\n";
    meta.close();
    if (!meta.good() || std::rename(meta_tmp.c_str(), meta_path.c_str()) != 0) {
        std::fprintf(stderr, "cannot commit E1 output metadata: %s\n", meta_path.c_str());
        std::abort();
    }
}

// MVT candidates deliberately live behind one process-wide allow-list.  In particular,
// old one-off lab knobs must not make a benchmark a candidate run without its nonce and
// spec binding being recorded.  The default (no ESCHA_CANDIDATE_ID) is the unchanged
// control route; setting the attestation triplet with no candidate ID records that fact.
enum class escha_mvt_candidate {
    control,
    prefill_warp_rotate_half,
    prefill_pipelined_decode,
    prefill_finalize_fused_single_slice,
    prefill_direct_b_fragment,
    prefill_direct_b_fragment_bn64,
    prefill_mma_bn64,
    prefill_pipelined_decode_bn64,
    prefill_swiglu_up_finalize_mul_single_slice,
    prefill_swiglu_gate_silu_up_finalize_mul_single_slice,
    decode_gen_target_mul_2,
    decode_gen_target_mul_4,
    decode_gen_target_mul_8,
};

struct escha_mvt_config {
    escha_mvt_candidate candidate = escha_mvt_candidate::control;
    std::string candidate_id;
    std::string nonce;
    std::string spec_sha256;
    std::string attestation_path;
    std::string source_identity_sha256;
    std::string binary_sha256;
    std::string runtime_identity_sha256;
    std::string model_sha256;
    std::string gguf_sha256;
    std::string input_sha256;
    std::string config_sha256;
    std::string model;
    std::string workload;
    std::string dimensions;
    std::string codec;
    int gen_target_mul = ESCHA_GEN_TARGET_MUL;

    bool candidate_enabled() const {
        return candidate != escha_mvt_candidate::control;
    }

    bool attestation_enabled() const {
        return !attestation_path.empty();
    }
};

struct escha_mvt_stats {
    std::atomic<uint64_t> target_route_count{0};
    std::atomic<uint64_t> control_route_count{0};
    std::atomic<uint64_t> unaffected_route_count{0};
    std::atomic<uint64_t> fallback_count{0};
    // Current candidates select one complete route per call, so this remains zero.  It
    // is retained in the schema to make a future partial/mixed candidate auditable.
    std::atomic<uint64_t> mixed_route_count{0};
    std::mutex observed_mutex;
    std::set<std::string> codecs;
    std::set<std::string> dimensions;
    std::set<int> rows;
};

static escha_mvt_config g_escha_mvt_config;
static escha_mvt_stats g_escha_mvt_stats;
static std::once_flag g_escha_mvt_once;

static const char * escha_mvt_env(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr ? value : "";
}

static bool escha_mvt_sha256(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(),
        [](unsigned char c) { return std::isxdigit(c) != 0 && !(c >= 'A' && c <= 'F'); });
}

static std::string escha_mvt_json_escape(const std::string & value) {
    std::ostringstream escaped;
    for (const unsigned char c : value) {
        switch (c) {
            case '\\': escaped << "\\\\"; break;
            case '"':  escaped << "\\\""; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (c < 0x20) {
                    escaped << "\\u00" << "0123456789abcdef"[c >> 4]
                            << "0123456789abcdef"[c & 0x0f];
                } else {
                    escaped << static_cast<char>(c);
                }
        }
    }
    return escaped.str();
}

static void escha_mvt_json_string_array(std::ostringstream & out, const std::set<std::string> & values) {
    out << '[';
    bool first = true;
    for (const std::string & value : values) {
        if (!first) out << ',';
        first = false;
        out << '"' << escha_mvt_json_escape(value) << '"';
    }
    out << ']';
}

static void escha_mvt_json_int_array(std::ostringstream & out, const std::set<int> & values) {
    out << '[';
    bool first = true;
    for (const int value : values) {
        if (!first) out << ',';
        first = false;
        out << value;
    }
    out << ']';
}

static void escha_mvt_attest_atexit() {
    const escha_mvt_config & config = g_escha_mvt_config;
    if (!config.attestation_enabled()) {
        return;
    }

    std::set<std::string> codecs;
    std::set<std::string> dimensions;
    std::set<int> rows;
    {
        std::lock_guard<std::mutex> lock(g_escha_mvt_stats.observed_mutex);
        codecs = g_escha_mvt_stats.codecs;
        dimensions = g_escha_mvt_stats.dimensions;
        rows = g_escha_mvt_stats.rows;
    }

    std::ostringstream json;
    json << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"nonce\": \"" << escha_mvt_json_escape(config.nonce) << "\",\n"
         << "  \"candidate_id\": \"" << escha_mvt_json_escape(config.candidate_id) << "\",\n"
         << "  \"spec_sha256\": \"" << escha_mvt_json_escape(config.spec_sha256) << "\",\n"
         << "  \"candidate_enabled\": " << (config.candidate_enabled() ? "true" : "false") << ",\n"
         << "  \"target_route_count\": " << g_escha_mvt_stats.target_route_count.load(std::memory_order_relaxed) << ",\n"
         << "  \"control_route_count\": " << g_escha_mvt_stats.control_route_count.load(std::memory_order_relaxed) << ",\n"
         << "  \"unaffected_route_count\": " << g_escha_mvt_stats.unaffected_route_count.load(std::memory_order_relaxed) << ",\n"
         << "  \"fallback_count\": " << g_escha_mvt_stats.fallback_count.load(std::memory_order_relaxed) << ",\n"
         << "  \"mixed_route_count\": " << g_escha_mvt_stats.mixed_route_count.load(std::memory_order_relaxed) << ",\n"
         << "  \"binding\": {\n"
         << "    \"source_identity_sha256\": \"" << escha_mvt_json_escape(config.source_identity_sha256) << "\",\n"
         << "    \"binary_sha256\": \"" << escha_mvt_json_escape(config.binary_sha256) << "\",\n"
         << "    \"runtime_identity_sha256\": \"" << escha_mvt_json_escape(config.runtime_identity_sha256) << "\",\n"
         << "    \"model_sha256\": \"" << escha_mvt_json_escape(config.model_sha256) << "\",\n"
         << "    \"gguf_sha256\": \"" << escha_mvt_json_escape(config.gguf_sha256) << "\",\n"
         << "    \"input_sha256\": \"" << escha_mvt_json_escape(config.input_sha256) << "\",\n"
         << "    \"config_sha256\": \"" << escha_mvt_json_escape(config.config_sha256) << "\",\n"
         << "    \"shape\": \"" << escha_mvt_json_escape(config.dimensions) << "\",\n"
         << "    \"codec\": \"" << escha_mvt_json_escape(config.codec) << "\",\n"
         << "    \"model\": \"" << escha_mvt_json_escape(config.model) << "\",\n"
         << "    \"workload\": \"" << escha_mvt_json_escape(config.workload) << "\"\n"
         << "  },\n"
         << "  \"observed\": { \"codecs\": ";
    escha_mvt_json_string_array(json, codecs);
    json << ", \"dimensions\": ";
    escha_mvt_json_string_array(json, dimensions);
    json << ", \"rows\": ";
    escha_mvt_json_int_array(json, rows);
    json << " },\n"
         << "  \"resolved_parameters\": { ";
    if (config.candidate == escha_mvt_candidate::decode_gen_target_mul_2 ||
        config.candidate == escha_mvt_candidate::decode_gen_target_mul_4 ||
        config.candidate == escha_mvt_candidate::decode_gen_target_mul_8) {
        json << "\"generation_target_multiplier\": " << config.gen_target_mul;
    } else if (config.candidate == escha_mvt_candidate::prefill_warp_rotate_half) {
        json << "\"route\": \"warp-register-half-rotation\"";
    } else if (config.candidate == escha_mvt_candidate::prefill_pipelined_decode) {
        json << "\"schedule\": \"pipelined-decode-2barrier\"";
    } else if (config.candidate == escha_mvt_candidate::prefill_finalize_fused_single_slice) {
        json << "\"epilogue\": \"single-slice-shared-16x128\"";
    } else if (config.candidate == escha_mvt_candidate::prefill_direct_b_fragment) {
        json << "\"dataflow\": \"direct-packed-to-mma-b-fragment\"";
    } else if (config.candidate == escha_mvt_candidate::prefill_direct_b_fragment_bn64) {
        json << "\"dataflow\": \"direct-packed-to-mma-b-fragment\", \"BN\": 64";
    } else if (config.candidate == escha_mvt_candidate::prefill_mma_bn64) {
        json << "\"dataflow\": \"shared-b\", \"BN\": 64";
    } else if (config.candidate == escha_mvt_candidate::prefill_pipelined_decode_bn64) {
        json << "\"schedule\": \"pipelined-decode-2barrier\", \"BN\": 64";
    } else if (config.candidate == escha_mvt_candidate::prefill_swiglu_up_finalize_mul_single_slice) {
        json << "\"epilogue\": \"up-finalizer-f32-silu-gate-mul\","
             << "\"input_columns\": 5120, \"output_columns\": 17408,"
             << "\"rows\": 512, \"n_slices\": 1";
    } else if (config.candidate == escha_mvt_candidate::prefill_swiglu_gate_silu_up_finalize_mul_single_slice) {
        json << "\"epilogue\": \"gate-f32-silu-up-finalizer-f32-mul\","
             << "\"input_columns\": 5120, \"output_columns\": 17408,"
             << "\"rows\": 512, \"n_slices\": 1";
    } else {
        json << "\"control\": true";
    }
    json << " }\n"
         << "}\n";

#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = static_cast<int>(getpid());
#endif
    const std::string temporary = config.attestation_path + ".tmp." + std::to_string(pid);
    std::ofstream out(temporary, std::ios::out | std::ios::trunc);
    if (!out.good()) {
        std::fprintf(stderr, "escha MVT: cannot create attestation: %s\n", temporary.c_str());
        return;
    }
    out << json.str();
    out.close();
    if (!out.good() || std::rename(temporary.c_str(), config.attestation_path.c_str()) != 0) {
        std::fprintf(stderr, "escha MVT: cannot atomically commit attestation: %s (%s)\n",
                     config.attestation_path.c_str(), std::strerror(errno));
    }
}

static void escha_mvt_init() {
    std::call_once(g_escha_mvt_once, [] {
        escha_mvt_config config;
        config.candidate_id = escha_mvt_env("ESCHA_CANDIDATE_ID");
        config.nonce = escha_mvt_env("ESCHA_RUN_NONCE");
        config.spec_sha256 = escha_mvt_env("ESCHA_SPEC_SHA256");
        config.attestation_path = escha_mvt_env("ESCHA_ATTESTATION_PATH");
        config.source_identity_sha256 = escha_mvt_env("ESCHA_SOURCE_IDENTITY_SHA256");
        config.binary_sha256 = escha_mvt_env("ESCHA_BINARY_SHA256");
        config.runtime_identity_sha256 = escha_mvt_env("ESCHA_RUNTIME_IDENTITY_SHA256");
        config.model_sha256 = escha_mvt_env("ESCHA_MODEL_SHA256");
        config.gguf_sha256 = escha_mvt_env("ESCHA_GGUF_SHA256");
        config.input_sha256 = escha_mvt_env("ESCHA_INPUT_SHA256");
        config.config_sha256 = escha_mvt_env("ESCHA_CONFIG_SHA256");
        config.model = escha_mvt_env("ESCHA_MODEL_ID");
        config.workload = escha_mvt_env("ESCHA_WORKLOAD");
        config.dimensions = escha_mvt_env("ESCHA_EXPECTED_DIMENSIONS");
        config.codec = escha_mvt_env("ESCHA_EXPECTED_CODEC");

        const bool any_attestation_field = !config.nonce.empty() || !config.spec_sha256.empty() ||
                                           !config.attestation_path.empty();
        if (any_attestation_field && (config.nonce.empty() || config.spec_sha256.empty() || config.attestation_path.empty())) {
            GGML_ABORT("escha MVT: ESCHA_RUN_NONCE, ESCHA_SPEC_SHA256, and ESCHA_ATTESTATION_PATH must be supplied together");
        }
        if (!config.candidate_id.empty() && !any_attestation_field) {
            GGML_ABORT("escha MVT: ESCHA_CANDIDATE_ID requires nonce, spec SHA-256, and attestation path");
        }
        if (any_attestation_field &&
            (!escha_mvt_sha256(config.spec_sha256) || !escha_mvt_sha256(config.source_identity_sha256) ||
             !escha_mvt_sha256(config.binary_sha256) || !escha_mvt_sha256(config.runtime_identity_sha256) ||
             !escha_mvt_sha256(config.model_sha256) || !escha_mvt_sha256(config.gguf_sha256) ||
             !escha_mvt_sha256(config.input_sha256) || !escha_mvt_sha256(config.config_sha256) ||
             (config.model != "W2" && config.model != "E3") ||
             (config.workload != "decode" && config.workload != "prefill") ||
             config.dimensions != "IC=5120,OC=17408" ||
             (config.codec != "K2" && config.codec != "K3" && config.codec != "K2+K3"))) {
            GGML_ABORT("escha MVT: incomplete or invalid identity binding");
        }

        if (config.candidate_id == "prefill_warp_rotate_half") {
            config.candidate = escha_mvt_candidate::prefill_warp_rotate_half;
        } else if (config.candidate_id == "prefill_pipelined_decode") {
            config.candidate = escha_mvt_candidate::prefill_pipelined_decode;
        } else if (config.candidate_id == "prefill_finalize_fused_single_slice") {
            config.candidate = escha_mvt_candidate::prefill_finalize_fused_single_slice;
        } else if (config.candidate_id == "prefill_direct_b_fragment") {
            config.candidate = escha_mvt_candidate::prefill_direct_b_fragment;
        } else if (config.candidate_id == "prefill_direct_b_fragment_bn64") {
            config.candidate = escha_mvt_candidate::prefill_direct_b_fragment_bn64;
        } else if (config.candidate_id == "prefill_mma_bn64") {
            config.candidate = escha_mvt_candidate::prefill_mma_bn64;
        } else if (config.candidate_id == "prefill_pipelined_decode_bn64") {
            config.candidate = escha_mvt_candidate::prefill_pipelined_decode_bn64;
        } else if (config.candidate_id == "prefill_swiglu_up_finalize_mul_single_slice") {
            config.candidate = escha_mvt_candidate::prefill_swiglu_up_finalize_mul_single_slice;
        } else if (config.candidate_id == "prefill_swiglu_gate_silu_up_finalize_mul_single_slice") {
            config.candidate = escha_mvt_candidate::prefill_swiglu_gate_silu_up_finalize_mul_single_slice;
        } else if (config.candidate_id == "decode_gen_target_mul_2") {
            config.candidate = escha_mvt_candidate::decode_gen_target_mul_2;
            config.gen_target_mul = 2;
        } else if (config.candidate_id == "decode_gen_target_mul_4") {
            config.candidate = escha_mvt_candidate::decode_gen_target_mul_4;
            config.gen_target_mul = 4;
        } else if (config.candidate_id == "decode_gen_target_mul_8") {
            config.candidate = escha_mvt_candidate::decode_gen_target_mul_8;
            config.gen_target_mul = 8;
        } else if (!config.candidate_id.empty()) {
            // This happens before the first CUDA launch in ggml_cuda_op_escha_mul_mat.
            GGML_ABORT("escha MVT: unknown ESCHA_CANDIDATE_ID '%s'", config.candidate_id.c_str());
        }

        g_escha_mvt_config = std::move(config);
        if (g_escha_mvt_config.attestation_enabled()) {
            std::atexit(escha_mvt_attest_atexit);
        }
    });
}

bool ggml_backend_cuda_escha_mvt_candidate_enabled(const char * candidate_id) {
    escha_mvt_init();
    return candidate_id != nullptr && g_escha_mvt_config.candidate_enabled() &&
           g_escha_mvt_config.attestation_enabled() &&
           g_escha_mvt_config.candidate_id == candidate_id;
}

static void escha_mvt_observe(int K, int IC, int OC, int n_rows) {
    const escha_mvt_config & config = g_escha_mvt_config;
    if (!config.attestation_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_escha_mvt_stats.observed_mutex);
    g_escha_mvt_stats.codecs.insert("K" + std::to_string(K));
    g_escha_mvt_stats.dimensions.insert("IC=" + std::to_string(IC) + ",OC=" + std::to_string(OC));
    g_escha_mvt_stats.rows.insert(n_rows);
}

// Row tile used by the generation (decode/spec-verify) instantiation.
//
// The row index of this instantiation is blockIdx.x, so `R` rows share one weight tile
// only if they share a block.  With R == 1 every row re-reads its (oc0, slice) tile, so a
// 4-row MTP verification batch pays four passes over the weights instead of one and the
// batch does not amortize.  R=64 is not the answer either: acc[R] is a per-thread register
// array, so a large R costs occupancy and crippled batch-1 generation (measured 1.84 t/s).
// The tile must therefore follow the batch size.  M=1 keeps R=1 exactly, so the
// single-token decode path is byte-for-byte unchanged.
static int escha_gen_row_tile(int n_rows) {
    // Lab-gated sweep override.  Row blocking trades weight-tile reuse (more rows per block =
    // fewer weight re-reads) against parallelism (fewer row-blocks = fewer CTAs to fill the
    // device).  With n_ocb=136 and a 128-SM device, R=4 leaves only n_rb*n_ocb*n_slices CTAs
    // for the M=4 verify, so the optimum is not obviously the widest tile.  Only applies to
    // n_rows > 1, so the single-token path is untouched.
    if (n_rows > 1) {
        if (const char * v = std::getenv("ESCHA_GEN_ROW_TILE")) {
            const int t = (int) std::strtol(v, nullptr, 10);
            if (t >= 1 && t <= ESCHA_GEN_MAX_ROWS) {
                return t;
            }
        }
    }
    // Exact-row tiles.  The multi-row kernels unroll their accumulate loop over the
    // compile-time R (`for m < R: acc[m] += ...`), so an R wider than the batch performs
    // R rows of FMA arithmetic with only (batch) of them meaningful -- the staged u is
    // zero-masked but the arithmetic is not.  Measured before this change (E3, warm):
    // M=3 28.25 ms (== M=4, both R=4) and M=5 43.44 ms (R=8).  Row counts above 8 still
    // round up to ESCHA_GEN_MAX_ROWS.
    if (n_rows <= 1) { return 1; }
    if (n_rows <= ESCHA_GEN_MAX_ROWS) { return n_rows; }
    return ESCHA_GEN_MAX_ROWS;
}

static void ggml_cuda_op_escha_mul_mat_impl(ggml_backend_cuda_context & ctx,
                                             ggml_tensor * dst,
                                             const ggml_tensor * gate_silu,
                                             ggml_tensor * final_dst,
                                             const bool finalize_silu) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * lut  = dst->src[3];
    const ggml_tensor * dep  = dst->src[4];
    const ggml_tensor * x    = dst->src[5];

    GGML_ASSERT(code->type == GGML_TYPE_I16 && dep->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16 && lut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int K   = code->ne[0]/16;
    const int OC  = code->ne[1]*16;
    const int IC  = code->ne[2]*16;
    const int nit = IC/ESCHA_TILE;

    const int n_rows = x->ne[1]*x->ne[2];
    const int n_ocb  = OC/ESCHA_NT;
    // The external cubin is qualified through 8 all-output rows.  At 9..16 rows its
    // logits diverge sharply even when every row is requested, so those shapes use the
    // internal small-row kernel.  Production N5 verification is at most M=6.
    const bool raw_decode_graph = ggml_get_op_params_i32(dst, 0) != 0 && n_rows <= 8;

    // Parse and validate the MVT environment before allocating or launching anything.
    // An unknown candidate therefore cannot silently fall through into a measurement.
    escha_mvt_init();
    const escha_mvt_config & mvt = g_escha_mvt_config;

    cudaStream_t stream = ctx.stream();

    // The tensor-core path wants its activations already in fp16 so cp.async can move them
    // verbatim, so the rotation has to know its consumer before it runs.
    const bool gen = n_rows <= ESCHA_GEN_MAX_ROWS;
    const bool use_mma = !gen
                      && ggml_cuda_info().devices[ctx.device].cc >= GGML_CUDA_CC_TURING
                      && OC % ESCHA_MMA_BN == 0
                      && getenv("ESCHA_NO_MMA") == nullptr;
    const char * actual_route = gen ? "generation"
        : (OC % ESCHA_BN != 0 ? "prefill_fma"
        : (use_mma ? "prefill_mma" : "prefill_scalar_tiled"));

    if (escha_official_bridge_requested() && escha_official_raw_bridge_requested()) {
        GGML_ABORT("escha: select only one official bridge route");
    }

    if (escha_official_raw_decode_bridge_requested() && !escha_official_raw_bridge_requested()) {
        GGML_ABORT("escha: raw decode bridge requires the raw prefill bridge to prepare shared state");
    }

    const bool raw_decode_gate = finalize_silu && gate_silu == nullptr && final_dst != nullptr;
    const bool raw_decode_up = !finalize_silu && gate_silu != nullptr && final_dst != nullptr;
    const bool raw_decode_plain = !finalize_silu && gate_silu == nullptr && final_dst == nullptr;
    const bool raw_decode_swiglu = escha_official_raw_decode_swiglu_fusion_requested() &&
        IC == 5120 && OC == 17408 && (raw_decode_gate || raw_decode_up);

    // R224: gate and up are independent projections over the same activation.
    // Defer the K2 gate until the K3 up is visited, run their retained raw main
    // kernels concurrently, then join before the up finalizer reads gate_silu.
    // This route is intentionally narrower than the normal raw-decode bridge;
    // any shape/order mismatch fails closed instead of silently changing math.
    if (raw_decode_graph && escha_official_raw_decode_gate_up_overlap_requested() &&
            gen && raw_decode_swiglu) {
        if (mvt.attestation_enabled()) {
            GGML_ABORT("escha: gate/up overlap cannot run with an MVT attestation");
        }
        if (!escha_official_raw_decode_native_epilogue_enabled() ||
            !escha_official_raw_decode_f32_input_requested() ||
            escha_official_raw_decode_rotated_f16_input_requested()) {
            GGML_ABORT("escha: gate/up overlap requires native epilogue and direct F32 input");
        }
        GGML_ASSERT(n_rows == 1 && ggml_is_contiguous(x) && ggml_is_contiguous(dst));
        GGML_ASSERT(final_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(final_dst));

        escha_decode_overlap_resources & resources =
            escha_decode_overlap_resources_get(ctx.device, stream);
        escha_decode_gate_pending & pending = g_escha_decode_gate_pending;

        if (raw_decode_gate) {
            if (K != 2 || pending.valid) {
                GGML_ABORT("escha: gate/up overlap expected one pending K2 gate");
            }
            pending.valid = true;
            pending.device = ctx.device;
            pending.main_stream = stream;
            pending.code = (const int16_t *) code->data;
            pending.rin = (const half *) rin->data;
            pending.rout = (const half *) rout->data;
            pending.x = (const float *) x->data;
            pending.gate_silu = (float *) final_dst->data;
            pending.gate_nb1 = final_dst->nb[1];
            pending.gate_nb2 = final_dst->nb[2];
            return;
        }

        if (K != 3 || !pending.valid || pending.device != ctx.device ||
            pending.main_stream != stream || pending.x != (const float *) x->data ||
            pending.gate_silu != (const float *) gate_silu->data) {
            GGML_ABORT("escha: gate/up overlap found an unmatched K3 up projection");
        }
        GGML_ASSERT(gate_silu->type == GGML_TYPE_F32 && ggml_is_contiguous(gate_silu));
        GGML_ASSERT(ggml_are_same_shape(gate_silu, dst));

        const auto overlap_splits = [&](const char * name, int fallback) {
            const char * value = std::getenv(name);
            if (value == nullptr) {
                return fallback;
            }
            char * end = nullptr;
            errno = 0;
            const long parsed = std::strtol(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || parsed < 1 || parsed > IC/128) {
                GGML_ABORT("escha: invalid %s=%s", name, value);
            }
            return (int) parsed;
        };
        const int gate_splits = overlap_splits(
            "ESCHA_OFFICIAL_RAW_DECODE_OVERLAP_GATE_SPLITS", 8);
        const int up_splits = overlap_splits(
            "ESCHA_OFFICIAL_RAW_DECODE_OVERLAP_UP_SPLITS", 8);
        const char * up_first_value =
            std::getenv("ESCHA_OFFICIAL_RAW_DECODE_OVERLAP_UP_FIRST");
        const bool up_first = up_first_value != nullptr &&
                              std::strcmp(up_first_value, "1") == 0;
        const char * up_aux_value =
            std::getenv("ESCHA_OFFICIAL_RAW_DECODE_OVERLAP_UP_AUX");
        const bool up_aux = up_aux_value != nullptr &&
                            std::strcmp(up_aux_value, "1") == 0;
        if (up_aux && up_first) {
            GGML_ABORT("escha: overlap up-aux and up-first selectors are mutually exclusive");
        }
        ggml_cuda_pool_alloc<float> gate_partial(
            ctx.pool(), (size_t) n_rows*OC*gate_splits);
        ggml_cuda_pool_alloc<float> up_partial(
            ctx.pool(), (size_t) n_rows*OC*up_splits);
        const escha_official_bridge_api & bridge = escha_official_bridge_load();
        if (bridge.decode_main_f32 == nullptr) {
            GGML_ABORT("escha: official bridge is missing its F32-input decode ABI");
        }

        const auto launch_gate = [&](cudaStream_t gate_stream) {
            const int rc = bridge.decode_main_f32(
                gate_stream, ctx.device, gate_partial.get(), pending.x,
                pending.code, pending.rin, n_rows, IC, OC, 2, gate_splits);
            if (rc == 0) {
                escha_finalize_dense_warp_swiglu<true, false>
                    <<<dim3(n_rows, OC/128), 32, 0, gate_stream>>>(
                    pending.rout, gate_partial.get(), nullptr, pending.gate_silu,
                    OC, (int) x->ne[1], n_rows, gate_splits,
                    pending.gate_nb1, pending.gate_nb2);
            }
            return rc;
        };
        const auto launch_up = [&](cudaStream_t up_stream) {
            return bridge.decode_main_f32(
                up_stream, ctx.device, up_partial.get(), (const float *) x->data,
                (const int16_t *) code->data, rin->data,
                n_rows, IC, OC, 3, up_splits);
        };

        CUDA_CHECK(cudaEventRecord(resources.fork, stream));
        int gate_rc;
        int up_rc;
        if (up_aux) {
            CUDA_CHECK(cudaStreamWaitEvent(resources.stream, resources.fork, 0));
            up_rc = launch_up(resources.stream);
            CUDA_CHECK(cudaEventRecord(resources.done, resources.stream));
            gate_rc = launch_gate(stream);
        } else if (up_first) {
            up_rc = launch_up(stream);
            CUDA_CHECK(cudaStreamWaitEvent(resources.stream, resources.fork, 0));
            gate_rc = launch_gate(resources.stream);
            CUDA_CHECK(cudaEventRecord(resources.done, resources.stream));
        } else {
            CUDA_CHECK(cudaStreamWaitEvent(resources.stream, resources.fork, 0));
            gate_rc = launch_gate(resources.stream);
            CUDA_CHECK(cudaEventRecord(resources.done, resources.stream));
            up_rc = launch_up(stream);
        }
        if (gate_rc != 0) {
            const char * error = bridge.error();
            GGML_ABORT("escha: overlapped gate projection failed: %s",
                       error != nullptr ? error : "unknown error");
        }
        if (up_rc != 0) {
            const char * error = bridge.error();
            GGML_ABORT("escha: overlapped up projection failed: %s",
                       error != nullptr ? error : "unknown error");
        }
        CUDA_CHECK(cudaStreamWaitEvent(stream, resources.done, 0));
        escha_finalize_dense_warp_swiglu<false, true>
            <<<dim3(n_rows, OC/128), 32, 0, stream>>>(
            (const half *) rout->data, up_partial.get(),
            (const float *) gate_silu->data, (float *) final_dst->data,
            OC, (int) x->ne[1], n_rows, up_splits,
            final_dst->nb[1], final_dst->nb[2]);
        pending.valid = false;
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (raw_decode_graph && escha_official_raw_decode_bridge_requested() &&
            gen && (K == 2 || K == 3)
            && IC % 128 == 0 && OC % 128 == 0
            && (raw_decode_plain || raw_decode_swiglu)) {
        if (mvt.attestation_enabled()) {
            GGML_ABORT("escha: official raw decode bridge cannot run with an MVT attestation");
        }
        GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
        const int output_groups = OC/128;
        int splits = 1;
        const int split_target = (1024 + output_groups - 1)/output_groups;
        while (splits < split_target) {
            splits *= 2;
        }
        splits = MIN(splits, IC/128);
        const char * k3_split_div2 = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_K3_SPLIT_DIV2");
        if (k3_split_div2 != nullptr && std::strcmp(k3_split_div2, "1") == 0 &&
                K == 3 && splits >= 2 &&
                ((IC == 5120 && OC == 17408) || (IC == 17408 && OC == 5120))) {
            // These remain 544-640 CTAs on SM89 after halving, enough to fill
            // the device while reducing partial/finalizer and CTA overhead.
            splits /= 2;
        }
        if (const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_K3_DOWN_SPLITS")) {
            if (std::strcmp(value, "12") != 0) {
                GGML_ABORT("escha: K3 down split override value must be 12");
            }
            if (K == 3 && IC == 17408 && OC == 5120) {
                splits = 12;
            }
        }
        if (const char * value = std::getenv("ESCHA_OFFICIAL_RAW_DECODE_K2_SHAPE_SPLITS")) {
            const bool all_shapes = std::strcmp(value, "1") == 0;
            const bool only_10240 = std::strcmp(value, "10240") == 0;
            if (!all_shapes && !only_10240) {
                GGML_ABORT("escha: K2 shape split selector must be 1 or 10240");
            }
            if (K == 2 && IC == 5120 && all_shapes) {
                switch (OC) {
                    case 5120:  splits = 40; break;
                    case 6144:  splits = 40; break;
                    case 10240: splits = 8;  break;
                    case 12288: splits = 20; break;
                    default: break;
                }
            }
            if (K == 2 && IC == 5120 && OC == 10240 && only_10240) {
                splits = 8;
            }
        }

        const bool native_epilogue = escha_official_raw_decode_native_epilogue_enabled();
        const bool f32_input = escha_official_raw_decode_f32_input_requested();
        const bool rotated_f16_input = escha_official_raw_decode_rotated_f16_input_requested();
        ggml_cuda_pool_alloc<half> input_f16(ctx.pool());
        if (!f32_input || rotated_f16_input) {
            input_f16.alloc((size_t) n_rows*IC);
        }
        ggml_cuda_pool_alloc<float> partial_f32(ctx.pool(), (size_t) n_rows*OC*splits);
        ggml_cuda_pool_alloc<half> output_f16(ctx.pool());
        if (!native_epilogue) {
            output_f16.alloc((size_t) n_rows*OC);
        }
        const size_t input_n = (size_t) n_rows*IC;
        const escha_official_bridge_api & bridge = escha_official_bridge_load();
        if (raw_decode_swiglu && !native_epilogue) {
            GGML_ABORT("escha: decode SwiGLU fusion requires the native F32 epilogue");
        }
        if (f32_input && !native_epilogue) {
            GGML_ABORT("escha: F32-input decode requires the native F32 epilogue");
        }
        if (f32_input && bridge.decode_main_f32 == nullptr) {
            GGML_ABORT("escha: official bridge is missing its F32-input decode ABI");
        }
        if (raw_decode_swiglu) {
            GGML_ASSERT(final_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(final_dst));
            if (raw_decode_up) {
                GGML_ASSERT(gate_silu->type == GGML_TYPE_F32 && ggml_is_contiguous(gate_silu));
                GGML_ASSERT(ggml_are_same_shape(gate_silu, dst));
            }
        }
        if (rotated_f16_input) {
            GGML_ASSERT(n_rows == 1 && ggml_cuda_info().devices[ctx.device].warp_size == 32);
            escha_rotate_in_dense_warp<half><<<n_rows, 256, 0, stream>>>(
                (const half *) rin->data, (const float *) x->data, input_f16.get(),
                IC, (int) x->ne[1], x->nb[1], x->nb[2]);
        } else if (!f32_input) {
            escha_copy_f32_f16<<<(input_n/2 + 255)/256, 256, 0, stream>>>(
                (const float *) x->data, input_f16.get(), input_n);
        }
        const float * decode_input = rotated_f16_input
            ? reinterpret_cast<const float *>(input_f16.get())
            : (const float *) x->data;
        const int rc = f32_input ? bridge.decode_main_f32(
                stream, ctx.device, partial_f32.get(), decode_input,
                (const int16_t *) code->data, rin->data,
                n_rows, IC, OC, K, splits)
            : native_epilogue ? bridge.decode_main(
                stream, ctx.device, partial_f32.get(), input_f16.get(),
                (const int16_t *) code->data, rin->data,
                n_rows, IC, OC, K, splits)
            : bridge.decode(
                stream, ctx.device, partial_f32.get(), output_f16.get(), input_f16.get(),
                (const int16_t *) code->data, rin->data, rout->data,
                n_rows, IC, OC, K, splits);
        if (rc != 0) {
            const char * error = bridge.error();
            GGML_ABORT("escha: official raw decode bridge failed: %s", error != nullptr ? error : "unknown error");
        }
        if (raw_decode_gate) {
            escha_finalize_dense_warp_swiglu<true, false><<<dim3(n_rows, output_groups), 32, 0, stream>>>(
                (const half *) rout->data, partial_f32.get(), nullptr, (float *) final_dst->data,
                OC, (int) x->ne[1], n_rows, splits, final_dst->nb[1], final_dst->nb[2]);
        } else if (raw_decode_up) {
            escha_finalize_dense_warp_swiglu<false, true><<<dim3(n_rows, output_groups), 32, 0, stream>>>(
                (const half *) rout->data, partial_f32.get(), (const float *) gate_silu->data,
                (float *) final_dst->data, OC, (int) x->ne[1], n_rows, splits,
                final_dst->nb[1], final_dst->nb[2]);
        } else if (native_epilogue) {
            escha_finalize_dense_warp<<<dim3(n_rows, output_groups), 32, 0, stream>>>(
                (const half *) rout->data, partial_f32.get(), (float *) dst->data,
                OC, (int) x->ne[1], n_rows, splits, dst->nb[1], dst->nb[2]);
        } else {
            const size_t output_n = (size_t) n_rows*OC;
            escha_copy_f16_f32<<<(output_n/2 + 255)/256, 256, 0, stream>>>(
                output_f16.get(), (float *) dst->data, output_n);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // The official bridge is a narrow experiment, not a format migration. It
    // consumes the model's current packed code and folded fp16 rotations in
    // place, and writes the ordinary fp32 projection result. Decode and fused
    // graph candidates intentionally stay on the native implementation.
    if (escha_official_bridge_requested() && use_mma && (K == 2 || K == 3)
            && IC % 128 == 0 && OC % 128 == 0
            && gate_silu == nullptr && final_dst == nullptr && !finalize_silu) {
        if (mvt.attestation_enabled()) {
            GGML_ABORT("escha: official bridge cannot run with an MVT attestation");
        }
        GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
        const escha_official_bridge_api & bridge = escha_official_bridge_load();
        // Existing checkpoints use codebook A. Mixed accumulation is the
        // qualified policy for the IC<=6144 projections; wider reductions use
        // fp32 accumulation, exactly as the current native route does.
        const int acc_mode = escha_official_bridge_acc_mode(code, K, IC);
        const int rc = bridge.gemm(stream, ctx.device, (float *) dst->data,
                                   (const float *) x->data, (const int16_t *) code->data,
                                   rin->data, rout->data, n_rows, IC, OC, K,
                                   1, 0, acc_mode);
        if (rc != 0) {
            const char * error = bridge.error();
            GGML_ABORT("escha: official bridge failed: %s", error != nullptr ? error : "unknown error");
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    ggml_cuda_pool_alloc<char> u_buf(ctx.pool(),
        (size_t) n_rows*IC*(use_mma ? sizeof(half) : sizeof(float)));

    const bool use_warp_rotate = !use_mma
                              && IC % 128 == 0
                              && ggml_cuda_info().devices[ctx.device].warp_size == 32
                              && getenv("ESCHA_NO_WARP_ROTATE") == nullptr;

    // E1 candidate probe only.  This is the existing warp-register/half route,
    // now selected solely through the nonce-bound MVT allow-list rather than a
    // free-standing knob.  The default MMA path remains shared-memory rotation.
    const bool use_e1_warp_rotate_half = use_mma
                                      && IC % 128 == 0
                                      && ggml_cuda_info().devices[ctx.device].warp_size == 32
                                      && mvt.candidate == escha_mvt_candidate::prefill_warp_rotate_half;

    if (use_mma) {
        if (use_e1_warp_rotate_half) {
            escha_rotate_in_dense_warp<half><<<n_rows, 256, 0, stream>>>(
                (const half *) rin->data, (const float *) x->data, (half *) u_buf.get(),
                IC, (int) x->ne[1], x->nb[1], x->nb[2]);
        } else {
            escha_rotate_in_dense<half><<<n_rows, 256, 0, stream>>>(
                (const half *) rin->data, (const float *) x->data, (half *) u_buf.get(),
                IC, (int) x->ne[1], x->nb[1], x->nb[2]);
        }
    } else if (use_warp_rotate) {
        escha_rotate_in_dense_warp<float><<<n_rows, 256, 0, stream>>>(
            (const half *) rin->data, (const float *) x->data, (float *) u_buf.get(),
            IC, (int) x->ne[1], x->nb[1], x->nb[2]);
    } else {
        escha_rotate_in_dense<float><<<n_rows, 256, 0, stream>>>(
            (const half *) rin->data, (const float *) x->data, (float *) u_buf.get(),
            IC, (int) x->ne[1], x->nb[1], x->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());

    const bool raw_swiglu_gate = finalize_silu && gate_silu == nullptr && final_dst != nullptr;
    const bool raw_swiglu_up   = !finalize_silu && gate_silu != nullptr && final_dst != nullptr;
    if (escha_official_raw_swiglu_fusion_requested() && use_mma && n_rows == 2048 &&
            (K == 2 || K == 3) && IC == 5120 && OC == 17408 &&
            (raw_swiglu_gate || raw_swiglu_up)) {
        if (mvt.attestation_enabled()) {
            GGML_ABORT("escha: official raw SwiGLU fusion cannot run with an MVT attestation");
        }
        GGML_ASSERT(ggml_is_contiguous(x));
        GGML_ASSERT(final_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(final_dst));
        if (raw_swiglu_up) {
            GGML_ASSERT(gate_silu->type == GGML_TYPE_F32 && ggml_is_contiguous(gate_silu));
            GGML_ASSERT(ggml_are_same_shape(gate_silu, dst));
        }

        const size_t output_n = (size_t) n_rows*OC;
        ggml_cuda_pool_alloc<half> output_f16(ctx.pool(), output_n);
        const escha_official_bridge_api & bridge = escha_official_bridge_load();
        const int rc = bridge.pretransformed(
            stream, ctx.device, output_f16.get(), u_buf.get(),
            (const int16_t *) code->data, rout->data, n_rows, IC, OC, K,
            escha_official_bridge_acc_mode(code, K, IC));
        if (rc != 0) {
            const char * error = bridge.error();
            GGML_ABORT("escha: official raw SwiGLU projection failed: %s",
                       error != nullptr ? error : "unknown error");
        }

        if (raw_swiglu_gate) {
            escha_f16_silu_f32<<<(output_n/2 + 255)/256, 256, 0, stream>>>(
                output_f16.get(), (float *) final_dst->data, output_n);
        } else {
            escha_f16_mul_f32<<<(output_n/2 + 255)/256, 256, 0, stream>>>(
                output_f16.get(), (const float *) gate_silu->data,
                (float *) final_dst->data, output_n);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (escha_official_raw_bridge_requested() && use_mma && (K == 2 || K == 3)
            && IC % 128 == 0 && OC % 128 == 0
            && gate_silu == nullptr && final_dst == nullptr && !finalize_silu) {
        if (mvt.attestation_enabled()) {
            GGML_ABORT("escha: official raw bridge cannot run with an MVT attestation");
        }
        GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
        ggml_cuda_pool_alloc<half> output_f16(ctx.pool(), (size_t) n_rows*OC);
        const escha_official_bridge_api & bridge = escha_official_bridge_load();
        const int rc = bridge.pretransformed(
            stream, ctx.device, output_f16.get(), u_buf.get(),
            (const int16_t *) code->data, rout->data,
            n_rows, IC, OC, K, escha_official_bridge_acc_mode(code, K, IC));
        if (rc != 0) {
            const char * error = bridge.error();
            GGML_ABORT("escha: official raw bridge failed: %s", error != nullptr ? error : "unknown error");
        }
        const size_t n = (size_t) n_rows*OC;
        escha_copy_f16_f32<<<(n/2 + 255)/256, 256, 0, stream>>>(
            output_f16.get(), (float *) dst->data, n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // slice the IC reduction only as far as it takes to fill the device: at batch 1 the
    // natural grid is just n_ocb blocks, but a long prompt already has plenty of rows
    // Generation sizes its row tile to the batch so a multi-row verify reads each weight
    // tile once; prefill keeps the 64-row amortization tile.  See escha_gen_row_tile.
    const int  R   = gen ? escha_gen_row_tile(n_rows) : ESCHA_ROWS_DENSE;

    const int n_rb = (n_rows + R - 1)/R;
    // the tiled prefill kernel blocks over BM rows x BN columns instead
    const int n_tb = (n_rows + ESCHA_BM - 1)/ESCHA_BM;
    const int n_cb = OC/ESCHA_BN;
    // batch 1 has only n_ocb blocks before slicing (136 for the FFN), which leaves an 82-SM
    // device mostly idle, so generation slices the reduction much harder than prefill
    // The generation-target candidates affect only this multiplier.  All other
    // paths keep the compiled default and the existing n_slices calculation.
    const int target = gen ? mvt.gen_target_mul*ESCHA_TARGET : ESCHA_TARGET;
    // n_slices must be derived from the R == 1 block count, not from the widened row tile.
    // A wider tile lowers n_rb, which would raise n_slices and therefore split the IC
    // reduction differently -- changing the reduction tree and the accumulated rounding.
    // Measured cost of getting this wrong: max_abs_diff 4.1e-3, rel_rms 2.1e-4 against the
    // control (numerically equivalent, but it fails the bit-exactness contract).  Keeping the
    // single-token slice structure means each row accumulates the identical sequence of
    // partial products, while the block still decodes its weight tile once and reuses it
    // across the R rows -- which is where the speed comes from.
    const int n_rb_slices = gen ? n_rows : n_rb;
    int n_slices = target/MAX(1, gen ? n_rb_slices*n_ocb : n_tb*n_cb);
    n_slices = MIN(MAX(n_slices, 1), nit);

    const bool e1_activation_selected =
        std::getenv("ESCHA_E1_ACTIVATION_CAPTURE_DIR") != nullptr &&
        escha_e1_filter_matches(std::getenv("ESCHA_E1_ACTIVATION_CAPTURE_TENSORS"), code->name);
    const bool e1_partial_selected =
        std::getenv("ESCHA_E1_PARTIAL_CAPTURE_DIR") != nullptr &&
        escha_e1_filter_matches(std::getenv("ESCHA_E1_PARTIAL_CAPTURE_TENSORS"), code->name);
    const bool e1_output_selected =
        std::getenv("ESCHA_E1_OUTPUT_CAPTURE_DIR") != nullptr &&
        escha_e1_filter_matches(std::getenv("ESCHA_E1_OUTPUT_CAPTURE_TENSORS"), code->name);
    const bool e1_selected = e1_activation_selected || e1_partial_selected || e1_output_selected;
    const uint64_t e1_invocation = e1_selected
        ? g_escha_e1_capture_invocation.fetch_add(1, std::memory_order_relaxed) + 1
        : 0;
    if (e1_activation_selected) {
        escha_e1_capture_activation(code, x, e1_invocation, n_rows, n_rows, IC, OC, K,
                                     n_slices, !mvt.candidate_enabled(), actual_route, stream);
    }

    const bool gen_target_candidate = mvt.candidate == escha_mvt_candidate::decode_gen_target_mul_2 ||
                                      mvt.candidate == escha_mvt_candidate::decode_gen_target_mul_4 ||
                                      mvt.candidate == escha_mvt_candidate::decode_gen_target_mul_8;
    const bool pipelined_decode_bn64_candidate = use_mma
        && mvt.candidate == escha_mvt_candidate::prefill_pipelined_decode_bn64;
    // Recovered promoted EXP-04 Stage-2 route: async A staging, BN128, and
    // native mixed accumulation (FP16 for IC<=6144).  It re-qualified on SM89
    // at +9.14% W2 / +8.01% E3 p2048 with neutral decode.  Keep the former
    // BN64 pipeline available as the immediate same-binary rollback.
    const bool use_mixedacc = use_mma
        && std::getenv("ESCHA_NO_PREFILL_MIXEDACC") == nullptr;
    // The BN64 pipelined route is the verified former SM89 prefill default.
    // It remains the rollback selected by ESCHA_NO_PREFILL_MIXEDACC=1; do not
    // overlap it with a different nonce-bound MVT candidate.
    const bool pipelined_decode_bn64_default = use_mma
        && !mvt.candidate_enabled()
        && !use_mixedacc
        && std::getenv("ESCHA_NO_PREFILL_PIPE_BN64") == nullptr;
    const bool use_pipelined_decode_bn64 = pipelined_decode_bn64_candidate
        || pipelined_decode_bn64_default;
    const bool pipelined_decode_candidate = use_mma
        && (mvt.candidate == escha_mvt_candidate::prefill_pipelined_decode ||
            pipelined_decode_bn64_candidate);
    const bool use_pipelined_decode = pipelined_decode_candidate
        || pipelined_decode_bn64_default;
    const bool fused_finalize_candidate = use_mma && n_slices == 1
        && mvt.candidate_enabled() && mvt.attestation_enabled()
        && mvt.candidate == escha_mvt_candidate::prefill_finalize_fused_single_slice;
    const bool direct_b_bn64_candidate = use_mma
        && mvt.candidate == escha_mvt_candidate::prefill_direct_b_fragment_bn64;
    const bool mma_bn64_candidate = use_mma
        && mvt.candidate == escha_mvt_candidate::prefill_mma_bn64;
    const bool direct_b_candidate = use_mma
        && (mvt.candidate == escha_mvt_candidate::prefill_direct_b_fragment ||
            direct_b_bn64_candidate);
    const bool swiglu_up_finalize_candidate = !gen && n_slices == 1
        && gate_silu != nullptr
        && (mvt.candidate == escha_mvt_candidate::prefill_swiglu_up_finalize_mul_single_slice ||
            mvt.candidate == escha_mvt_candidate::prefill_swiglu_gate_silu_up_finalize_mul_single_slice);
    const bool swiglu_gate_silu_candidate = !gen && n_slices == 1 && finalize_silu
        && mvt.candidate == escha_mvt_candidate::prefill_swiglu_gate_silu_up_finalize_mul_single_slice;
    const bool candidate_route = use_e1_warp_rotate_half || pipelined_decode_candidate
                                 || direct_b_candidate
                                 || mma_bn64_candidate
                                 || fused_finalize_candidate
                                 || swiglu_up_finalize_candidate
                                 || swiglu_gate_silu_candidate
                                 || (gen && gen_target_candidate);
    if (mvt.attestation_enabled()) {
        if (mvt.candidate_enabled()) {
            if (candidate_route) {
                g_escha_mvt_stats.target_route_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Candidate runs stay executable for ineligible shapes, but the MVT
                // record makes that control fallback explicit rather than claiming it.
                // fallback_count is a labeled subset of the actual control-route count.
                g_escha_mvt_stats.control_route_count.fetch_add(1, std::memory_order_relaxed);
                const bool candidate_target_node =
                    (mvt.candidate == escha_mvt_candidate::prefill_swiglu_up_finalize_mul_single_slice &&
                     gate_silu != nullptr) || swiglu_gate_silu_candidate;
                if (candidate_target_node) {
                    g_escha_mvt_stats.fallback_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    g_escha_mvt_stats.unaffected_route_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } else {
            g_escha_mvt_stats.control_route_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    escha_mvt_observe(K, IC, OC, n_rows);

    ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_slices*n_rows*OC);

    if (gen) {
        // widest slice any block gets, since lo/hi split nit unevenly by at most one tile
        const int tiles_max = (nit + n_slices - 1)/n_slices;
        // s_u is [R][ESCHA_TILE].  A wider row tile lowers n_slices, so the staged row
        // count that matters is the max of the two, not tiles_max alone.
        const size_t smem = ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) MAX(tiles_max, R)*ESCHA_TILE*sizeof(float);
        GGML_ASSERT(smem <= 48*1024 && "escha: staged u exceeds the default shared budget");
        // ---- multi-row generation: chunk-staged u + payload prefetch + hoisted schedule ----
        // These live in a SEPARATE kernel (escha_matmul_dense_adv) on purpose.  The R == 1
        // single-token instantiation must compile from the pristine text: an earlier revision
        // restructured the shared kernel so every instantiation got the multi-row loop nest,
        // which cost the no-MTP path 9.5 % on E3 (56.60 -> 51.25 t/s, same session, identity
        // verified) while flattering the MTP ratio.  Keep the two paths apart.
        const char * pfp_env = std::getenv("ESCHA_GEN_PFP");
        const char * sc_env  = std::getenv("ESCHA_GEN_SCHED");
        const char * cu_env  = std::getenv("ESCHA_GEN_R4_CHUNKU");
        const bool pfp   = (pfp_env == nullptr || pfp_env[0] != '0');
        const bool sched = (sc_env  == nullptr || sc_env[0]  != '0');
        const size_t smem_chunku = ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint2)
                                 + (size_t) 16*R*ESCHA_TILE*sizeof(float);
        // REJECTED 2026-09-18: extending chunku to R>8 measured WORSE (E3 M=9 94.31 -> 129.87 ms,
        // M=12 107.34 -> 137.15, M=16 129.95 -> 150.16; numerics byte-identical).  The plain
        // non-chunked staging is the better structure for deep R on this kernel.  Kept at R <= 8.
        const bool chunku = R > 1 && R <= 8
                         && (cu_env == nullptr || cu_env[0] != '0')
                         && smem_chunku <= 48*1024;
        const bool adv_pfp   = chunku && pfp;
        const bool adv_sched = adv_pfp && sched;
        const size_t smem_adv = adv_pfp ? smem_chunku : smem;
        const int16_t * k_code = (const int16_t *) code->data;
        const half    * k_lut  = (const half *)    lut->data;
        const int16_t * k_dep  = (const int16_t *) dep->data;
        const float   * k_u    = (const float *)   u_buf.get();
        float         * k_out  = p_buf.get();
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_rb, n_ocb, n_slices), ESCHA_NT, smem, stream>>>(
                k_code, k_lut, k_dep, k_u, k_out, IC, OC, n_rows, n_slices);
        };
        auto launch_adv = [&](auto kernel) {
            kernel<<<dim3(n_rb, n_ocb, n_slices), ESCHA_NT, smem_adv, stream>>>(
                k_code, k_lut, k_dep, k_u, k_out, IC, OC, n_rows, n_slices);
        };
        // One instantiation per row tile so R stays a compile-time constant for the
        // register-resident accumulator.  R == 1 is the unchanged single-token path.
        switch (K) {
            case 2:
                switch (R) {
                    case 1:  launch(escha_matmul_dense<2, 1>); break;
                    case 2:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 2, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 2, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 2>);
                        }
                        break;
                    case 3:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 3, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 3, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 3>);
                        }
                        break;
                    case 5:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 5, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 5, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 5>);
                        }
                        break;
                    case 6:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 6, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 6, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 6>);
                        }
                        break;
                    case 7:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 7, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 7, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 7>);
                        }
                        break;
                    case 4:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 4, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 4, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 4>);
                        }
                        break;
                    case 8:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 8, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 8, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 8>);
                        }
                        break;
                    case 9:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 9, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 9, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 9>);
                        }
                        break;
                    case 10:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 10, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 10, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 10>);
                        }
                        break;
                    case 11:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 11, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 11, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 11>);
                        }
                        break;
                    case 12:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 12, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 12, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 12>);
                        }
                        break;
                    case 13:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 13, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 13, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 13>);
                        }
                        break;
                    case 14:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 14, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 14, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 14>);
                        }
                        break;
                    case 15:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, 15, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, 15, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, 15>);
                        }
                        break;
                    default:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<2, ESCHA_GEN_MAX_ROWS, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<2, ESCHA_GEN_MAX_ROWS, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<2, ESCHA_GEN_MAX_ROWS>);
                        }
                        break;
                }
                break;
            case 3:
                switch (R) {
                    case 1:  launch(escha_matmul_dense<3, 1>); break;
                    case 2:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 2, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 2, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 2>);
                        }
                        break;
                    case 3:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 3, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 3, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 3>);
                        }
                        break;
                    case 5:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 5, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 5, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 5>);
                        }
                        break;
                    case 6:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 6, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 6, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 6>);
                        }
                        break;
                    case 7:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 7, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 7, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 7>);
                        }
                        break;
                    case 4:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 4, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 4, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 4>);
                        }
                        break;
                    case 8:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 8, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 8, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 8>);
                        }
                        break;
                    case 9:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 9, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 9, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 9>);
                        }
                        break;
                    case 10:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 10, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 10, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 10>);
                        }
                        break;
                    case 11:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 11, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 11, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 11>);
                        }
                        break;
                    case 12:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 12, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 12, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 12>);
                        }
                        break;
                    case 13:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 13, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 13, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 13>);
                        }
                        break;
                    case 14:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 14, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 14, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 14>);
                        }
                        break;
                    case 15:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, 15, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, 15, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, 15>);
                        }
                        break;
                    default:
                        if (adv_sched) {
                            launch_adv(escha_matmul_dense_adv<3, ESCHA_GEN_MAX_ROWS, true, true, true>);
                        } else if (adv_pfp) {
                            launch_adv(escha_matmul_dense_adv<3, ESCHA_GEN_MAX_ROWS, true, true, false>);
                        } else {
                            launch(escha_matmul_dense<3, ESCHA_GEN_MAX_ROWS>);
                        }
                        break;
                }
                break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    } else if (OC % ESCHA_BN != 0) {
        // the tiled kernel blocks the output axis in exact BN steps; a ragged OC would
        // silently leave the tail columns unwritten. Every projection in this checkpoint is
        // 128-aligned, but that is a property of the model, not of the format.
        const size_t smem = ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) ESCHA_ROWS_DENSE*ESCHA_TILE*sizeof(float);
        auto launch = [&](auto kernel) {
            kernel<<<dim3((n_rows + ESCHA_ROWS_DENSE - 1)/ESCHA_ROWS_DENSE, n_ocb, n_slices),
                     ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch(escha_matmul_dense<2, ESCHA_ROWS_DENSE>); break;
            case 3: launch(escha_matmul_dense<3, ESCHA_ROWS_DENSE>); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    } else if (use_mma) {
        // tensor-core prefill. Weights are exact; activations are rounded to fp16, which is
        // what escha's runtime does. ESCHA_NO_MMA=1 falls back to the fp32 FMA kernel.
        const bool use_pipe = use_pipelined_decode;
        const int candidate_bn = (direct_b_bn64_candidate || mma_bn64_candidate ||
                                  use_pipelined_decode_bn64) ? 64 : ESCHA_MMA_BN;
        // pipelined variant: s_pay triple-buffered, s_u and s_w double-buffered
        const size_t smem = use_pipe
            ? (size_t) 3*(candidate_bn/ESCHA_TILE)*ESCHA_MAX_W*sizeof(uint2)
            + (size_t) 2*ESCHA_MMA_BM*ESCHA_TILE*sizeof(half)
            + (size_t) 2*candidate_bn*ESCHA_TILE*sizeof(half)
            : (size_t) (candidate_bn/ESCHA_TILE)*ESCHA_MAX_W*sizeof(uint2)
            + (size_t) 2*ESCHA_MMA_BM*ESCHA_TILE*sizeof(half)
            + (direct_b_candidate ? 0 : (size_t) candidate_bn*ESCHA_TILE*sizeof(half));
        const int n_tb_mma = (n_rows + ESCHA_MMA_BM - 1)/ESCHA_MMA_BM;
        const int n_cb_mma = OC/candidate_bn;
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_tb_mma, n_cb_mma, n_slices), dim3(32, 256/32), smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const half *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices,
                (const half *) rout->data, (float *) dst->data,
                (int) x->ne[1], dst->nb[1], dst->nb[2]);
        };
        auto launch_pipe = [&](auto kernel) {
            kernel<<<dim3(n_tb_mma, n_cb_mma, n_slices), dim3(32, 256/32), smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const half *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        if (use_mixedacc) {
            switch (K) {
                case 2:
                    if (IC <= 6144) {
                        launch((escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, ESCHA_MMA_BN, false, false, true>));
                    } else {
                        launch((escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, ESCHA_MMA_BN, false, false, false>));
                    }
                    break;
                case 3:
                    if (IC <= 6144) {
                        launch((escha_matmul_dense_tiled_mma<3, ESCHA_MMA_BM, ESCHA_MMA_BN, false, false, true>));
                    } else {
                        launch((escha_matmul_dense_tiled_mma<3, ESCHA_MMA_BM, ESCHA_MMA_BN, false, false, false>));
                    }
                    break;
                default: GGML_ABORT("escha: unsupported K=%d", K);
            }
        } else if (use_pipelined_decode_bn64) {
            switch (K) {
                case 2: launch_pipe((escha_matmul_dense_tiled_mma_pipe<2, ESCHA_MMA_BM, 64>)); break;
                case 3: launch_pipe((escha_matmul_dense_tiled_mma_pipe<3, ESCHA_MMA_BM, 64>)); break;
                default: GGML_ABORT("escha: unsupported K=%d", K);
            }
        } else if (use_pipe) {
            switch (K) {
                case 2: launch_pipe((escha_matmul_dense_tiled_mma_pipe<2, ESCHA_MMA_BM, ESCHA_MMA_BN>)); break;
                case 3: launch_pipe((escha_matmul_dense_tiled_mma_pipe<3, ESCHA_MMA_BM, ESCHA_MMA_BN>)); break;
                default: GGML_ABORT("escha: unsupported K=%d", K);
            }
        } else if (mma_bn64_candidate) {
            switch (K) {
                case 2: launch((escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, 64, false, false>)); break;
                case 3: launch((escha_matmul_dense_tiled_mma<3, ESCHA_MMA_BM, 64, false, false>)); break;
                default: GGML_ABORT("escha: unsupported K=%d", K);
            }
        } else if (direct_b_bn64_candidate) {
            switch (K) {
                case 2: launch((escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, 64, false, true>)); break;
                case 3: launch((escha_matmul_dense_tiled_mma<3, ESCHA_MMA_BM, 64, false, true>)); break;
                default: GGML_ABORT("escha: unsupported K=%d", K);
            }
        } else if (direct_b_candidate) {
            switch (K) {
                case 2: launch((escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, ESCHA_MMA_BN, false, true>)); break;
                case 3: launch((escha_matmul_dense_tiled_mma<3, ESCHA_MMA_BM, ESCHA_MMA_BN, false, true>)); break;
                default: GGML_ABORT("escha: unsupported K=%d", K);
            }
        } else if (fused_finalize_candidate) {
            switch (K) {
                case 2: launch((escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, ESCHA_MMA_BN, true>)); break;
                case 3: launch((escha_matmul_dense_tiled_mma<3, ESCHA_MMA_BM, ESCHA_MMA_BN, true>)); break;
                default: GGML_ABORT("escha: unsupported K=%d", K);
            }
        } else {
            switch (K) {
                case 2: launch((escha_matmul_dense_tiled_mma<2, ESCHA_MMA_BM, ESCHA_MMA_BN, false>)); break;
                case 3: launch((escha_matmul_dense_tiled_mma<3, ESCHA_MMA_BM, ESCHA_MMA_BN, false>)); break;
                default: GGML_ABORT("escha: unsupported K=%d", K);
            }
        }
    } else {
        constexpr int NT  = (ESCHA_BM/ESCHA_TM)*(ESCHA_BN/ESCHA_TN);
        constexpr int NTJ = ESCHA_BN/ESCHA_TILE;
        const size_t smem = NTJ*ESCHA_MAX_W*sizeof(uint32_t)
                          + (size_t) ESCHA_TILE*ESCHA_BN*sizeof(float)
                          + (size_t) 2*ESCHA_TILE*ESCHA_BM*sizeof(float);
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_tb, n_cb, n_slices), NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch((escha_matmul_dense_tiled<2, ESCHA_BM, ESCHA_BN, ESCHA_TM, ESCHA_TN>)); break;
            case 3: launch((escha_matmul_dense_tiled<3, ESCHA_BM, ESCHA_BN, ESCHA_TM, ESCHA_TN>)); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    if (!fused_finalize_candidate) {
        escha_e1_capture_partial(code, p_buf.get(), (size_t) n_slices*n_rows*OC,
                                 e1_invocation, n_rows, IC, K, n_slices, n_rows, OC,
                                 actual_route, !mvt.candidate_enabled(), stream);

        if (swiglu_gate_silu_candidate) {
            GGML_ASSERT(final_dst != nullptr && final_dst->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(final_dst));
            escha_finalize_dense_silu<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
                (const half *) rout->data, p_buf.get(), (float *) final_dst->data,
                OC, (int) x->ne[1], n_rows, n_slices,
                final_dst->nb[1], final_dst->nb[2]);
        } else if (swiglu_up_finalize_candidate) {
            GGML_ASSERT(gate_silu->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_are_same_shape(gate_silu, dst));
            GGML_ASSERT(ggml_is_contiguous(gate_silu) && ggml_is_contiguous(dst));
            escha_finalize_dense_mul<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
                (const half *) rout->data, p_buf.get(), (const float *) gate_silu->data,
                (float *) final_dst->data, OC, (int) x->ne[1], n_rows, n_slices,
                final_dst->nb[1], final_dst->nb[2]);
        // The register/shuffle finalizer wins once there are at least 48
        // output groups; below that point, long slice sums favor the 128-thread
        // shared implementation.  Qualified bitwise on W2 and E3 decode.
        } else if (gen && n_ocb >= 48 &&
                   std::getenv("ESCHA_NO_DECODE_WARP_FINALIZE") == nullptr) {
            escha_finalize_dense_warp<<<dim3(n_rows, n_ocb), 32, 0, stream>>>(
                (const half *) rout->data, p_buf.get(), (float *) dst->data,
                OC, (int) x->ne[1], n_rows, n_slices, dst->nb[1], dst->nb[2]);
            escha_e1_capture_output(code, x, dst, e1_invocation, n_rows, n_rows, IC, OC, K,
                                    n_slices, !mvt.candidate_enabled(), actual_route, stream);
        } else {
            escha_finalize_dense<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
                (const half *) rout->data, p_buf.get(), (float *) dst->data,
                OC, (int) x->ne[1], n_rows, n_slices, dst->nb[1], dst->nb[2]);
            // Control-route output capture: only after the normal finalizer has
            // written dst.  Inert unless the ESCHA_E1_OUTPUT_CAPTURE_* pair is
            // supplied, and the hook itself refuses a candidate run.
            escha_e1_capture_output(code, x, dst, e1_invocation, n_rows, n_rows, IC, OC, K,
                                    n_slices, !mvt.candidate_enabled(), actual_route, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

bool ggml_cuda_escha_mul_mat_is_single_slice(const ggml_tensor * up) {
    if (up == nullptr || up->src[0] == nullptr || up->src[5] == nullptr ||
        up->ne[0] <= 0 || up->src[5]->ne[1] <= 0 || up->src[5]->ne[2] <= 0) {
        return false;
    }

    const int IC = up->src[0]->ne[2]*16;
    const int OC = up->ne[0];
    const int n_rows = up->src[5]->ne[1]*up->src[5]->ne[2];
    const int nit = IC/ESCHA_TILE;
    const int n_tb = (n_rows + ESCHA_BM - 1)/ESCHA_BM;
    const int n_cb = OC/ESCHA_BN;
    int n_slices = ESCHA_TARGET/MAX(1, n_tb*n_cb);
    n_slices = MIN(MAX(n_slices, 1), nit);
    return n_slices == 1;
}

void ggml_cuda_op_escha_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_escha_mul_mat_impl(ctx, dst, nullptr, nullptr, false);
}

bool ggml_cuda_escha_decode_down_add_rms_is_eligible(
        const ggml_tensor * down, const ggml_tensor * add,
        const ggml_tensor * rms, const ggml_tensor * mul) {
    const char * enabled =
        std::getenv("ESCHA_OFFICIAL_RAW_DECODE_DOWN_ADD_RMS_FUSION");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0 ||
        !escha_official_raw_decode_bridge_requested() ||
        !escha_official_raw_decode_native_epilogue_enabled() ||
        !escha_official_raw_decode_f32_input_requested() ||
        escha_official_raw_decode_rotated_f16_input_requested() ||
        down == nullptr || add == nullptr || rms == nullptr || mul == nullptr ||
        down->op != GGML_OP_ESCHA_MUL_MAT || add->op != GGML_OP_ADD ||
        rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL ||
        down->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
        rms->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        down->src[0] == nullptr || down->src[1] == nullptr ||
        down->src[2] == nullptr || down->src[5] == nullptr ||
        down->src[0]->type != GGML_TYPE_I16 ||
        down->src[0]->ne[0] != 48 || down->src[0]->ne[1]*16 != 5120 ||
        down->src[0]->ne[2]*16 != 17408 || down->src[1]->type != GGML_TYPE_F16 ||
        down->src[2]->type != GGML_TYPE_F16 || down->src[5]->type != GGML_TYPE_F32 ||
        down->ne[0] != 5120 || ggml_nrows(down) != 1 ||
        (add->src[0] != down && add->src[1] != down) ||
        rms->src[0] != add || (mul->src[0] != rms && mul->src[1] != rms) ||
        !ggml_are_same_shape(add->src[0], add->src[1]) ||
        !ggml_are_same_shape(down, add) || !ggml_are_same_shape(add, rms) ||
        !ggml_are_same_shape(rms, mul) ||
        !ggml_is_contiguous(down->src[5]) || !ggml_is_contiguous(down) ||
        !ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous(add->src[1]) ||
        !ggml_is_contiguous(add) || !ggml_is_contiguous(mul)) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    return weight != nullptr && weight->type == GGML_TYPE_F32 &&
           weight->ne[0] == 5120 && ggml_nrows(weight) == 1 &&
           ggml_is_contiguous(weight);
}

void ggml_cuda_op_escha_mul_mat_fused_down_add_rms(
        ggml_backend_cuda_context & ctx, ggml_tensor * down,
        ggml_tensor * add, ggml_tensor * rms, ggml_tensor * mul) {
    GGML_ASSERT(ggml_cuda_escha_decode_down_add_rms_is_eligible(
        down, add, rms, mul));
    constexpr int n_rows = 1;
    constexpr int ic = 17408;
    constexpr int oc = 5120;
    constexpr int splits = 12;
    constexpr int groups = oc/ESCHA_NT;
    constexpr int norm_threads = 1024;

    const ggml_tensor * residual_src = add->src[0] == down ? add->src[1] : add->src[0];
    const ggml_tensor * weight = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    float eps = 0.0f;
    std::memcpy(&eps, rms->op_params, sizeof(float));

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) oc*splits);
    ggml_cuda_pool_alloc<float> group_ss(ctx.pool(), groups);
    const escha_official_bridge_api & bridge = escha_official_bridge_load();
    if (bridge.decode_main_f32 == nullptr) {
        GGML_ABORT("escha: fused down/add/rms requires the F32-input decode ABI");
    }
    const int rc = bridge.decode_main_f32(
        stream, ctx.device, partial.get(), (const float *) down->src[5]->data,
        (const int16_t *) down->src[0]->data, down->src[1]->data,
        n_rows, ic, oc, 3, splits);
    if (rc != 0) {
        const char * error = bridge.error();
        GGML_ABORT("escha: fused down projection failed: %s",
                   error != nullptr ? error : "unknown error");
    }
    escha_finalize_dense_warp_add_ss<<<groups, 32, 0, stream>>>(
        (const half *) down->src[2]->data, partial.get(),
        (const float *) residual_src->data, (float *) add->data, group_ss.get(),
        oc, splits);
    escha_rms_mul_from_group_ss<norm_threads>
        <<<1, norm_threads, 32*sizeof(float), stream>>>(
        (const float *) add->data, group_ss.get(), (const float *) weight->data,
        (float *) mul->data, oc, groups, eps);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_escha_decode_qkv_z_is_eligible(const ggml_tensor * qkv,
                                              const ggml_tensor * z) {
    if (!escha_official_raw_decode_qkv_z_overlap_requested() ||
        qkv == nullptr || z == nullptr ||
        qkv->op != GGML_OP_ESCHA_MUL_MAT || z->op != GGML_OP_ESCHA_MUL_MAT ||
        qkv->type != GGML_TYPE_F32 || z->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(qkv) || !ggml_is_contiguous(z) ||
        qkv->ne[0] != 10240 || z->ne[0] != 6144 ||
        qkv->src[0] == nullptr || z->src[0] == nullptr ||
        qkv->src[1] == nullptr || z->src[1] == nullptr ||
        qkv->src[2] == nullptr || z->src[2] == nullptr ||
        qkv->src[5] == nullptr || qkv->src[5] != z->src[5] ||
        qkv->src[0]->type != GGML_TYPE_I16 || z->src[0]->type != GGML_TYPE_I16 ||
        qkv->src[0]->ne[0] != 32 || z->src[0]->ne[0] != 32 ||
        qkv->src[0]->ne[2]*16 != 5120 || z->src[0]->ne[2]*16 != 5120 ||
        qkv->src[5]->type != GGML_TYPE_F32 || qkv->src[5]->ne[0] != 5120) {
        return false;
    }
    return qkv->src[5]->ne[1]*qkv->src[5]->ne[2] == 1;
}

bool ggml_cuda_escha_decode_qkv_z_beta_alpha_is_eligible(
        const ggml_tensor * qkv, const ggml_tensor * z,
        const ggml_tensor * beta, const ggml_tensor * alpha) {
    const char * enabled =
        std::getenv("ESCHA_OFFICIAL_RAW_DECODE_QKV_Z_BETA_ALPHA_OVERLAP");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0 ||
        !ggml_cuda_escha_decode_qkv_z_is_eligible(qkv, z) ||
        beta == nullptr || alpha == nullptr ||
        beta->op != GGML_OP_MUL_MAT || alpha->op != GGML_OP_MUL_MAT ||
        beta->type != GGML_TYPE_F32 || alpha->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(beta) || !ggml_is_contiguous(alpha) ||
        beta->src[0] == nullptr || beta->src[1] == nullptr ||
        alpha->src[0] == nullptr || alpha->src[1] == nullptr ||
        beta->src[0]->type != GGML_TYPE_F16 || alpha->src[0]->type != GGML_TYPE_F16 ||
        beta->src[1] != qkv->src[5] || alpha->src[1] != qkv->src[5] ||
        beta->src[1]->type != GGML_TYPE_F32 || alpha->src[1]->type != GGML_TYPE_F32 ||
        beta->src[0]->ne[0] != 5120 || alpha->src[0]->ne[0] != 5120 ||
        beta->src[0]->ne[1] != beta->ne[0] || alpha->src[0]->ne[1] != alpha->ne[0]) {
        return false;
    }
    return beta->ne[1] == 1 && alpha->ne[1] == 1;
}

void ggml_cuda_op_escha_mul_mat_fused_qkv_z_beta_alpha(
        ggml_backend_cuda_context & ctx, ggml_tensor * qkv, ggml_tensor * z,
        ggml_tensor * beta, ggml_tensor * alpha) {
    GGML_ASSERT(ggml_cuda_escha_decode_qkv_z_beta_alpha_is_eligible(
        qkv, z, beta, alpha));
    if (!escha_official_raw_decode_native_epilogue_enabled() ||
        !escha_official_raw_decode_f32_input_requested() ||
        escha_official_raw_decode_rotated_f16_input_requested()) {
        GGML_ABORT("escha: fused qkv/z/beta/alpha overlap requires native epilogue and direct F32 input");
    }

    constexpr int n_rows = 1;
    constexpr int ic = 5120;
    constexpr int qkv_oc = 10240;
    constexpr int z_oc = 6144;
    constexpr int qkv_splits = 16;
    constexpr int z_splits = 32;
    cudaStream_t stream = ctx.stream();
    escha_decode_overlap_resources & resources =
        escha_decode_overlap_resources_get(ctx.device, stream);
    ggml_cuda_pool_alloc<float> qkv_partial(ctx.pool(), (size_t) qkv_oc*qkv_splits);
    ggml_cuda_pool_alloc<float> z_partial(ctx.pool(), (size_t) z_oc*z_splits);
    const escha_official_bridge_api & bridge = escha_official_bridge_load();
    if (bridge.decode_main_f32 == nullptr) {
        GGML_ABORT("escha: official bridge is missing its F32-input decode ABI");
    }

    CUDA_CHECK(cudaEventRecord(resources.fork, stream));
    CUDA_CHECK(cudaStreamWaitEvent(resources.stream, resources.fork, 0));
    const int z_rc = bridge.decode_main_f32(
        resources.stream, ctx.device, z_partial.get(), (const float *) z->src[5]->data,
        (const int16_t *) z->src[0]->data, z->src[1]->data,
        n_rows, ic, z_oc, 2, z_splits);
    if (z_rc != 0) {
        const char * error = bridge.error();
        GGML_ABORT("escha: fused z projection failed: %s",
                   error != nullptr ? error : "unknown error");
    }
    escha_finalize_dense_warp<<<dim3(n_rows, z_oc/128), 32, 0, resources.stream>>>(
        (const half *) z->src[2]->data, z_partial.get(), (float *) z->data,
        z_oc, (int) z->src[5]->ne[1], n_rows, z_splits, z->nb[1], z->nb[2]);
    ggml_cuda_mul_mat_vec_f(ctx, beta->src[0], beta->src[1], nullptr, beta,
                            nullptr, resources.stream);
    ggml_cuda_mul_mat_vec_f(ctx, alpha->src[0], alpha->src[1], nullptr, alpha,
                            nullptr, resources.stream);
    CUDA_CHECK(cudaEventRecord(resources.done, resources.stream));

    const int qkv_rc = bridge.decode_main_f32(
        stream, ctx.device, qkv_partial.get(), (const float *) qkv->src[5]->data,
        (const int16_t *) qkv->src[0]->data, qkv->src[1]->data,
        n_rows, ic, qkv_oc, 2, qkv_splits);
    if (qkv_rc != 0) {
        const char * error = bridge.error();
        GGML_ABORT("escha: fused qkv projection failed: %s",
                   error != nullptr ? error : "unknown error");
    }
    escha_finalize_dense_warp<<<dim3(n_rows, qkv_oc/128), 32, 0, stream>>>(
        (const half *) qkv->src[2]->data, qkv_partial.get(), (float *) qkv->data,
        qkv_oc, (int) qkv->src[5]->ne[1], n_rows, qkv_splits, qkv->nb[1], qkv->nb[2]);
    CUDA_CHECK(cudaStreamWaitEvent(stream, resources.done, 0));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_escha_mul_mat_fused_qkv_z(ggml_backend_cuda_context & ctx,
                                            ggml_tensor * qkv,
                                            ggml_tensor * z) {
    GGML_ASSERT(ggml_cuda_escha_decode_qkv_z_is_eligible(qkv, z));
    if (!escha_official_raw_decode_native_epilogue_enabled() ||
        !escha_official_raw_decode_f32_input_requested() ||
        escha_official_raw_decode_rotated_f16_input_requested()) {
        GGML_ABORT("escha: fused qkv/z overlap requires native epilogue and direct F32 input");
    }

    constexpr int n_rows = 1;
    constexpr int ic = 5120;
    constexpr int qkv_oc = 10240;
    constexpr int z_oc = 6144;
    constexpr int qkv_splits = 16;
    constexpr int z_splits = 32;
    cudaStream_t stream = ctx.stream();
    escha_decode_overlap_resources & resources =
        escha_decode_overlap_resources_get(ctx.device, stream);
    ggml_cuda_pool_alloc<float> qkv_partial(
        ctx.pool(), (size_t) qkv_oc*qkv_splits);
    ggml_cuda_pool_alloc<float> z_partial(
        ctx.pool(), (size_t) z_oc*z_splits);
    const escha_official_bridge_api & bridge = escha_official_bridge_load();
    if (bridge.decode_main_f32 == nullptr) {
        GGML_ABORT("escha: official bridge is missing its F32-input decode ABI");
    }

    CUDA_CHECK(cudaEventRecord(resources.fork, stream));
    CUDA_CHECK(cudaStreamWaitEvent(resources.stream, resources.fork, 0));
    const int z_rc = bridge.decode_main_f32(
        resources.stream, ctx.device, z_partial.get(), (const float *) z->src[5]->data,
        (const int16_t *) z->src[0]->data, z->src[1]->data,
        n_rows, ic, z_oc, 2, z_splits);
    if (z_rc != 0) {
        const char * error = bridge.error();
        GGML_ABORT("escha: fused z projection failed: %s",
                   error != nullptr ? error : "unknown error");
    }
    escha_finalize_dense_warp<<<dim3(n_rows, z_oc/128), 32, 0, resources.stream>>>(
        (const half *) z->src[2]->data, z_partial.get(), (float *) z->data,
        z_oc, (int) z->src[5]->ne[1], n_rows, z_splits, z->nb[1], z->nb[2]);
    CUDA_CHECK(cudaEventRecord(resources.done, resources.stream));

    const int qkv_rc = bridge.decode_main_f32(
        stream, ctx.device, qkv_partial.get(), (const float *) qkv->src[5]->data,
        (const int16_t *) qkv->src[0]->data, qkv->src[1]->data,
        n_rows, ic, qkv_oc, 2, qkv_splits);
    if (qkv_rc != 0) {
        const char * error = bridge.error();
        GGML_ABORT("escha: fused qkv projection failed: %s",
                   error != nullptr ? error : "unknown error");
    }
    escha_finalize_dense_warp<<<dim3(n_rows, qkv_oc/128), 32, 0, stream>>>(
        (const half *) qkv->src[2]->data, qkv_partial.get(), (float *) qkv->data,
        qkv_oc, (int) qkv->src[5]->ne[1], n_rows, qkv_splits, qkv->nb[1], qkv->nb[2]);
    CUDA_CHECK(cudaStreamWaitEvent(stream, resources.done, 0));
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_escha_decode_attn_qkv_is_eligible(const ggml_tensor * q,
                                                 const ggml_tensor * k,
                                                 const ggml_tensor * v) {
    if (!escha_official_raw_decode_attn_qkv_overlap_requested() ||
        q == nullptr || k == nullptr || v == nullptr ||
        q->op != GGML_OP_ESCHA_MUL_MAT ||
        k->op != GGML_OP_ESCHA_MUL_MAT ||
        v->op != GGML_OP_ESCHA_MUL_MAT ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v) ||
        q->ne[0] != 12288 || k->ne[0] != 1024 || v->ne[0] != 1024 ||
        q->src[0] == nullptr || k->src[0] == nullptr || v->src[0] == nullptr ||
        q->src[1] == nullptr || k->src[1] == nullptr || v->src[1] == nullptr ||
        q->src[2] == nullptr || k->src[2] == nullptr || v->src[2] == nullptr ||
        q->src[5] == nullptr || q->src[5] != k->src[5] || q->src[5] != v->src[5] ||
        q->src[0]->type != GGML_TYPE_I16 ||
        k->src[0]->type != GGML_TYPE_I16 || v->src[0]->type != GGML_TYPE_I16 ||
        q->src[0]->ne[0] != 32 || k->src[0]->ne[0] != 32 || v->src[0]->ne[0] != 32 ||
        q->src[0]->ne[2]*16 != 5120 ||
        k->src[0]->ne[2]*16 != 5120 || v->src[0]->ne[2]*16 != 5120 ||
        q->src[5]->type != GGML_TYPE_F32 || q->src[5]->ne[0] != 5120) {
        return false;
    }
    return q->src[5]->ne[1]*q->src[5]->ne[2] == 1;
}

void ggml_cuda_op_escha_mul_mat_fused_attn_qkv(ggml_backend_cuda_context & ctx,
                                               ggml_tensor * q,
                                               ggml_tensor * k,
                                               ggml_tensor * v) {
    GGML_ASSERT(ggml_cuda_escha_decode_attn_qkv_is_eligible(q, k, v));
    if (!escha_official_raw_decode_native_epilogue_enabled() ||
        !escha_official_raw_decode_f32_input_requested() ||
        escha_official_raw_decode_rotated_f16_input_requested()) {
        GGML_ABORT("escha: fused attention QKV overlap requires native epilogue and direct F32 input");
    }

    constexpr int n_rows = 1;
    constexpr int ic = 5120;
    constexpr int q_oc = 12288;
    constexpr int kv_oc = 1024;
    constexpr int q_splits = 16;
    constexpr int kv_splits = 40;
    cudaStream_t stream = ctx.stream();
    escha_decode_overlap_resources & resources =
        escha_decode_overlap_resources_get(ctx.device, stream);
    ggml_cuda_pool_alloc<float> q_partial(ctx.pool(), (size_t) q_oc*q_splits);
    ggml_cuda_pool_alloc<float> k_partial(ctx.pool(), (size_t) kv_oc*kv_splits);
    ggml_cuda_pool_alloc<float> v_partial(ctx.pool(), (size_t) kv_oc*kv_splits);
    const escha_official_bridge_api & bridge = escha_official_bridge_load();
    if (bridge.decode_main_f32 == nullptr) {
        GGML_ABORT("escha: official bridge is missing its F32-input decode ABI");
    }

    CUDA_CHECK(cudaEventRecord(resources.fork, stream));
    CUDA_CHECK(cudaStreamWaitEvent(resources.stream, resources.fork, 0));
    const int k_rc = bridge.decode_main_f32(
        resources.stream, ctx.device, k_partial.get(), (const float *) k->src[5]->data,
        (const int16_t *) k->src[0]->data, k->src[1]->data,
        n_rows, ic, kv_oc, 2, kv_splits);
    if (k_rc != 0) {
        const char * error = bridge.error();
        GGML_ABORT("escha: fused attention K projection failed: %s",
                   error != nullptr ? error : "unknown error");
    }
    escha_finalize_dense_warp<<<dim3(n_rows, kv_oc/128), 32, 0, resources.stream>>>(
        (const half *) k->src[2]->data, k_partial.get(), (float *) k->data,
        kv_oc, (int) k->src[5]->ne[1], n_rows, kv_splits, k->nb[1], k->nb[2]);

    const int v_rc = bridge.decode_main_f32(
        resources.stream, ctx.device, v_partial.get(), (const float *) v->src[5]->data,
        (const int16_t *) v->src[0]->data, v->src[1]->data,
        n_rows, ic, kv_oc, 2, kv_splits);
    if (v_rc != 0) {
        const char * error = bridge.error();
        GGML_ABORT("escha: fused attention V projection failed: %s",
                   error != nullptr ? error : "unknown error");
    }
    escha_finalize_dense_warp<<<dim3(n_rows, kv_oc/128), 32, 0, resources.stream>>>(
        (const half *) v->src[2]->data, v_partial.get(), (float *) v->data,
        kv_oc, (int) v->src[5]->ne[1], n_rows, kv_splits, v->nb[1], v->nb[2]);
    CUDA_CHECK(cudaEventRecord(resources.done, resources.stream));

    const int q_rc = bridge.decode_main_f32(
        stream, ctx.device, q_partial.get(), (const float *) q->src[5]->data,
        (const int16_t *) q->src[0]->data, q->src[1]->data,
        n_rows, ic, q_oc, 2, q_splits);
    if (q_rc != 0) {
        const char * error = bridge.error();
        GGML_ABORT("escha: fused attention Q projection failed: %s",
                   error != nullptr ? error : "unknown error");
    }
    escha_finalize_dense_warp<<<dim3(n_rows, q_oc/128), 32, 0, stream>>>(
        (const half *) q->src[2]->data, q_partial.get(), (float *) q->data,
        q_oc, (int) q->src[5]->ne[1], n_rows, q_splits, q->nb[1], q->nb[2]);
    CUDA_CHECK(cudaStreamWaitEvent(stream, resources.done, 0));
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_escha_mvt_up_finalize_is_eligible(const ggml_tensor * up) {
    const bool candidate_enabled =
        ggml_backend_cuda_escha_mvt_candidate_enabled(
            "prefill_swiglu_up_finalize_mul_single_slice") ||
        ggml_backend_cuda_escha_mvt_candidate_enabled(
            "prefill_swiglu_gate_silu_up_finalize_mul_single_slice");
    const bool raw_candidate = escha_official_raw_swiglu_fusion_requested();
    const bool raw_decode_candidate = escha_official_raw_decode_swiglu_fusion_requested();
    if ((!candidate_enabled && !raw_candidate && !raw_decode_candidate) ||
        up == nullptr || up->src[0] == nullptr || up->src[5] == nullptr ||
        up->type != GGML_TYPE_F32 || !ggml_is_contiguous(up) || up->ne[0] != 17408 ||
        up->src[0]->ne[2]*16 != 5120 || up->src[5]->ne[0] != 5120 ||
        up->src[5]->ne[1] <= 0 || up->src[5]->ne[2] <= 0) {
        return false;
    }
    const int n_rows = up->src[5]->ne[1]*up->src[5]->ne[2];
    if (raw_candidate || raw_decode_candidate) {
        return (raw_candidate && n_rows == 2048) ||
               (raw_decode_candidate && n_rows == 1);
    }
    if (n_rows != 512) {
        return false;
    }
    const int nit = 5120/ESCHA_TILE;
    const int n_tb = (512 + ESCHA_BM - 1)/ESCHA_BM;
    const int n_cb = 17408/ESCHA_BN;
    const int n_slices = MIN(MAX(ESCHA_TARGET/MAX(1, n_tb*n_cb), 1), nit);
    return 512 > ESCHA_GEN_MAX_ROWS && n_slices == 1;
}

void ggml_cuda_op_escha_mul_mat_fused_up(ggml_backend_cuda_context & ctx,
                                         ggml_tensor * up,
                                         ggml_tensor * mul) {
    const ggml_tensor * gate_silu = up != nullptr ? up->src[6] : nullptr;
    GGML_ASSERT(ggml_cuda_escha_mvt_up_finalize_is_eligible(up));
    GGML_ASSERT(gate_silu != nullptr && mul != nullptr);
    GGML_ASSERT(gate_silu->op == GGML_OP_UNARY && ggml_get_unary_op(gate_silu) == GGML_UNARY_OP_SILU);
    GGML_ASSERT(gate_silu->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(gate_silu, up) && ggml_are_same_shape(up, mul));
    GGML_ASSERT(ggml_is_contiguous(gate_silu) && ggml_is_contiguous(up) && ggml_is_contiguous(mul));
    ggml_cuda_op_escha_mul_mat_impl(ctx, up, gate_silu, mul, false);
}

bool ggml_cuda_escha_mvt_gate_silu_is_eligible(const ggml_tensor * gate,
                                              const ggml_tensor * silu) {
    const bool candidate_enabled = ggml_backend_cuda_escha_mvt_candidate_enabled(
        "prefill_swiglu_gate_silu_up_finalize_mul_single_slice");
    const bool raw_candidate = escha_official_raw_swiglu_fusion_requested();
    const bool raw_decode_candidate = escha_official_raw_decode_swiglu_fusion_requested();
    if ((!candidate_enabled && !raw_candidate && !raw_decode_candidate) ||
        gate == nullptr || silu == nullptr || gate->src[0] == nullptr ||
        gate->src[5] == nullptr || gate->src[6] != nullptr ||
        gate->op != GGML_OP_ESCHA_MUL_MAT || silu->op != GGML_OP_UNARY ||
        ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != gate ||
        gate->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(gate) || !ggml_is_contiguous(silu) ||
        !ggml_are_same_shape(gate, silu) || gate->ne[0] != 17408 ||
        gate->src[0]->ne[2]*16 != 5120 || gate->src[5]->ne[0] != 5120 ||
        gate->src[5]->ne[1] <= 0 || gate->src[5]->ne[2] <= 0) {
        return false;
    }
    const int n_rows = gate->src[5]->ne[1]*gate->src[5]->ne[2];
    if (raw_candidate || raw_decode_candidate) {
        return (raw_candidate && n_rows == 2048) ||
               (raw_decode_candidate && n_rows == 1);
    }
    if (n_rows != 512) {
        return false;
    }
    const int nit = 5120/ESCHA_TILE;
    const int n_tb = (512 + ESCHA_BM - 1)/ESCHA_BM;
    const int n_cb = 17408/ESCHA_BN;
    const int n_slices = MIN(MAX(ESCHA_TARGET/MAX(1, n_tb*n_cb), 1), nit);
    return 512 > ESCHA_GEN_MAX_ROWS && n_slices == 1;
}

void ggml_cuda_op_escha_mul_mat_fused_gate_silu(ggml_backend_cuda_context & ctx,
                                                ggml_tensor * gate,
                                                ggml_tensor * silu) {
    GGML_ASSERT(ggml_cuda_escha_mvt_gate_silu_is_eligible(gate, silu));
    ggml_cuda_op_escha_mul_mat_impl(ctx, gate, nullptr, silu, true);
}

#include <cuda.h>
#include "../gdn-matched-sm86/launch-metadata.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

constexpr int T = 512; // maximum allocation; dispatch uses padded runtime length
constexpr int H = 48;
constexpr int HG = 16;
constexpr int D = 128;
constexpr int NT = T / 64;

thread_local std::string last_error;

void cuda_ok(cudaError_t status, const char * what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void cu_ok(CUresult status, const char * what) {
    if (status == CUDA_SUCCESS) return;
    const char * text = nullptr;
    cuGetErrorString(status, &text);
    throw std::runtime_error(std::string(what) + ": " + (text ? text : "unknown driver error"));
}

__global__ void cast_grouped_qk(
        const float * q, const float * k, __nv_bfloat16 * qg, __nv_bfloat16 * kg,
        int64_t token_stride, int padded, int valid) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) padded * HG * D;
    if (i >= n) return;
    const int d = i % D;
    const size_t x = i / D;
    const int hg = x % HG;
    const int t = x / HG;
    const int source_head = token_stride == HG*D ? hg : hg*3;
    const size_t src = (size_t) t * token_stride + source_head * D + d;
    qg[i] = t < valid ? __float2bfloat16_rn(q[src]) : __float2bfloat16_rn(0.0f);
    kg[i] = t < valid ? __float2bfloat16_rn(k[src]) : __float2bfloat16_rn(0.0f);
}

__global__ void cast_compact_v(const float * v, __nv_bfloat16 * compact, int64_t token_stride, int padded, int valid) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) padded * H * D;
    if (i >= n) return;
    const int d = i % D;
    const size_t x = i / D;
    const int h = x % H;
    const int t = x / H;
    compact[i] = t < valid ? __float2bfloat16_rn(v[(size_t) t * token_stride + h * D + d]) : __float2bfloat16_rn(0.0f);
}

__global__ void state_to_fla(const float * physical, float * fla) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) H * D * D;
    if (i >= n) return;
    const int col = i % D;
    const size_t x = i / D;
    const int row = x % D;
    const int head = x / D;
    fla[i] = physical[((size_t) head * D + col) * D + row];
}

__global__ void state_from_fla(const float * fla, float * physical) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) H * D * D;
    if (i >= n) return;
    const int col = i % D;
    const size_t x = i / D;
    const int row = x % D;
    const int head = x / D;
    physical[((size_t) head * D + col) * D + row] = fla[i];
}

__global__ void output_to_f32(const __nv_bfloat16 * src, float * dst, int padded, int valid) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) padded * H * D;
    if (i < (size_t) valid * H * D) dst[i] = __bfloat162float(src[i]);
}

// Zero beta and log-decay make padded positions leave the state unchanged.
__global__ void pad_g_beta(const float * g, const float * beta, float * gp, float * bp,
                          int padded, int valid) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) padded * H) return;
    gp[i] = i < (size_t) valid * H ? g[i] : 0.0f;
    bp[i] = i < (size_t) valid * H ? beta[i] : 0.0f;
}

struct Kernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

struct Workspace {
    int device = -1;
    Kernel cumsum, kkt, solve, merge, recompute, state, output;
    __nv_bfloat16 * q = nullptr;
    __nv_bfloat16 * k = nullptr;
    __nv_bfloat16 * v = nullptr;
    __nv_bfloat16 * Ai = nullptr;
    __nv_bfloat16 * w = nullptr;
    __nv_bfloat16 * u = nullptr;
    __nv_bfloat16 * h = nullptr;
    __nv_bfloat16 * vnew = nullptr;
    __nv_bfloat16 * out = nullptr;
    float * gsum = nullptr;
    float * gpad = nullptr;
    float * bpad = nullptr;
    float * A = nullptr;
    float * Ad = nullptr;
    float * state_fla = nullptr;
    int * indices = nullptr;
};

Workspace workspace;
std::once_flag init_once;
std::mutex owner_mutex;
bool ready = false;
cudaStream_t owner = nullptr;

Kernel load_kernel(const std::string & path, const char * name, int shared) {
    Kernel k;
    cu_ok(cuModuleLoad(&k.module, path.c_str()), "cuModuleLoad");
    cu_ok(cuModuleGetFunction(&k.function, k.module, name), "cuModuleGetFunction");
    if (shared > 48 * 1024) {
        cu_ok(cuFuncSetAttribute(k.function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared),
              "cuFuncSetAttribute");
    }
    return k;
}

void allocate(void ** ptr, size_t bytes) { cuda_ok(cudaMalloc(ptr, bytes), "cudaMalloc GDN workspace"); }

void initialize(int device) {
    cuda_ok(cudaSetDevice(device), "cudaSetDevice");
    cu_ok(cuInit(0), "cuInit");
    const char * root_env = std::getenv("L0XRE_GDN_MASKED_CUBIN_ROOT");
    if (root_env == nullptr || root_env[0] == '\0') {
        throw std::runtime_error("L0XRE_GDN_MASKED_CUBIN_ROOT is required");
    }
    const std::string root = root_env;
    workspace.device = device;
    workspace.cumsum = load_kernel(root + "/cumsum.cubin", GDN_CUMSUM_NAME, GDN_CUMSUM_SHARED);
    workspace.kkt = load_kernel(root + "/kkt.cubin", GDN_KKT_NAME, GDN_KKT_SHARED);
    workspace.solve = load_kernel(root + "/solve.cubin", GDN_SOLVE_NAME, GDN_SOLVE_SHARED);
    workspace.merge = load_kernel(root + "/merge.cubin", GDN_MERGE_NAME, GDN_MERGE_SHARED);
    workspace.recompute = load_kernel(root + "/recompute.cubin", GDN_RECOMPUTE_NAME, GDN_RECOMPUTE_SHARED);
    workspace.state = load_kernel(root + "/state.cubin", GDN_STATE_NAME, GDN_STATE_SHARED);
    workspace.output = load_kernel(root + "/output.cubin", GDN_OUTPUT_NAME, GDN_OUTPUT_SHARED);
    allocate((void **) &workspace.q, (size_t) T*HG*D*2);
    allocate((void **) &workspace.k, (size_t) T*HG*D*2);
    allocate((void **) &workspace.v, (size_t) T*H*D*2);
    allocate((void **) &workspace.gsum, (size_t) T*H*4);
    allocate((void **) &workspace.gpad, (size_t) T*H*4);
    allocate((void **) &workspace.bpad, (size_t) T*H*4);
    allocate((void **) &workspace.A, (size_t) T*H*64*4);
    allocate((void **) &workspace.Ad, (size_t) T*H*16*4);
    allocate((void **) &workspace.Ai, (size_t) T*H*64*2);
    allocate((void **) &workspace.w, (size_t) T*H*D*2);
    allocate((void **) &workspace.u, (size_t) T*H*D*2);
    allocate((void **) &workspace.h, (size_t) NT*H*D*D*2);
    allocate((void **) &workspace.vnew, (size_t) T*H*D*2);
    allocate((void **) &workspace.out, (size_t) T*H*D*2);
    allocate((void **) &workspace.state_fla, (size_t) H*D*D*4);
    allocate((void **) &workspace.indices, sizeof(int));
    cuda_ok(cudaMemset(workspace.indices, 0, sizeof(int)), "cudaMemset state index");
}

void launch(CUfunction fn, dim3 grid, int warps, int shared, void ** args, cudaStream_t stream) {
    cu_ok(cuLaunchKernel(fn, grid.x, grid.y, grid.z, warps*32, 1, 1, shared,
                        reinterpret_cast<CUstream>(stream), args, nullptr), "cuLaunchKernel");
}

} // namespace

extern "C" int escha_gdn_chunk_prefill_masked_v1(
        cudaStream_t stream, int device,
        const float * q32, const float * k32, const float * v32,
        const float * g32, const float * beta32, const float * state_in,
        float * dst32, float * state_out,
        int64_t S_v, int64_t heads, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sv1, int64_t sv2,
        int64_t sb1, int64_t sb2, float scale, int64_t valid_tokens) {
    try {
        last_error.clear();
        // Serialize the entire shared-workspace launch sequence, including
        // callers from different host threads on the same CUDA stream.
        std::lock_guard<std::mutex> dispatch_guard(owner_mutex);
        if (q32 == nullptr || k32 == nullptr || v32 == nullptr || g32 == nullptr ||
            beta32 == nullptr || state_in == nullptr || dst32 == nullptr || state_out == nullptr ||
            device != 0 || S_v != D || heads != H || n_tokens < 64 || n_tokens > T ||
            n_tokens % 64 != 0 || valid_tokens <= 0 || valid_tokens > n_tokens || n_seqs != 1 ||
            // BeeLlama's ggml tensors are laid out as [D,H,T,B]: the
            // element stride is one and the token stride is H*D.  The
            // earlier standalone probe accidentally required D for the
            // element stride, which could never match the runtime tensor.
            // sq1/sv1 are head strides in float elements, not scalar strides.
            sq1 != D || sq2 != H*D || sv1 != D || sv2 < H*D ||
            sb1 != 1 || sb2 != H || std::abs(scale - 1.0f/std::sqrt((float)D)) > 1e-6f) {
            return 1; // unsupported: no output writes
        }
        // Allocate only outside capture; callers must warm up before graph replay.
        // One owner stream prevents cross-stream reuse of the shared scratch.
        {
            if (!ready) {
                cudaStreamCaptureStatus capture;
                cuda_ok(cudaStreamIsCapturing(stream, &capture), "capture status");
                if (capture != cudaStreamCaptureStatusNone) return 1;
                // Decline before module/workspace initialization under memory
                // pressure. The caller can execute native without output writes.
                size_t free_bytes = 0, total_bytes = 0;
                cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "workspace headroom");
                constexpr size_t required_bytes =
                    2ull*T*HG*D*2 + 5ull*T*H*D*2 + 3ull*T*H*4 +
                    1ull*T*H*64*4 + 1ull*T*H*16*4 + 1ull*T*H*64*2 +
                    1ull*NT*H*D*D*2 + 1ull*H*D*D*4 + sizeof(int);
                static_assert(required_bytes == 62685188, "workspace accounting");
                constexpr size_t module_margin = 8ull*1024*1024;
                if (free_bytes < required_bytes + module_margin) return 1;
                std::call_once(init_once, [device] { initialize(device); });
                owner = stream;
                ready = true;
            }
            if (workspace.device != device || owner != stream) return 1;
        }
        cuda_ok(cudaSetDevice(device), "cudaSetDevice dispatch");
        const int padded = (int) n_tokens;
        const int valid = (int) valid_tokens;
        const int chunks = padded / 64;
        constexpr int threads = 256;
        cast_grouped_qk<<<((size_t)padded*HG*D+threads-1)/threads,threads,0,stream>>>(
            q32,k32,workspace.q,workspace.k,sq2,padded,valid);
        cast_compact_v<<<((size_t)padded*H*D+threads-1)/threads,threads,0,stream>>>(
            v32,workspace.v,sv2,padded,valid);
        pad_g_beta<<<((size_t)padded*H+threads-1)/threads,threads,0,stream>>>(
            g32,beta32,workspace.gpad,workspace.bpad,padded,valid);
        state_to_fla<<<((size_t)H*D*D+threads-1)/threads,threads,0,stream>>>(
            state_in,workspace.state_fla);

        const int t = padded;
        CUdeviceptr null_scratch=0;
        CUdeviceptr pg=(CUdeviceptr)workspace.gpad, pgs=(CUdeviceptr)workspace.gsum;
        // Non-varlen Triton ABI: slots 2/3 and 5/6 are retained but unused;
        // T is slot 4; this fresh kernel set is explicitly non-varlen.
        void * ac[]={&pg,&pgs,&null_scratch,&null_scratch,const_cast<int *>(&t),&null_scratch,&null_scratch};
        launch(workspace.cumsum.function,dim3(chunks,H,1),GDN_CUMSUM_WARPS,GDN_CUMSUM_SHARED,ac,stream);
        CUdeviceptr pk=(CUdeviceptr)workspace.k,pb=(CUdeviceptr)workspace.bpad,pA=(CUdeviceptr)workspace.A;
        // KKT: k,beta,g,A,cu_seqlens,chunk_indices,T plus two unused slots.
        void * ak[]={&pk,&pb,&pgs,&pA,&null_scratch,&null_scratch,const_cast<int *>(&t),&null_scratch,&null_scratch};
        launch(workspace.kkt.function,dim3(chunks,H,1),GDN_KKT_WARPS,GDN_KKT_SHARED,ak,stream);
        CUdeviceptr pAd=(CUdeviceptr)workspace.Ad;
        // solve: A,Ai,cu_seqlens,chunk_indices,T plus two unused slots.
        void * as[]={&pA,&pAd,&null_scratch,&null_scratch,const_cast<int *>(&t),&null_scratch,&null_scratch};
        launch(workspace.solve.function,dim3(padded/16,H,1),GDN_SOLVE_WARPS,GDN_SOLVE_SHARED,as,stream);
        CUdeviceptr pAi=(CUdeviceptr)workspace.Ai;
        // Fresh v0.10.2 merge ABI: A,Ad,Ai,cu_seqlens,chunk_indices,T,global scratch,profile scratch.
        void * am[]={&pA,&pAd,&pAi,&null_scratch,&null_scratch,const_cast<int *>(&t),&null_scratch,&null_scratch};
        launch(workspace.merge.function,dim3(chunks,H,1),GDN_MERGE_WARPS,GDN_MERGE_SHARED,am,stream);
        CUdeviceptr pv=(CUdeviceptr)workspace.v,pw=(CUdeviceptr)workspace.w,pu=(CUdeviceptr)workspace.u;
        // recompute: k,v,beta,w,u,A,g,cu_seqlens,chunk_indices,T.
        void * ar[]={&pk,&pv,&pb,&pw,&pu,&pAi,&pgs,&null_scratch,&null_scratch,const_cast<int *>(&t),&null_scratch,&null_scratch};
        launch(workspace.recompute.function,dim3(chunks,H,1),GDN_RECOMPUTE_WARPS,GDN_RECOMPUTE_SHARED,ar,stream);
        CUdeviceptr pvnew=(CUdeviceptr)workspace.vnew,ph=(CUdeviceptr)workspace.h;
        CUdeviceptr ps=(CUdeviceptr)workspace.state_fla,pi=(CUdeviceptr)workspace.indices;
        // Fresh state ABI: k,v,w,v_new,g,h,h0,ht,cu_seqlens,chunk_offsets,T.
        // h0/ht alias is safe: each program owns disjoint columns and loads
        // its complete initial state before writing the final state.
        void * ah[]={&pk,&pu,&pw,&pvnew,&pgs,&ph,&ps,&ps,
                     &null_scratch,&null_scratch,const_cast<int *>(&t),
                     &null_scratch,&null_scratch};
        launch(workspace.state.function,dim3(2,H,1),GDN_STATE_WARPS,GDN_STATE_SHARED,ah,stream);
        cuda_ok(cudaMemsetAsync(workspace.out,0,(size_t)padded*H*D*2,stream),"cudaMemset output");
        CUdeviceptr pq=(CUdeviceptr)workspace.q,po=(CUdeviceptr)workspace.out;
        // output: q,k,v,h,g,o,cu_seqlens,chunk_indices,scale,T plus two unused slots.
        void * ao[]={&pq,&pk,&pvnew,&ph,&pgs,&po,&null_scratch,&null_scratch,
                     &scale,const_cast<int *>(&t),&null_scratch,&null_scratch};
        launch(workspace.output.function,dim3(2,chunks,H),GDN_OUTPUT_WARPS,GDN_OUTPUT_SHARED,ao,stream);
        output_to_f32<<<((size_t)valid*H*D+threads-1)/threads,threads,0,stream>>>(workspace.out,dst32,padded,valid);
        state_from_fla<<<((size_t)H*D*D+threads-1)/threads,threads,0,stream>>>(workspace.state_fla,state_out);
        cuda_ok(cudaPeekAtLastError(), "GDN bridge launch");
        return 0;
    } catch (const std::exception & e) {
        last_error=e.what();
        return -1;
    } catch (...) {
        last_error="unknown GDN bridge error";
        return -2;
    }
}

extern "C" const char * escha_gdn_chunk_error() { return last_error.c_str(); }

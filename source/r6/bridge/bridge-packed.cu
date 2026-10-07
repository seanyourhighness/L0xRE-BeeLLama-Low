#include "decode-transform.cuh"
#include "decode-transform-swizzled.cuh"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "escha_official_bridge_v1.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Independent GEMM route for the oracle: materialize working INT8 weights,
// then use cuBLAS INT8->INT32. Quantization helpers are shared deliberately;
// agreement checks packed-tile indexing/MMA/epilogue, not model quality.
#include "int8-prefill-helpers.cuh"
#include <cublas_v2.h>

template<int K>
static __global__ void materialize_i8(const int16_t * code,int8_t * dense,int IC,int OC) {
    constexpr int NW=8*K;
    __shared__ uint32_t payload[8*NW];
    const int tid=threadIdx.x,ct=blockIdx.y*8,kt=blockIdx.x,nct=OC/16;
    for(int j=tid;j<8*NW;j+=256) payload[j]=((const uint32_t*)(code+(int64_t)(kt*nct+ct)*(16*K)))[j];
    __syncthreads();
    const int r=tid%16,c=tid/16;
    int shift=((32-K)-K*(dep_pi(r)+32*c+4*(c>>3)))%(NW*32);
    if(shift<0)shift+=NW*32;
    int g=shift>>5,w=g?(NW-g):0,prev=w?(w-1):(NW-1);
#pragma unroll
    for(int k=0;k<8;k++) {
        uint32_t idx=__funnelshift_r(payload[k*NW+w],payload[k*NW+prev],shift&31)&65535;
        int8_t q=(int8_t)__float2int_rn(__half2float(codebook(idx))*32.0f);
        dense[(int64_t)((ct+k)*16+c)*IC+kt*16+r]=q;
    }
}
// One warp handles each H128 output block, with four values per lane.
// Preserve the original seven butterfly stages and both final multiplications.
static __global__ void finish_i8(const int *accum,const float *scales,const half *rout,half *out,int OC,int M) {
    const int lane=threadIdx.x&31,group=blockIdx.x*8+threadIdx.x/32;
    const int groups_per_row=OC/128;
    if(group>=M*groups_per_row)return;
    const int row=group/groups_per_row,col=(group%groups_per_row)*128;
    const float scale=scales[row]/32.0f;
    float v[4];
#pragma unroll
    for(int j=0;j<4;j++)v[j]=float(accum[(int64_t)row*OC+col+j*32+lane])*scale;
#pragma unroll
    for(int len=1;len<32;len<<=1) {
#pragma unroll
        for(int j=0;j<4;j++) {
            const float other=__shfl_xor_sync(0xffffffff,v[j],len);
            v[j]=(lane&len)?other-v[j]:v[j]+other;
        }
    }
    // Butterfly len32, then len64, in exactly the shared-memory order.
    {float a=v[0],b=v[1];v[0]=a+b;v[1]=a-b;}
    {float a=v[2],b=v[3];v[2]=a+b;v[3]=a-b;}
    {float a=v[0],b=v[2];v[0]=a+b;v[2]=a-b;}
    {float a=v[1],b=v[3];v[1]=a+b;v[3]=a-b;}
#pragma unroll
    for(int j=0;j<4;j++) {
        v[j]*=rsqrtf(128.0f);
        out[(int64_t)row*OC+col+j*32+lane]=__float2half_rn(v[j]*__half2float(rout[col+j*32+lane]));
    }
}

// Isolated single-GPU, single-slot experiment. Not a general concurrent bridge.
// Preload initializes process-lifetime scratch before any CUDA graph capture.
// Only the designated prefill stream may enqueue here; all other paths retain B74.
#include <cstdio>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
namespace l0xre_i8 {
constexpr int MAX_IC=17408, MAX_OC=17408, CAP=512;
constexpr size_t WEIGHT_ELEMS=size_t(17408)*5120;
constexpr size_t CUBLAS_WORK=4*1024*1024;
struct State {
    int mode=0, mask=1; bool initialized=false, owner_set=false;
    cudaStream_t owner=nullptr;
    int8_t *q=nullptr,*weight=nullptr;float *scales=nullptr;int *accum=nullptr;
    void *work=nullptr;cublasHandle_t handle=nullptr;
    unsigned long calls[8]={}, other_stream[8]={};
    std::mutex mutex;
};
static State & state() { static State *s=new State;return *s; }
static void must(cudaError_t e,const char *what) {
    if(e!=cudaSuccess) {fprintf(stderr,"L0XRE_I8_INIT_FAILED %s: %s\n",what,cudaGetErrorString(e));_exit(91);}
}
static void cb(cublasStatus_t e,const char *what) {
    if(e!=CUBLAS_STATUS_SUCCESS) {fprintf(stderr,"L0XRE_I8_CUBLAS_FAILED %s: %d\n",what,int(e));_exit(92);}
}
#ifndef _WIN32
__attribute__((constructor))
#endif
static void startup() {
    const char *env=getenv("L0XRE_INT8_PREFILL");
    if(!env || (strcmp(env,"1") && strcmp(env,"2")))return;
    auto &s=state();s.mode=atoi(env);
    if(const char *mask=getenv("L0XRE_INT8_PROJ_MASK")) {
        char *end=nullptr;long value=strtol(mask,&end,10);
        if(!*mask || *end || value<0 || value>255){fprintf(stderr,"L0XRE_I8_BAD_MASK\n");_exit(93);}
        s.mask=int(value);
    }
    must(cudaSetDevice(0),"device");
    must(cudaMalloc(&s.weight,WEIGHT_ELEMS),"weight");
    must(cudaMalloc(&s.q,size_t(CAP)*MAX_IC),"activations");
    must(cudaMalloc(&s.accum,size_t(CAP)*MAX_OC*sizeof(int)),"accumulators");
    must(cudaMalloc(&s.scales,CAP*sizeof(float)),"scales");
    must(cudaMalloc(&s.work,CUBLAS_WORK),"cublas workspace");
    cb(cublasCreate(&s.handle),"create");
    cb(cublasSetWorkspace(s.handle,s.work,CUBLAS_WORK),"workspace");
    must(cudaMemset(s.weight,0,WEIGHT_ELEMS),"zero weights");
    must(cudaMemset(s.q,0,size_t(CAP)*MAX_IC),"zero activations");
    const int alpha=1,beta=0;
    const int shapes[][2]={{17408,5120},{5120,17408},{5120,10240},{6144,5120},{5120,6144},{5120,12288},{5120,1024}};
    for(const auto &shape: shapes) { const int IC=shape[0],OC=shape[1];
    for(int m: {256,512})cb(cublasGemmEx(s.handle,CUBLAS_OP_T,CUBLAS_OP_N,OC,m,IC,
        &alpha,s.weight,CUDA_R_8I,IC,s.q,CUDA_R_8I,IC,&beta,s.accum,CUDA_R_32I,OC,
        CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"warmup");
    }
    must(cudaDeviceSynchronize(),"warmup synchronize");s.initialized=true;
    fprintf(stderr,"L0XRE_I8_READY mode=%d mask=%d capacity=%d explicit_bytes=%zu single_slot_only=1\n",s.mode,s.mask,CAP,
        WEIGHT_ELEMS+size_t(CAP)*(MAX_IC+MAX_OC*sizeof(int)+sizeof(float))+CUBLAS_WORK);
}
// 1 means fall through to untouched B74; 0 means success; -1 means failure.
static int run(cudaStream_t stream,int device,void *output,const void *input,
        const int16_t *code,const void *rout,int M,int ic,int oc,int K,int acc) {
    auto &s=state();
    if(!s.initialized || s.mode!=1 || device!=0 || M<256 || M>CAP)return 1;
    const int role=(ic==17408 && oc==5120 && K==3 && acc==0)?0:
                   (ic==5120 && oc==17408 && K==2 && acc==1)?1:
                   (ic==5120 && oc==17408 && K==3 && acc==1)?2:
                   (ic==5120 && oc==10240 && K==2 && acc==1)?3:
                   (ic==6144 && oc==5120 && K==2 && acc==1)?4:
                   (ic==5120 && oc==6144 && K==2 && acc==1)?5:
                   (ic==5120 && oc==12288 && K==2 && acc==1)?6:
                   (ic==5120 && oc==1024 && K==2 && acc==1)?7:-1;
    if(role<0 || !(s.mask&(1<<role)))return 1;
    const int IC=ic,OC=oc;
    std::lock_guard<std::mutex> lock(s.mutex);
    if(s.owner_set && s.owner!=stream) {
        if(s.other_stream[role]++==0)fprintf(stderr,"L0XRE_I8_OTHER_STREAM_FALLBACK role=%d\n",role);
        return 1;
    }
    if(!s.owner_set){s.owner=stream;s.owner_set=true;}
    cb(cublasSetStream(s.handle,stream),"stream");
    // cublasSetStream resets workspace: restore it on every enqueue.
    cb(cublasSetWorkspace(s.handle,s.work,CUBLAS_WORK),"workspace");
    must(cudaGetLastError(),"entry last error");
    if(K==2)materialize_i8<2><<<dim3(IC/16,OC/128),256,0,stream>>>(code,s.weight,IC,OC);
    else materialize_i8<3><<<dim3(IC/16,OC/128),256,0,stream>>>(code,s.weight,IC,OC);
    must(cudaGetLastError(),"materialize launch");
    quantize_rows<<<M,256,0,stream>>>((const half*)input,s.q,s.scales,IC);
    must(cudaGetLastError(),"quantize launch");
    const int alpha=1,beta=0;
    cb(cublasGemmEx(s.handle,CUBLAS_OP_T,CUBLAS_OP_N,OC,M,IC,
        &alpha,s.weight,CUDA_R_8I,IC,s.q,CUDA_R_8I,IC,&beta,s.accum,CUDA_R_32I,OC,
        CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"gemm");
    finish_i8<<<(M*(OC/128)+7)/8,256,0,stream>>>(s.accum,s.scales,(const half*)rout,(half*)output,OC,M);
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){fprintf(stderr,"L0XRE_I8_LAUNCH_FAILED %s\n",cudaGetErrorString(err));return -1;}
    if(s.calls[role]++<2)fprintf(stderr,"L0XRE_I8_ENQUEUE role=%d M=%d IC=%d OC=%d K=%d stream=%p\n",role,M,IC,OC,K,(void*)stream);
    return 0;
}
} // namespace l0xre_i8
#ifdef _WIN32
extern "C" __declspec(dllexport) void l0xre_int8_prefill_init_v1() {
    static std::once_flag once;
    std::call_once(once, [] { l0xre_i8::startup(); });
}
#endif

// CUDA-only replacement for the Sprint/Bridge C ABI.  The code-GEMM and
// F32-input decode cubins are extracted from the retained Escha wheel; this
// adapter deliberately does not call torch, SGLang, or the native BeeLlama
// prefill implementation.

namespace {

thread_local std::string g_error;
std::mutex g_mutex;

CUmodule g_code_module = nullptr;
std::unordered_map<std::string, CUfunction> g_code_functions;
std::unordered_map<uint64_t, float *> g_ones;

struct decode_module_key {
    int device;
    std::string path;

    bool operator == (const decode_module_key & other) const {
        return device == other.device && path == other.path;
    }
};

struct decode_module_key_hash {
    size_t operator()(const decode_module_key & key) const {
        size_t h = std::hash<int>{}(key.device);
        h ^= std::hash<std::string>{}(key.path) + size_t(0x9e3779b9) + (h << 6) + (h >> 2);
        return h;
    }
};

struct decode_module_functions {
    CUmodule module = nullptr;
    CUfunction k2 = nullptr;
    CUfunction k3 = nullptr;
};

std::unordered_map<decode_module_key, decode_module_functions, decode_module_key_hash>
    g_decode_modules;

struct input_buffer_key {
    int device;
    uintptr_t stream;
    size_t length;

    bool operator == (const input_buffer_key & other) const {
        return device == other.device && stream == other.stream && length == other.length;
    }
};

struct input_buffer_key_hash {
    size_t operator()(const input_buffer_key & key) const {
        size_t h = std::hash<int>{}(key.device);
        h ^= std::hash<uintptr_t>{}(key.stream) + size_t(0x9e3779b9) + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.length) + size_t(0x9e3779b9) + (h << 6) + (h >> 2);
        return h;
    }
};

// The conversion is asynchronous.  A buffer keyed only by (device, length)
// can be overwritten by a second stream while the first decode still reads
// it.  Stream-keying preserves the no-hot-allocation behavior while making
// concurrent gate/up or DFlash-style streams independent.
std::unordered_map<input_buffer_key, uint16_t *, input_buffer_key_hash> g_input_f16;
std::unordered_map<input_buffer_key, float *, input_buffer_key_hash> g_rounded_f32;

void set_driver_error(const char * what, CUresult status) {
    const char * text = nullptr;
    cuGetErrorString(status, &text);
    g_error = std::string(what) + ": " + (text ? text : "unknown driver error");
}

bool ensure_context(int device) {
    if (cudaSetDevice(device) != cudaSuccess || cuInit(0) != CUDA_SUCCESS) {
        g_error = "failed to initialize CUDA context";
        return false;
    }
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
        g_error = "cuCtxGetCurrent failed";
        return false;
    }
    if (current == nullptr) {
        CUdevice cu_device = 0;
        if (cuDeviceGet(&cu_device, device) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&current, cu_device) != CUDA_SUCCESS ||
            cuCtxSetCurrent(current) != CUDA_SUCCESS) {
            g_error = "failed to establish CUDA primary context";
            return false;
        }
    }
    return true;
}

bool load_code_module(int device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_code_module != nullptr) {
        return true;
    }
    const char * path = std::getenv("ESCHA_OFFICIAL_CODE_GEMM_CUBIN");
    if (path == nullptr || path[0] == '\0') {
        g_error = "ESCHA_OFFICIAL_CODE_GEMM_CUBIN is required";
        return false;
    }
    if (!ensure_context(device)) {
        return false;
    }
    const CUresult status = cuModuleLoad(&g_code_module, path);
    if (status != CUDA_SUCCESS) {
        set_driver_error("cuModuleLoad code-GEMM cubin failed", status);
        g_code_module = nullptr;
        return false;
    }
    return true;
}

bool load_decode_module(int device, const char * requested_path,
                        CUfunction * k2_out, CUfunction * k3_out) {
    const char * path = requested_path;
    if (path == nullptr || path[0] == '\0') {
        path = std::getenv("ESCHA_OFFICIAL_F32_DECODE_CUBIN");
    }
    if (path == nullptr || path[0] == '\0') {
        g_error = "ESCHA_OFFICIAL_F32_DECODE_CUBIN is required";
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    const decode_module_key key { device, path };
    auto cached = g_decode_modules.find(key);
    if (cached != g_decode_modules.end()) {
        if (k2_out != nullptr) {
            *k2_out = cached->second.k2;
        }
        if (k3_out != nullptr) {
            *k3_out = cached->second.k3;
        }
        return true;
    }
    if (!ensure_context(device)) {
        return false;
    }
    decode_module_functions functions;
    CUresult status = cuModuleLoad(&functions.module, path);
    if (status != CUDA_SUCCESS) {
        set_driver_error("cuModuleLoad F32 decode cubin failed", status);
        return false;
    }
    constexpr const char * k2_symbol =
        "_ZN12escha_escham21escham_gemv_bw_kernelILi1ELi2ELb0ELb0ELb0EEEvPfPK6__halfPKtS4_PKfiiiiiPS2_S4_S8_Pi";
    constexpr const char * k3_symbol =
        "_ZN12escha_escham21escham_gemv_bw_kernelILi1ELi3ELb0ELb0ELb0EEEvPfPK6__halfPKtS4_PKfiiiiiPS2_S4_S8_Pi";
    status = cuModuleGetFunction(&functions.k2, functions.module, k2_symbol);
    if (status == CUDA_SUCCESS) {
        status = cuModuleGetFunction(&functions.k3, functions.module, k3_symbol);
    }
    if (status != CUDA_SUCCESS) {
        set_driver_error("cuModuleGetFunction F32 decode failed", status);
        cuModuleUnload(functions.module);
        return false;
    }

    auto inserted = g_decode_modules.emplace(key, functions);
    if (k2_out != nullptr) {
        *k2_out = inserted.first->second.k2;
    }
    if (k3_out != nullptr) {
        *k3_out = inserted.first->second.k3;
    }
    return true;
}

const char * code_symbol(int K, bool small, int acc_mode) {
    if (K == 2 && !small && acc_mode == 0) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi2ELi128ELi64ELi2ELb0ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 2 && !small && acc_mode == 1) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi2ELi128ELi64ELi2ELb1ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 3 && !small && acc_mode == 0) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi3ELi128ELi64ELi2ELb0ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 3 && !small && acc_mode == 1) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi3ELi128ELi64ELi2ELb1ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 2 && small && acc_mode == 0) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi2ELi64ELi32ELi3ELb0ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 2 && small && acc_mode == 1) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi2ELi64ELi32ELi3ELb1ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 3 && small && acc_mode == 0) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi3ELi64ELi32ELi3ELb0ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 3 && small && acc_mode == 1) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi3ELi64ELi32ELi3ELb1ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    return nullptr;
}

CUfunction resolve_code(int device, int K, bool small, int acc_mode) {
    if (!load_code_module(device)) {
        return nullptr;
    }
    const char * symbol = code_symbol(K, small, acc_mode);
    if (symbol == nullptr) {
        g_error = "unsupported raw Escha code-GEMM selection";
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_code_functions.find(symbol);
    if (it != g_code_functions.end()) {
        return it->second;
    }
    CUfunction function = nullptr;
    const CUresult status = cuModuleGetFunction(&function, g_code_module, symbol);
    if (status != CUDA_SUCCESS) {
        set_driver_error("cuModuleGetFunction code-GEMM failed", status);
        return nullptr;
    }
    g_code_functions.emplace(symbol, function);
    return function;
}

float * ones_for(int device, int length) {
    const uint64_t key = (uint64_t(uint32_t(device)) << 32) | uint32_t(length);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_ones.find(key);
    if (it != g_ones.end()) {
        return it->second;
    }
    std::vector<float> host((size_t) length, 1.0f);
    float * device_ptr = nullptr;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMalloc(&device_ptr, host.size() * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(device_ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        if (device_ptr != nullptr) {
            cudaFree(device_ptr);
        }
        g_error = "failed to allocate Escha scale vector";
        return nullptr;
    }
    g_ones.emplace(key, device_ptr);
    return device_ptr;
}

uint16_t * input_f16_for(int device, cudaStream_t stream, size_t length) {
    const input_buffer_key key {
        device,
        reinterpret_cast<uintptr_t>(stream),
        length,
    };
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_input_f16.find(key);
    if (it != g_input_f16.end()) {
        return it->second;
    }
    uint16_t * device_ptr = nullptr;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMalloc(&device_ptr, length * sizeof(uint16_t)) != cudaSuccess) {
        g_error = "failed to allocate Escha FP16 decode input";
        if (device_ptr != nullptr) {
            cudaFree(device_ptr);
        }
        return nullptr;
    }
    g_input_f16.emplace(key, device_ptr);
    return device_ptr;
}

float * rounded_f32_for(int device, cudaStream_t stream, size_t length) {
    const input_buffer_key key {
        device,
        reinterpret_cast<uintptr_t>(stream),
        length,
    };
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_rounded_f32.find(key);
    if (it != g_rounded_f32.end()) return it->second;
    float * device_ptr = nullptr;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMalloc(&device_ptr, length * sizeof(float)) != cudaSuccess) {
        g_error = "failed to allocate rounded FP32 decode input";
        if (device_ptr != nullptr) cudaFree(device_ptr);
        return nullptr;
    }
    g_rounded_f32.emplace(key, device_ptr);
    return device_ptr;
}

// The retained code-GEMM decode symbols consume the original F16 activation
// ABI.  The retargeted f32_input cubin is different: the PTX changes the four
// activation loads to ld.global.nc.f32 while retaining the historical mangled
// name.  Make that ABI choice explicit instead of silently feeding one type to
// the other (which produces plausible-looking but incorrect tokens).
bool decode_input_is_f32_abi() {
    const char * value = std::getenv("ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "f16") == 0) {
        return false;
    }
    if (std::strcmp(value, "f32") == 0) {
        return true;
    }
    g_error = "ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI must be f16 or f32";
    return false;
}

bool require_direct_f32_for_batched_decode() {
    const char * value = std::getenv("ESCHA_OFFICIAL_REQUIRE_DIRECT_F32");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

__global__ void convert_f32_to_f16(const float * src, uint16_t * dst, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        reinterpret_cast<__half *>(dst)[i] = __float2half(src[i]);
    }
}

__global__ void round_f32_via_f16(const float * src, float * dst, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(__float2half(src[i]));
}

int decode_main_f32_launch(
        cudaStream_t stream,
        int device,
        float * partial_f32,
        const float * x_f32,
        const int16_t * code,
        const void * rin_f16,
        int M,
        int IC,
        int OC,
        int K,
        int splits,
        uint32_t input_abi,
        uint32_t flags,
        const char * cubin_path) {
    if (partial_f32 == nullptr || x_f32 == nullptr || code == nullptr || rin_f16 == nullptr ||
        M <= 0 || M > 16 || IC <= 0 || OC <= 0 || (K != 2 && K != 3) ||
        IC % 128 != 0 || OC % 128 != 0 || splits <= 0 || splits > IC / 128) {
        g_error = "invalid F32-input decode-main arguments";
        return -1;
    }
    if (input_abi != ESCHA_OFFICIAL_BRIDGE_INPUT_F16 &&
        input_abi != ESCHA_OFFICIAL_BRIDGE_INPUT_F32) {
        g_error = "unsupported Escha decode input ABI";
        return -1;
    }
    if ((flags & ESCHA_OFFICIAL_BRIDGE_FLAG_ROTATED_F16_INPUT) != 0) {
        g_error = "rotated-F16 input is not supported by the decode-main descriptor";
        return -1;
    }

    const char * transformed_path = std::getenv("L0XRE_DECODE_TRANSFORM_CUBIN");
    // Preserve existing input precision semantics; only the current F16-rounded route qualifies.
    const bool transformed = transformed_path && transformed_path[0] && M <= 8 &&
                             input_abi == ESCHA_OFFICIAL_BRIDGE_INPUT_F16;
    const char * vector_k3_path = std::getenv("L0XRE_K3_VECTOR_CUBIN");
    const bool vector_k3 = K == 3 && vector_k3_path != nullptr && vector_k3_path[0] != '\0';
    CUfunction decode_k2 = nullptr;
    CUfunction decode_k3 = nullptr;
    if (!load_decode_module(device, transformed ? transformed_path : (vector_k3 ? vector_k3_path : cubin_path),
                            &decode_k2, &decode_k3)) {
        return -1;
    }
    float * s_in = ones_for(device, IC);
    if (s_in == nullptr) {
        return -1;
    }
    const bool input_f32_abi = input_abi == ESCHA_OFFICIAL_BRIDGE_INPUT_F32;
    if (M > 1 && require_direct_f32_for_batched_decode() && !input_f32_abi && !vector_k3) {
        g_error = "batched decode requires ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32";
        return -1;
    }

    void * input_arg = nullptr;
    const size_t input_count = size_t(M) * size_t(IC);
    if (transformed) {
        uint16_t * rotated = input_f16_for(device, stream, input_count);
        if (!rotated) return -1;
        const char * packed_flag = std::getenv("L0XRE_PACKED_DECODE_INPUT");
        if (packed_flag && std::strcmp(packed_flag,"1") == 0) {
        decode_transform_swizzled<<<(M*(IC/128)+7)/8,256,0,stream>>>(x_f32,
            reinterpret_cast<const half *>(rin_f16),s_in,
            reinterpret_cast<half *>(rotated),IC,M);
        } else {
        decode_transform<<<(M*(IC/128)+7)/8,256,0,stream>>>(x_f32,
            reinterpret_cast<const half *>(rin_f16),s_in,
            reinterpret_cast<half *>(rotated),IC,M);
        }
        const cudaError_t transform_error = cudaGetLastError();
        if (transform_error != cudaSuccess) {
            g_error = std::string("decode activation transform launch failed: ") + cudaGetErrorString(transform_error);
            return -1;
        }
        input_arg=rotated;
    } else if (vector_k3) {
        float * rounded = rounded_f32_for(device, stream, input_count);
        if (rounded == nullptr) return -1;
        round_f32_via_f16<<<(input_count + 255) / 256, 256, 0, stream>>>(
            x_f32, rounded, input_count);
        if (cudaGetLastError() != cudaSuccess) {
            g_error = "failed to launch Escha FP16-rounded FP32 decode conversion";
            return -1;
        }
        input_arg = rounded;
    } else if (input_f32_abi) {
        input_arg = const_cast<float *>(x_f32);
    } else {
        uint16_t * input_f16 = input_f16_for(device, stream, input_count);
        if (input_f16 == nullptr) {
            return -1;
        }
        convert_f32_to_f16<<<(input_count + 255) / 256, 256, 0, stream>>>(
            x_f32, input_f16, input_count);
        if (cudaGetLastError() != cudaSuccess) {
            g_error = "failed to launch Escha FP32-to-FP16 decode conversion";
            return -1;
        }
        input_arg = input_f16;
    }

    const uint16_t * packed = reinterpret_cast<const uint16_t *>(code);
    const void * rin = rin_f16;
    int output_blocks = OC / 16;
    void * bias = nullptr;
    void * bias_scale = nullptr;
    float * correction = nullptr;
    int * route = nullptr;
    void * arguments[] = {
        &partial_f32, &input_arg, const_cast<uint16_t **>(&packed),
        const_cast<void **>(&rin), &s_in, &M, &IC, &OC, &output_blocks, &splits,
        &bias, &bias_scale, &correction, &route,
    };
    const CUresult status = cuLaunchKernel(
        K == 2 ? decode_k2 : decode_k3,
        (unsigned) (OC / 128), (unsigned) splits, 1,
        256, 1, 1, 0, reinterpret_cast<CUstream>(stream), arguments, nullptr);
    if (status != CUDA_SUCCESS) {
        set_driver_error("raw Escha F32 decode launch failed", status);
        return -1;
    }
    return 0;
}

int unsupported(const char * name) {
    g_error = std::string(name) + " is unavailable in the CUDA-only raw bridge";
    return -1;
}

} // namespace

extern "C" int escha_official_code_gemm(
        cudaStream_t, int, float *, const float *, const int16_t *, const void *, const void *,
        int, int, int, int, int, int, int) {
    return unsupported("escha_official_code_gemm");
}

extern "C" int escha_official_code_gemm_pretransformed(
        cudaStream_t stream,
        int device,
        void * dst_f16,
        const void * x_rotated_f16,
        const int16_t * code,
        const void * rout_f16,
        int M,
        int IC,
        int OC,
        int K,
        int acc_mode) {
    try {
        g_error.clear();
        if (dst_f16 == nullptr || x_rotated_f16 == nullptr || code == nullptr ||
            rout_f16 == nullptr || M <= 0 || IC <= 0 || OC <= 0 ||
            (K != 2 && K != 3) || IC % 128 != 0 || OC % 128 != 0 ||
            (acc_mode != 0 && acc_mode != 1)) {
            g_error = "invalid pretransformed Escha GEMM argument";
            return -1;
        }
        const int i8_result=l0xre_i8::run(stream,device,dst_f16,x_rotated_f16,
            code,rout_f16,M,IC,OC,K,acc_mode);
        if(i8_result<=0) {
            if(i8_result<0)g_error="experimental INT8 FFN prefill launch failed";
            return i8_result;
        }
        const bool small = OC <= 1024;
        CUfunction function = resolve_code(device, K, small, acc_mode);
        if (function == nullptr) {
            return -1;
        }
        float * accumulation = nullptr;
        auto * output = static_cast<uint16_t *>(dst_f16);
        const auto * input = static_cast<const uint16_t *>(x_rotated_f16);
        const auto * packed = reinterpret_cast<const uint16_t *>(code);
        const auto * rout = static_cast<const uint16_t *>(rout_f16);
        float * s_out = ones_for(device, OC);
        if (s_out == nullptr) {
            return -1;
        }
        const int output_blocks = OC / 16;
        void * arguments[] = {
            &accumulation, &output, &input, &packed, &rout, &s_out,
            &M, &IC, &OC, const_cast<int *>(&output_blocks),
        };
        const int BM = small ? 64 : 128;
        const CUresult status = cuLaunchKernel(
            function, (unsigned) (OC / 128), (unsigned) ((M + BM - 1) / BM), 1,
            256, 1, 1, 0, reinterpret_cast<CUstream>(stream), arguments, nullptr);
        if (status != CUDA_SUCCESS) {
            set_driver_error("raw Escha code-GEMM launch failed", status);
            return -1;
        }
        return 0;
    } catch (...) {
        g_error = "unexpected exception in raw Escha code-GEMM";
        return -2;
    }
}

extern "C" int escha_official_decode_gemv_raw(
        cudaStream_t, int, float *, void *, const void *, const int16_t *, const void *, const void *,
        int, int, int, int, int) {
    return unsupported("escha_official_decode_gemv_raw");
}

extern "C" int escha_official_decode_gemv_main_raw(
        cudaStream_t, int, float *, const void *, const int16_t *, const void *,
        int, int, int, int, int) {
    return unsupported("escha_official_decode_gemv_main_raw");
}

extern "C" int escha_official_decode_gemv_main_f32_raw(
        cudaStream_t stream,
        int device,
        float * partial_f32,
        const float * x_f32,
        const int16_t * code,
        const void * rin_f16,
        int M,
        int IC,
        int OC,
        int K,
    int splits) {
    try {
        g_error.clear();
        const bool input_f32_abi = decode_input_is_f32_abi();
        if (!g_error.empty()) {
            return -1;
        }
        return decode_main_f32_launch(
            stream, device, partial_f32, x_f32, code, rin_f16,
            M, IC, OC, K, splits,
            input_f32_abi ? ESCHA_OFFICIAL_BRIDGE_INPUT_F32
                          : ESCHA_OFFICIAL_BRIDGE_INPUT_F16,
            0, nullptr);
    } catch (...) {
        g_error = "unexpected exception in raw Escha F32 decode";
        return -2;
    }
}

extern "C" int escha_official_bridge_abi_version() {
    return ESCHA_OFFICIAL_BRIDGE_ABI_V1;
}

extern "C" int escha_official_decode_gemv_main_desc_v1(
        const escha_official_bridge_decode_desc_v1 * desc) {
    try {
        g_error.clear();
        if (desc == nullptr || desc->abi_version != ESCHA_OFFICIAL_BRIDGE_ABI_V1 ||
            desc->struct_bytes < sizeof(*desc)) {
            g_error = "invalid Escha bridge decode descriptor version or size";
            return -1;
        }
        return decode_main_f32_launch(
            desc->stream, desc->device, desc->partial_f32, desc->x_f32,
            desc->code, desc->rin_f16, desc->M, desc->IC, desc->OC,
            desc->K, desc->splits, desc->input_abi, desc->flags,
            desc->cubin_path);
    } catch (...) {
        g_error = "unexpected exception in Escha bridge decode descriptor";
        return -2;
    }
}

extern "C" int escha_official_decode_gemv_main_batch_v1(
        const escha_official_bridge_batch_v1 * batch) {
    try {
        g_error.clear();
        if (batch == nullptr || batch->abi_version != ESCHA_OFFICIAL_BRIDGE_ABI_V1 ||
            batch->struct_bytes < sizeof(*batch) || batch->projection_count <= 0 ||
            batch->projection_count > 64 || batch->projections == nullptr ||
            (batch->input_abi != ESCHA_OFFICIAL_BRIDGE_INPUT_F16 &&
             batch->input_abi != ESCHA_OFFICIAL_BRIDGE_INPUT_F32)) {
            g_error = "invalid Escha bridge batch descriptor version, size, or count";
            return -1;
        }
        for (int i = 0; i < batch->projection_count; ++i) {
            const escha_official_bridge_decode_desc_v1 & desc = batch->projections[i];
            if (desc.abi_version != ESCHA_OFFICIAL_BRIDGE_ABI_V1 ||
                desc.struct_bytes < sizeof(desc) || desc.device != batch->device ||
                desc.stream != batch->stream || desc.input_abi != batch->input_abi) {
                g_error = "Escha bridge batch contains an incompatible projection descriptor";
                return -1;
            }
            const char * path = desc.cubin_path != nullptr ? desc.cubin_path : batch->cubin_path;
            const int rc = decode_main_f32_launch(
                batch->stream, batch->device, desc.partial_f32, desc.x_f32,
                desc.code, desc.rin_f16, desc.M, desc.IC, desc.OC,
                desc.K, desc.splits, batch->input_abi, batch->flags | desc.flags, path);
            if (rc != 0) {
                return rc;
            }
        }
        return 0;
    } catch (...) {
        g_error = "unexpected exception in Escha bridge batch descriptor";
        return -2;
    }
}

extern "C" const char * escha_official_bridge_error() {
    return g_error.c_str();
}

extern "C" int escha_official_bridge_probe() {
    return load_code_module(0) ? 0 : -1;
}

// Experimental packed E3 prefill oracle. Never installed in production.
// E3 codebook, dependency permutation and Hadamard follow BeeLlama ab1698c.
// Weight files remain packed E3. This working-tile arithmetic is approximate.
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include "mma.cuh"

static __device__ __forceinline__ half codebook(uint32_t idx) {
    uint32_t x=idx*0xcbac1fedu;
    asm("lop3.b32 %0, %1, %2, %3, 0x6a;" : "=r"(x)
        : "r"(x), "n"(0x8fff8fffu), "n"(0x3b603b60u));
    __half2 h;
    memcpy(&h,&x,sizeof(h));
    return __hadd(__low2half(h),__high2half(h));
}
static __device__ __forceinline__ int dep_pi(int r) {
    return (r&1)|(((r>>3)&1)<<1)|(((r>>1)&3)<<3);
}
static __device__ __forceinline__ void hadamard(float * v,int n,int tid) {
    for(int len=1;len<128;len<<=1) {
        for(int idx=tid;idx<(n/128)*64;idx+=256) {
            int blk=idx/64,j=idx%64,i=(j/len)*(2*len)+(j%len);
            float * p=v+blk*128+i;
            float a=p[0],b=p[len];
            p[0]=a+b;p[len]=a-b;
        }
        __syncthreads();
    }
    for(int i=tid;i<n;i+=256) v[i]*=rsqrtf(128.0f);
    __syncthreads();
}

static __global__ void quantize_rows(const half * x,int8_t * q,float * scales,int IC) {
    __shared__ float maxima[8];
    const int tid=threadIdx.x,row=blockIdx.x;
    float amax=0;
    for(int i=tid;i<IC;i+=256) amax=fmaxf(amax,fabsf(__half2float(x[(int64_t)row*IC+i])));
    for(int d=16;d;d>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d));
    if((tid&31)==0) maxima[tid/32]=amax;
    __syncthreads();
    if(tid<32) {
        amax=tid<8?maxima[tid]:0;
        for(int d=16;d;d>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d));
        if(tid==0) { maxima[0]=amax;scales[row]=amax/127.0f; }
    }
    __syncthreads();
    const float inv=maxima[0]>0?127.0f/maxima[0]:0;
    for(int i=tid;i<IC;i+=256) {
        int v=__float2int_rn(__half2float(x[(int64_t)row*IC+i])*inv);
        q[(int64_t)row*IC+i]=(int8_t)max(-127,min(127,v));
    }
}

template<int BM=128>
static __global__ __launch_bounds__(256,2) void packed_i8_down(
    const int16_t * code,const int8_t * u,const float * scales,
    const half * rout,half * dst,int M,int IC,int OC) {
    constexpr int BN=128,NTJ=8,NWD=24,NT=256,STEP=32;
    constexpr int WM=4,WN=2,MT=BM/16/WM,NTT=BN/8/WN;
    extern __shared__ __align__(16) unsigned char shared[];
    uint2 * pay=(uint2*)shared; // [2][8][24] paired cyclic payload words
    int8_t * su=(int8_t*)(pay+2*NTJ*NWD); // [2][BM][32]
    int8_t * sw=su+2*BM*STEP; // [128][32]
    const int lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane;
    const int row0=blockIdx.x*BM,col0=blockIdx.y*BN,nct=OC/16;
    const int wm=warp/WN,wn=warp%WN,nit=IC/STEP;
    constexpr int CPB=BM*STEP/NT;
    const int cp_m=tid/(STEP/CPB),cp_k=(tid%(STEP/CPB))*CPB;
    const int dr=tid%16,dc=tid/16;
    int shift=((32-3)-3*(dep_pi(dr)+32*dc+4*(dc>>3)))%768;
    if(shift<0) shift+=768;
    const int g=shift>>5,word=g?(NWD-g):0,sh=shift&31;

    using tc=ggml_cuda_mma::tile<16,8,int>;
    using ta=ggml_cuda_mma::tile<16,8,int>;
    using tb=ggml_cuda_mma::tile<8,8,int>;
    tc acc[MT][NTT];
    uint32_t next_words[2]={0,0};
#pragma unroll
    for(int p=0;p<2;p++) {
        int j=tid+p*NT;
        if(j<2*NTJ*NWD) {
            int kt=j/(NTJ*NWD),ot=(j/NWD)%NTJ,pw=j%NWD;
            next_words[p]=((const uint32_t*)(code+(int64_t)(kt*nct+col0/16+ot)*48))[pw];
        }
    }
    {
        int row=row0+cp_m,src=row<M?row:0;
        __pipeline_memcpy_async(su+cp_m*STEP+cp_k,u+(int64_t)src*IC+cp_k,CPB,row<M?0:CPB);
        __pipeline_commit();
    }
    for(int ti=0;ti<nit;ti++) {
        int8_t * cur=su+(ti&1)*BM*STEP;
        int8_t * next=su+((ti&1)^1)*BM*STEP;
#pragma unroll
        for(int p=0;p<2;p++) {
            int j=tid+p*NT;
            if(j<2*NTJ*NWD) {
                int kt=j/(NTJ*NWD),ot=(j/NWD)%NTJ,pw=j%NWD;
                uint2 * pair=pay+(kt*NTJ+ot)*NWD;
                pair[pw].y=next_words[p];
                pair[(pw+1)%NWD].x=next_words[p];
                if(ti+1<nit)
                    next_words[p]=((const uint32_t*)(code+(int64_t)(((ti+1)*2+kt)*nct+col0/16+ot)*48))[pw];
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();
        if(ti+1<nit) {
            int row=row0+cp_m,src=row<M?row:0;
            __pipeline_memcpy_async(next+cp_m*STEP+cp_k,
                u+(int64_t)src*IC+(ti+1)*STEP+cp_k,CPB,row<M?0:CPB);
            __pipeline_commit();
        }
#pragma unroll
        for(int k=0;k<16;k++) {
            int kt=k/8,ot=k%8,c=dc+16*ot;
            const uint2 * p=pay+(kt*NTJ+ot)*NWD;
            half w=codebook(__funnelshift_r(p[word].y,p[word].x,sh)&0xffffu);
            // All codebook values lie in [-3.94921875,3.94921875],
            // so round(w*32) fits signed int8 without clipping.
            sw[c*STEP+kt*16+dr]=(int8_t)__float2int_rn(__half2float(w)*32.0f);
        }
        __syncthreads();
        const int * a32=(const int*)cur,*b32=(const int*)sw;
        ta A[MT];tb B[NTT];
#pragma unroll
        for(int i=0;i<MT;i++) ggml_cuda_mma::load_ldmatrix(A[i],a32+(wm*(16*MT)+i*16)*8,8);
#pragma unroll
        for(int j=0;j<NTT;j++) ggml_cuda_mma::load_ldmatrix(B[j],b32+(wn*(8*NTT)+j*8)*8,8);
#pragma unroll
        for(int i=0;i<MT;i++) {
#pragma unroll
            for(int j=0;j<NTT;j++) ggml_cuda_mma::mma(acc[i][j],A[i],B[j]);
        }
        __syncthreads();
    }
    // The input/weight staging is dead; its combined area can hold16x128 F32.
    static_assert(2*BM*STEP+BN*STEP>=16*BN*sizeof(float),"epilogue workspace");
    float * ep=(float*)su;
    for(int group=0;group<BM/16;group++) {
#pragma unroll
        for(int i=0;i<MT;i++) {
#pragma unroll
            for(int j=0;j<NTT;j++) {
#pragma unroll
                for(int l=0;l<tc::ne;l++) {
                    int m=wm*(16*MT)+i*16+tc::get_i(l),n=wn*(8*NTT)+j*8+tc::get_j(l);
                    if(m/16==group) {
                        float s=(row0+m<M)?scales[row0+m]/32.0f:0;
                        ep[(m-group*16)*BN+n]=float(acc[i][j].x[l])*s;
                    }
                }
            }
        }
        __syncthreads();
        hadamard(ep,16*BN,tid);
        if(tid<BN) for(int r=0;r<16;r++) {
            int row=row0+group*16+r;
            if(row<M) dst[(int64_t)row*OC+col0+tid]=__float2half_rn(ep[r*BN+tid]*__half2float(rout[col0+tid]));
        }
        __syncthreads();
    }
}


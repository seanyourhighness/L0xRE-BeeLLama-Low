#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
template<int T,int ROWS,int K,int SPLITS,int WPART> __global__ void head_fused(const uint8_t *codes,const half *scales,const uint8_t *zps,const half *x,float *out,int D,int V){
 constexpr int THREADS=2*ROWS*WPART,STRIDE=K+8;
 __shared__ __align__(16) half sw[ROWS][STRIDE];
 __shared__ __align__(16) half sx[8][STRIDE];
 __shared__ float final_sum[WPART][T][ROWS];
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,t=lane&3;
 int row0=blockIdx.x*ROWS,row=(warp%(ROWS/16))*16+g,KB=D*3/8,G=D/128;
 float c0=0,c1=0,c2=0,c3=0;
 int tile_start=(D/K)*blockIdx.y/SPLITS,tile_end=(D/K)*(blockIdx.y+1)/SPLITS;
 for(int tile=tile_start;tile<tile_end;tile++){
  int kb=tile*K;
  for(int j=tid;j<ROWS*(K/8);j+=THREADS){
   int r=j/(K/8),p=j%(K/8),vr=row0+r;
   uint4 raw={0,0,0,0};
   if(vr<V){
    const uint8_t *cr=codes+(size_t)vr*KB+3*(kb/8+p);
    unsigned bits=unsigned(cr[0])|(unsigned(cr[1])<<8)|(unsigned(cr[2])<<16);
    float scale=__half2float(scales[(size_t)vr*G+(kb+p*8)/128]);float zp=zps[(size_t)vr*G+(kb+p*8)/128];
    unsigned result[4];
    unsigned z=(0x6400u+unsigned(zp))*0x10001u;
    half2 sh=__half2half2(__float2half_rn(scale));
#pragma unroll
    for(int i=0;i<4;i++) {
     unsigned q=0x64006400u|((bits>>(6*i))&7u)|(((bits>>(6*i+3))&7u)<<16);
     half2 v=__hmul2(__hsub2(*reinterpret_cast<half2*>(&q),*reinterpret_cast<half2*>(&z)),sh);
     result[i]=*reinterpret_cast<unsigned*>(&v);
    }
    raw=make_uint4(result[0],result[1],result[2],result[3]);
   }
   *reinterpret_cast<uint4*>(&sw[r][p*8])=raw;
  }
  for(int j=tid;j<8*(K/2);j+=THREADS){int tok=j/(K/2),p=j%(K/2);*reinterpret_cast<unsigned*>(&sx[tok][p*2])=tok<T?*reinterpret_cast<const unsigned*>(x+(size_t)tok*D+kb+2*p):0;}
  __syncthreads();
#pragma unroll
  for(int k=(warp/(ROWS/16))*(K/WPART);k<(warp/(ROWS/16)+1)*(K/WPART);k+=16){
   unsigned a0=*reinterpret_cast<unsigned*>(&sw[row][k+2*t]);
   unsigned a1=*reinterpret_cast<unsigned*>(&sw[row+8][k+2*t]);
   unsigned a2=*reinterpret_cast<unsigned*>(&sw[row][k+2*t+8]);
   unsigned a3=*reinterpret_cast<unsigned*>(&sw[row+8][k+2*t+8]);
   unsigned b0=*reinterpret_cast<unsigned*>(&sx[g][k+2*t]);
   unsigned b1=*reinterpret_cast<unsigned*>(&sx[g][k+2*t+8]);
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  __syncthreads();
 }
 int tok=2*t,part=warp/(ROWS/16);
 if(tok<T){final_sum[part][tok][row]=c0;final_sum[part][tok][row+8]=c2;}
 if(tok+1<T){final_sum[part][tok+1][row]=c1;final_sum[part][tok+1][row+8]=c3;}
 __syncthreads();
 for(int i=tid;i<T*ROWS;i+=THREADS){int tok=i/ROWS,r=i%ROWS;if(row0+r<V){float sum=final_sum[0][tok][r];for(int j=1;j<WPART;j++)sum+=final_sum[j][tok][r];out[((size_t)blockIdx.y*T+tok)*V+row0+r]=sum;}}

}
template<int S> __global__ void head_partials(const float*in,float*out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=in[i];for(int s=1;s<S;s++)v+=in[(size_t)s*n+i];out[i]=v;}}

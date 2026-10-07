#pragma once
#include <cassert>

// Each row is [final(D), retained_base(D), K(capacity*H*S),
// g(capacity*H), delta(capacity*H*S), saved_count(1)].
__device__ __forceinline__ float l0xre_compact_replay_element(
        const float * cache, int S, int H, int capacity, int64_t id, int count) {
    const int64_t D=int64_t(S)*S*H;
    const int h=id/(S*S), col=(id%(S*S))/S, row=id%S;
    float state=cache[D+id];
    for(int t=0;t<count;t++) {
        const float gate=cache[2*D+capacity*H*S+t*H+h];
        const float key=cache[2*D+(t*H+h)*S+row];
        const float delta=cache[2*D+capacity*H*(S+1)+(t*H+h)*S+col];
        state=gate*state+key*delta;
    }
    return state;
}

__global__ void l0xre_compact_restore_cuda(
        const float * cache, const int32_t * rollback, float * out, int S, int H, int capacity) {
    const int64_t D=int64_t(S)*S*H;
    const int64_t id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=D)return;
    const int n=int(cache[2*D+capacity*H*(2*S+1)]), rewind=rollback[0];
    if(!(n>=0 && n<=capacity && rewind>=0 && rewind<=n))asm volatile("trap;");
    out[id]=rewind==0 ? cache[id] : l0xre_compact_replay_element(cache,S,H,capacity,id,n-rewind);
}

#include "gdn-compact-kernel.cuh"

static void l0xre_compact_gdn_op(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode=ggml_get_op_params_i32(dst,4), capacity=ggml_get_op_params_i32(dst,0);
    GGML_ASSERT(capacity>=1 && capacity<=8);
    cudaStream_t stream=ctx.stream();
    if(mode==2) {
        const int S=ggml_get_op_params_i32(dst,5), H=ggml_get_op_params_i32(dst,6);
        GGML_ASSERT(S==128 && H>0 && dst->src[0]->type==GGML_TYPE_F32 && dst->src[1]->type==GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(dst->src[0])==ggml_gdn_compact_elements(S,H,capacity));
        l0xre_compact_restore_cuda<<<(int64_t(S)*S*H+255)/256,256,0,stream>>>(
                (const float*)dst->src[0]->data,(const int32_t*)dst->src[1]->data,(float*)dst->data,S,H,capacity);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    GGML_ASSERT(mode==1 && ggml_get_op_params_i32(dst,3)==0);
    const ggml_tensor *q=dst->src[0], *k=dst->src[1], *v=dst->src[2], *g=dst->src[3], *b=dst->src[4];
    const ggml_tensor *initial=dst->src[5], *cache=dst->src[8], *rewind=dst->src[9];
    const int64_t S=v->ne[0], H=v->ne[1], T=v->ne[2];
    GGML_ASSERT(S==128 && v->ne[3]==1 && g->ne[0]==1 && q->ne[1]==k->ne[1]);
    GGML_ASSERT(ggml_are_same_stride(q,k) && ggml_is_contiguous_rows(q) && ggml_is_contiguous_rows(v));
    GGML_ASSERT(ggml_is_contiguous(initial) && ggml_is_contiguous(g) && ggml_is_contiguous(b));
    const int64_t sq1=q->nb[1]/4, sq2=q->nb[2]/4, sq3=q->nb[3]/4;
    const int64_t sv1=v->nb[1]/4, sv2=v->nb[2]/4, sv3=v->nb[3]/4;
    const int64_t sb1=b->nb[1]/4, sb2=b->nb[2]/4, sb3=b->nb[3]/4;
    const bool grouped=ggml_get_op_params_i32(dst,1)!=0;
    const float eps=grouped ? ggml_get_op_params_f32(dst,2) : 0.0f;
    const float scale=1.0f/sqrtf(float(S));
    const uint3 heads=init_fastdiv_values(q->ne[1]);
    const uint3 group=init_fastdiv_values(H/q->ne[1]);
    const uint3 seq=init_fastdiv_values(1);
    float *attention=(float*)dst->data, *packed=attention+S*H*T;
    const float *old=(const float*)cache->data;
    GGML_ASSERT(old!=packed);
    const float *qd=(const float*)q->data,*kd=(const float*)k->data,*vd=(const float*)v->data;
    const float *gd=(const float*)g->data,*bd=(const float*)b->data,*sd=(const float*)initial->data;
    int64_t count=T;
    ggml_cuda_pool_alloc<float> prefix_state(ctx.pool());
    // Preserve the existing qualified masked-prefill prefix and its exact tail.
    if(!grouped && T>=256 && T<=1024 && capacity+1<T && q->ne[1]==H &&
            sq1==S && sq2==H*S && sv1==S && sv2>=H*S && sb1==1 && sb2==H &&
            (ggml_cuda_info().devices[ctx.device].cc==860 || ggml_cuda_info().devices[ctx.device].cc==890) && l0xre_gdn_masked_requested()) {
        const int64_t native_tail=capacity+1; // native rollback slots include the final state
        const int64_t prefix=T-native_tail, padded=((prefix+63)/64)*64;
        prefix_state.alloc(S*S*H);
        const auto &api=l0xre_gdn_masked_load();
        const int status=api.prefill(stream,ctx.device,qd,kd,vd,gd,bd,sd,
                                    attention,prefix_state.get(),S,H,padded,1,
                                    sq1,sq2,sv1,sv2,sb1,sb2,scale,prefix);
        if(status<0)GGML_ABORT("Compact GDN masked prefix failed: %s",api.error());
        if(status==0) {
            qd+=prefix*sq2;kd+=prefix*sq2;vd+=prefix*sv2;gd+=prefix*sb2;bd+=prefix*sb2;
            sd=prefix_state.get();attention+=prefix*H*S;count=native_tail;
        } else GGML_ASSERT(status==1);
    }
    const ggml_cuda_kernel_launch_params launch(dim3(H,1,S/4),dim3(32,4),0,stream);
    if(grouped) {
        ggml_cuda_kernel_launch(gdn_compact_forward_cuda<128,false,false,true,false>,launch,
                qd,kd,vd,gd,bd,nullptr,nullptr,sd,attention,packed,H,count,int64_t(1),
                sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,heads,group,seq,scale,eps,S*S*H,capacity,
                old,(const int32_t*)rewind->data,packed);
    } else {
        ggml_cuda_kernel_launch(gdn_compact_forward_cuda<128,false,false,false,false>,launch,
                qd,kd,vd,gd,bd,nullptr,nullptr,sd,attention,packed,H,count,int64_t(1),
                sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,heads,group,seq,scale,eps,S*S*H,capacity,
                old,(const int32_t*)rewind->data,packed);
    }
    CUDA_CHECK(cudaGetLastError());
}

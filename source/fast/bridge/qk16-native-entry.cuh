#pragma once
#include "fattn-common.cuh"
#include "fattn-mma-kvarn-case-decl.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include "fattn-qk16-prefill.cuh"
static bool enabled(const char*name){const char*x=std::getenv(name);return x&&std::atoi(x)!=0;}
static bool eligible(ggml_backend_cuda_context &ctx,ggml_tensor *dst){
 const char*mode=std::getenv("L0XRE_KVARN_QK_FP16_ACC");if(!mode||(std::strcmp(mode,"1")&&std::strcmp(mode,"2")))return false;
 if(!dst||!dst->src[0]||enabled("GGML_KVARN_TEST_FORCE_MATERIALIZE_FATTN")||enabled("GGML_KVARN_TEST_FORCE_PORTABLE_FATTN")||enabled("GGML_KVARN_TEST_DISABLE_LONG_SPECIALIZED_DECODE"))return false;
 // Preserve exact production prefill arithmetic for short contexts.
 if(!dst->src[1] || dst->src[1]->ne[1] <= 16384)return false;
 auto q=dst->src[0];ggml_cuda_fattn_kvarn_plan plan;
 if(ctx.device!=0||ggml_cuda_info().devices[ctx.device].cc!=860||q->ne[0]!=256||q->ne[1]<128||q->ne[1]>1024||dst->ne[0]!=256||!ggml_cuda_fattn_kvarn_view_supported(ctx.device,dst,&plan)||plan.k.bits!=4||plan.v.bits!=4||q->ne[2]/dst->src[1]->ne[2]!=6||dst->src[4]||ggml_cuda_fattn_kvarn_domain(dst)!=GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V)return false;
 float bias,softcap;std::memcpy(&bias,(const float*)dst->op_params+1,4);std::memcpy(&softcap,(const float*)dst->op_params+2,4);if(bias!=0||softcap!=0)return false;
 const char*w=std::getenv("GGML_KVARN_WINDOW");return !w||std::atoi(w)!=0;
}

static bool l0xre_qk16_prefill_entry(ggml_backend_cuda_context &ctx, ggml_tensor *dst, ggml_cuda_fattn_kvarn_entry_path path) {
    if ((int(path)==0 || int(path)==1) && eligible(ctx,dst)) {
        const bool result=l0xre_qk16_dispatch(ctx,dst);
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) std::fprintf(stderr,"L0XRE_QK_CONTEXT_GATE min_kv_exclusive=16384 original_tag=%d gpu_entry=0 original_decode_backend=1 M=%lld\n",int(path),(long long)dst->src[0]->ne[1]);
        return result;
    }
    return false;
}

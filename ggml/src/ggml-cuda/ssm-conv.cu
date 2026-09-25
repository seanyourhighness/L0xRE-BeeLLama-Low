#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

// R260 lab kernel: the M=1 recurrent convolution can consume the three cached
// columns and current projected value directly, then shift the cache in place.
// One thread owns one channel, so all four inputs are loaded before that
// channel's cache row is updated.
static __global__ void escha_ssm_conv4_update_silu_f32(
        char       * state_ptr,
        const char * current_ptr,
        const char * weight_ptr,
        char       * dst_ptr,
        const int64_t channels,
        const int64_t state_nb1,
        const int64_t current_nb1,
        const int64_t weight_nb1,
        const int64_t dst_nb0) {
    ggml_cuda_pdl_lc();
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }

    float * state = (float *) (state_ptr + c * state_nb1);
    const float current = *(const float *) (current_ptr + c * current_nb1);
    const float * weight = (const float *) (weight_ptr + c * weight_nb1);

    ggml_cuda_pdl_sync();
    const float x0 = state[0];
    const float x1 = state[1];
    const float x2 = state[2];

    float sumf = 0.0f;
    sumf += x0      * weight[0];
    sumf += x1      * weight[1];
    sumf += x2      * weight[2];
    sumf += current * weight[3];

    *(float *) (dst_ptr + c * dst_nb0) = ggml_cuda_op_silu_single(sumf);
    state[0] = x1;
    state[1] = x2;
    state[2] = current;
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

bool ggml_cuda_ssm_conv_update_is_eligible(
        const ggml_tensor * concat,
        const ggml_tensor * state_cpy,
        const ggml_tensor * ssm_conv,
        const ggml_tensor * silu) {
    if (concat == nullptr || state_cpy == nullptr || ssm_conv == nullptr || silu == nullptr ||
        concat->op != GGML_OP_CONCAT || state_cpy->op != GGML_OP_CPY ||
        ssm_conv->op != GGML_OP_SSM_CONV || silu->op != GGML_OP_UNARY ||
        ssm_conv->src[2] != nullptr ||
        ggml_get_op_params_i32(concat, 0) != 0 ||
        ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        return false;
    }

    const ggml_tensor * state   = concat->src[0];
    const ggml_tensor * current = concat->src[1];
    const ggml_tensor * weight  = ssm_conv->src[1];
    const ggml_tensor * cpy_src = state_cpy->src[0];
    const ggml_tensor * cpy_dst = state_cpy->src[1];

    const bool edges = cpy_src != nullptr && cpy_src->view_src == concat &&
        ssm_conv->src[0] == concat && silu->src[0] == ssm_conv;
    const bool types = state != nullptr && current != nullptr && weight != nullptr && cpy_dst != nullptr &&
        state->type == GGML_TYPE_F32 && current->type == GGML_TYPE_F32 &&
        concat->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32 &&
        ssm_conv->type == GGML_TYPE_F32 && silu->type == GGML_TYPE_F32 &&
        cpy_dst->type == GGML_TYPE_F32;
    if (!edges || !types) {
        return false;
    }

    const int64_t channels = state->ne[1];
    const bool shapes = state->ne[0] == 3 && state->ne[2] == 1 && state->ne[3] == 1 &&
        current->ne[0] == 1 && current->ne[1] == channels && current->ne[2] == 1 && current->ne[3] == 1 &&
        concat->ne[0] == 4 && concat->ne[1] == channels && concat->ne[2] == 1 && concat->ne[3] == 1 &&
        weight->ne[0] == 4 && weight->ne[1] == channels &&
        silu->ne[0] == channels && silu->ne[1] == 1 && silu->ne[2] == 1 && silu->ne[3] == 1 &&
        ggml_nelements(cpy_dst) == 3 * channels;
    const bool strides = state->nb[0] == sizeof(float) && state->nb[1] == 3 * sizeof(float) &&
        current->nb[1] == sizeof(float) && weight->nb[0] == sizeof(float) &&
        weight->nb[1] == 4 * sizeof(float) && silu->nb[0] == sizeof(float);
    const bool inplace = state->data != nullptr && cpy_dst->data == state->data;

    return shapes && strides && inplace;
}

void ggml_cuda_op_ssm_conv_update_fused(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * concat,
        ggml_tensor * state_cpy,
        ggml_tensor * ssm_conv,
        ggml_tensor * silu) {
    GGML_ASSERT(ggml_cuda_ssm_conv_update_is_eligible(concat, state_cpy, ssm_conv, silu));
    ggml_tensor * state         = concat->src[0];
    const ggml_tensor * current = concat->src[1];
    const ggml_tensor * weight  = ssm_conv->src[1];

    constexpr int threads = 128;
    const int64_t channels = state->ne[1];
    const dim3 blocks((channels + threads - 1) / threads, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(escha_ssm_conv4_update_silu_f32, launch_params,
        (char *) state->data, (const char *) current->data,
        (const char *) weight->data, (char *) silu->data,
        channels, (int64_t) state->nb[1], (int64_t) current->nb[1],
        (int64_t) weight->nb[1], (int64_t) silu->nb[0]);
}

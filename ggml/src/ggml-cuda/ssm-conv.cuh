#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
bool ggml_cuda_ssm_conv_update_is_eligible(const ggml_tensor * concat,
                                           const ggml_tensor * state_cpy,
                                           const ggml_tensor * ssm_conv,
                                           const ggml_tensor * silu);
void ggml_cuda_op_ssm_conv_update_fused(ggml_backend_cuda_context & ctx,
                                        ggml_tensor * concat,
                                        ggml_tensor * state_cpy,
                                        ggml_tensor * ssm_conv,
                                        ggml_tensor * silu);

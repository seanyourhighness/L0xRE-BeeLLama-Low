#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

// Decode-only RMSNorm(x)*weight followed by SiLU(gate) multiplication.
// The opt-in caller guarantees the exact four-node F32, contiguous shape.
void ggml_cuda_op_rms_norm_silu_mul_fused(ggml_backend_cuda_context & ctx,
                                          ggml_tensor *               rms_node,
                                          ggml_tensor *               mul_node,
                                          ggml_tensor *               silu_node,
                                          ggml_tensor *               out_node);

void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               add_tensor,
                                     ggml_tensor *               rms_tensor,
                                     ggml_tensor *               mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

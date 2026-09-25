#include "common.cuh"

extern "C" bool ggml_backend_cuda_escha_mvt_candidate_enabled(const char * candidate_id);

void ggml_cuda_op_escha_moe(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_escha_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_escha_mul_mat_fused_down_add_rms(ggml_backend_cuda_context & ctx,
                                                   ggml_tensor * down,
                                                   ggml_tensor * add,
                                                   ggml_tensor * rms,
                                                   ggml_tensor * mul);
void ggml_cuda_op_escha_mul_mat_fused_up(ggml_backend_cuda_context & ctx,
                                         ggml_tensor * up,
                                         ggml_tensor * mul);
void ggml_cuda_op_escha_mul_mat_fused_gate_silu(ggml_backend_cuda_context & ctx,
                                                ggml_tensor * gate,
                                                ggml_tensor * silu);
void ggml_cuda_op_escha_mul_mat_fused_qkv_z(ggml_backend_cuda_context & ctx,
                                            ggml_tensor * qkv,
                                            ggml_tensor * z);
void ggml_cuda_op_escha_mul_mat_fused_qkv_z_beta_alpha(ggml_backend_cuda_context & ctx,
                                                       ggml_tensor * qkv,
                                                       ggml_tensor * z,
                                                       ggml_tensor * beta,
                                                       ggml_tensor * alpha);
void ggml_cuda_op_escha_mul_mat_fused_attn_qkv(ggml_backend_cuda_context & ctx,
                                               ggml_tensor * q,
                                               ggml_tensor * k,
                                               ggml_tensor * v);
bool ggml_cuda_escha_mvt_up_finalize_is_eligible(const ggml_tensor * up);
bool ggml_cuda_escha_mul_mat_is_single_slice(const ggml_tensor * up);
bool ggml_cuda_escha_mvt_gate_silu_is_eligible(const ggml_tensor * gate,
                                              const ggml_tensor * silu);
bool ggml_cuda_escha_decode_qkv_z_is_eligible(const ggml_tensor * qkv,
                                              const ggml_tensor * z);
bool ggml_cuda_escha_decode_qkv_z_beta_alpha_is_eligible(const ggml_tensor * qkv,
                                                         const ggml_tensor * z,
                                                         const ggml_tensor * beta,
                                                         const ggml_tensor * alpha);
bool ggml_cuda_escha_decode_attn_qkv_is_eligible(const ggml_tensor * q,
                                                 const ggml_tensor * k,
                                                 const ggml_tensor * v);
bool ggml_cuda_escha_decode_down_add_rms_is_eligible(const ggml_tensor * down,
                                                     const ggml_tensor * add,
                                                     const ggml_tensor * rms,
                                                     const ggml_tensor * mul);

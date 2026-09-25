#include "models.h"
#include "llama-model-loader.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "ggml.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

//
// llama_model_escha
//
// Escha 2/3-bit code with a vocab that is either:
//   W2: int8 + F16 per-row scale, dequantized to F16 at load (this file)
//   E3: LowGPU v1 3-bit, kept packed (handled by llama_model_qwen35)
//
// The body (gated delta net + full attention hybrid) is identical to Qwen3.5,
// so llama_model_escha inherits from llama_model_qwen35 and only overrides the
// vocab loading path. The graph builder is inherited unchanged.
//

void llama_model_escha::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen35::load_arch_hparams(ml);

    ml.get_key(LLM_KV_ESCHA_VERSION, escha_version, false);
    if (escha_version != 1) {
        throw std::runtime_error(format("escha model requires escha.version == 1 (got %u)", escha_version));
    }

    ml.get_key(LLM_KV_LOWGPU_VERSION, lowgpu_version, false);
    if (lowgpu_version != 0 && lowgpu_version != 1) {
        throw std::runtime_error(format("unsupported escha LowGPU codec version %u", lowgpu_version));
    }
}

void llama_model_escha::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const std::string embd_w = tn(LLM_TENSOR_TOKEN_EMBD, "weight").str();
    const std::string embd_s = tn(LLM_TENSOR_TOKEN_EMBD, "weight_scale").str();
    const std::string out_w  = tn(LLM_TENSOR_OUTPUT, "weight").str();
    const std::string out_s  = tn(LLM_TENSOR_OUTPUT, "weight_scale").str();

    const std::string embd_lg_code  = tn(LLM_TENSOR_TOKEN_EMBD_LOWGPU_CODE).str();
    const std::string embd_lg_scale = tn(LLM_TENSOR_TOKEN_EMBD_LOWGPU_SCALE).str();
    const std::string embd_lg_zp    = tn(LLM_TENSOR_TOKEN_EMBD_LOWGPU_ZP).str();
    const std::string out_lg_code   = tn(LLM_TENSOR_OUTPUT_LOWGPU_CODE).str();
    const std::string out_lg_scale  = tn(LLM_TENSOR_OUTPUT_LOWGPU_SCALE).str();
    const std::string out_lg_zp     = tn(LLM_TENSOR_OUTPUT_LOWGPU_ZP).str();

    const bool has_embd_w = ml.get_weight(embd_w.c_str()) != nullptr;
    const bool has_embd_s = ml.get_weight(embd_s.c_str()) != nullptr;
    const bool has_out_w  = ml.get_weight(out_w.c_str()) != nullptr;
    const bool has_out_s  = ml.get_weight(out_s.c_str()) != nullptr;

    const bool has_embd_lg_code  = ml.get_weight(embd_lg_code.c_str()) != nullptr;
    const bool has_embd_lg_scale = ml.get_weight(embd_lg_scale.c_str()) != nullptr;
    const bool has_embd_lg_zp    = ml.get_weight(embd_lg_zp.c_str()) != nullptr;
    const bool has_out_lg_code   = ml.get_weight(out_lg_code.c_str()) != nullptr;
    const bool has_out_lg_scale  = ml.get_weight(out_lg_scale.c_str()) != nullptr;
    const bool has_out_lg_zp     = ml.get_weight(out_lg_zp.c_str()) != nullptr;

    const bool any_dense = has_embd_w || has_embd_s || has_out_w || has_out_s;
    const bool all_dense = has_embd_w && has_embd_s && has_out_w && has_out_s;
    const bool any_lowgpu = has_embd_lg_code || has_embd_lg_scale || has_embd_lg_zp || has_out_lg_code || has_out_lg_scale || has_out_lg_zp;
    const bool all_lowgpu = has_embd_lg_code && has_embd_lg_scale && has_embd_lg_zp && has_out_lg_code && has_out_lg_scale && has_out_lg_zp;

    if (lowgpu_version == 1) {
        if (!all_lowgpu || any_dense) {
            throw std::runtime_error("escha E3 requires exactly six LowGPU vocab tensors and no dense vocab tensors");
        }
        llama_model_qwen35::load_arch_tensors(ml);
        return;
    }

    if (!all_dense || any_lowgpu) {
        throw std::runtime_error("escha W2 requires both I8/F16 row-scale vocab pairs and no LowGPU vocab tensors");
    }

    // W2 variant: int8 vocab + F16 per-row scale, dequantized to F16 at load.

    const auto * w_embd   = ml.get_weight(embd_w.c_str());
    const auto * w_embd_s = ml.get_weight(embd_s.c_str());
    const auto * w_out    = ml.get_weight(out_w.c_str());
    const auto * w_out_s  = ml.get_weight(out_s.c_str());

    if (!w_embd || !w_embd_s || !w_out || !w_out_s) {
        throw std::runtime_error(
            "escha W2 model is missing int8 vocab or F16 weight_scale tensors");
    }

    // Verify the source tensors are the expected types.
    if (w_embd->tensor->type   != GGML_TYPE_I8)  {
        throw std::runtime_error(format(
            "escha W2 token_embd.weight must be I8 (got type %d)", (int)w_embd->tensor->type));
    }
    if (w_embd_s->tensor->type != GGML_TYPE_F16) {
        throw std::runtime_error(format(
            "escha W2 token_embd.weight_scale must be F16 (got type %d)", (int)w_embd_s->tensor->type));
    }
    if (w_out->tensor->type    != GGML_TYPE_I8)  {
        throw std::runtime_error(format(
            "escha W2 output.weight must be I8 (got type %d)", (int)w_out->tensor->type));
    }
    if (w_out_s->tensor->type  != GGML_TYPE_F16) {
        throw std::runtime_error(format(
            "escha W2 output.weight_scale must be F16 (got type %d)", (int)w_out_s->tensor->type));
    }

    // Verify shapes: [n_embd, n_vocab] for weights, [n_vocab] for scales.
    if (w_embd->tensor->ne[0] != n_embd || w_embd->tensor->ne[1] != n_vocab) {
        throw std::runtime_error(format(
            "escha W2 token_embd.weight shape [%lld,%lld] != [n_embd,n_vocab]=[%lld,%lld]",
            (long long)w_embd->tensor->ne[0], (long long)w_embd->tensor->ne[1],
            (long long)n_embd, (long long)n_vocab));
    }
    if (w_embd_s->tensor->ne[0] != n_vocab) {
        throw std::runtime_error(format(
            "escha W2 token_embd.weight_scale shape [%lld] != [n_vocab]=[%lld]",
            (long long)w_embd_s->tensor->ne[0], (long long)n_vocab));
    }
    if (w_out->tensor->ne[0] != n_embd || w_out->tensor->ne[1] != n_vocab) {
        throw std::runtime_error(format(
            "escha W2 output.weight shape [%lld,%lld] != [n_embd,n_vocab]=[%lld,%lld]",
            (long long)w_out->tensor->ne[0], (long long)w_out->tensor->ne[1],
            (long long)n_embd, (long long)n_vocab));
    }
    if (w_out_s->tensor->ne[0] != n_vocab) {
        throw std::runtime_error(format(
            "escha W2 output.weight_scale shape [%lld] != [n_vocab]=[%lld]",
            (long long)w_out_s->tensor->ne[0], (long long)n_vocab));
    }

    // Lab-gated direct INT8 head: keep output.weight in its source format and
    // consume output.weight_scale in the CUDA graph.  Default remains the
    // established I8 -> F16 load-time transform.
    const char * i8_head_env = std::getenv("ESCHA_W2_I8_HEAD");
    const bool keep_i8_head = i8_head_env != nullptr && std::string(i8_head_env) == "1";

    // Register transforms BEFORE the base creates runtime tensors so
    // create_tensor applies the I8 -> F16 type override where requested.
    ml.register_transform(embd_w, embd_w, embd_s, GGML_TYPE_F16);
    if (!keep_i8_head) {
        ml.register_transform(out_w, out_w, out_s, GGML_TYPE_F16);
    }

    // Let the base create tok_embd / output (F16 via the transform) plus all
    // the body tensors (escha LUT, layers, output_norm, ...).
    llama_model_qwen35::load_arch_tensors(ml);

    // Create the F16 weight_scale tensors (not created by the base). They are
    // loaded normally (F16 -> F16, no transform) and must exist as runtime
    // tensors so n_created == n_tensors holds.
    if (ml.get_weight(embd_s.c_str())) {
        create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight_scale"), { n_vocab }, 0);
    }
    if (ml.get_weight(out_s.c_str())) {
        ggml_tensor * scale = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight_scale"), { n_vocab }, 0);
        if (keep_i8_head) {
            output_s = scale;
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_escha::build_arch_graph(const llm_graph_params & params) const {
    // The body (gated delta net + full attention hybrid) is identical to Qwen3.5.
    return llama_model_qwen35::build_arch_graph(params);
}

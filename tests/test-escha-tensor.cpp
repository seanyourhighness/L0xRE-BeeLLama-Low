// File-backed Escha tensor-stage tests (Round 1 R7-tensor and R8).
// These fixtures deliberately contain no model body: they prove the vocabulary
// schema boundary and loader transform without requiring the multi-GiB artifact.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-loader.h"
#include "../src/llama-model-saver.h"
#include "../src/models/models.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const char * msg) {
    fprintf(stderr, "test-escha-tensor FAIL: %s\n", msg);
    std::exit(1);
}
#define CHECK(c, m) do { if (!(c)) fail(m); } while (0)

struct fixture_tensor { std::string name; ggml_type type; std::vector<int64_t> ne; std::vector<uint8_t> data; };

static void add_hparams(gguf_context * ctx, uint32_t lowgpu_version) {
    llama_model_saver ms(LLM_ARCH_ESCHA, ctx);
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(LLM_ARCH_ESCHA));
    ms.add_kv(LLM_KV_VOCAB_SIZE, uint32_t(2));
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(128));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(128));
    ms.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, uint32_t(384));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 0.0001f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>{16, 24, 0, 0});
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL, uint32_t(4));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE, uint32_t(256));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE, uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, uint32_t(2));
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));
    ms.add_kv(LLM_KV_ESCHA_VERSION, uint32_t(1));
    ms.add_kv(LLM_KV_LOWGPU_VERSION, lowgpu_version);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
}

static FILE * make_fixture(uint32_t lowgpu_version, const std::vector<fixture_tensor> & ts) {
    gguf_context * gguf = gguf_init_empty();
    add_hparams(gguf, lowgpu_version);
    ggml_init_params p = { 1 << 20, nullptr, true };
    ggml_context * tctx = ggml_init(p);
    CHECK(tctx != nullptr, "tensor fixture context");
    for (const auto & f : ts) {
        ggml_tensor * t = ggml_new_tensor(tctx, f.type, (int) f.ne.size(), f.ne.data());
        ggml_set_name(t, f.name.c_str());
        CHECK(ggml_nbytes(t) == f.data.size(), "fixture tensor byte count");
        gguf_add_tensor(gguf, t);
        gguf_set_tensor_data(gguf, f.name.c_str(), f.data.data());
    }
    FILE * file = tmpfile();
    CHECK(file != nullptr, "tmpfile");
    CHECK(gguf_write_to_file_ptr(gguf, file, false), "write GGUF fixture");
    rewind(file);
    ggml_free(tctx);
    gguf_free(gguf);
    return file;
}

static fixture_tensor tensor(const char * name, ggml_type type, std::vector<int64_t> ne) {
    size_t n = type == GGML_TYPE_F16 ? 2 : 1;
    for (int64_t d : ne) n *= (size_t) d;
    return { name, type, std::move(ne), std::vector<uint8_t>(n) };
}

static std::string tensor_error(uint32_t lowgpu_version, const std::vector<fixture_tensor> & ts) {
    FILE * file = make_fixture(lowgpu_version, ts);
    std::vector<std::string> splits;
    std::string err;
    try {
        llama_model_loader ml(nullptr, nullptr, nullptr, "", splits, file, LLAMA_LOAD_MODE_NONE, false, true, false, nullptr, nullptr);
        llama_model_params params = llama_model_default_params();
        llama_model_escha model(params);
        model.arch = LLM_ARCH_ESCHA;
        model.load_hparams(ml);
        model.load_arch_tensors(ml);
    } catch (const std::exception & e) { err = e.what(); }
    return err;
}

static std::string complete_vocab_error(uint32_t lowgpu_version, const std::vector<fixture_tensor> & ts) {
    FILE * file = make_fixture(lowgpu_version, ts);
    std::vector<std::string> splits;
    std::string err;
    try {
        llama_model_loader ml(nullptr, nullptr, nullptr, "", splits, file, LLAMA_LOAD_MODE_NONE, false, true, false, nullptr, nullptr);
        llama_model_params params = llama_model_default_params();
        llama_model_escha model(params);
        model.arch = LLM_ARCH_ESCHA;
        model.load_hparams(ml);
        model.load_vocab(ml);
        model.load_tensors(ml);
    } catch (const std::exception & e) { err = e.what(); }
    return err;
}

static void test_r7_file_backed_schema() {
    const auto dense_e = tensor("token_embd.weight", GGML_TYPE_I8, {128, 2});
    const auto dense_s = tensor("token_embd.weight_scale", GGML_TYPE_F16, {2});
    const auto out_e   = tensor("output.weight", GGML_TYPE_I8, {128, 2});
    const auto out_s   = tensor("output.weight_scale", GGML_TYPE_F16, {2});
    const auto ec = tensor("token_embd.lowgpu_codes", GGML_TYPE_I8, {48, 2});
    const auto es = tensor("token_embd.lowgpu_scales", GGML_TYPE_F16, {1, 2});
    const auto ez = tensor("token_embd.lowgpu_zps", GGML_TYPE_I8, {1, 2});
    const auto oc = tensor("output.lowgpu_codes", GGML_TYPE_I8, {48, 2});
    const auto os = tensor("output.lowgpu_scales", GGML_TYPE_F16, {1, 2});
    const auto oz = tensor("output.lowgpu_zps", GGML_TYPE_I8, {1, 2});

    CHECK(tensor_error(0, {dense_e}) == "escha W2 requires both I8/F16 row-scale vocab pairs and no LowGPU vocab tensors", "R7 W2 partial exact rejection");
    CHECK(tensor_error(0, {dense_e, dense_s, out_e, out_s, ec}) == "escha W2 requires both I8/F16 row-scale vocab pairs and no LowGPU vocab tensors", "R7 W2 mixed exact rejection");
    CHECK(tensor_error(1, {ec, es, ez, oc, os}) == "escha E3 requires exactly six LowGPU vocab tensors and no dense vocab tensors", "R7 E3 partial exact rejection");
    CHECK(tensor_error(1, {ec, es, ez, oc, os, oz, dense_e}) == "escha E3 requires exactly six LowGPU vocab tensors and no dense vocab tensors", "R7 E3 mixed exact rejection");

    // W2 is the CPU-loadable positive control: its complete dense vocab passes
    // Escha validation and reaches the first required Qwen3.5 body tensor.
    const std::string positive = complete_vocab_error(0, {dense_e, dense_s, out_e, out_s});
    CHECK(positive == "check_tensor_dims: tensor 'output_norm.weight' not found",
          "R7 complete W2 reaches the first missing Qwen3.5 body tensor");
}

static void test_r8_transform(llama_load_mode mode) {
    const int64_t hidden = 4, vocab = 3;
    std::vector<int8_t> src = {-128, -2, 0, 127, 1, -1, 2, -3, 4, 5, -6, 7};
    std::vector<ggml_fp16_t> scale = {ggml_fp32_to_fp16(0.5f), ggml_fp32_to_fp16(1.5f), ggml_fp32_to_fp16(-2.0f)};
    fixture_tensor s = tensor("src", GGML_TYPE_I8, {hidden, vocab});
    fixture_tensor q = tensor("scale", GGML_TYPE_F16, {vocab});
    memcpy(s.data.data(), src.data(), s.data.size());
    memcpy(q.data.data(), scale.data(), q.data.size());
    FILE * file = make_fixture(0, {s, q});
    std::vector<std::string> splits;
    llama_model_loader ml(nullptr, nullptr, nullptr, "", splits, file, mode, false, true, false, nullptr, nullptr);
    if (mode == LLAMA_LOAD_MODE_MMAP) ml.init_mappings(false);
    ml.register_transform("dst", "src", "scale", GGML_TYPE_F16);
    ggml_init_params p = { 1 << 16, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    CHECK(ctx != nullptr, "R8 destination context");
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden, vocab);
    ggml_set_name(dst, "dst");
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(dst));
    CHECK(buf != nullptr, "R8 CPU buffer allocation");
    CHECK(ggml_backend_tensor_alloc(buf, dst, ggml_backend_buffer_get_base(buf)) == GGML_STATUS_SUCCESS,
          "R8 CPU destination initialization");
    CHECK(ml.fill_transformed(dst) == src.size(), "R8 source byte progress");
    std::vector<ggml_fp16_t> got(src.size());
    ggml_backend_tensor_get(dst, got.data(), 0, ggml_nbytes(dst));
    for (size_t row = 0; row < (size_t) vocab; ++row) for (size_t col = 0; col < (size_t) hidden; ++col) {
        const size_t i = row*hidden + col;
        const ggml_fp16_t expected = ggml_fp32_to_fp16((float) src[i] * ggml_fp16_to_fp32(scale[row]));
        CHECK(memcmp(&got[i], &expected, sizeof(expected)) == 0, "R8 byte-exact I8 x F16 result");
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

} // namespace

int main() {
    test_r7_file_backed_schema();
    test_r8_transform(LLAMA_LOAD_MODE_NONE);
    test_r8_transform(LLAMA_LOAD_MODE_MMAP);
    printf("test-escha-tensor: all checks passed\n");
    return 0;
}

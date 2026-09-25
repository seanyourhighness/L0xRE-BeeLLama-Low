// Focused first-class Escha architecture / schema tests (Round 1, G1 gate).
//
// Covers the R6 (architecture purity) and R7 (schema rejection) focused tests
// from round1/PLAN-DOd-round1-v2.md. These exercise the real
// `llama_model_escha` code paths without needing the 10 GB W2/E3 artifact.
//
//   R6  llm_arch_* table checks  -> escha is a dedicated hybrid arch that
//       constructs llama_model_escha, supports rollback, rejects tensor-split
//       (sm_tensor=false), and is named "escha".
//
//   R7  escha schema rejection at the *hparams* stage. We call
//       llama_model_base::load_hparams(...) directly on a llama_model_escha
//       instance (it sets n_layer_all, then dispatches to
//       llama_model_escha::load_arch_hparams) and assert the *exact* error
//       message. Asserting the message (not just a failure) is what makes this
//       a true test: a positive control proves the fixture is complete enough
//       that the escha validation is the first thing to fire, so the negative
//       cases cannot be passing for some unrelated reason.
//         - escha.version != 1         -> "escha model requires escha.version == 1"
//         - lowgpu.version not in{0,1} -> "unsupported escha LowGPU codec version"
//
// NOTE: the file-backed tensor-stage checks for R7 (mixed/partial W2/E3 vocab)
// and the R8 transform checks live in test-escha-tensor.cpp. R9 (packed-LoRA
// fail-closed) still requires a loaded E3 model plus an active adapter and is
// tracked separately; the full artifact lane is constrained by the host
// cgroup OOM (see round1/evidence/G2/diagnosis.md).

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model.h"
#include "../src/llama-model-loader.h"
#include "../src/llama-model-saver.h"
#include "../src/models/models.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Build a GGUF fixture carrying exactly the hparams that the base
// llama_model_base::load_hparams + the Qwen3.5/escha load_arch_hparams need,
// plus the two Escha version keys under test. The body tensors are NOT
// declared (hparams loading never touches them).
static gguf_context * escha_hparams_gguf(uint32_t escha_version, uint32_t lowgpu_version) {
    gguf_context * ctx = gguf_init_empty();
    llama_model_saver ms(LLM_ARCH_ESCHA, ctx);

    // base load_hparams (required)
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(LLM_ARCH_ESCHA));
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,       uint32_t(128));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,     uint32_t(256));
    ms.add_kv(LLM_KV_BLOCK_COUNT,          uint32_t(2));
    // base load_hparams (scalar, broadcast to per-layer arrays)
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,  uint32_t(384));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(2));
    // Qwen3.5 load_arch_hparams (required)
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 0.0001f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,     std::vector<uint32_t>{16, 24, 0, 0});
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              uint32_t(256));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          uint32_t(2));
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,             uint32_t(2));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,     uint32_t(2));
    // the Escha-specific KV under test
    ms.add_kv(LLM_KV_ESCHA_VERSION,    escha_version);
    ms.add_kv(LLM_KV_LOWGPU_VERSION,   lowgpu_version);
    return ctx;
}

// Run the base hparams load (which dispatches to escha::load_arch_hparams) on
// a throwaway escha instance. Returns the exception message, or "" on success.
static std::string escha_hparams_error(uint32_t escha_version, uint32_t lowgpu_version) {
    gguf_context * ctx = escha_hparams_gguf(escha_version, lowgpu_version);
    llama_model_params params = llama_model_default_params();
    llama_model_escha  model(params);
    model.arch = LLM_ARCH_ESCHA; // normally set by the factory (llama_model.cpp:339)
    std::vector<std::string> splits; // no-file (user-init) branch: unused
    llama_model_loader ml(ctx, nullptr, nullptr, std::string(), splits,
                          /*file*/ nullptr, LLAMA_LOAD_MODE_NONE,
                          /*check_tensors*/ false, /*no_alloc*/ true,
                          /*load_mtp*/ false, nullptr, nullptr);
    std::string err;
    try {
        model.load_hparams(ml);
    } catch (const std::exception & e) {
        err = e.what();
    }
    gguf_free(ctx);
    return err;
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        fprintf(stderr, "usage: %s  (no args; self-contained)\n", argv[0]);
        return 2;
    }

    int failures = 0;
    auto check = [&](bool cond, const char * what) {
        if (cond) {
            printf("PASS: %s\n", what);
        } else {
            printf("FAIL: %s\n", what);
            ++failures;
        }
    };

    // ------------------------------------------------------------------
    // R6 — architecture purity: escha is a dedicated hybrid arch.
    // ------------------------------------------------------------------
    check(std::strcmp(llm_arch_name(LLM_ARCH_ESCHA), "escha") == 0,
          "R6: llm_arch_name(ESCHA) == \"escha\"");
    check(llm_arch_is_hybrid(LLM_ARCH_ESCHA),
          "R6: llm_arch_is_hybrid(ESCHA) == true (gated-delta-net + full-attn)");
    check(llm_arch_supports_rs_rollback(LLM_ARCH_ESCHA),
          "R6: llm_arch_supports_rs_rollback(ESCHA) == true (hybrid rollback)");
    check(!llm_arch_supports_sm_tensor(LLM_ARCH_ESCHA),
          "R6: llm_arch_supports_sm_tensor(ESCHA) == false (rejects tensor-split)");

    // ------------------------------------------------------------------
    // R7 — schema rejection at the hparams stage (exact-message assertions).
    // ------------------------------------------------------------------
    // Positive control first: a valid (escha.version==1, lowgpu.version==0)
    // fixture must load hparams with NO error. This proves the fixture is
    // complete enough that the escha validation is the first thing to fire,
    // so the negative cases below cannot be passing for an unrelated reason.
    check(escha_hparams_error(1, 0).empty(),
          "R7: positive control — valid (escha.version==1, lowgpu.version==0) hparams load cleanly");

    // Exact-equality assertions (not substring) so a reworded/lengthened message
    // cannot slip through. These match escha.cpp:26-34 verbatim.
    check(escha_hparams_error(0, 0) == "escha model requires escha.version == 1 (got 0)",
          "R7: escha.version == 0 rejected with exact message (requires == 1)");
    check(escha_hparams_error(2, 0) == "escha model requires escha.version == 1 (got 2)",
          "R7: escha.version == 2 rejected with exact message (requires == 1)");
    check(escha_hparams_error(1, 2) == "unsupported escha LowGPU codec version 2",
          "R7: lowgpu.version == 2 rejected with exact message (must be 0 or 1)");

    // ------------------------------------------------------------------
    // R8 is covered by test-escha-tensor; R9 still needs a loaded-model
    // adapter exercise (see round1/STATUS.md).
    // ------------------------------------------------------------------
    printf("NOTE: R9 (packed-LoRA fail-closed) still needs the loaded-model\n"
           "      adapter exercise; R8 is covered by test-escha-tensor.\n");

    if (failures == 0) {
        printf("test-escha-arch: all checks passed\n");
        return 0;
    }
    printf("test-escha-arch: FAILURE (%d failed)\n", failures);
    return 1;
}

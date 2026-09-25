// Focused LowGPU reference + backend-capability test (Escha Round-1, plan items 2 & 3).
//
// Covers:
//   (A) ggml_lowgpu_get_rows: the selected-row packed decode must match the canonical
//       F16-rounded oracle exactly, using fixed packed bytes that cross every 3-byte
//       code boundary and the group 127/128 boundary (dims 120..135).
//   (B) CPU backend capability (item 3 fail-closed): CPU accepts LOWGPU_GET_ROWS but
//       rejects LOWGPU_MUL_MAT (the packed 3-bit LM head executes only on CUDA in
//       Round 1; a full V x n_embd F16 expansion is forbidden by the DoD).
//
// The CUDA numerical contract (decode F16 -> F32, round activation to F16, accumulate
// F32) is exercised by the G4 5090 smoke; this CPU test pins the shared decode format
// and the placement gate.
//
// NOTE: checks are explicit (not assert()) so they run under Release/NDEBUG too.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

[[noreturn]] static void fail(const char * msg) {
    fprintf(stderr, "test-lowgpu FAIL: %s\n", msg);
    std::exit(1);
}

#define CHECK(cond, msg) do { if (!(cond)) fail(msg); } while (0)

// Canonical LowGPU decode of one row, matching ggml_compute_forward_lowgpu_get_rows:
//   x_hat = (q - zp) * scale, rounded through F16, emitted as F32.
static float lowgpu_oracle(const uint8_t * code, const ggml_fp16_t * scale,
                           const uint8_t * zp, int64_t dim) {
    const int64_t t = dim / 8;
    const int     i = dim % 8;
    const uint8_t b0 = code[3*t + 0];
    const uint8_t b1 = code[3*t + 1];
    const uint8_t b2 = code[3*t + 2];
    const uint8_t c[8] = {
        (uint8_t) (b0 & 7),
        (uint8_t) ((b0 >> 3) & 7),
        (uint8_t) (((b0 >> 6) | (b1 << 2)) & 7),
        (uint8_t) ((b1 >> 1) & 7),
        (uint8_t) ((b1 >> 4) & 7),
        (uint8_t) (((b1 >> 7) | (b2 << 1)) & 7),
        (uint8_t) ((b2 >> 2) & 7),
        (uint8_t) ((b2 >> 5) & 7),
    };
    const int64_t g = (t*8) / 128;   // 8 codes per triple, 128 dims per group
    const float s = ggml_fp16_to_fp32(scale[g]);
    const float z = (float) zp[g];
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(((float) c[i] - z) * s));
}

} // namespace

int main() {
    // Geometry: n_embd = 512 -> KB = 192 bytes/row, G = 4 groups of 128 dims.
    const int64_t n_embd = 512;
    const int64_t KB     = n_embd*3/8;   // 192
    const int64_t G      = n_embd/128;   // 4
    const int64_t V      = 8;            // rows
    const int64_t n_ids  = 4;            // selected rows: 0,1,2,3

    CHECK(KB == 192, "KB must be 192");
    CHECK(G  == 4,   "G must be 4");

    // Fixed, deterministic packed bytes. The pattern is chosen so that:
    //   * every one of the KB/3 = 64 triples is exercised, and
    //   * dims 120..135 (triples t=15 and t=16) straddle the group 127/128 boundary.
    std::vector<uint8_t>     code(V*KB);
    std::vector<ggml_fp16_t> scale(V*G);
    std::vector<uint8_t>     zp(V*G);
    for (int64_t r = 0; r < V; ++r) {
        for (int64_t b = 0; b < KB; ++b) {
            code[r*KB + b] = (uint8_t) ((r*37 + b*13 + 5) & 0xFF);
        }
        for (int64_t g = 0; g < G; ++g) {
            scale[r*G + g] = ggml_fp32_to_fp16(0.25f + 0.125f*g);
            zp   [r*G + g] = (uint8_t) ((r*3 + g*2) & 0x07);
        }
    }
    const int32_t ids[4] = { 0, 1, 2, 3 };

    // ---- build the graph -------------------------------------------------
    // no_alloc: tensor data is allocated by the backend buffer below.
    ggml_init_params params = { .mem_size = 1<<20, .mem_buffer = nullptr, .no_alloc = true };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "ggml_init failed");

    ggml_tensor * t_code  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8,  KB, V);
    ggml_tensor * t_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, G,  V);
    ggml_tensor * t_zp    = ggml_new_tensor_2d(ctx, GGML_TYPE_I8,  G,  V);
    ggml_tensor * t_ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ids);

    ggml_tensor * out = ggml_lowgpu_get_rows(ctx, t_code, t_scale, t_zp, t_ids);
    CHECK(out != nullptr, "ggml_lowgpu_get_rows returned null");
    CHECK(out->op == GGML_OP_LOWGPU_GET_ROWS, "out op mismatch");
    CHECK(out->ne[0] == n_embd && out->ne[1] == n_ids, "out shape mismatch");

    // ---- allocate + compute on CPU ---------------------------------------
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr, "ggml_backend_cpu_init failed");

    // (B1) CPU must ACCEPT the selected-row embedding decode.
    CHECK(ggml_backend_supports_op(backend, out),
          "CPU must support LOWGPU_GET_ROWS (selected-row decode)");

    // Let the scheduler allocate all tensors, then copy inputs into the buffers.
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, 64, false, false);
    CHECK(sched != nullptr, "ggml_backend_sched_new failed");
    CHECK(ggml_backend_sched_alloc_graph(sched, graph),
          "sched_alloc_graph failed");

    // Inputs live in the context; copy them into the allocated tensor buffers.
    memcpy(t_code ->data,  code .data(), V*KB);
    memcpy(t_scale->data,  (const uint8_t *) scale.data(), V*G*2);
    memcpy(t_zp   ->data,  zp   .data(), V*G);
    memcpy(t_ids  ->data,  ids, n_ids*4);

    const ggml_status st = ggml_backend_sched_graph_compute(sched, graph);
    CHECK(st == GGML_STATUS_SUCCESS, "lowgpu_get_rows scheduled compute failed");

    // (A) Exact match against the F16-rounded oracle for every selected row/dim.
    {
        int64_t mismatches = 0;
        for (int64_t r = 0; r < n_ids; ++r) {
            const float * got = (const float *) (out->data + r*out->nb[1]);
            for (int64_t d = 0; d < n_embd; ++d) {
                const float want = lowgpu_oracle(
                    &code[(int64_t) ids[r]*KB],
                    &scale[(int64_t) ids[r]*G],
                    &zp  [(int64_t) ids[r]*G],
                    d);
                if (got[d] != want) {
                    if (mismatches < 8) {
                        printf("  MISMATCH row=%lld dim=%lld got=%f want=%f\n",
                               (long long) r, (long long) d, got[d], want);
                    }
                    ++mismatches;
                }
            }
        }
        CHECK(mismatches == 0, "lowgpu_get_rows must match the F16-rounded oracle exactly");
    }

    // Boundary sanity: the group-127/128 straddle (dims 127/128) must decode to the
    // group-0 and group-1 values respectively.
    {
        const float * row0 = (const float *) out->data;
        const float w127 = lowgpu_oracle(&code[0], &scale[0], &zp[0], 127);
        const float w128 = lowgpu_oracle(&code[0], &scale[0], &zp[0], 128);
        CHECK(row0[127] == w127, "dim 127 decode mismatch (group 0)");
        CHECK(row0[128] == w128, "dim 128 decode mismatch (group 1)");
        // group 0 and group 1 use different scale/zp, so the values must differ.
        CHECK(row0[127] != row0[128], "group 127/128 boundary must decode to different values");
    }

    // (B2) CPU must REJECT the packed LM head (Round-1: CUDA-only, no F16 expansion).
    {
        // supports_op inspects only types/shapes, so no input data is required.
        ggml_tensor * t_x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ggml_tensor * head = ggml_lowgpu_mul_mat(ctx, t_code, t_scale, t_zp, t_x);
        CHECK(head != nullptr, "ggml_lowgpu_mul_mat returned null");
        CHECK(head->op == GGML_OP_LOWGPU_MUL_MAT, "head op mismatch");
        CHECK(!ggml_backend_supports_op(backend, head),
              "CPU must REJECT LOWGPU_MUL_MAT in Round 1 (fail-closed)");
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);

    printf("test-lowgpu: OK (get_rows oracle exact, group 127/128 boundary, "
           "CPU accepts GET_ROWS / rejects MUL_MAT)\n");
    return 0;
}

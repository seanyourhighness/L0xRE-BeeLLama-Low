// Focused CPU regression coverage for the dense Escha matrix operation.
//
// The production decode path commonly has a single input row.  Keep that case
// alongside short prompt batches so changes to the CPU work partition cannot
// trade decode correctness for prompt correctness (or vice versa).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const char * msg) {
    fprintf(stderr, "test-escha-cpu FAIL: %s\n", msg);
    std::exit(1);
}

#define CHECK(c, m) do { if (!(c)) fail(m); } while (0)

struct run_result {
    std::vector<float> output;
    std::vector<float> oracle;
    double elapsed_ms;
};

static void hadamard_128(float * values, int64_t n) {
    const float scale = 1.0f/std::sqrt(128.0f);
    for (int64_t off = 0; off < n; off += 128) {
        float * block = values + off;
        for (int len = 1; len < 128; len <<= 1) {
            for (int i = 0; i < 128; i += 2*len) {
                for (int j = 0; j < len; ++j) {
                    const float a = block[i + j];
                    const float b = block[i + j + len];
                    block[i + j]       = a + b;
                    block[i + j + len] = a - b;
                }
            }
        }
        for (int i = 0; i < 128; ++i) {
            block[i] *= scale;
        }
    }
}

static float codebook(uint32_t idx) {
    const uint32_t packed = ((idx*UINT32_C(0xcbac1fed)) & UINT32_C(0x8fff8fff)) ^ UINT32_C(0x3b603b60);
    ggml_fp16_t lo;
    ggml_fp16_t hi;
    std::memcpy(&lo, (const uint8_t *) &packed, sizeof(lo));
    std::memcpy(&hi, (const uint8_t *) &packed + sizeof(lo), sizeof(hi));
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(ggml_fp16_to_fp32(lo) + ggml_fp16_to_fp32(hi)));
}

// Independent bit-by-bit construction of the published cyclic dependency table.
// The production extractor must match these physical payload positions exactly.
static std::vector<int16_t> make_cyclic_dep(int k) {
    const int nw = 8*k;
    const int nb = 32*nw;
    std::vector<int16_t> dep(16*256);
    for (int r = 0; r < 16; ++r) {
        const int pi = (r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3);
        for (int c = 0; c < 16; ++c) {
            int sp = ((32 - k) - k*(pi + 32*c + 4*(c >> 3))) % nb;
            if (sp < 0) {
                sp += nb;
            }
            for (int b = 0; b < 16; ++b) {
                const int stream_bit = (sp + b) % nb;
                const int group = stream_bit >> 5;
                const int word = group ? nw - group : 0;
                dep[(r*16 + c)*16 + b] = (int16_t) (word*32 + (stream_bit & 31));
            }
        }
    }
    return dep;
}

// The frozen source tables are NumPy v1 files with a fixed 128-byte header and
// C-order shape [bit, weight].  GGML stores the transpose as [weight, bit].
// This loader deliberately does not share the production affine-map expression.
static std::vector<int16_t> load_frozen_dep(int k) {
    const std::string path = std::string(ESCHA_DEP_DIR) + (k == 2 ? "/dep_k2.npy" : "/dep_k3.npy");
    std::ifstream file(path, std::ios::binary);
    CHECK(file.good(), "open frozen Escha dependency table");
    file.seekg(128);
    std::vector<int16_t> source(16*256);
    file.read((char *) source.data(), (std::streamsize) (source.size()*sizeof(int16_t)));
    CHECK(file.gcount() == (std::streamsize) (source.size()*sizeof(int16_t)), "read frozen Escha dependency table");

    std::vector<int16_t> dep(16*256);
    for (int p = 0; p < 256; ++p) {
        for (int b = 0; b < 16; ++b) {
            dep[p*16 + b] = source[b*256 + p];
        }
    }
    return dep;
}

static uint16_t reference_index(const uint8_t * payload, const int16_t * dep, int r, int c) {
    uint16_t index = 0;
    const int16_t * bits = dep + (r*16 + c)*16;
    for (int b = 0; b < 16; ++b) {
        index |= (uint16_t) (((payload[bits[b] >> 3] >> (bits[b] & 7)) & 1) << b);
    }
    return index;
}

static void test_cyclic_index_oracle() {
    for (const int k : { 2, 3 }) {
        const int tile_bytes = 32*k;
        const std::vector<int16_t> generated_dep = make_cyclic_dep(k);
        const std::vector<int16_t> dep = load_frozen_dep(k);
        CHECK(generated_dep == dep, "cyclic map must equal frozen Escha dependency table");
        std::vector<uint8_t> payload(tile_bytes);
        for (int sample = 0; sample < 66; ++sample) {
            for (int i = 0; i < tile_bytes; ++i) {
                payload[i] = sample == 0 ? 0x00 : sample == 1 ? 0xff :
                             (uint8_t) ((i*73 + sample*41 + (i >> 1)*19) & 0xff);
            }
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    const uint16_t want = reference_index(payload.data(), dep.data(), r, c);
                    const uint16_t got = ggml_escha_decode_index_cyclic(payload.data(), k, r, c);
                    if (got != want) {
                        fprintf(stderr, "test-escha-cpu cyclic index: K=%d sample=%d r=%d c=%d got=%u want=%u\n",
                                k, sample, r, c, (unsigned) got, (unsigned) want);
                        fail("cyclic Escha index differs from dependency-table oracle");
                    }
                }
            }
        }
    }
}

static std::vector<float> reference_dense(
        int k, int64_t nrows, int64_t IC, int64_t OC,
        const std::vector<uint8_t> & code_data,
        const std::vector<int16_t> & dep_data,
        const std::vector<ggml_fp16_t> & rin_data,
        const std::vector<ggml_fp16_t> & rout_data,
        const std::vector<float> & x_data) {
    const int64_t nit = IC/16;
    const int64_t nct = OC/16;
    const int64_t tile_bytes = 16*k*sizeof(int16_t);
    std::vector<float> output((size_t) OC*nrows);
    std::vector<float> u(IC);
    std::vector<float> acc(OC);
    float tile[256];

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t i = 0; i < IC; ++i) {
            u[i] = x_data[(size_t) row*IC + i]*ggml_fp16_to_fp32(rin_data[i]);
        }
        hadamard_128(u.data(), IC);
        std::fill(acc.begin(), acc.end(), 0.0f);

        for (int64_t it = 0; it < nit; ++it) {
            for (int64_t ot = 0; ot < nct; ++ot) {
                const uint8_t * payload = code_data.data() + (it*nct + ot)*tile_bytes;
                for (int p = 0; p < 256; ++p) {
                    uint32_t index = 0;
                    for (int b = 0; b < 16; ++b) {
                        const int bit = dep_data[(size_t) p*16 + b];
                        index |= (uint32_t) ((payload[bit >> 3] >> (bit & 7)) & 1) << b;
                    }
                    tile[p] = codebook(index);
                }
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        acc[(size_t) ot*16 + c] += u[(size_t) it*16 + r]*tile[r*16 + c];
                    }
                }
            }
        }
        hadamard_128(acc.data(), OC);
        for (int64_t i = 0; i < OC; ++i) {
            output[(size_t) row*OC + i] = acc[i]*ggml_fp16_to_fp32(rout_data[i]);
        }
    }
    return output;
}

static run_result run_escha_dense(
        int k, int64_t nrows, int n_threads,
        int64_t IC = 128, int64_t OC = 128, bool with_oracle = true) {
    CHECK(IC % 128 == 0 && OC % 128 == 0, "Escha dimensions must be 128-aligned");
    const int64_t nit = IC/16;
    const int64_t nct = OC/16;
    const int64_t tile_stride = 16*k;

    ggml_init_params params = {
        /* .mem_size   = */ 2u << 20,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "ggml_init");

    ggml_tensor * code = ggml_new_tensor_3d(ctx, GGML_TYPE_I16, tile_stride, nct, nit);
    ggml_tensor * rin  = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, IC);
    ggml_tensor * rout = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, OC);
    ggml_tensor * lut  = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 65536);
    ggml_tensor * dep  = ggml_new_tensor_2d(ctx, GGML_TYPE_I16, 16, 256);
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, nrows);
    ggml_tensor * out  = ggml_escha_mul_mat(ctx, code, rin, rout, lut, dep, x);
    CHECK(out != nullptr && out->op == GGML_OP_ESCHA_MUL_MAT, "dense Escha graph construction");
    CHECK(out->ne[0] == OC && out->ne[1] == nrows, "dense Escha output shape");

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr, "CPU backend initialization");
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    CHECK(ggml_backend_supports_op(backend, out), "CPU must support dense Escha matmul");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr, "CPU tensor allocation");

    // Exercise every payload byte against the format's cyclic dependency table.
    std::vector<uint8_t> code_data(ggml_nbytes(code));
    for (size_t i = 0; i < code_data.size(); ++i) {
        code_data[i] = (uint8_t) ((i*29 + k*17 + 3) & 0xff);
    }
    std::vector<int16_t> dep_data = make_cyclic_dep(k);

    std::vector<ggml_fp16_t> rin_data(IC);
    std::vector<ggml_fp16_t> rout_data(OC);
    for (int64_t i = 0; i < IC; ++i) {
        rin_data[i] = ggml_fp32_to_fp16(0.5f + 0.03125f*(float) (i % 11));
    }
    for (int64_t i = 0; i < OC; ++i) {
        rout_data[i] = ggml_fp32_to_fp16(0.75f + 0.015625f*(float) (i % 13));
    }
    std::vector<ggml_fp16_t> lut_data(ggml_nelements(lut), ggml_fp32_to_fp16(0.0f));
    std::vector<float> x_data((size_t) IC*nrows);
    for (int64_t r = 0; r < nrows; ++r) {
        for (int64_t i = 0; i < IC; ++i) {
            x_data[(size_t) r*IC + i] = ((float) ((r*23 + i*7) % 31) - 15.0f)/32.0f;
        }
    }

    ggml_backend_tensor_set(code, code_data.data(), 0, code_data.size());
    ggml_backend_tensor_set(rin,  rin_data.data(),  0, ggml_nbytes(rin));
    ggml_backend_tensor_set(rout, rout_data.data(), 0, ggml_nbytes(rout));
    ggml_backend_tensor_set(lut,  lut_data.data(),  0, ggml_nbytes(lut));
    ggml_backend_tensor_set(dep,  dep_data.data(),  0, ggml_nbytes(dep));
    ggml_backend_tensor_set(x,    x_data.data(),    0, ggml_nbytes(x));

    const auto start = std::chrono::steady_clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    const auto stop = std::chrono::steady_clock::now();
    CHECK(status == GGML_STATUS_SUCCESS, "dense Escha CPU compute");

    run_result result;
    result.output.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.output.data(), 0, ggml_nbytes(out));
    if (with_oracle) {
        result.oracle = reference_dense(k, nrows, IC, OC, code_data, dep_data, rin_data, rout_data, x_data);
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return result;
}

static void require_oracle(const run_result & result, int k, int64_t nrows, int n_threads) {
    CHECK(result.output.size() == result.oracle.size(), "oracle output size");
    for (size_t i = 0; i < result.output.size(); ++i) {
        const float want = result.oracle[i];
        const float got  = result.output[i];
        // The independent oracle uses the portable fp16 conversion helpers while
        // the CPU backend may use ISA conversions; exact index equality above is
        // the decoder contract. Keep the established numerical tolerance here.
        const float tolerance = 1e-5f*std::max(1.0f, std::fabs(want));
        if (!std::isfinite(want) || !std::isfinite(got) || std::fabs(got - want) > tolerance) {
            fprintf(stderr,
                    "test-escha-cpu oracle: K=%d rows=%lld threads=%d index=%zu got=%g want=%g tolerance=%g\n",
                    k, (long long) nrows, n_threads, i, (double) got, (double) want, (double) tolerance);
            fail("dense Escha scalar-oracle parity");
        }
    }
}

static void require_parity(const run_result & reference, const run_result & candidate,
                           int k, int64_t nrows, int n_threads) {
    CHECK(reference.output.size() == candidate.output.size(), "thread parity output size");
    for (size_t i = 0; i < reference.output.size(); ++i) {
        const float want = reference.output[i];
        const float got  = candidate.output[i];
        if (!std::isfinite(want) || !std::isfinite(got)) {
            fail("dense Escha output must be finite");
        }
        if (std::memcmp(&got, &want, sizeof(got)) != 0) {
            fprintf(stderr,
                    "test-escha-cpu parity: K=%d rows=%lld threads=%d index=%zu got=%g want=%g\n",
                    k, (long long) nrows, n_threads, i, (double) got, (double) want);
            fail("dense Escha thread-count parity");
        }
    }
}

} // namespace

int main() {
    const bool benchmark = std::getenv("ESCHA_CPU_BENCH") != nullptr;
    const int thread_counts[] = { 2, 4, 8, 16 };

    test_cyclic_index_oracle();

    for (const int k : { 2, 3 }) {
        for (const int64_t nrows : { INT64_C(1), INT64_C(2), INT64_C(8), INT64_C(32) }) {
            const run_result reference = run_escha_dense(k, nrows, 1);
            require_oracle(reference, k, nrows, 1);
            if (benchmark) {
                printf("ESCHA_CPU_BENCH K=%d rows=%lld threads=1 elapsed_ms=%.3f\n",
                       k, (long long) nrows, reference.elapsed_ms);
            }
            for (const int n_threads : thread_counts) {
                const run_result candidate = run_escha_dense(k, nrows, n_threads);
                require_oracle(candidate, k, nrows, n_threads);
                require_parity(reference, candidate, k, nrows, n_threads);
                if (benchmark) {
                    printf("ESCHA_CPU_BENCH K=%d rows=%lld threads=%d elapsed_ms=%.3f speedup=%.3f\n",
                           k, (long long) nrows, n_threads, candidate.elapsed_ms,
                           reference.elapsed_ms/candidate.elapsed_ms);
                }
            }
        }
    }

    if (benchmark) {
        // One row is the production decode shape; the wider row counts exercise
        // the scheduler transition into prefill.  Keep real FFN dimensions opt-in
        // so ordinary CTest remains small and deterministic.
        int benchmark_threads = 8;
        if (const char * value = std::getenv("ESCHA_CPU_BENCH_THREADS")) {
            const long parsed = std::strtol(value, nullptr, 10);
            if (parsed >= 1 && parsed <= 256) {
                benchmark_threads = (int) parsed;
            }
        }
        int repetitions = 5;
        if (const char * value = std::getenv("ESCHA_CPU_BENCH_REPS")) {
            const long parsed = std::strtol(value, nullptr, 10);
            if (parsed >= 1 && parsed <= 20) {
                repetitions = (int) parsed;
            }
        }
        int benchmark_rows = 0;
        if (const char * value = std::getenv("ESCHA_CPU_BENCH_ROWS")) {
            const long parsed = std::strtol(value, nullptr, 10);
            if (parsed == 1 || parsed == 2 || parsed == 8 || parsed == 32) {
                benchmark_rows = (int) parsed;
            }
        }

        const auto median_ms = [](std::vector<double> samples) {
            std::sort(samples.begin(), samples.end());
            return samples[samples.size()/2];
        };

        for (const int k : { 2, 3 }) {
            for (const auto & shape : { std::pair<int64_t, int64_t>{ 5120, 17408 },
                                       std::pair<int64_t, int64_t>{ 17408, 5120 } }) {
                for (const int64_t rows : { INT64_C(1), INT64_C(2), INT64_C(8), INT64_C(32) }) {
                    if (benchmark_rows != 0 && rows != benchmark_rows) {
                        continue;
                    }
                    // Warm up allocation and code paths without incorporating it
                    // into the measurement.  Each timed run is separately checked
                    // against the serial result so a speed number never hides a
                    // work-partitioning error.
                    (void) run_escha_dense(k, rows, benchmark_threads, shape.first, shape.second, false);

                    std::vector<double> serial_samples;
                    std::vector<double> parallel_samples;
                    for (int rep = 0; rep < repetitions; ++rep) {
                        const run_result serial = run_escha_dense(k, rows, 1,                 shape.first, shape.second, false);
                        const run_result parallel = run_escha_dense(k, rows, benchmark_threads, shape.first, shape.second, false);
                        require_parity(serial, parallel, k, rows, benchmark_threads);
                        serial_samples.push_back(serial.elapsed_ms);
                        parallel_samples.push_back(parallel.elapsed_ms);
                    }

                    const double serial_median = median_ms(serial_samples);
                    const double parallel_median = median_ms(parallel_samples);
                    printf("ESCHA_CPU_BENCH_REP K=%d IC=%lld OC=%lld rows=%lld reps=%d threads=1 median_ms=%.3f\n",
                           k, (long long) shape.first, (long long) shape.second,
                           (long long) rows, repetitions, serial_median);
                    printf("ESCHA_CPU_BENCH_REP K=%d IC=%lld OC=%lld rows=%lld reps=%d threads=%d median_ms=%.3f speedup=%.3f\n",
                           k, (long long) shape.first, (long long) shape.second,
                           (long long) rows, repetitions, benchmark_threads,
                           parallel_median, serial_median/parallel_median);
                }
            }
        }
    }

    printf("test-escha-cpu: K2/K3 thread parity passed for 1/2/8/32 rows\n");
    return 0;
}

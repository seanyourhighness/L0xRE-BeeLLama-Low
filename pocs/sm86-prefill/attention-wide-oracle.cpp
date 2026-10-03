// Reuse the repository's independent record-generation and attention-reference
// helpers. The original suite remains unchanged; this executable requires CUDA.
#define main retained_kvarn_suite_main
#include "../../tests/test-kvarn.cpp"
#undef main

int main() {
    ggml_backend_load_all();
    ggml_backend_t gpu = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, true);
    struct Shape { int d, q, kv, bits, qh, kvh; };
    const Shape shapes[] = {
        {256, 1,   512,   4, 24, 4},
        {256, 3,   512,   4, 24, 4},
        {256, 128, 4096,  4, 24, 4},
        {256, 326, 8192,  4, 24, 4},
        {256, 512, 32768, 4, 24, 4},
        {256, 512, 77824, 4, 24, 4},
        {256, 768, 8192,  4, 24, 4},
        {256, 1024,32768, 4, 24, 4},
        {256, 1024,77824, 4, 24, 4},
        {256, 256, 4096,  3, 24, 4},
        {128, 256, 4096,  4, 24, 4},
        {256, 256, 4096,  4, 32, 4},
    };
    for (const Shape & s : shapes) {
        std::fprintf(stderr, "TILE_ORACLE_CASE D=%d Q=%d KV=%d bits=%d heads=%d/%d\n",
                     s.d, s.q, s.kv, s.bits, s.qh, s.kvh);
        std::fflush(stderr);
        auto evaluate = [&] {
            return test_native_flash_attention_output(
                gpu, true, true, s.d, s.bits, s.bits, s.q, s.qh, s.kvh,
                s.kv, 3, false, nullptr, false, 128, true,
                GGML_TYPE_F16, 0, false, true, -1, true, false, false, 0, 16384);
        };
        std::vector<float> reference;
        {
            scoped_test_env disable_window("GGML_KVARN_WINDOW", "0");
            reference = evaluate();
        }
        std::vector<float> candidate = evaluate();
        require(reference.size() == candidate.size() && !candidate.empty(), "attention oracle size mismatch");
        double sum = 0.0, max_abs = 0.0;
        for (size_t i = 0; i < candidate.size(); ++i) {
            require(std::isfinite(reference[i]) && std::isfinite(candidate[i]), "attention oracle non-finite output");
            const double delta = double(candidate[i]) - double(reference[i]);
            sum += delta * delta;
            max_abs = std::max(max_abs, std::fabs(delta));
        }
        const double rmse = std::sqrt(sum / candidate.size());
        // Existing repository prefill-route parity tolerance, including merges.
        require(rmse <= 3e-4, "attention oracle differs from direct-record reference");
        std::printf("{\"D\":%d,\"Q\":%d,\"KV\":%d,\"bits\":%d,\"query_heads\":%d,\"kv_heads\":%d,\"values\":%zu,\"rmse\":%.12g,\"max_abs\":%.12g}\n",
                    s.d, s.q, s.kv, s.bits, s.qh, s.kvh, candidate.size(), rmse, max_abs);
        std::fflush(stdout);
    }
    ggml_backend_free(gpu);
    return 0;
}

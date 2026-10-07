#pragma once
#include "llama.h"
#include <cmath>
#include <cstdint>
#include <limits>

// Experimental shared-Gumbel coupling. The counter follows accepted tokens,
// including prompt accepts, and is copied/reset through the upstream sampler API.
struct l0xre_coupling_state {
    bool enabled = false;
    uint32_t seed = 0;
    uint64_t counter = 0;
    float temperature = 1.0f;
};

inline uint64_t l0xre_mix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

inline double l0xre_gumbel(uint32_t seed, uint64_t counter, llama_token token) {
    const uint64_t key = l0xre_mix64(uint64_t(seed) ^ UINT64_C(0x6a09e667f3bcc909));
    const uint64_t h = l0xre_mix64(l0xre_mix64(counter ^ key) ^ uint32_t(token));
    // 52 bits plus a half-unit offset keeps U strictly inside (0, 1).
    const double u = (double(h >> 12) + 0.5) * 0x1p-52;
    return -std::log(-std::log(u));
}

struct l0xre_coupled_ctx {
    llama_sampler * dist;
    uint32_t seed;
    uint64_t counter;
};

inline llama_sampler_i * l0xre_coupled_iface() {
    static llama_sampler_i iface = [] {
        llama_sampler_i i = {};
        i.name = [](const llama_sampler *) { return "l0xre-coupled-dist"; };
        i.accept = [](llama_sampler * s, llama_token t) {
            auto * c = static_cast<l0xre_coupled_ctx *>(s->ctx);
            llama_sampler_accept(c->dist, t);
            ++c->counter;
        };
        i.apply = [](llama_sampler * s, llama_token_data_array * a) {
            auto * c = static_cast<l0xre_coupled_ctx *>(s->ctx);
            // Retain the official distribution calculation after every filter.
            llama_sampler_apply(c->dist, a);
            double best = -std::numeric_limits<double>::infinity();
            for (size_t k = 0; k < a->size; ++k) {
                if (a->data[k].p <= 0.0f) { continue; }
                const double score = std::log(double(a->data[k].p)) +
                    l0xre_gumbel(c->seed, c->counter, a->data[k].id);
                if (score > best) { best = score; a->selected = int64_t(k); }
            }
        };
        i.reset = [](llama_sampler * s) {
            auto * c = static_cast<l0xre_coupled_ctx *>(s->ctx);
            llama_sampler_reset(c->dist);
            c->seed = llama_sampler_get_seed(c->dist);
            c->counter = 0;
        };
        i.clone = [](const llama_sampler * s) {
            auto * c = static_cast<const l0xre_coupled_ctx *>(s->ctx);
            return llama_sampler_init(l0xre_coupled_iface(),
                new l0xre_coupled_ctx{llama_sampler_clone(c->dist), c->seed, c->counter});
        };
        i.free = [](llama_sampler * s) {
            auto * c = static_cast<l0xre_coupled_ctx *>(s->ctx);
            llama_sampler_free(c->dist);
            delete c;
        };
        return i;
    }();
    return &iface;
}

inline llama_sampler * l0xre_coupled_init(uint32_t seed) {
    auto * dist = llama_sampler_init_dist(seed);
    return llama_sampler_init(l0xre_coupled_iface(),
        new l0xre_coupled_ctx{dist, llama_sampler_get_seed(dist), 0});
}

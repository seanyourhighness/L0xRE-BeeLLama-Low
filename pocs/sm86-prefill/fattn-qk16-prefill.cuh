// GENERATED RESEARCH COPY. Source commit ab1698c739b3bc13b9e06fb8febdc774ff106d23.
// Retains BeeLlama/ggml MIT licensing. Generate with prepare-qk16-experiment.py.
// Namespaced to preserve original template kernels and avoid symbol collisions.
#pragma once
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-mma-kvarn-impl.cuh"
#include "fattn-swizzle.cuh"
#include <atomic>
#ifdef GGML_CUDA_FATTN_MMA_CONFIG_CASE
#undef GGML_CUDA_FATTN_MMA_CONFIG_CASE
#endif
namespace l0xre_qk16 {

using namespace ggml_cuda_mma;

// Config options for the MMA kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};           \
    }                                                                                                                                                              \

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 128, 2, 128,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 128, 2, 128,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 128, 2, 128,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 128, 256, 1,  32,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 4,  64,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 4,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  32,  96,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 128, 256, 1,  32, 128, 128, 128, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_turing(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);

    // TODO tune specifically for Volta
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 1, true);
    // Bee KVarN wide 16x8 verification tile.
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 128, 256, 1,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  64,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  64,  96,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 256, 2,  32, 128, 128,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 128, 256, 1,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 128, 2,  32, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 128, 2,  32, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_cdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 1,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 256, 4,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 256, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 256, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 256, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 256, 2,  64,  64,  64,  64, 1, true);
    // Bee KVarN wide 16x8 verification tile.
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 128, 512, 1,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 512, 1,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 512, 1,  64, 128, 128,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 128, 512, 1,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  64, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  64, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  64, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (ampere_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    if (turing_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    }
    if (amd_mfma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
    }
    if (amd_wmma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
    }
    GGML_ASSERT(volta_mma_available(cc));
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols) {
#if defined(AMPERE_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
#elif defined(TURING_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
#elif defined(AMD_MFMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
#elif defined(VOLTA_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
#elif defined(AMD_WMMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __host__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nthreads;
}

static __host__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).occupancy;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_fa;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_K2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_V2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_combine;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_combine;
}

static __host__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nstages_target;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nstages_target;
}

static __host__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).Q_in_reg;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).Q_in_reg;
}

static constexpr __device__ int get_cols_per_thread() {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    return 1; // AMD has a single column per thread.
#else
    return 2; // This is specifically KQ columns, Volta only has a single VKQ column.
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static __host__ int get_cols_per_warp(const int cc) {
    if (turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc)) {
        return 16;
    } else {
        // Volta
        return 32;
    }
}

// ------------------------------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
    return cp_async_available(cc) && ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2, cc) : 0;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const bool use_sparse) {
#ifdef CP_ASYNC_AVAILABLE
    const int nstages_target = ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2) : 0;
    // sparse gather is not implemented for multi-stage loading
    return use_sparse && nstages_target > 1 ? 1 : nstages_target;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2, use_sparse);
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

template<int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV,
        const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    // The minimum granularity is 16 bytes.
    constexpr int h2_per_chunk = 16/sizeof(half2);
    const int chunks_per_row = D2 / h2_per_chunk;
    if constexpr (use_cp_async) {
        static_assert(warp_size == 32, "bad warp_size");
        static_assert(!oob_check || use_sparse, "OOB check not compatible with cp_async");
        constexpr int preload = 64;

        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);

        auto load = [&] __device__ (auto n) {
            const int stride_k = warp_size >> n;
            const int k0_start = stride_k == warp_size ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

                int64_t i_KV;
                if constexpr (use_sparse) {
                    // padded slots gather row 0, the -inf mask removes their contribution
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : 0;
                    i_KV = index >= 0 ? index : 0;
                } else {
                    i_KV = k_VKQ_0 + i;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    if constexpr (swz) {
                        const int smem_offs_b = ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(i, k*h2_per_chunk);
                        cp_async_cg_16<preload>(tile_KV_32 + smem_offs_b, KV + i_KV*stride_KV + k*h2_per_chunk);
                    } else {
                        cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV + i_KV*stride_KV + k*h2_per_chunk);
                    }
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    } else {
        const half2 zero[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
        auto load = [&] __device__ (const int n) {
            const int stride_k = 32 >> n;
            const int k0_start = stride_k == 32 ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                      chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const half2 * src;
                    if constexpr (use_sparse) {
                        const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                        src = index >= 0 ? KV + int64_t(index)*stride_KV + k*h2_per_chunk : zero;
                    } else {
                        src = !oob_check || i < i_sup ? KV + int64_t(k_VKQ_0 + i)*stride_KV + k*h2_per_chunk : zero;
                    }
                    if constexpr (swz) {
                        ggml_cuda_memcpy_1<16>((char *) tile_KV + ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(i, k*h2_per_chunk), src);
                    } else {
                        ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*4, src);
                    }
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    }
}

template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
        const int stride_mask, const int k_VKQ_0, const int i_sup, const int j0, const uint3 ne01,
        const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    if constexpr (use_cp_async) {
        static_assert(nbatch_fa <= 8*warp_size && nbatch_fa % 8 == 0, "bad nbatch_fa");
        static_assert(!oob_check, "OOB check incompatible with cp_async");
        static_assert(!use_sparse, "sparse gather incompatible with cp_async");
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = 8 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j_sram*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half), mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
        }
    } else if constexpr (oob_check || use_sparse) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                if constexpr (use_sparse) {
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = index >= 0 ? mask_h[int64_t(j_vram)*stride_mask + index] : half(-INFINITY);
                } else {
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[int64_t(j_vram)*stride_mask + k_VKQ_0 + i] : half(0.0f);
                }
            }
        }
    } else if constexpr (nbatch_fa < 2*warp_size) {
        constexpr int cols_per_warp = 2*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (warp_size/cols_per_warp);

            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*warp_size) {
                const int i = i0 + 2*threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
            }
        }
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        half         * const __restrict__ kvarn_smem,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    constexpr int  warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols);
    constexpr bool is_kvarn_kv     = ggml_cuda_fattn_kvarn_template_type(type_K) || ggml_cuda_fattn_kvarn_template_type(type_V);
    constexpr int  nstages         = is_kvarn_kv ? 0 : ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, use_sparse);
    static_assert(!is_kvarn_kv || !use_sparse, "sparse KVarN record loads are not qualified");

    // swizzle the tile stride for K and V based on the batch size.
    constexpr int stride_tile_K = is_kvarn_kv ? nbatch_K2 + 4 : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2);
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K :
        (is_kvarn_kv ? nbatch_V2 + 4 : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2));
    constexpr bool swz_K = !is_kvarn_kv && ggml_cuda_fattn_smem_swizzle::enabled(nbatch_K2);
    constexpr bool swz_V = V_is_K_view ? swz_K : !is_kvarn_kv && ggml_cuda_fattn_smem_swizzle::enabled(nbatch_V2);

    const int k_VKQ_0 = kb0 * nbatch_fa;
#if defined(TURING_MMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#else // Volta
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    if constexpr (nstages > 1) {
        static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        flash_attn_ext_f16_load_tile<stride_tile_V, swz_V, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
            (V_h2, tile_V, nbatch_V2, stride_V, k_VKQ_0, k_VKQ_sup, nullptr);
    } else {
        // the sparse mask values are gathered per element, always load them synchronously
        constexpr bool use_cp_async = nstages == 1 && !use_sparse;
        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (mask_h, tile_mask, stride_mask, k_VKQ_0, k_VKQ_sup, jt*ncols1, ne01, indices);
        }
    }

    // Experimental only: retain half accumulators across the D256 dot product,
    // then expand exactly once to the original float tile for softmax/masking.
    static_assert(std::is_same_v<decltype(T_C_KQ::x), float[T_C_KQ::ne]>, "QK accumulator must start as F32");
    using T_C_KQ_HALF = tile<T_C_KQ::I, T_C_KQ::J/2, half2>;
    T_C_KQ_HALF KQ_HALF[sizeof(KQ_C)/sizeof(KQ_C[0])];
    // For MLA K and V have the same data.
    // Therefore, iterate over K in reverse and later re-use the data if possible.
#pragma unroll
    for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;

        if constexpr (nstages <= 1) {
                if constexpr (ggml_cuda_fattn_kvarn_template_type(type_K)) {
                    constexpr bool kvarn_original_domain = type_K == GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE;
                    constexpr int nthreads_kvarn = nwarps * ggml_cuda_get_physical_warp_size();
                    flash_attn_ext_kvarn_load_tile<DKQ, stride_tile_K, nbatch_fa, nthreads_kvarn, oob_check,
                        kvarn_original_domain, false,
                        type_K == GGML_CUDA_FATTN_KVARN_TYPE && type_V == GGML_CUDA_FATTN_KVARN_TYPE>
                        ((const char *) K_h2, tile_K, k_VKQ_0, k_VKQ_sup, k0_start, k0_stop - k0_start, kvarn_smem);
                } else {
                    const int k0_diff = k0_stop - k0_start;
                    constexpr bool use_cp_async = nstages == 1;
                    flash_attn_ext_f16_load_tile<stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                        (K_h2 + k0_start, tile_K, k0_diff, stride_K, k_VKQ_0, k_VKQ_sup, indices);
                    if (use_cp_async) {
                        cp_async_wait_all();
                    }
                }
            __syncthreads();
        }

        // Calculate tile of KQ:
        if constexpr (Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    T_A_KQ K_A;
                    ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_K, swz_K>(K_A, tile_K, i_KQ_0, k_KQ_0 - k0_start);
                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_HALF[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_HALF[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_HALF[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
                }
            }
        } else {
            constexpr int stride_tile_Q = DKQ/2 + 4;
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;

                    T_A_KQ K_A;
                    ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_K, swz_K>(K_A, tile_K, i_KQ_0, k_KQ_0 - k0_start);

                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_HALF[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_HALF[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_HALF[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
                }
            }
        }

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }

#pragma unroll
    for(int t=0;t<int(sizeof(KQ_C)/sizeof(KQ_C[0]));++t) {
#pragma unroll
        for(int l=0;l<T_C_KQ_HALF::ne;++l) {
            const float2 f=__half22float2(KQ_HALF[t].x[l]);
            KQ_C[t].x[2*l]=f.x;KQ_C[t].x[2*l+1]=f.y;
        }
    }

    if (use_logit_softcap) {
        constexpr int stride = cols_per_warp == 8 ? np*T_C_KQ::I : np*T_C_KQ::J;
        static_assert(nbatch_fa % stride == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < nbatch_fa/stride; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (ncols2 > 1 || mask_h) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

        // Values per KQ column are spread across 8 threads:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFFULL, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else { // not Turing mma or T_B_KQ::I > 8
        if (ncols2 > 1 || mask_h) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;

                // The mask is stored as 16 bit half values, loading them as 32 bit half2 values is preferred in terms of speed.
                // However, this is not possible for RDNA3 where 2 consecutive l indices are not consecutive in the mask memory layout.
#ifdef RDNA3
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_j(l);
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::J)].x[l] += __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
#else
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                    const float2 tmp = __half22float2(((const half2 *)tile_mask)[j*(nbatch_fa/2 + 4) + i]);
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                }
#endif // RDNA3
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
            // Values per KQ column are spread across 4 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 1;
#elif defined(AMD_MFMA_AVAILABLE)
            // MFMA: 4 threads per Q column (threadIdx.x % 16 == col, spaced by 16).
            constexpr int offset_first = 32;
            constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 16;
            constexpr int offset_last  = 16;
#else // Volta
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFFULL, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                } else {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            // Rescale in fp32 to avoid double-rounding the scale to half first.
            const float scale_f32 = KQ_max_scale[0];
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    float2 acc_f32 = __half22float2(VKQ_C[i].x[l]);
                    acc_f32.x *= scale_f32;
                    acc_f32.y *= scale_f32;
                    VKQ_C[i].x[l] = make_half2(acc_f32.x, acc_f32.y);
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    if constexpr (nstages > 1) {
        static_assert(!use_sparse, "sparse gather not implemented for multi-stage loading");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        // Preload K tile for next iteration:
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if (!last_iter) {
            if (ncols2 > 1 || mask_h) {
                flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                    (mask_h, tile_mask, stride_mask, k_VKQ_0 + nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr);
            }
            flash_attn_ext_f16_load_tile<stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (K_h2, tile_K, nbatch_K2, stride_K, k_VKQ_0 + nbatch_fa, k_VKQ_sup, nullptr);
        }
    }


    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;

        if constexpr (nstages <= 1) {
                if constexpr (ggml_cuda_fattn_kvarn_template_type(type_V)) {
                    constexpr bool kvarn_original_domain = type_V == GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE;
                    constexpr int nthreads_kvarn = nwarps * ggml_cuda_get_physical_warp_size();
                    flash_attn_ext_kvarn_load_tile<DV, stride_tile_V, nbatch_fa, nthreads_kvarn, oob_check,
                        kvarn_original_domain, false,
                        type_K == GGML_CUDA_FATTN_KVARN_TYPE && type_V == GGML_CUDA_FATTN_KVARN_TYPE>
                        ((const char *) V_h2, tile_V, k_VKQ_0, k_VKQ_sup, i0_start / 2, nbatch_V2, kvarn_smem);
                    __syncthreads();
                } else {
                    if (!V_is_K_view || i0_stop > 2*nbatch_K2) {
                        const int i0_diff = i0_stop - i0_start;
                        constexpr bool use_cp_async = nstages == 1;
                        flash_attn_ext_f16_load_tile<stride_tile_V, swz_V, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                            (V_h2 + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_0, k_VKQ_sup, indices);
                        if (use_cp_async) {
                            cp_async_wait_all();
                        }
                        __syncthreads();
                    }
                }
        }
        const half2 * tile_V_i = !V_is_K_view || i0_stop > 2*nbatch_K2 ? tile_V : tile_V + i0_start/2;

#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
                ggml_cuda_fattn_smem_swizzle::load_ldmatrix_trans<stride_tile_V, swz_V>(A, tile_V, (int)(tile_V_i - tile_V) + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    // Wide version of VKQ_C is column-major.
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    // AMD matrix C is column-major.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
#else
                    // swap A and B for CUDA.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                }
            }
        }
#else // Volta
        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert(nbatch_fa % (np*T_A_VKQ::I) == 0, "bad loop size");
            static_assert(2*T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa; k00 += np*T_A_VKQ::I) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::I;

                T_A_VKQ A; // Transposed in both SRAM and registers, load normally.
                ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_V, swz_V>(A, tile_V, (int)(tile_V_i - tile_V) + k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
            }
        }
#endif // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

// KVarN-only tile selector. Forwards to mma_tile_sizes except RDNA3 D128/D256,
// which use the fp32-accumulator specializations below. Lets the dense path
// keep its qualified tiles while KVarN uses the qualified fp32 ones. The
// member lookups are dependent and resolve after the arch regions below.
template<int DV, int ncols> struct mma_tile_sizes;
template<int DV, int ncols> struct mma_tile_sizes_kvarn {
    using T_A_KQ  = typename mma_tile_sizes<DV, ncols>::T_A_KQ;
    using T_B_KQ  = typename mma_tile_sizes<DV, ncols>::T_B_KQ;
    using T_C_KQ  = typename mma_tile_sizes<DV, ncols>::T_C_KQ;
    using T_A_VKQ = typename mma_tile_sizes<DV, ncols>::T_A_VKQ;
    using T_B_VKQ = typename mma_tile_sizes<DV, ncols>::T_B_VKQ;
    using T_C_VKQ = typename mma_tile_sizes<DV, ncols>::T_C_VKQ;
};

#if defined(TURING_MMA_AVAILABLE)
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<int DV> struct mma_tile_sizes<DV, 8> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
#elif defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
// KVarN-only fp32-accumulator tiles (stew675 f32-VKQ guidance): DV=128/256
// with fp16 PV accumulator show ~3e-4/tile error compounding over 64 layers
// on gfx1100. Selected explicitly for the KVarN path via mma_tile_sizes_kvarn
// (forwarding primary declared above the region chain); dense keeps the
// primary half2 tile.
template<int ncols> struct mma_tile_sizes_kvarn<128, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes_kvarn<256, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#else
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR_SCRAMBLED>; // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
#endif // RDNA3
#elif defined(AMD_MFMA_AVAILABLE)
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
#else // Volta
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile< 8,  4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_KQ  = tile<32,  8, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile< 8,  4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>; // column-major
    using T_B_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#endif // defined(TURING_MMA_AVAILABLE)

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        float2       * const __restrict__ dst_final_meta,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols = ncols1 * ncols2;
    constexpr bool is_kvarn_kv = ggml_cuda_fattn_kvarn_template_type(type_K) || ggml_cuda_fattn_kvarn_template_type(type_V);
    using tile_sizes_sel = typename std::conditional<is_kvarn_kv,
        mma_tile_sizes_kvarn<DV, ncols>, mma_tile_sizes<DV, ncols>>::type;
    using     T_A_KQ    = typename tile_sizes_sel::T_A_KQ;
    using     T_B_KQ    = typename tile_sizes_sel::T_B_KQ;
    using     T_C_KQ    = typename tile_sizes_sel::T_C_KQ;
    using     T_A_VKQ   = typename tile_sizes_sel::T_A_VKQ;
    using     T_B_VKQ   = typename tile_sizes_sel::T_B_VKQ;
    using     T_C_VKQ   = typename tile_sizes_sel::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);
    constexpr int  nstages         = is_kvarn_kv ? 0 : ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, use_sparse);
    static_assert(!is_kvarn_kv || !use_sparse, "sparse KVarN record loads are not qualified");

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = DKQ/2     + 4;
    // swizzle the tile stride for K and V based on the batch size.
    constexpr int stride_tile_K = is_kvarn_kv ? nbatch_K2 + 4 : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2);
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K :
        (is_kvarn_kv ? nbatch_V2 + 4 : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2));
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;
    constexpr bool swz_K = !is_kvarn_kv && ggml_cuda_fattn_smem_swizzle::enabled(nbatch_K2);
    constexpr bool swz_V = V_is_K_view ? swz_K : !is_kvarn_kv && ggml_cuda_fattn_smem_swizzle::enabled(nbatch_V2);

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
    half2 * tile_V    =           nstages > 1 ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (nstages > 1 ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);
    half * kvarn_smem = tile_mask + ncols1 * (nbatch_fa + 8);

    constexpr bool cache_kvarn_record_axes =
        type_K == GGML_CUDA_FATTN_KVARN_TYPE && type_V == GGML_CUDA_FATTN_KVARN_TYPE;
    if constexpr (cache_kvarn_record_axes) {
        static_assert(DKQ == DV, "rotated KVarN K/V axes require matching attention head dimensions");
        constexpr int axis_tag_count = 2 * (DKQ / GGML_CUDA_FATTN_KVARN_DIM);
        int * axis_group_tags = (int *) (kvarn_smem + 6 * DKQ);
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (int i = 0; i < axis_tag_count; ++i) {
                axis_group_tags[i] = -1;
            }
        }
        __syncthreads();
    }

    T_B_KQ    Q_B[(Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1)];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    // Entry count mirrors the rescale loops: half2 accumulators fold two
    // stacked K-halves per entry via the opsel pair (DV/32 for DV%32==0),
    // float accumulators keep one 16-row tile per entry (DV/16 always).
    static constexpr int VKQ_C_COUNT = std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>
        ? DV/T_C_VKQ::J
        : (DV % 32 != 0 ? DV/T_C_VKQ::J : DV/(2*T_C_VKQ::J));
    T_C_VKQ VKQ_C[VKQ_C_COUNT];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    if (Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    int kb0 = kb0_start;

    // Preload mask and K data for first iteration when using cp_async with multiple stages:
    if constexpr (nstages > 1) {
        static_assert(!use_sparse, "sparse gather not implemented for multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi-stage pipeline");
        constexpr bool use_cp_async = true;
        constexpr bool oob_check    = false;
        constexpr int  k_VKQ_sup    = nbatch_fa;
        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (mask_h, tile_mask, stride_mask, kb0*nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr);
        }
        flash_attn_ext_f16_load_tile<stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
            (K_h2, tile_K, nbatch_K2, stride_K, kb0*nbatch_fa, k_VKQ_sup, nullptr);
    }

    // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
    if constexpr (ncols2 == 1 || use_sparse) {
        constexpr bool oob_check = true;
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, type_K, type_V>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, kvarn_smem, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
        }
        constexpr bool last_iter = true;
        const     int  k_VKQ_sup = ne11 - kb0*nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
              T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, type_K, type_V>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, kvarn_smem, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
    } else {
        constexpr bool oob_check = false;
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, type_K, type_V>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, kvarn_smem, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
        }
        constexpr bool last_iter = true;
        constexpr int  k_VKQ_sup = nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, type_K, type_V>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, kvarn_smem, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
    }

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if constexpr (nstages > 1 && nwarps*cols_per_warp > nbatch_fa) {
        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#elif defined(AMD_MFMA_AVAILABLE)
        // The partial sums are spread across 4 threads (wavefront64, 16 cols).
        constexpr int offset_first = 32;
        constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFFULL, KQ_rowsum[col], offset, warp_size);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchronization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = (threadIdx.y/np)*cols_per_warp + (cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2*col));
            const int c = jc % ncols2;
            const bool valid_head = ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio;
            if (!valid_head) {
                KQ_max_scale[col] = 1.0f;
                continue;
            }
            const float sink = sinks_f[c];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            // Rescale in fp32 to avoid double-rounding the scale to half first.
            const float scale_f32 = KQ_max_scale[0];
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    float2 acc_f32 = __half22float2(VKQ_C[i].x[l]);
                    acc_f32.x *= scale_f32;
                    acc_f32.y *= scale_f32;
                    VKQ_C[i].x[l] = make_half2(acc_f32.x, acc_f32.y);
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const int col = (threadIdx.x / 2) % 2;
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    constexpr bool combine_needs_sync = swz_K || swz_V;

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if constexpr (combine_needs_sync) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            // KVarN whole-tile blocks must publish final (max, rowsum) too: the
            // tail merge reads body_meta for every row, and the stream-k fixup
            // skips tiles whose K range aligns exactly to tile boundaries, so
            // without this store those rows keep zero meta and their (correct)
            // body values are silently discarded by the merge.
            if (!needs_fixup && !is_fixup && dst_final_meta && threadIdx.x < T_B_KQ::I) {
                const int j = jc_cwm / ncols2;
                const int c = jc_cwm % ncols2;
                if (jt*ncols1 + j < int(ne01.z) && zt_gqa*ncols2 + c < gqa_ratio) {
                    dst_final_meta[(jt*ncols1 + j)*ne02 + c] = KQ_cmr;
                }
            }
        }
    } else {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
#else // Volta
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if constexpr (combine_needs_sync) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            // KVarN whole-tile blocks must publish final (max, rowsum) too: the
            // tail merge reads body_meta for every row, and the stream-k fixup
            // skips tiles whose K range aligns exactly to tile boundaries, so
            // without this store those rows keep zero meta and their (correct)
            // body values are silently discarded by the merge.
            if (!needs_fixup && !is_fixup && dst_final_meta && thread_should_write) {
                const int j = jc_cwm / ncols2;
                const int c = jc_cwm % ncols2;
                if (jt*ncols1 + j < int(ne01.z) && zt_gqa*ncols2 + c < gqa_ratio) {
                    dst_final_meta[(jt*ncols1 + j)*ne02 + c] = KQ_cmr;
                }
            }
        }
    }

    if (np > 1 && threadIdx.y % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        constexpr int nmeta = np*cols_per_warp >= warp_size ? np*cols_per_warp/warp_size : 1;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < warp_size ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < warp_size) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFFULL, KQ_cmn, offset, warp_size));
            }
        }

        float KQ_cms[nmeta]; // KQ combine max scale per warp.
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < warp_size) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFFULL, KQ_crs, offset, warp_size);
            }
        }

        __syncthreads();

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= warp_size || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        // Combined KQ max + rowsum.
        static_assert(cols_per_warp <= warp_size);
        if (needs_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        // KVarN whole-tile blocks must publish final (max, rowsum) too: the
        // tail merge reads body_meta for every row, and the stream-k fixup
        // skips tiles whose K range aligns exactly to tile boundaries, so
        // without this store those rows keep zero meta and their (correct)
        // body values are silently discarded by the merge.
        if (!needs_fixup && !is_fixup && dst_final_meta &&
                (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
            const int jc = (threadIdx.y/np)*cols_per_warp + threadIdx.x;
            if (jc < ncols) {
                const int j = jc / ncols2;
                const int c = jc % ncols2;
                if (jt*ncols1 + j < int(ne01.z) && zt_gqa*ncols2 + c < gqa_ratio) {
                    dst_final_meta[(jt*ncols1 + j)*ne02 + c] = make_float2(KQ_cmn, KQ_crs);
                }
            }
        }
    } else if (np > 1) {
        // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
        // Therefore, all other warps also need to execute a __syncthreads().
        // Otherwise the points at which warps synchronize with each other would become misaligned.
        __syncthreads();
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>, "bad VKQ type");
            const int jc_cwd = threadIdx.y*T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = threadIdx.y*cols_per_warp;
            if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
                if constexpr (T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR) {
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                        for (int l = 0; l < T_C_VKQ::ne; ++l) {
                            const int j = j0 + T_C_VKQ::get_i(l);
                            const int k = k1 + T_C_VKQ::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                        }
                    }
                } else {
                    static_assert(T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR_SCRAMBLED, "bad T_C_VKQ data layout");
                    using T_C_VKQ_us = tile<T_C_VKQ::I, T_C_VKQ::J, half2, DATA_LAYOUT_I_MAJOR>; // us == unscrambled
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
                        const T_C_VKQ_us VKQ_C_us = unscramble(VKQ_C[(k00 + k1)/T_C_VKQ::J]);
#pragma unroll
                        for (int l = 0; l < T_C_VKQ_us::ne; ++l) {
                            const int j = j0 + T_C_VKQ_us::get_i(l);
                            const int k = k1 + T_C_VKQ_us::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C_us.x[l];
                        }
                    }
                }
            } else {
                static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
                half * tile_Q_h = (half *) tile_Q;
#pragma unroll
                for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J/2) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        const int j = j0 + T_C_VKQ::get_i(l);
                        const int k = 2*k1 + T_C_VKQ::get_j(l);

                        tile_Q_h[j*(2*tile_stride) + k] = VKQ_C[(k00 + k1)/(T_C_VKQ::J/2)].x[l];
                    }
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
                const int k0_start  = stride_k == warp_size ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = warp_size / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && ((ncols1 > 1 && jt*ncols1 + j_dst >= int(ne01.z)) || (ncols2 > 1 && zt_gqa*ncols2 + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        // The tile_Q buffer is reused for the next k00 iteration, so all warps must sync here
        // before its data is overwritten. With np > 1 only some warps read back, but they all write.
        if (np > 1 || k00 + nbatch_combine < DV/2) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dstk_fixup, dst_final_meta,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
        jt, kb0_start, kb0_stop);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static constexpr __host__ __device__ bool ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(
        const int DKQ, const int DV, const int ncols1, const int ncols2) {
    return (DKQ == 512 && DV == 512 && ncols1 == 1 && ncols2 == 8) ||
           (DKQ == 576 && DV == 512 && ncols1 == 1 && ncols2 == 16);
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view, bool use_sparse = false,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_f16(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        float2     * dst_final_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_sync(); // TODO optimize placement
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    const char * GGML_CUDA_RESTRICT Q              = Q_ptr;
    const char * GGML_CUDA_RESTRICT K              = K_ptr;
    const char * GGML_CUDA_RESTRICT V              = V_ptr;
    const char * GGML_CUDA_RESTRICT mask           = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks          = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max         = use_sparse ? nullptr : KV_max_ptr;
    const int  * GGML_CUDA_RESTRICT sparse_indices = use_sparse ? KV_max_ptr : nullptr;
    float      * GGML_CUDA_RESTRICT dst            = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta       = dst_meta_ptr;
    float2     * GGML_CUDA_RESTRICT dst_final_meta = dst_final_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256 || DKQ == 512)) {
        NO_DEVICE_CODE;
        return;
    }
    if (DKQ == 192 && ncols2 != 8 && ncols2 != 16) {
        NO_DEVICE_CODE;
        return;
    }

    if (!ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2) && use_sparse) {
        NO_DEVICE_CODE;
        return;
    }
#ifdef VOLTA_MMA_AVAILABLE
    if (ncols1*ncols2 < 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

#if defined(AMD_WMMA_AVAILABLE)
    // Mirrored by ggml_cuda_fattn_kvarn_amd_mma_eligibility on the host.
    // Keep this final invariant for callers outside the KVarN dispatcher.
    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 256) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)

#if defined(AMD_MFMA_AVAILABLE)
    // Mirrored by ggml_cuda_fattn_kvarn_amd_mma_eligibility on the host.
    if (ncols1*ncols2 < 16 || DKQ > 256) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_MFMA_AVAILABLE)

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / warp_size;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = V_is_K_view ? stride_K : nb21 / sizeof(half2);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);
        float2       * dst_final_meta_tile = dst_final_meta ? dst_final_meta + sequence*ne01.z*ne02 + zt_Q : nullptr;

        const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
        const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;
        const int32_t * indices = use_sparse ? sparse_indices + (int64_t(sequence % ne33)*ne31 + jt*ncols1)*ne11 : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, type_K, type_V>
                (Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dst_meta, dst_final_meta_tile, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, type_K, type_V>
                (Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dst_meta, dst_final_meta_tile, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
        }

        // The next process_tile call reuses the tile_Q buffer for its Q/K tiles, so all warps must
        // have finished reading the combined results before any of them starts the next call.
        // (With np == 1 the end-of-k00 barrier does not fire, so this is required for correctness.)
        __syncthreads();

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);
    float2       * dst_final_meta_tile = dst_final_meta ? dst_final_meta + sequence*ne01.z*ne02 + zt_Q : nullptr;

    const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
    const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;
    const int32_t * indices = use_sparse ? sparse_indices + (int64_t(sequence % ne33)*ne31 + jt*ncols1)*ne11 : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, type_K, type_V>
        (Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dst_meta, dst_final_meta_tile, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, dst_final_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = DKQ == 576; // Guaranteed by the kernel selection logic in fattn.cu

    // KV tile strides must match flash_attn_ext_f16_iter / _process_tile.
    const int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2, cc);
    const int stride_tile_V = V_is_K_view ? stride_tile_K : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2, cc);
    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(stride_tile_K,  stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (stride_tile_K + stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    bool use_sparse = false;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2)) {
            if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ctx, dst)) {
                constexpr bool use_sparse_kernel = true;
                fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel>;
                use_sparse = true;

                static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
                if (!shared_memory_limit_raised[id]) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                    shared_memory_limit_raised[id] = true;
                }
            } else {
                constexpr bool use_sparse_kernel = false;
                fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel>;

                static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
                if (!shared_memory_limit_raised[id]) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                    shared_memory_limit_raised[id] = true;
                }
            }
        } else
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
            constexpr bool use_sparse_kernel = false;
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel>;

#if !defined(GGML_USE_MUSA)
            static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
            if (!shared_memory_limit_raised[id]) {
                CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                shared_memory_limit_raised[id] = true;
            }
#endif // !defined(GGML_USE_MUSA)
        }
    } else {
        constexpr bool use_logit_softcap = true;
        constexpr bool use_sparse_kernel = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, true, true, true, use_sparse, warp_size_host);
}







#if defined(GGML_USE_HIP)
using ggml_cuda_fattn_kernel_attr_ptr_t = const void *;
#else
using ggml_cuda_fattn_kernel_attr_ptr_t = fattn_kernel_t;
#endif

// Keep prefill single-window through 64K to avoid an additional floating-point
// partial merge. Lower values remain available through GGML_KVARN_WINDOW_CHUNK
// when concurrent long prompts require less transient K/V scratch.
static constexpr int GGML_CUDA_FATTN_KVARN_WINDOW_CHUNK = 65536;

static inline bool ggml_cuda_fattn_kvarn_window_enabled() {
    const char * env = getenv("GGML_KVARN_WINDOW");
    return env == nullptr || atoi(env) != 0;
}

static inline int ggml_cuda_fattn_kvarn_window_chunk(const ggml_tensor * dst, const int n_kv) {
    const int context_chunk = ggml_get_op_params_i32(
            dst, GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_WINDOW_CHUNK);
    if (context_chunk > 0) {
        return std::min(n_kv, context_chunk);
    }

    const char * env = getenv("GGML_KVARN_WINDOW_CHUNK");
    if (env == nullptr) {
        return std::min(n_kv, GGML_CUDA_FATTN_KVARN_WINDOW_CHUNK);
    }
    const int override_chunk = atoi(env);
    return override_chunk > 0 ? std::min(n_kv, override_chunk) :
        std::min(n_kv, GGML_CUDA_FATTN_KVARN_WINDOW_CHUNK);
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
static inline fattn_kernel_t ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel(
        bool k_original_domain,
        bool v_original_domain) {
    constexpr bool V_is_K_view = false;

    if (k_original_domain) {
        GGML_ASSERT(v_original_domain);
        return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false,
            GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE, GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE>;
    }

    if (v_original_domain) {
        return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false,
            GGML_CUDA_FATTN_KVARN_TYPE, GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE>;
    }

    return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false,
        GGML_CUDA_FATTN_KVARN_TYPE, GGML_CUDA_FATTN_KVARN_TYPE>;
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void ggml_cuda_fattn_kvarn_window_f16_partial_kernel(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        float2 * partial_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne33, const int32_t nb31, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    constexpr int ncols = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads (DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / ggml_cuda_get_physical_warp_size();

    const int gqa_ratio = ne02 / ne12;
    const int iter_j = (ne01.z + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;

    const int tile = blockIdx.x;
    const int sequence = tile / (iter_j * iter_z_gqa * ne12);
    const int rem0 = tile - sequence * iter_j * iter_z_gqa * ne12;
    const int z_KV = rem0 / (iter_j * iter_z_gqa);
    const int rem1 = rem0 - z_KV * iter_j * iter_z_gqa;
    const int zt_gqa = rem1 / iter_j;
    const int jt = rem1 - zt_gqa * iter_j;
    const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2;

    const float2 * Q_f2 = (const float2 *) (Q_ptr + nb03 * sequence + nb02 * zt_Q);
    const half2  * K_h2 = (const half2  *) (K_ptr + nb13 * sequence + nb12 * z_KV);
    const half2  * V_h2 = (const half2  *) (V_ptr + nb23 * sequence + nb22 * z_KV);
    const half   * mask_h = ncols2 == 1 && !mask_ptr ? nullptr :
        (const half *) (mask_ptr + nb33 * (sequence % ne33));
    const float  * sinks_f = sinks_ptr ? (const float *) sinks_ptr + zt_Q : nullptr;

    float2 * dstk = nullptr;
    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;
    const int iter_k = (ne11 + nbatch_fa - 1) / nbatch_fa;
    constexpr bool V_is_K_view = false;
    constexpr bool needs_fixup = false;
    constexpr bool is_fixup = true;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, false, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h, nullptr, sinks_f, dstk, partial_ptr, nullptr, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, nb01 / (int32_t) sizeof(float2), nb02 / (int32_t) sizeof(float2),
         nb11 / (int32_t) sizeof(half2), nb21 / (int32_t) sizeof(half2), nb31 / (int32_t) sizeof(half),
         jt, zt_gqa, 0, iter_k);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, partial_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23, ne33, nb31, nb33);
    NO_DEVICE_CODE;
#endif
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void ggml_cuda_fattn_kvarn_window_f16_direct_kernel(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        float * dst_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne33, const int32_t nb31, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    constexpr int ncols = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads (DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / ggml_cuda_get_physical_warp_size();

    const int gqa_ratio = ne02 / ne12;
    const int iter_j = (ne01.z + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;

    const int tile = blockIdx.x;
    const int sequence = tile / (iter_j * iter_z_gqa * ne12);
    const int rem0 = tile - sequence * iter_j * iter_z_gqa * ne12;
    const int z_KV = rem0 / (iter_j * iter_z_gqa);
    const int rem1 = rem0 - z_KV * iter_j * iter_z_gqa;
    const int zt_gqa = rem1 / iter_j;
    const int jt = rem1 - zt_gqa * iter_j;
    const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2;

    const float2 * Q_f2 = (const float2 *) (Q_ptr + nb03 * sequence + nb02 * zt_Q);
    const half2  * K_h2 = (const half2  *) (K_ptr + nb13 * sequence + nb12 * z_KV);
    const half2  * V_h2 = (const half2  *) (V_ptr + nb23 * sequence + nb22 * z_KV);
    const half   * mask_h = ncols2 == 1 && !mask_ptr ? nullptr :
        (const half *) (mask_ptr + nb33 * (sequence % ne33));
    const float  * sinks_f = sinks_ptr ? (const float *) sinks_ptr + zt_Q : nullptr;
    float2       * dstk = ((float2 *) dst_ptr) + ((size_t) sequence * ne01.z * ne02 + zt_Q) * (DV / 2);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;
    const int iter_k = (ne11 + nbatch_fa - 1) / nbatch_fa;
    constexpr bool V_is_K_view = false;
    constexpr bool needs_fixup = false;
    constexpr bool is_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, false, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h, nullptr, sinks_f, dstk, nullptr, nullptr, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, nb01 / (int32_t) sizeof(float2), nb02 / (int32_t) sizeof(float2),
         nb11 / (int32_t) sizeof(half2), nb21 / (int32_t) sizeof(half2), nb31 / (int32_t) sizeof(half),
         jt, zt_gqa, 0, iter_k);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23, ne33, nb31, nb33);
    NO_DEVICE_CODE;
#endif
}

template<int D, int ncols1, int ncols2>
__launch_bounds__(D, 1)
static __global__ void ggml_cuda_fattn_kvarn_window_merge_kernel(
        const float2 * partial_ptr,
        float * acc_ptr,
        float2 * acc_meta_ptr,
        bool init,
        const uint3 ne01,
        const int ne02,
        const int ne12,
        const int gqa_ratio,
        const int ntiles_dst) {
    constexpr int ncols = ncols1 * ncols2;
    const int tile = blockIdx.x;
    const int jc = blockIdx.y;
    const int d = threadIdx.x;
    const int j = jc / ncols2;
    const int c = jc - j * ncols2;

    const int iter_j = (ne01.z + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int sequence = tile / (iter_j * iter_z_gqa * ne12);
    const int rem0 = tile - sequence * iter_j * iter_z_gqa * ne12;
    const int z_KV = rem0 / (iter_j * iter_z_gqa);
    const int rem1 = rem0 - z_KV * iter_j * iter_z_gqa;
    const int zt_gqa = rem1 / iter_j;
    const int jt = rem1 - zt_gqa * iter_j;

    const int q = jt * ncols1 + j;
    if (q >= (int) ne01.z || zt_gqa * ncols2 + c >= gqa_ratio) {
        return;
    }

    const int q_head = z_KV * gqa_ratio + zt_gqa * ncols2 + c;
    const size_t out_off = ((size_t) sequence * ne01.z * ne02 + (size_t) q * ne02 + q_head) * D + d;
    const size_t row_off = ((size_t) sequence * ne01.z + q) * ne02 + q_head;

    const float2 part_meta = partial_ptr[((size_t) ntiles_dst + tile) * ncols + jc];
    const float2 * partial_data = partial_ptr + (size_t) ntiles_dst * (2 * ncols) +
        ((size_t) tile * ncols + jc) * (D / 2);
    const float part = ((const float *) partial_data)[d];

    if (init) {
        const bool has_data = part_meta.y > 0.0f;
        acc_ptr[out_off] = has_data ? part : 0.0f;
        if (d == 0) {
            acc_meta_ptr[row_off] = has_data ? part_meta : make_float2(0.0f, 0.0f);
        }
        return;
    }

    const float2 acc_meta = acc_meta_ptr[row_off];
    __syncthreads();
    if (part_meta.y <= 0.0f) {
        return;
    }
    if (acc_meta.y <= 0.0f) {
        acc_ptr[out_off] = part;
        if (d == 0) {
            acc_meta_ptr[row_off] = part_meta;
        }
        return;
    }

    const float max_new = fmaxf(acc_meta.x, part_meta.x);
    const float acc_diff = acc_meta.x - max_new;
    const float part_diff = part_meta.x - max_new;
    const float acc_scale = acc_diff >= SOFTMAX_FTZ_THRESHOLD ? expf(acc_diff) : 0.0f;
    const float part_scale = part_diff >= SOFTMAX_FTZ_THRESHOLD ? expf(part_diff) : 0.0f;
    acc_ptr[out_off] = acc_scale * acc_ptr[out_off] + part_scale * part;
    if (d == 0) {
        acc_meta_ptr[row_off] = make_float2(max_new, acc_scale * acc_meta.y + part_scale * part_meta.y);
    }
}

template<int D, int ncols1, int ncols2>
__launch_bounds__(D, 1)
static __global__ void ggml_cuda_fattn_kvarn_window_single_finalize_kernel(
        const float2 * partial_ptr,
        float * dst_ptr,
        float2 * dst_meta_ptr,
        const uint3 ne01,
        const int ne02,
        const int ne12,
        const int gqa_ratio,
        const int ntiles_dst) {
    constexpr int ncols = ncols1 * ncols2;
    const int tile = blockIdx.x;
    const int jc = blockIdx.y;
    const int d = threadIdx.x;
    const int j = jc / ncols2;
    const int c = jc - j * ncols2;

    const int iter_j = (ne01.z + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int sequence = tile / (iter_j * iter_z_gqa * ne12);
    const int rem0 = tile - sequence * iter_j * iter_z_gqa * ne12;
    const int z_KV = rem0 / (iter_j * iter_z_gqa);
    const int rem1 = rem0 - z_KV * iter_j * iter_z_gqa;
    const int zt_gqa = rem1 / iter_j;
    const int jt = rem1 - zt_gqa * iter_j;

    const int q = jt * ncols1 + j;
    if (q >= (int) ne01.z || zt_gqa * ncols2 + c >= gqa_ratio) {
        return;
    }

    const int q_head = z_KV * gqa_ratio + zt_gqa * ncols2 + c;
    const size_t out_off = ((size_t) sequence * ne01.z * ne02 + (size_t) q * ne02 + q_head) * D + d;
    const size_t row_off = ((size_t) sequence * ne01.z + q) * ne02 + q_head;

    const float2 part_meta = partial_ptr[((size_t) ntiles_dst + tile) * ncols + jc];
    const float2 * partial_data = partial_ptr + (size_t) ntiles_dst * (2 * ncols) +
        ((size_t) tile * ncols + jc) * (D / 2);
    const float part = ((const float *) partial_data)[d];
    dst_ptr[out_off] = part_meta.y > 0.0f ? part / part_meta.y : 0.0f;
    if (d == 0 && dst_meta_ptr != nullptr) {
        dst_meta_ptr[row_off] = part_meta;
    }
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
static bool ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case_impl(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan,
        size_t nbytes_shared_total) {
    if constexpr (DKQ != DV || (DKQ != 128 && DKQ != 256 && DKQ != 512)) {
        GGML_UNUSED_VARS(ctx, dst, plan, nbytes_shared_total);
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float2 * const dst_meta = dst->src[8] != nullptr ? (float2 *) dst->src[8]->data : nullptr;
    const enum ggml_flash_attn_ext_kvarn_domain domain = ggml_cuda_fattn_kvarn_domain(dst);
    if (!ggml_cuda_fattn_kvarn_window_enabled() ||
            Q->ne[1] <= 1 || sinks != nullptr ||
            domain != GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V) {
        return false;
    }

#if !defined(GGML_USE_MUSA)
    // The partial kernel does not share the FlashAttention signature, so it is
    // passed as an opaque entry pointer instead of being cast to fattn_kernel_t.
    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<const void *>(
            ggml_cuda_fattn_kvarn_window_f16_partial_kernel<DKQ, DV, ncols1, ncols2, use_logit_softcap>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    constexpr int ncols = ncols1 * ncols2;
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc);
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols, cc);
    const int nwarps = nthreads / warp_size_host;
    const int window_chunk = ggml_cuda_fattn_kvarn_window_chunk(dst, plan.n_kv);

    if (getenv("GGML_CUDA_FA_ROUTE_DEBUG") != nullptr) {
        fprintf(stderr,
            "CUDA_FA_ROUTE_EXEC_DISPATCH kernel=KVARN_WINDOWED "
            "Q=[%lld,%lld,%lld,%lld] n_kv=%d n_kv_heads=%d n_stream=%d q_stream=%lld "
            "chunk=%d ncols=[%d,%d] domain=%s bits=[%d,%d]\n",
            (long long) Q->ne[0], (long long) Q->ne[1],
            (long long) Q->ne[2], (long long) Q->ne[3],
            plan.n_kv, plan.n_kv_heads, plan.n_stream, (long long) Q->ne[3],
            window_chunk, ncols1, ncols2, ggml_cuda_fattn_kvarn_domain_name(dst), plan.k.bits, plan.v.bits);
        fflush(stderr);
    }

    const size_t n_desc = (size_t) plan.n_stream * plan.n_kv_heads;
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> k_desc(pool, n_desc);
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> v_desc(pool, n_desc);
    const bool k_original_domain = ggml_cuda_fattn_kvarn_k_original_domain(dst);
    const bool v_original_domain = ggml_cuda_fattn_kvarn_v_original_domain(dst);
    ggml_cuda_fattn_kvarn_init_descs(plan, k_desc.get(), v_desc.get(),
            k_original_domain ? 1 : 0, v_original_domain ? 1 : 0, stream);

    const int chunk_cap = window_chunk;
    ggml_cuda_pool_alloc<half> k_f16(pool, (size_t) chunk_cap * plan.n_kv_heads * plan.n_stream * DKQ);
    ggml_cuda_pool_alloc<half> v_f16(pool, (size_t) chunk_cap * plan.n_kv_heads * plan.n_stream * DV);

    const int gqa_ratio = Q->ne[2] / plan.n_kv_heads;
    const int ntiles_x = (Q->ne[1] + ncols1 - 1) / ncols1;
    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int ntiles_dst = ntiles_x * ntiles_z_gqa * plan.n_kv_heads * Q->ne[3];
    const int n_rows = (int) ((size_t) Q->ne[1] * Q->ne[2] * Q->ne[3]);
    const size_t partial_fixup_f2 = (size_t) ntiles_dst * (2 * ncols + ncols * (DV / 2));

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    static const ggml_cuda_fattn_kvarn_window_dequant_kernel_t dequant_kernel =
        ggml_cuda_fattn_kvarn_window_dequant_get_kernel<DKQ>();
    static const ggml_cuda_fattn_kvarn_window_finalize_kernel_t finalize_kernel =
        ggml_cuda_fattn_kvarn_window_finalize_get_kernel<DV>();
    const dim3 dequant_block((uint32_t) (2 * plan.slices * warp_size_host), 1, 1);
    const dim3 partial_block((uint32_t) warp_size_host,
            (uint32_t) nwarps, 1);
    const dim3 partial_grid((uint32_t) ntiles_dst, 1, 1);
    const dim3 merge_block(DV, 1, 1);
    const dim3 merge_grid((uint32_t) ntiles_dst, ncols, 1);

    // The single-window path below may dequantize the full active window, but it
    // is bounded by window_chunk and kept as a transient scratch allocation.
    // It is not a graph-level KVarN materialize fallback.
    if (window_chunk >= plan.n_kv) {
        const int chunk_len = plan.n_kv;
        const dim3 dequant_grid((uint32_t) chunk_len, (uint32_t) plan.n_kv_heads, (uint32_t) plan.n_stream);
        ggml_cuda_kernel_launch_params dequant_params(dequant_grid, dequant_block, 0, stream);
        ggml_cuda_kernel_launch(dequant_kernel, dequant_params,
            k_desc.get(), v_desc.get(), k_f16.get(), v_f16.get(), 0, chunk_len, plan.n_kv_heads);

        const char * mask_data = mask ? (const char *) mask->data : nullptr;
        if (Q->ne[1] >= 512) {
            ggml_tensor k_win = *dst->src[1];
            ggml_tensor v_win = *dst->src[2];

            k_win.type = GGML_TYPE_F16;
            k_win.data = k_f16.get();
            k_win.view_src = nullptr;
            k_win.view_offs = 0;
            k_win.ne[0] = DKQ;
            k_win.ne[1] = chunk_len;
            k_win.ne[2] = plan.n_kv_heads;
            k_win.ne[3] = plan.n_stream;
            k_win.nb[0] = sizeof(half);
            k_win.nb[1] = DKQ * (int64_t) sizeof(half);
            k_win.nb[2] = chunk_len * DKQ * (int64_t) sizeof(half);
            k_win.nb[3] = (int64_t) plan.n_kv_heads * chunk_len * DKQ * (int64_t) sizeof(half);

            v_win.type = GGML_TYPE_F16;
            v_win.data = v_f16.get();
            v_win.view_src = nullptr;
            v_win.view_offs = 0;
            v_win.ne[0] = DV;
            v_win.ne[1] = chunk_len;
            v_win.ne[2] = plan.n_kv_heads;
            v_win.ne[3] = plan.n_stream;
            v_win.nb[0] = sizeof(half);
            v_win.nb[1] = DV * (int64_t) sizeof(half);
            v_win.nb[2] = chunk_len * DV * (int64_t) sizeof(half);
            v_win.nb[3] = (int64_t) plan.n_kv_heads * chunk_len * DV * (int64_t) sizeof(half);

            fattn_kernel_t f16_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false>;
#if !defined(GGML_USE_MUSA)
            CUDA_CHECK(cudaFuncSetAttribute(
                reinterpret_cast<ggml_cuda_fattn_kernel_attr_ptr_t>(f16_kernel),
                cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif

            ggml_tensor * orig_k = dst->src[1];
            ggml_tensor * orig_v = dst->src[2];
            dst->src[1] = &k_win;
            dst->src[2] = &v_win;
            launch_fattn<DV, ncols1, ncols2>(
                ctx, dst, f16_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, false, warp_size_host);
            dst->src[1] = orig_k;
            dst->src[2] = orig_v;
            return true;
        }

        ggml_cuda_pool_alloc<float2> partial(pool, partial_fixup_f2);
        ggml_cuda_kernel_launch_params partial_params(partial_grid, partial_block, nbytes_shared_total, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_f16_partial_kernel<DKQ, DV, ncols1, ncols2, use_logit_softcap>,
            partial_params,
            (const char *) Q->data,
            (const char *) k_f16.get(),
            (const char *) v_f16.get(),
            mask_data,
            nullptr,
            partial.get(),
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            DKQ, chunk_len, plan.n_kv_heads, plan.n_stream,
            DKQ * (int32_t) sizeof(half),
            chunk_len * DKQ * (int32_t) sizeof(half),
            (int64_t) plan.n_kv_heads * chunk_len * DKQ * (int64_t) sizeof(half),
            DV * (int32_t) sizeof(half),
            chunk_len * DV * (int32_t) sizeof(half),
            (int64_t) plan.n_kv_heads * chunk_len * DV * (int64_t) sizeof(half),
            mask ? (int32_t) mask->ne[3] : 1,
            mask ? (int32_t) mask->nb[1] : 0,
            mask ? (int64_t) mask->nb[3] : 0);
        ggml_cuda_kernel_launch_params single_finalize_params(merge_grid, merge_block, 0, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_single_finalize_kernel<DV, ncols1, ncols2>, single_finalize_params,
            partial.get(), (float *) dst->data, dst_meta, ne01, Q->ne[2], plan.n_kv_heads, gqa_ratio, ntiles_dst);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    ggml_cuda_pool_alloc<float2> partial(pool, partial_fixup_f2);
    ggml_cuda_pool_alloc<float2> acc_meta(pool, n_rows);

    bool init = true;
    for (int chunk_begin = 0; chunk_begin < plan.n_kv; chunk_begin += window_chunk) {
        const int chunk_len = std::min(window_chunk, plan.n_kv - chunk_begin);
        const dim3 dequant_grid((uint32_t) chunk_len, (uint32_t) plan.n_kv_heads, (uint32_t) plan.n_stream);
        ggml_cuda_kernel_launch_params dequant_params(dequant_grid, dequant_block, 0, stream);
        ggml_cuda_kernel_launch(dequant_kernel, dequant_params,
            k_desc.get(), v_desc.get(), k_f16.get(), v_f16.get(), chunk_begin, chunk_len, plan.n_kv_heads);

        const char * mask_data = mask ? (const char *) mask->data + (size_t) chunk_begin * mask->nb[0] : nullptr;
        ggml_cuda_kernel_launch_params partial_params(partial_grid, partial_block, nbytes_shared_total, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_f16_partial_kernel<DKQ, DV, ncols1, ncols2, use_logit_softcap>,
            partial_params,
            (const char *) Q->data,
            (const char *) k_f16.get(),
            (const char *) v_f16.get(),
            mask_data,
            nullptr,
            partial.get(),
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            DKQ, chunk_len, plan.n_kv_heads, plan.n_stream,
            DKQ * (int32_t) sizeof(half),
            chunk_len * DKQ * (int32_t) sizeof(half),
            (int64_t) plan.n_kv_heads * chunk_len * DKQ * (int64_t) sizeof(half),
            DV * (int32_t) sizeof(half),
            chunk_len * DV * (int32_t) sizeof(half),
            (int64_t) plan.n_kv_heads * chunk_len * DV * (int64_t) sizeof(half),
            mask ? (int32_t) mask->ne[3] : 1,
            mask ? (int32_t) mask->nb[1] : 0,
            mask ? (int64_t) mask->nb[3] : 0);

        ggml_cuda_kernel_launch_params merge_params(merge_grid, merge_block, 0, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_merge_kernel<DV, ncols1, ncols2>, merge_params,
            partial.get(), (float *) dst->data, acc_meta.get(), init, ne01, Q->ne[2], plan.n_kv_heads, gqa_ratio, ntiles_dst);
        init = false;
    }

    const dim3 finalize_grid((uint32_t) n_rows, 1, 1);
    ggml_cuda_kernel_launch_params finalize_params(finalize_grid, merge_block, 0, stream);
    ggml_cuda_kernel_launch(finalize_kernel, finalize_params,
        (float *) dst->data, acc_meta.get(), dst_meta, n_rows);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int DKQ, int DV, int ncols1, int ncols2>
static bool ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan,
        size_t nbytes_shared_total,
        bool use_logit_softcap) {
    if constexpr (DKQ != DV || (DKQ != 128 && DKQ != 256 && DKQ != 512)) {
        GGML_UNUSED_VARS(ctx, dst, plan, nbytes_shared_total, use_logit_softcap);
        return false;
    } else {
        if (use_logit_softcap) {
            return ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case_impl<DKQ, DV, ncols1, ncols2, true>(
                ctx, dst, plan, nbytes_shared_total);
        }
        return ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case_impl<DKQ, DV, ncols1, ncols2, false>(
            ctx, dst, plan, nbytes_shared_total);
    }
}

template <int DKQ, int DV, int ncols1, int ncols2>
static size_t ggml_cuda_fattn_kvarn_mma_shared_bytes(
        int cc,
        int warp_size_host,
        bool has_original_domain) {
    constexpr int ncols = ncols1 * ncols2;
    const int nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg      = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int nwarps         = nthreads / warp_size_host;
    const int cols_per_warp  = std::min(ncols, get_cols_per_warp(cc));

    const size_t nbytes_shared_KV = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q = ncols * (DKQ/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_mask = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t nbytes_shared_kvarn_rotated =
        6 * std::max(DKQ, DV) * sizeof(half) +
        2 * (std::max(DKQ, DV) / GGML_CUDA_FATTN_KVARN_DIM) * sizeof(int);
    const size_t nbytes_shared_kvarn_original =
        3 * GGML_CUDA_FATTN_KVARN_DIM * sizeof(half) +
        2 * nwarps * GGML_CUDA_FATTN_KVARN_DIM * sizeof(float);
    const size_t nbytes_shared_kvarn = has_original_domain ?
        nbytes_shared_kvarn_original : nbytes_shared_kvarn_rotated;
    const size_t nbytes_shared_KV_mask_kvarn = nbytes_shared_KV + nbytes_shared_mask + nbytes_shared_kvarn;
    return std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q, nbytes_shared_KV_mask_kvarn) :
                 nbytes_shared_Q + nbytes_shared_KV_mask_kvarn);
}

#if !defined(GGML_USE_MUSA)
template <int DKQ, int DV, int ncols1, int ncols2>
bool ggml_cuda_fattn_kvarn_wide_mma_supported(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst) {
    const int device = ctx.device;
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const bool k_original_domain = ggml_cuda_fattn_kvarn_k_original_domain(dst);
    const bool v_original_domain = ggml_cuda_fattn_kvarn_v_original_domain(dst);
    const int cache_key = 4 * (int) k_original_domain + 2 * (int) v_original_domain + (logit_softcap != 0.0f);
    static std::atomic<int> cached_active_blocks[GGML_CUDA_MAX_DEVICES][8] = {};
    const int cached = cached_active_blocks[device][cache_key].load(std::memory_order_relaxed);
    if (cached != 0) {
        return cached > 0;
    }

    const auto & device_info = ggml_cuda_info().devices[device];
    const size_t nbytes_shared_total = ggml_cuda_fattn_kvarn_mma_shared_bytes<DKQ, DV, ncols1, ncols2>(
        device_info.cc, device_info.warp_size, k_original_domain || v_original_domain);
    if (nbytes_shared_total > device_info.smpbo) {
        cached_active_blocks[device][cache_key].store(-1, std::memory_order_relaxed);
        return false;
    }

    fattn_kernel_t fattn_kernel = logit_softcap == 0.0f ?
        ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, false>(
            k_original_domain, v_original_domain) :
        ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, true>(
            k_original_domain, v_original_domain);
    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<ggml_cuda_fattn_kernel_attr_ptr_t>(fattn_kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        nbytes_shared_total));

    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1 * ncols2, device_info.cc);
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        reinterpret_cast<ggml_cuda_fattn_kernel_attr_ptr_t>(fattn_kernel),
        nthreads,
        nbytes_shared_total));
    cached_active_blocks[device][cache_key].store(active_blocks > 0 ? active_blocks : -1, std::memory_order_relaxed);
    return active_blocks > 0;
}
#endif

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_kvarn_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_fattn_kvarn_plan plan;
    GGML_ASSERT(ggml_cuda_fattn_kvarn_view_supported(ctx.device, dst, &plan));

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps = nthreads / warp_size_host;
    const bool k_original_domain = ggml_cuda_fattn_kvarn_k_original_domain(dst);
    const bool v_original_domain = ggml_cuda_fattn_kvarn_v_original_domain(dst);
    const bool has_original_domain = k_original_domain || v_original_domain;

    const size_t nbytes_shared_KV = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q = ncols * (DKQ/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_mask = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t nbytes_shared_total = ggml_cuda_fattn_kvarn_mma_shared_bytes<DKQ, DV, ncols1, ncols2>(
        cc, warp_size_host, has_original_domain);
    const int nstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc);
    const size_t nbytes_shared_KV_f16_1stage = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_f16_2stage = nbatch_fa * (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_f16 = nstages <= 1 ? nbytes_shared_KV_f16_1stage : nbytes_shared_KV_f16_2stage;
    const size_t nbytes_shared_total_f16 = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q, nbytes_shared_KV_f16 + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV_f16 + nbytes_shared_mask);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const size_t n_desc = (size_t) plan.n_stream * plan.n_kv_heads;
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> k_desc(pool, n_desc);
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> v_desc(pool, n_desc);
    ggml_cuda_fattn_kvarn_init_descs(plan, k_desc.get(), v_desc.get(),
            k_original_domain ? 1 : 0, v_original_domain ? 1 : 0, stream);

    ggml_tensor K_desc = *dst->src[1];
    ggml_tensor V_desc = *dst->src[2];
    K_desc.data = k_desc.get();
    V_desc.data = v_desc.get();
    K_desc.type = GGML_TYPE_F16;
    V_desc.type = GGML_TYPE_F16;
    K_desc.view_src = nullptr;
    V_desc.view_src = nullptr;
    K_desc.view_offs = 0;
    V_desc.view_offs = 0;
    K_desc.nb[0] = sizeof(half);
    V_desc.nb[0] = sizeof(half);
    K_desc.nb[1] = 0;
    V_desc.nb[1] = 0;
    K_desc.nb[2] = sizeof(ggml_cuda_fattn_kvarn_desc);
    V_desc.nb[2] = sizeof(ggml_cuda_fattn_kvarn_desc);
    K_desc.nb[3] = sizeof(ggml_cuda_fattn_kvarn_desc) * plan.n_kv_heads;
    V_desc.nb[3] = sizeof(ggml_cuda_fattn_kvarn_desc) * plan.n_kv_heads;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif
    fattn_kernel_t fattn_kernel;
    fattn_kernel_t fattn_kernel_no_softcap = ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, false>(
        k_original_domain, v_original_domain);
    fattn_kernel_t fattn_kernel_softcap = ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, true>(
        k_original_domain, v_original_domain);
    if (logit_softcap == 0.0f) {
        fattn_kernel = fattn_kernel_no_softcap;
#if !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
    } else {
        fattn_kernel = fattn_kernel_softcap;
#if !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
    }

    if (ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case<DKQ, DV, ncols1, ncols2>(
            ctx, dst, plan, nbytes_shared_total_f16, logit_softcap != 0.0f)) {
        return;
    }

    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];
    dst->src[1] = &K_desc;
    dst->src[2] = &V_desc;
    // need_f16_K=false, need_f16_V=false: KVarN K/V stay descriptor-backed.
    // Mixed prefill reconstructs original-domain V in the native loader.
    // use_sparse=false: record loads are not qualified for sparse gathers.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, false, warp_size_host);
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;
}


} // namespace l0xre_qk16
#undef GGML_CUDA_FATTN_MMA_CONFIG_CASE

static bool l0xre_qk16_dispatch(ggml_backend_cuda_context &ctx,ggml_tensor *dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const char *env=std::getenv("L0XRE_KVARN_QK_FP16_ACC");
    if(!env || (std::strcmp(env,"1")&&std::strcmp(env,"2")))return false;
    const ggml_tensor *q=dst->src[0];
    ggml_cuda_fattn_kvarn_plan plan;
    if(ggml_cuda_info().devices[ctx.device].cc!=860 || q->ne[0]!=256 ||
       q->ne[1]<128 || q->ne[1]>1024 || dst->ne[0]!=256 ||
       !ggml_cuda_fattn_kvarn_view_supported(ctx.device,dst,&plan) ||
       plan.k.bits!=4 || plan.v.bits!=4 || q->ne[2]/dst->src[1]->ne[2]!=6 ||
       dst->src[4]!=nullptr ||
       ggml_cuda_fattn_kvarn_domain(dst)!=GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V)
        return false;
    float softcap;std::memcpy(&softcap,(const float*)dst->op_params+2,sizeof(float));
    if(softcap!=0.0f)return false;
    const char *window=std::getenv("GGML_KVARN_WINDOW");
    if(window && std::atoi(window)==0)return false;
    const int mode=std::atoi(env);
    if(mode==1)l0xre_qk16::ggml_cuda_flash_attn_ext_mma_kvarn_case<256,256,8,8>(ctx,dst);
    else l0xre_qk16::ggml_cuda_flash_attn_ext_mma_kvarn_case<256,256,32,2>(ctx,dst);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))
        std::fprintf(stderr,"L0XRE_QK_FP16_ACC M=%lld tile=%dx%d softmax=F32 cc=860 bits=4/4\n",
                     (long long)q->ne[1],mode==1?8:32,mode==1?8:2);
    return true;
#else
    GGML_UNUSED(ctx);GGML_UNUSED(dst);return false;
#endif
}

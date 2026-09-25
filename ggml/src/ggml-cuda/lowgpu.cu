#include "lowgpu.cuh"

// LowGPU v1 3-bit vocab kernels (see source/lowgpu/format.py for the packing):
//   * 3 bits/code, 8 levels, group size 128 along the hidden dim;
//   * x_hat = (q - zp) * scale -> fp16;
//   * 8 codes packed per 3 bytes, little-endian bit order.

#define LOWGPU_GROUP 128
#define LOWGPU_MROW_MAX 8   // widest MTP verification batch this variant serves
#define LOWGPU_ROWS_R 4      // vocabulary rows per CTA in the rows variant
#define LOWGPU_ROWS_MAXT 4   // activation rows it serves (draft M=1 and n3 M=4)
#define LOWGPU_RT_MAXT 8     // widest token count the rt (rows x tokens) head variant serves

// decode one vocab row into fp16 [n_embd]
static __global__ void lowgpu_dequant_row_kernel(
        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        half          * __restrict__ out,
        const int KB, const int G, const int ntri) {
    const int tid = threadIdx.x;
    const int r   = blockIdx.x;

    const uint8_t * cr = codes + (size_t) r*KB;
    const half    * sr = scales + (size_t) r*G;
    const uint8_t * zr = zps    + (size_t) r*G;
    half          * orow = out + (size_t) r*(KB*8/3);

    for (int t = tid; t < ntri; t += blockDim.x) {
        const uint8_t b0 = cr[3*t + 0];
        const uint8_t b1 = cr[3*t + 1];
        const uint8_t b2 = cr[3*t + 2];

        const uint8_t c0 = b0 & 7;
        const uint8_t c1 = (b0 >> 3) & 7;
        const uint8_t c2 = ((b0 >> 6) | (b1 << 2)) & 7;
        const uint8_t c3 = (b1 >> 1) & 7;
        const uint8_t c4 = (b1 >> 4) & 7;
        const uint8_t c5 = ((b1 >> 7) | (b2 << 1)) & 7;
        const uint8_t c6 = (b2 >> 2) & 7;
        const uint8_t c7 = (b2 >> 5) & 7;

        const int g = t*8/LOWGPU_GROUP;
        const float s = __half2float(sr[g]);
        const float z = (float) zr[g];

        orow[8*t + 0] = __float2half(((float) c0 - z) * s);
        orow[8*t + 1] = __float2half(((float) c1 - z) * s);
        orow[8*t + 2] = __float2half(((float) c2 - z) * s);
        orow[8*t + 3] = __float2half(((float) c3 - z) * s);
        orow[8*t + 4] = __float2half(((float) c4 - z) * s);
        orow[8*t + 5] = __float2half(((float) c5 - z) * s);
        orow[8*t + 6] = __float2half(((float) c6 - z) * s);
        orow[8*t + 7] = __float2half(((float) c7 - z) * s);
    }
}

// decode the row selected by ids[token] directly into dst[token]
static __global__ void lowgpu_get_rows_kernel(
        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        const int32_t * __restrict__ ids,
        float         * __restrict__ out,
        const int KB, const int G, const int ntri, const int n_embd, const int V) {
    const int tid = threadIdx.x;
    const int tok = blockIdx.x;

    const int r = ids[tok];
    if (r < 0 || r >= V) {
        return;
    }

    const uint8_t * cr = codes + (size_t) r*KB;
    const half    * sr = scales + (size_t) r*G;
    const uint8_t * zr = zps    + (size_t) r*G;
    float         * orow = out + (size_t) tok*n_embd;

    for (int t = tid; t < ntri; t += blockDim.x) {
        const uint8_t b0 = cr[3*t + 0];
        const uint8_t b1 = cr[3*t + 1];
        const uint8_t b2 = cr[3*t + 2];

        const uint8_t c0 = b0 & 7;
        const uint8_t c1 = (b0 >> 3) & 7;
        const uint8_t c2 = ((b0 >> 6) | (b1 << 2)) & 7;
        const uint8_t c3 = (b1 >> 1) & 7;
        const uint8_t c4 = (b1 >> 4) & 7;
        const uint8_t c5 = ((b1 >> 7) | (b2 << 1)) & 7;
        const uint8_t c6 = (b2 >> 2) & 7;
        const uint8_t c7 = (b2 >> 5) & 7;

        const int g = t*8/LOWGPU_GROUP;
        const float s = __half2float(sr[g]);
        const float z = (float) zr[g];

        orow[8*t + 0] = __half2float(__float2half(((float) c0 - z) * s));
        orow[8*t + 1] = __half2float(__float2half(((float) c1 - z) * s));
        orow[8*t + 2] = __half2float(__float2half(((float) c2 - z) * s));
        orow[8*t + 3] = __half2float(__float2half(((float) c3 - z) * s));
        orow[8*t + 4] = __half2float(__float2half(((float) c4 - z) * s));
        orow[8*t + 5] = __half2float(__float2half(((float) c5 - z) * s));
        orow[8*t + 6] = __half2float(__float2half(((float) c6 - z) * s));
        orow[8*t + 7] = __half2float(__float2half(((float) c7 - z) * s));
    }
}

void ggml_cuda_op_lowgpu_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code  = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * zp    = dst->src[2];
    const ggml_tensor * ids   = dst->src[3];

    const int KB = code->ne[0];
    const int V  = code->ne[1];
    const int G  = scale->ne[0];
    const int n_ids = ids->ne[0];
    const int ntri  = KB/3;
    const int n_embd = KB*8/3;

    dim3 grid(n_ids, 1, 1);
    lowgpu_get_rows_kernel<<<grid, 256, 0, ctx.stream()>>>(
            (const uint8_t *) code->data,
            (const half *)    scale->data,
            (const uint8_t *) zp->data,
            (const int32_t *) ids->data,
            (float *) dst->data,
            KB, G, ntri, n_embd, V);
}

static __global__ void lowgpu_f32_to_f16_kernel(
        const float * __restrict__ in, half * __restrict__ out, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(in[i]);
    }
}

// W2 vocabulary format: one signed INT8 row and one F16 scale per output.
// Match the established load-time control exactly at its storage boundaries:
// both the scaled weight and activation are rounded through F16, then the dot
// accumulates in FP32.  Keeping the source row packed halves head bandwidth.
static __global__ void i8_row_scaled_gemv_legacy_kernel(
        const int8_t * __restrict__ weight,
        const half   * __restrict__ scales,
        const float  * __restrict__ x,
        float        * __restrict__ dst,
        const int n_embd, const int V) {
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= V) {
        return;
    }

    const int8_t * wr = weight + (size_t) row*n_embd;
    const float   * xv = x      + (size_t) tok*n_embd;
    const float scale = __half2float(scales[row]);

    float sum = 0.0f;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        const float a = __half2float(__float2half(xv[i]));
        const float w = __half2float(__float2half((float) wr[i]*scale));
        sum += w*a;
    }

    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst[(size_t) tok*V + row] = partial[0];
    }
}

// Four independent output rows per CTA, one warp per row.  The legacy kernel
// assigned a 256-thread CTA to every row, which launches 248k CTAs for the W2
// vocabulary and pays eight block-wide barriers per output.  SGLang's proven
// escha_gemv geometry uses 128 threads and V/4 CTAs for the same matrix.  Keep
// the established arithmetic contract here: both x and scaled weights round
// through F16 and each lane accumulates in FP32.
static __global__ void i8_row_scaled_gemv_warp4_kernel(
        const int8_t * __restrict__ weight,
        const half   * __restrict__ scales,
        const float  * __restrict__ x,
        float        * __restrict__ dst,
        const int n_embd, const int V) {
    constexpr int rows_per_block = 4;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x*rows_per_block + warp;
    const int tok = blockIdx.y;
    if (row >= V) {
        return;
    }

    const int8_t * wr = weight + (size_t) row*n_embd;
    const float   * xv = x      + (size_t) tok*n_embd;
    const float scale = __half2float(scales[row]);

    float sums[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    int i = lane;
    for (; i + 7*WARP_SIZE < n_embd; i += 8*WARP_SIZE) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int k = i + j*WARP_SIZE;
            const float a = __half2float(__float2half(xv[k]));
            const float w = __half2float(__float2half((float) wr[k]*scale));
            sums[j] += w*a;
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        sum += sums[j];
    }
    for (; i < n_embd; i += WARP_SIZE) {
        const float a = __half2float(__float2half(xv[i]));
        const float w = __half2float(__float2half((float) wr[i]*scale));
        sum += w*a;
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        dst[(size_t) tok*V + row] = sum;
    }
}

// Decode one packed vocabulary row and dot it with one hidden vector in the
// same block.  Materializing all V*hidden weights for a single decode token
// costs 2.5 GiB of transient writes before GEMM can start; this path reads the
// compact representation once and writes only one FP32 logit per row.
// VEC selects 16-byte activation loads.  See the dispatch site for the alignment contract.
template <bool VEC>
static __global__ void lowgpu_packed_gemv_kernel(
        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int KB, const int G, const int ntri, const int n_embd, const int V) {
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= V) {
        return;
    }

    const uint8_t * cr = codes  + (size_t) row*KB;
    const half    * sr = scales + (size_t) row*G;
    const uint8_t * zr = zps    + (size_t) row*G;
    const float   * xv = x + (size_t) tok*n_embd;

    float sum = 0.0f;
    for (int t = tid; t < ntri; t += blockDim.x) {
        const uint8_t b0 = cr[3*t + 0];
        const uint8_t b1 = cr[3*t + 1];
        const uint8_t b2 = cr[3*t + 2];
        const uint8_t q[8] = {
            (uint8_t) (b0 & 7), (uint8_t) ((b0 >> 3) & 7),
            (uint8_t) (((b0 >> 6) | (b1 << 2)) & 7), (uint8_t) ((b1 >> 1) & 7),
            (uint8_t) ((b1 >> 4) & 7), (uint8_t) (((b1 >> 7) | (b2 << 1)) & 7),
            (uint8_t) ((b2 >> 2) & 7), (uint8_t) ((b2 >> 5) & 7),
        };
        const float scale_v = __half2float(sr[t*8/LOWGPU_GROUP]);
        const float zp_v = zr[t*8/LOWGPU_GROUP];
        float xa[8];
        if (VEC) {
            const float4 * xr = reinterpret_cast<const float4 *>(xv + 8*t);
            const float4 p0 = xr[0];
            const float4 p1 = xr[1];
            xa[0] = p0.x; xa[1] = p0.y; xa[2] = p0.z; xa[3] = p0.w;
            xa[4] = p1.x; xa[5] = p1.y; xa[6] = p1.z; xa[7] = p1.w;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // Match the canonical LowGPU F16 weight boundary and the existing
            // GEMM path's FP32 -> FP16 activation boundary, so decode and
            // prefill logits share the same numeric contract.
            const float a = __half2float(__float2half(VEC ? xa[i] : xv[8*t + i]));
            const float w = __half2float(__float2half(((float) q[i] - zp_v)*scale_v));
            sum += w*a;
        }
    }

    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst[(size_t) tok*V + row] = partial[0];
    }
}

// MTP verification width: one CTA per vocabulary row, with the activation rows looped inside the
// CTA so a row's packed codes, scales and zero points are read ONCE and reused across the whole
// verification batch. The single-row kernel above launches one CTA per (row, token) pair, so at
// n_tokens=8 it re-reads every row eight times - which is why the coded head used to fall back to
// dequantizing the entire vocabulary into fp16 stripes for n_tokens>4, measured at ~7.9 ms per
// verification cycle (MTP-0029 profile: lowgpu_dequant_row_kernel 369 ms over 47 cycles).
// Numerics are the single-row kernel's contract: fp16 rounding of both weight and activation,
// fp32 accumulation, and the same 256-thread binary reduction tree per token.
static __global__ void lowgpu_packed_gemv_mrow_kernel(
        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        const half    * __restrict__ x,
        float         * __restrict__ dst,
        const int KB, const int G, const int ntri, const int n_embd, const int V,
        const int n_tokens) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= V || n_tokens > LOWGPU_MROW_MAX) {
        return;
    }

    const uint8_t * cr = codes  + (size_t) row*KB;
    const half    * sr = scales + (size_t) row*G;
    const uint8_t * zr = zps    + (size_t) row*G;

    float acc[LOWGPU_MROW_MAX];
#pragma unroll
    for (int tok = 0; tok < LOWGPU_MROW_MAX; ++tok) {
        acc[tok] = 0.0f;
    }

    for (int t = tid; t < ntri; t += blockDim.x) {
        const uint8_t b0 = cr[3*t + 0];
        const uint8_t b1 = cr[3*t + 1];
        const uint8_t b2 = cr[3*t + 2];
        const uint8_t q[8] = {
            (uint8_t) (b0 & 7), (uint8_t) ((b0 >> 3) & 7),
            (uint8_t) (((b0 >> 6) | (b1 << 2)) & 7), (uint8_t) ((b1 >> 1) & 7),
            (uint8_t) ((b1 >> 4) & 7), (uint8_t) (((b1 >> 7) | (b2 << 1)) & 7),
            (uint8_t) ((b2 >> 2) & 7), (uint8_t) ((b2 >> 5) & 7),
        };
        const float scale_v = __half2float(sr[t*8/LOWGPU_GROUP]);
        const float zp_v    = zr[t*8/LOWGPU_GROUP];
        float w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[i] = __half2float(__float2half(((float) q[i] - zp_v)*scale_v));
        }
        for (int tok = 0; tok < n_tokens; ++tok) {
            // x arrives already rounded to fp16 by the caller: one conversion per element for the
            // whole batch instead of one per row-block per token, and the load is vectorised. This is
            // the change that attacks the per-row activation traffic which dominated the first version.
            const half * xv = x + (size_t) tok*n_embd + 8*t;
            const float2 h01 = __half22float2(reinterpret_cast<const half2 *>(xv)[0]);
            const float2 h23 = __half22float2(reinterpret_cast<const half2 *>(xv)[1]);
            const float2 h45 = __half22float2(reinterpret_cast<const half2 *>(xv)[2]);
            const float2 h67 = __half22float2(reinterpret_cast<const half2 *>(xv)[3]);
            const float xs[8] = { h01.x, h01.y, h23.x, h23.y, h45.x, h45.y, h67.x, h67.y };
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s += w[i]*xs[i];
            }
            acc[tok] += s;
        }
    }

    __shared__ float partial[256];
#pragma unroll 1
    for (int tok = 0; tok < n_tokens; ++tok) {
        partial[tid] = acc[tok];
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            dst[(size_t) tok*V + row] = partial[0];
        }
        __syncthreads();
    }
}

// Rows variant for the narrow end (n_tokens 1..4), which is where the DRAFT lives: the MTP head
// calls this op once per draft step, so seven times per n7 cycle against once for the verification
// batch. The single-row kernel launches one CTA per (row, token) - 248,320 trivial CTAs at M=1,
// each re-reading the activation vector - while this one puts LOWGPU_ROWS_R vocabulary rows in a
// CTA and loops the activation rows inside the t-loop, so codes and activations are each read once
// per CTA. Arithmetic per output is deliberately identical to the single-row kernel (same per-thread
// t-stride accumulation, same fp16 rounding of weight and activation, same 256-thread binary
// reduction), so results are bit-identical and an A/B measures only speed.

// Multi-row x multi-token ("rt") variant for the MTP verification batch.
//
// NCU on the M=1 and M=4 head kernels (ncuhead/head-m1|m4.ncu-rep, 2026-09-18) shows the real limiter:
// at M=1 the head is DRAM-bound (DRAM 60.8% SOL, L2 hit rate 6.7%), but at M=4 DRAM is idle (19.3%)
// and L2 is idle (8.3%) while SM (79.9%) and the LSU pipe (75.0%) are saturated: it is an
// instruction/L1-traffic bound kernel that pays 4 activation loads + 8 half->float converts + 8 MACs
// per weight triplet PER TOKEN. The fix is therefore not more bandwidth, it is doing the
// activation-side work once per CTA and reusing it across R vocabulary rows.
//
// R and T are compile-time here. The first attempt used fixed MAXT arrays with a runtime n_tokens;
// the compiler then spilled acc[][] and the activation window to local memory, and the spill traffic
// cost more than the reuse saved (measured: rt2 +1.35 ms, rt4 +4.19 ms over mrow at M=4).
//
// Arithmetic per output is unchanged: same 3-byte code decode, same __float2half rounding of the
// reconstructed weight, same ascending 8-term dot, same accumulation over t with thread stride
// blockDim.x, same 256-thread binary-tree reduction over the same partial[tid] order.
template <int R, int T>
static __global__ void lowgpu_packed_gemv_rt_kernel(
        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        const half    * __restrict__ x,
        float         * __restrict__ dst,
        const int KB, const int G, const int ntri, const int n_embd, const int V) {
    const int row0 = blockIdx.x * R;
    const int tid  = threadIdx.x;

    float acc[R][T];
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
#pragma unroll
        for (int tok = 0; tok < T; ++tok) {
            acc[rr][tok] = 0.0f;
        }
    }

    for (int t = tid; t < ntri; t += blockDim.x) {
        float xw[T][8];
#pragma unroll
        for (int tok = 0; tok < T; ++tok) {
            const half * xv = x + (size_t) tok*n_embd + 8*t;
            // M-VERIFY-0015: one 16-byte load instead of four 8-byte half2 loads. The head is
            // LSU-issue bound (NCU: L1TEX ~80% of peak while L2 sits at 8%), and this removes 3 of
            // the ~12 activation-path load instructions per (t, token) with identical values.
            // Alignment: 2*(tok*n_embd + 8*t) bytes, with n_embd = 5120 these are 16-byte multiples
            // provided the f16 staging buffer is 16-byte aligned, which the pool guarantees.
            const bool vec16 = (reinterpret_cast<uintptr_t>(xv) & 15u) == 0;
            half2 h01, h23, h45, h67;
            if (vec16) {
                const uint4 raw = *reinterpret_cast<const uint4 *>(xv);
                h01 = *reinterpret_cast<const half2 *>(&raw.x);
                h23 = *reinterpret_cast<const half2 *>(&raw.y);
                h45 = *reinterpret_cast<const half2 *>(&raw.z);
                h67 = *reinterpret_cast<const half2 *>(&raw.w);
            } else {
                h01 = reinterpret_cast<const half2 *>(xv)[0];
                h23 = reinterpret_cast<const half2 *>(xv)[1];
                h45 = reinterpret_cast<const half2 *>(xv)[2];
                h67 = reinterpret_cast<const half2 *>(xv)[3];
            }
            const float2 f01 = __half22float2(h01);
            const float2 f23 = __half22float2(h23);
            const float2 f45 = __half22float2(h45);
            const float2 f67 = __half22float2(h67);
            xw[tok][0] = f01.x; xw[tok][1] = f01.y;
            xw[tok][2] = f23.x; xw[tok][3] = f23.y;
            xw[tok][4] = f45.x; xw[tok][5] = f45.y;
            xw[tok][6] = f67.x; xw[tok][7] = f67.y;
        }
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            if (row0 + rr < V) {
                const uint8_t * cr = codes  + (size_t) (row0 + rr)*KB;
                const half    * sr = scales + (size_t) (row0 + rr)*G;
                const uint8_t * zr = zps    + (size_t) (row0 + rr)*G;
                const uint8_t b0 = cr[3*t + 0];
                const uint8_t b1 = cr[3*t + 1];
                const uint8_t b2 = cr[3*t + 2];
                const uint8_t q[8] = {
                    (uint8_t) (b0 & 7), (uint8_t) ((b0 >> 3) & 7),
                    (uint8_t) (((b0 >> 6) | (b1 << 2)) & 7), (uint8_t) ((b1 >> 1) & 7),
                    (uint8_t) ((b1 >> 4) & 7), (uint8_t) (((b1 >> 7) | (b2 << 1)) & 7),
                    (uint8_t) ((b2 >> 2) & 7), (uint8_t) ((b2 >> 5) & 7),
                };
                const float scale_v = __half2float(sr[t*8/LOWGPU_GROUP]);
                const float zp_v    = zr[t*8/LOWGPU_GROUP];
                float w[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    w[i] = __half2float(__float2half(((float) q[i] - zp_v)*scale_v));
                }
#pragma unroll
                for (int tok = 0; tok < T; ++tok) {
                    float s = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        s += w[i]*xw[tok][i];
                    }
                    acc[rr][tok] += s;
                }
            }
        }
    }

    __shared__ float partial[256][R*T];
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
#pragma unroll
        for (int tok = 0; tok < T; ++tok) {
            partial[tid][rr*T + tok] = acc[rr][tok];
        }
    }
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int i = 0; i < R*T; ++i) {
                partial[tid][i] += partial[tid + stride][i];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            if (row0 + rr >= V) {
                break;
            }
#pragma unroll
            for (int tok = 0; tok < T; ++tok) {
                dst[(size_t) tok*V + (row0 + rr)] = partial[0][rr*T + tok];
            }
        }
    }
}

static __global__ void lowgpu_packed_gemv_rows_kernel(

        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int KB, const int G, const int ntri, const int n_embd, const int V,
        const int n_tokens) {
    const int row0 = blockIdx.x * LOWGPU_ROWS_R;
    const int tid  = threadIdx.x;
    if (row0 >= V || n_tokens > LOWGPU_ROWS_MAXT) {
        return;
    }

    float acc[LOWGPU_ROWS_R][LOWGPU_ROWS_MAXT];
#pragma unroll
    for (int rr = 0; rr < LOWGPU_ROWS_R; ++rr) {
#pragma unroll
        for (int tok = 0; tok < LOWGPU_ROWS_MAXT; ++tok) {
            acc[rr][tok] = 0.0f;
        }
    }

    for (int t = tid; t < ntri; t += blockDim.x) {
        // activations for every token, loaded once and reused across the row group
        float xa[LOWGPU_ROWS_MAXT][8];
        for (int tok = 0; tok < n_tokens; ++tok) {
            const float * xv = x + (size_t) tok*n_embd + 8*t;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                xa[tok][i] = __half2float(__float2half(xv[i]));
            }
        }
        for (int rr = 0; rr < LOWGPU_ROWS_R; ++rr) {
            const int row = row0 + rr;
            if (row >= V) {
                break;
            }
            const uint8_t * cr = codes  + (size_t) row*KB;
            const half    * sr = scales + (size_t) row*G;
            const uint8_t * zr = zps    + (size_t) row*G;
            const uint8_t b0 = cr[3*t + 0];
            const uint8_t b1 = cr[3*t + 1];
            const uint8_t b2 = cr[3*t + 2];
            const uint8_t q[8] = {
                (uint8_t) (b0 & 7), (uint8_t) ((b0 >> 3) & 7),
                (uint8_t) (((b0 >> 6) | (b1 << 2)) & 7), (uint8_t) ((b1 >> 1) & 7),
                (uint8_t) ((b1 >> 4) & 7), (uint8_t) (((b1 >> 7) | (b2 << 1)) & 7),
                (uint8_t) ((b2 >> 2) & 7), (uint8_t) ((b2 >> 5) & 7),
            };
            const float scale_v = __half2float(sr[t*8/LOWGPU_GROUP]);
            const float zp_v    = zr[t*8/LOWGPU_GROUP];
            float w[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                w[i] = __half2float(__float2half(((float) q[i] - zp_v)*scale_v));
            }
            for (int tok = 0; tok < n_tokens; ++tok) {
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    s += w[i]*xa[tok][i];
                }
                acc[rr][tok] += s;
            }
        }
    }

    // one reduction pass for all R x T outputs: 8 syncs per CTA instead of R*T*8
    __shared__ float partial[256][LOWGPU_ROWS_R*LOWGPU_ROWS_MAXT];
#pragma unroll
    for (int rr = 0; rr < LOWGPU_ROWS_R; ++rr) {
#pragma unroll
        for (int tok = 0; tok < LOWGPU_ROWS_MAXT; ++tok) {
            partial[tid][rr*LOWGPU_ROWS_MAXT + tok] = acc[rr][tok];
        }
    }
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int i = 0; i < LOWGPU_ROWS_R*LOWGPU_ROWS_MAXT; ++i) {
                partial[tid][i] += partial[tid + stride][i];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int rr = 0; rr < LOWGPU_ROWS_R; ++rr) {
            const int row = row0 + rr;
            if (row >= V) {
                break;
            }
            for (int tok = 0; tok < n_tokens; ++tok) {
                dst[(size_t) tok*V + row] = partial[0][rr*LOWGPU_ROWS_MAXT + tok];
            }
        }
    }
}

// Four rows per CTA. Each lane emulates eight legacy threads, preserving
// their accumulation streams and the exact 256-thread reduction tree.
// Packed codes/scales/zero points are consumed directly from the model.
static __global__ void lowgpu_packed_gemv_warp4_exact_kernel(
        const uint8_t * __restrict__ codes,
        const half    * __restrict__ scales,
        const uint8_t * __restrict__ zps,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int KB, const int G, const int ntri, const int n_embd, const int V) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x*4 + threadIdx.x/WARP_SIZE;
    const int tok = blockIdx.y;
    if (row >= V) {
        return;
    }
    const uint8_t * cr = codes + (size_t) row*KB;
    const half * sr = scales + (size_t) row*G;
    const uint8_t * zr = zps + (size_t) row*G;
    const float * xv = x + (size_t) tok*n_embd;
    float sums[8] = {};
    for (int base = lane; base < ntri; base += 256) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int t = base + j*WARP_SIZE;
            if (t < ntri) {
                const uint8_t b0 = cr[3*t];
                const uint8_t b1 = cr[3*t + 1];
                const uint8_t b2 = cr[3*t + 2];
                const uint8_t q[8] = {
                    (uint8_t) (b0 & 7), (uint8_t) ((b0 >> 3) & 7),
                    (uint8_t) (((b0 >> 6) | (b1 << 2)) & 7), (uint8_t) ((b1 >> 1) & 7),
                    (uint8_t) ((b1 >> 4) & 7), (uint8_t) (((b1 >> 7) | (b2 << 1)) & 7),
                    (uint8_t) ((b2 >> 2) & 7), (uint8_t) ((b2 >> 5) & 7),
                };
                const float scale_v = __half2float(sr[t*8/LOWGPU_GROUP]);
                const float zp_v = zr[t*8/LOWGPU_GROUP];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float a = __half2float(__float2half(xv[8*t + i]));
                    const float w = __half2float(__float2half(((float) q[i] - zp_v)*scale_v));
                    sums[j] += w*a;
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sums[j] += sums[j + 4];
    }
    sums[0] += sums[2];
    sums[1] += sums[3];
    float sum = sums[0] + sums[1];
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        dst[(size_t) tok*V + row] = sum;
    }
}

void ggml_cuda_op_lowgpu_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code  = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * zp    = dst->src[2];
    const ggml_tensor * x     = dst->src[3];

    // A null zero-point marks the Escha W2 row-scaled INT8 format.  This is a
    // complete direct path, not a materialization fallback.
    if (zp == nullptr) {
        GGML_ASSERT(code->type == GGML_TYPE_I8 && scale->type == GGML_TYPE_F16);
        GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        const int n_embd = code->ne[0];
        const int V = code->ne[1];
        const int n_tokens = x->ne[1]*x->ne[2];
        GGML_ASSERT(scale->ne[0] == V && x->ne[0] == n_embd);

        // The retained W2 INT8 head is a decode GEMV: it gives each generated
        // token four vocabulary rows per CTA.  It has no batched prefill GEMM
        // implementation.  Launching it for p2048 would create V/4 CTAs for
        // every token (about 127M CTAs for this model), and can corrupt a CUDA
        // graph capture/WSL dxg session instead of producing a useful result.
        // Keep the selector strictly decode-only until a real INT8 prefill GEMM
        // exists; the default F16 transformed head remains the prefill route.
        if (n_tokens != 1) {
            GGML_ABORT("escha W2 INT8 head is decode-only (n_tokens=%d); unset ESCHA_W2_I8_HEAD for prefill", n_tokens);
        }

        const char * legacy_env = std::getenv("ESCHA_W2_I8_HEAD_LEGACY_CTA");
        const bool use_legacy = legacy_env != nullptr && strcmp(legacy_env, "1") == 0;
        if (!use_legacy) {
            constexpr int rows_per_block = 4;
            i8_row_scaled_gemv_warp4_kernel<<<dim3((V + rows_per_block - 1)/rows_per_block, n_tokens), 128, 0, ctx.stream()>>>(
                    (const int8_t *) code->data,
                    (const half *) scale->data,
                    (const float *) x->data,
                    (float *) dst->data,
                    n_embd, V);
        } else {
            i8_row_scaled_gemv_legacy_kernel<<<dim3(V, n_tokens), 256, 0, ctx.stream()>>>(
                    (const int8_t *) code->data,
                    (const half *) scale->data,
                    (const float *) x->data,
                    (float *) dst->data,
                    n_embd, V);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int KB = code->ne[0];
    const int V  = code->ne[1];
    const int G  = scale->ne[0];
    const int n_embd = KB*8/3;
    const int n_tokens = x->ne[1]*x->ne[2];
    const int ntri = KB/3;

    // Autoregressive decode dominates serving.  Fusing unpack + dot avoids a
    // full dequantized vocabulary allocation and its bandwidth cost.  Prefill
    // uses GEMM, but it must not expand the entire vocabulary: that alone is
    // 2.37 GiB for this 248k x 5120 model and defeats its 12 GiB deployment.
    // ESCHA_E3_HEAD_ROWS=1 selects the rows-per-CTA kernel for n_tokens 1..4 (the draft path and
    // n3). Bit-identical per output to the single-row kernel by construction, so it is a pure
    // speed knob; see research/patches/mtp0037-rows/ for the A/B.
    const char * rows_env = std::getenv("ESCHA_E3_HEAD_ROWS");
    if (rows_env != nullptr && strcmp(rows_env, "1") == 0 && n_tokens <= LOWGPU_ROWS_MAXT) {
        lowgpu_packed_gemv_rows_kernel<<<dim3((V + LOWGPU_ROWS_R - 1)/LOWGPU_ROWS_R), 256, 0, ctx.stream()>>>(
                (const uint8_t *) code->data, (const half *) scale->data,
                (const uint8_t *) zp->data, (const float *) x->data,
                (float *) dst->data, KB, G, ntri, n_embd, V, n_tokens);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // MTP head: R vocabulary rows per CTA for the verification batch, with the activation window
    // reused across rows; R and T are both compile-time so nothing spills. See the kernel comment
    // for the NCU evidence that this kernel is instruction/L1-bound at M>1, not bandwidth-bound.
    const char * rt_env = std::getenv("ESCHA_E3_HEAD_RT");
    const int rt_rows = rt_env != nullptr ? atoi(rt_env) : 0;
    if (rt_rows > 0 && n_tokens >= 2 && n_tokens <= LOWGPU_RT_MAXT) {
        const int64_t nx_rt = (int64_t) n_tokens*n_embd;
        ggml_cuda_pool_alloc<half> xh_rt(ctx.pool(), (size_t) nx_rt);
        lowgpu_f32_to_f16_kernel<<<(int) ((nx_rt + 255)/256), 256, 0, ctx.stream()>>>(
                (const float *) x->data, xh_rt.ptr, nx_rt);
        const uint8_t * rc = (const uint8_t *) code->data;
        const half    * rs = (const half *)    scale->data;
        const uint8_t * rz = (const uint8_t *) zp->data;
        float         * rd = (float *)         dst->data;
        const dim3 g2((V + 1)/2), g4((V + 3)/4), g8((V + 7)/8);
        const int T = n_tokens;
        if (rt_rows == 8 && T <= 6) {
            if      (T == 2) lowgpu_packed_gemv_rt_kernel<8,2><<<g8, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 3) lowgpu_packed_gemv_rt_kernel<8,3><<<g8, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 4) lowgpu_packed_gemv_rt_kernel<8,4><<<g8, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 5) lowgpu_packed_gemv_rt_kernel<8,5><<<g8, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else             lowgpu_packed_gemv_rt_kernel<8,6><<<g8, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (rt_rows == 4) {
            if      (T == 2) lowgpu_packed_gemv_rt_kernel<4,2><<<g4, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 3) lowgpu_packed_gemv_rt_kernel<4,3><<<g4, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 4) lowgpu_packed_gemv_rt_kernel<4,4><<<g4, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 5) lowgpu_packed_gemv_rt_kernel<4,5><<<g4, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 6) lowgpu_packed_gemv_rt_kernel<4,6><<<g4, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 7) lowgpu_packed_gemv_rt_kernel<4,7><<<g4, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else             lowgpu_packed_gemv_rt_kernel<4,8><<<g4, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (rt_rows == 2) {
            if      (T == 2) lowgpu_packed_gemv_rt_kernel<2,2><<<g2, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 3) lowgpu_packed_gemv_rt_kernel<2,3><<<g2, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 4) lowgpu_packed_gemv_rt_kernel<2,4><<<g2, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 5) lowgpu_packed_gemv_rt_kernel<2,5><<<g2, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 6) lowgpu_packed_gemv_rt_kernel<2,6><<<g2, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else if (T == 7) lowgpu_packed_gemv_rt_kernel<2,7><<<g2, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            else             lowgpu_packed_gemv_rt_kernel<2,8><<<g2, 256, 0, ctx.stream()>>>(rc, rs, rz, xh_rt.ptr, rd, KB, G, ntri, n_embd, V);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const char * mrow_env = std::getenv("ESCHA_E3_HEAD_MROW");

    const int mrow_max = mrow_env != nullptr ? atoi(mrow_env) : 0;
    // MTP-0068: the rows-per-CTA kernel also serves the narrow MTP verification batches
    // (n_tokens 2..4, i.e. depths n1..n3). Those used to run the V x n_tokens single-row grid,
    // which is 248320 x n_tokens CTAs - about a million CTAs at n_tokens=4. n_tokens == 1 stays
    // on the single-row kernel, so plain decode and the MTP draft step are untouched.
    if (mrow_max > 0 && n_tokens > 1 && n_tokens <= LOWGPU_MROW_MAX && n_tokens <= mrow_max) {
        const int64_t nx_mrow = (int64_t) n_tokens*n_embd;
        ggml_cuda_pool_alloc<half> xh_mrow(ctx.pool(), (size_t) nx_mrow);
        lowgpu_f32_to_f16_kernel<<<(int) ((nx_mrow + 255)/256), 256, 0, ctx.stream()>>>(
                (const float *) x->data, xh_mrow.ptr, nx_mrow);
        lowgpu_packed_gemv_mrow_kernel<<<dim3(V), 256, 0, ctx.stream()>>>(
                (const uint8_t *) code->data, (const half *) scale->data,
                (const uint8_t *) zp->data, xh_mrow.ptr,
                (float *) dst->data, KB, G, ntri, n_embd, V, n_tokens);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (n_tokens <= 4) {
        const char * warp4 = std::getenv("ESCHA_E3_HEAD_WARP4");
        if (warp4 != nullptr && strcmp(warp4, "1") == 0) {
            lowgpu_packed_gemv_warp4_exact_kernel<<<dim3((V + 3)/4, n_tokens), 128, 0, ctx.stream()>>>(
                    (const uint8_t *) code->data, (const half *) scale->data,
                    (const uint8_t *) zp->data, (const float *) x->data,
                    (float *) dst->data, KB, G, ntri, n_embd, V);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        const bool x_vec_ok = (reinterpret_cast<uintptr_t>(x->data) % 16 == 0) && (n_embd % 4 == 0);
        if (x_vec_ok) {
            lowgpu_packed_gemv_kernel<true><<<dim3(V, n_tokens), 256, 0, ctx.stream()>>>(
                    (const uint8_t *) code->data,
                    (const half *)    scale->data,
                    (const uint8_t *) zp->data,
                    (const float *)   x->data,
                    (float *)         dst->data,
                    KB, G, ntri, n_embd, V);
        } else {
            lowgpu_packed_gemv_kernel<false><<<dim3(V, n_tokens), 256, 0, ctx.stream()>>>(
                    (const uint8_t *) code->data,
                    (const half *)    scale->data,
                    (const uint8_t *) zp->data,
                    (const float *)   x->data,
                    (float *)         dst->data,
                    KB, G, ntri, n_embd, V);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // activations to fp16 (same-type GEMM; mixed F16/F32 GemmEx is unsupported on sm_120)
    ggml_cuda_pool_alloc<half> xh_alloc(ctx.pool(), (size_t) n_tokens*n_embd);
    half * xh = xh_alloc.ptr;
    const int64_t nx = (int64_t) n_tokens*n_embd;
    const int64_t nblocks = (nx + 255)/256;
    lowgpu_f32_to_f16_kernel<<<(int) nblocks, 256, 0, ctx.stream()>>>(
            (const float *) x->data, xh, nx);

    // dst[V, n_tokens] = W^T @ x, W row-major [n_embd, V]
    // CUBLAS_COMPUTE_32F makes alpha/beta FP32 host scalars.  Passing half
    // pointers here lets cuBLAS read four bytes from two unrelated half
    // values, which scales every logit toward zero on Blackwell.
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    // Dequantize a small row stripe immediately before its GEMM.  This keeps
    // source weights packed in VRAM and caps temporary weight storage at 40
    // MiB, while preserving the exact fp16 weight/activation boundary of the
    // former full-vocabulary path.
    constexpr int LOWGPU_PREFILL_ROWS = 4096;
    const uint8_t * code_d = (const uint8_t *) code->data;
    const half    * scale_d = (const half *) scale->data;
    const uint8_t * zp_d = (const uint8_t *) zp->data;
    float * dst_d = (float *) dst->data;
    ggml_cuda_pool_alloc<half> w_alloc(ctx.pool(), (size_t) LOWGPU_PREFILL_ROWS*n_embd);
    for (int row0 = 0; row0 < V; row0 += LOWGPU_PREFILL_ROWS) {
        const int rows = MIN(LOWGPU_PREFILL_ROWS, V - row0);
        half * w = w_alloc.ptr;
        lowgpu_dequant_row_kernel<<<rows, 256, 0, ctx.stream()>>>(
                code_d  + (size_t) row0*KB,
                scale_d + (size_t) row0*G,
                zp_d    + (size_t) row0*G,
                w, KB, G, ntri);
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasGemmEx(
                ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                rows, n_tokens, n_embd,
                &alpha, w, CUDA_R_16F, n_embd,
                        xh, CUDA_R_16F, n_embd,
                &beta, dst_d + row0, CUDA_R_32F, V,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

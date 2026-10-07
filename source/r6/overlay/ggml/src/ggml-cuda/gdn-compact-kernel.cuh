template <int S_v, bool KDA, bool keep_rs_t, bool grouped_qk_norm, bool fuse_decode_prep>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gdn_compact_forward_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * decode_dt,
                                     const float * decode_a_log,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   q_group_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     float         qk_norm_eps,
                                     int64_t       state_slot_stride,
                                     int           K, const float * old_cache, const int32_t * rollback, float * packed) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = grouped_qk_norm
        ? fastdiv(h_idx, q_group_magic)
        : fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    __shared__ float decode_gate;
    __shared__ float decode_beta;
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    const int64_t D = H*S_v*S_v;
    const int old_count = int(old_cache[2*D+K*H*(2*S_v+1)]);
    const int rewind = rollback[0];
    if(!(old_count>=0 && old_count<=K && rewind>=0 && rewind<=old_count))asm volatile("trap;");
    const int valid_old = old_count-rewind;
    const int keep_old = n_tokens<K ? min(valid_old,int(K-n_tokens)) : 0;
    const int drop = valid_old-keep_old;
    if(h_idx==0 && col==0 && lane==0)packed[2*D+K*H*(2*S_v+1)]=float(min(int(n_tokens)+keep_old,K));
    if(col==0) {
        for(int t=0;t<keep_old;t++) {
            #pragma unroll
            for(int r=0;r<rows_per_lane;r++) {
                const int i=r*warp_size+lane;
                packed[2*D+(t*H+h_idx)*S_v+i]=old_cache[2*D+((t+drop)*H+h_idx)*S_v+i];
                packed[2*D+K*H*(S_v+1)+(t*H+h_idx)*S_v+i]=old_cache[2*D+K*H*(S_v+1)+((t+drop)*H+h_idx)*S_v+i];
            }
            if(lane==0)packed[2*D+K*H*S_v+t*H+h_idx]=old_cache[2*D+K*H*S_v+(t+drop)*H+h_idx];
        }
    }
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    if(n_tokens<K) {
        #pragma unroll
        for(int r=0;r<rows_per_lane;r++) {
            const int i=r*warp_size+lane;
            packed[D+h_idx*S_v*S_v+col*S_v+i]=keep_old==0 ? s_shard[r] :
                l0xre_compact_replay_element(old_cache,S_v,H,K,h_idx*S_v*S_v+col*S_v+i,drop);
        }
    }
    for (int t = 0; t < n_tokens; t++) {
        if(n_tokens>=K && t==n_tokens-K) {
            #pragma unroll
            for(int r=0;r<rows_per_lane;r++)packed[D+h_idx*S_v*S_v+col*S_v+r*warp_size+lane]=s_shard[r];
        }

        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val;
        if constexpr (fuse_decode_prep) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                const float biased = *g_t + decode_dt[h_idx];
                const float alpha = biased > 20.0f ? biased : logf(1.0f + expf(biased));
                const float decay = -expf(decode_a_log[h_idx]);
                volatile float prepared_gate = alpha * decay;
                decode_gate = expf(prepared_gate);
                decode_beta = 1.0f/(1.0f + expf(-(*beta_t)));
            }
            __syncthreads();
            beta_val = decode_beta;
        } else {
            beta_val = *beta_t;
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (grouped_qk_norm) {
            float k_norm = 0.0f;
            float q_norm = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_norm += k_reg[r] * k_reg[r];
                q_norm += q_reg[r] * q_reg[r];
            }
            k_norm = rsqrtf(fmaxf(warp_reduce_sum<warp_size>(k_norm), qk_norm_eps * qk_norm_eps));
            q_norm = rsqrtf(fmaxf(warp_reduce_sum<warp_size>(q_norm), qk_norm_eps * qk_norm_eps));
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] *= k_norm;
                q_reg[r] *= q_norm;
            }
        }

        if constexpr (!KDA) {
            const float g_val = fuse_decode_prep ? decode_gate : expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            const int log_t = t-max(int(n_tokens)-K,0);
            if(log_t>=0) {
                const int slot=keep_old+log_t;
                if(col==0) {
                    #pragma unroll
                    for(int r=0;r<rows_per_lane;r++)packed[2*D+(slot*H+h_idx)*S_v+r*warp_size+lane]=k_reg[r];
                    if(lane==0)packed[2*D+K*H*S_v+slot*H+h_idx]=g_val;
                }
                if(lane==0)packed[2*D+K*H*(S_v+1)+(slot*H+h_idx)*S_v+col]=delta_col;
            }
            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

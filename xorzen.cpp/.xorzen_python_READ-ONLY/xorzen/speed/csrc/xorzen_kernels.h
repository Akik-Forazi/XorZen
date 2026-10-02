/*
 * xorzen Lightning Kernels — Master Header
 * =========================================
 * All C++ kernel declarations used by the Cython bridge.
 * Compiled with GCC/G++ + OpenMP + AVX2 on Windows (MinGW32/64).
 *
 * Build flags:
 *   -O3 -march=native -fopenmp -mavx2 -mfma -ffast-math
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Cross-platform restrict keyword
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────────────────────────────────────────────────
 * SSM Parallel Prefix Scan
 * h_t = A * h_{t-1} + B_t   (diagonal A, element-wise)
 * ───────────────────────────────────────────────────────────────────── */

/*
 * ssm_parallel_scan_f32
 * ---------------------
 * in-place parallel prefix scan on a contiguous float32 buffer.
 *
 * u     : [B, T, H, N]  input  (B=batch, T=seq, H=hidden, N=state)
 * A     : [H, N]         state transition (diagonal, already discretised)
 * out   : [B, T, H, N]  output (may alias u)
 * B_sz  : batch size
 * T_sz  : sequence length
 * H_sz  : hidden dim
 * N_sz  : state dim
 */
void ssm_parallel_scan_f32(
    const float* RESTRICT u,
    const float* RESTRICT A,
    float*       RESTRICT out,
    int B_sz, int T_sz, int H_sz, int N_sz
);

/* ─────────────────────────────────────────────────────────────────────
 * Fused Window Attention
 * ───────────────────────────────────────────────────────────────────── */

/*
 * window_attention_f32
 * --------------------
 * Computes causal + windowed multi-head attention.
 * Q, K, V : [B, H, S, D]  (contiguous row-major)
 * out     : [B, H, S, D]
 * B_sz    : batch
 * H_sz    : heads
 * S_sz    : sequence length
 * D_sz    : head dim
 * window  : local window size (0 = full causal)
 * scale   : 1/sqrt(D) — caller computes this
 */
void window_attention_f32(
    const float* RESTRICT Q,
    const float* RESTRICT K,
    const float* RESTRICT V,
    float*       RESTRICT out,
    int B_sz, int H_sz, int S_sz, int D_sz,
    int window, float scale
);

/* ─────────────────────────────────────────────────────────────────────
 * Router MLP Fast Path
 * ───────────────────────────────────────────────────────────────────── */

/*
 * router_mlp_forward_f32
 * ----------------------
 * 3-layer MLP: Linear → GELU → Linear → GELU → Linear
 * Weights are pre-transposed for BLAS-friendly access.
 *
 * x      : [N, in_dim]   input tokens
 * w1,b1  : layer 1  [in_dim  → h1]
 * w2,b2  : layer 2  [h1      → h2]
 * w3,b3  : layer 3  [h2      → out_dim]
 * out    : [N, out_dim]
 * N      : number of tokens
 */
void router_mlp_forward_f32(
    const float* RESTRICT x,
    const float* RESTRICT w1, const float* RESTRICT b1, int h1,
    const float* RESTRICT w2, const float* RESTRICT b2, int h2,
    const float* RESTRICT w3, const float* RESTRICT b3,
    float*       RESTRICT out,
    int N, int in_dim, int out_dim
);

/* ─────────────────────────────────────────────────────────────────────
 * Expert Dispatch — Top-K + Capacity Constraint
 * ───────────────────────────────────────────────────────────────────── */

/*
 * expert_capacity_constraint_f32
 * --------------------------------
 * Vectorised capacity constraint via counting sort.
 * Modifies weights in-place; zeroes out tokens exceeding capacity.
 *
 * indices  : [N, top_k]  expert indices (int32)
 * weights  : [N, top_k]  routing weights (float32), modified in-place
 * N        : number of tokens
 * top_k    : number of experts per token
 * n_experts: total expert count
 * capacity : max tokens per expert
 */
void expert_capacity_constraint_f32(
    const int*   RESTRICT indices,
    float*       RESTRICT weights,
    int N, int top_k, int n_experts, int capacity
);

/*
 * expert_weighted_sum_f32
 * -------------------------
 * Accumulates weighted expert outputs into a destination buffer.
 * Uses SIMD for the inner hidden-dim loop.
 *
 * expert_out : [n_active, hidden]  stacked expert outputs
 * weights    : [n_active]          scalar weights
 * token_ids  : [n_active]          which output row to accumulate into
 * dst        : [N, hidden]         output buffer (zeroed by caller)
 * n_active   : number of (token, expert) pairs
 * N          : total tokens
 * hidden     : hidden dimension
 */
void expert_weighted_sum_f32(
    const float* RESTRICT expert_out,
    const float* RESTRICT weights,
    const int*   RESTRICT token_ids,
    float*       RESTRICT dst,
    int n_active, int N, int hidden
);

/* ─────────────────────────────────────────────────────────────────────
 * Fused Math Utilities
 * ───────────────────────────────────────────────────────────────────── */

/*
 * fused_layernorm_gelu_f32
 * -------------------------
 * Single-pass: LayerNorm → GELU in-place.
 * x    : [N, D]  input/output
 * gamma, beta : [D]  LN params
 * eps  : layernorm epsilon
 */
void fused_layernorm_gelu_f32(
    float* RESTRICT x,
    const float* RESTRICT gamma,
    const float* RESTRICT beta,
    int N, int D, float eps
);

/*
 * fused_residual_add_f32
 * ----------------------
 * dst = dst + src  (in-place, vectorised)
 * AVX2: processes 8 floats per cycle.
 */
void fused_residual_add_f32(
    float*       RESTRICT dst,
    const float* RESTRICT src,
    int n_elements
);

/*
 * gelu_f32_inplace
 * -----------------
 * GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
 * Vectorised over n_elements floats.
 */
void gelu_f32_inplace(float* RESTRICT x, int n_elements);

/*
 * softmax_f32_inplace
 * --------------------
 * Online (numerically stable) softmax over last dim.
 * x    : [rows, cols]  — softmax over cols
 */
void softmax_f32_inplace(float* RESTRICT x, int rows, int cols);

/* ─────────────────────────────────────────────────────────────────────
 * Extended kernels (fused_ops.cpp)
 * ───────────────────────────────────────────────────────────────────── */

/*
 * rms_norm_f32
 * -------------
 * RMSNorm: out = x / rms(x) * gamma   (no mean subtraction)
 * x, out : [N, D];   gamma : [D]
 */
void rms_norm_f32(
    const float* RESTRICT x,
    const float* RESTRICT gamma,
    float*       RESTRICT out,
    int N, int D, float eps);

/*
 * fused_swiglu_f32
 * -----------------
 * out = silu(gate) * up   — one pass, AVX2
 * gate, up, out : [N, D]
 */
void fused_swiglu_f32(
    const float* RESTRICT gate,
    const float* RESTRICT up,
    float*       RESTRICT out,
    int N, int D);

/*
 * diagonal_ssm_scan_f32
 * ----------------------
 * Efficient diagonal-A SSM scan: h_t = a * h_{t-1} + b_t
 * B_seq  : [batch, T, state]
 * A_diag : [state]            (diagonal of discretised A_bar)
 * states : [batch, T, state]  output
 */
void diagonal_ssm_scan_f32(
    const float* RESTRICT B_seq,
    const float* RESTRICT A_diag,
    float*       RESTRICT states,
    int batch, int T, int state);

#ifdef __cplusplus
}
#endif

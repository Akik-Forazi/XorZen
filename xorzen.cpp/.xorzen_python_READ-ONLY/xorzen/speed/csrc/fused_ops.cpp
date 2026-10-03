/*
 * fused_ops.cpp — Additional fused kernels for xorzen
 * ====================================================
 * 1. rms_norm_f32        : RMSNorm (no mean subtraction) — faster than LN
 * 2. fused_swiglu_f32    : SwiGLU in one pass (gate * silu(up))
 * 3. diagonal_ssm_scan_f32 : SSM scan assuming diagonal A (element-wise)
 *                            — avoids matrix_exp on the full NxN A matrix.
 *
 * These complement the existing kernels and are exposed via the Cython bridge.
 *
 * Compile flags: -O3 -march=native -mavx2 -mfma -fopenmp -std=c++17
 */

#include "xorzen_kernels.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef __AVX2__
  #include <immintrin.h>
#endif

/* ── fast exp approximation for SSM (Schraudolph 1999 variant) ──── */
static inline float fast_exp(float x) {
    /* Clamped to avoid overflow; accurate to ~2% for |x| < 10 */
    x = std::max(-88.0f, std::min(88.0f, x));
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x) + 1065353216;
    return u.f;
}

/* ─────────────────────────────────────────────────────────────────
 * 1.  RMSNorm
 *     out[i] = x[i] / rms(x_row) * gamma[i]
 *     No beta — same as LLaMA / xorzen RMSNorm.
 * ───────────────────────────────────────────────────────────────── */

extern "C"
void rms_norm_f32(
    const float* __restrict x,       /* [N, D] input  */
    const float* __restrict gamma,   /* [D]   scale   */
    float*       __restrict out,     /* [N, D] output */
    int N, int D, float eps)
{
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const float* row = x   + n * D;
        float*       dst = out + n * D;

        /* RMS = sqrt(mean(x^2) + eps) */
        float sum2 = 0.0f;
#ifdef __AVX2__
        {
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i <= D - 8; i += 8) {
                __m256 v = _mm256_loadu_ps(row + i);
                acc = _mm256_fmadd_ps(v, v, acc);
            }
            /* horizontal sum */
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s  = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            sum2 = _mm_cvtss_f32(s);
            for (; i < D; ++i) sum2 += row[i] * row[i];
        }
#else
        for (int i = 0; i < D; ++i) sum2 += row[i] * row[i];
#endif
        float inv_rms = 1.0f / std::sqrt(sum2 / D + eps);

        /* Scale */
#ifdef __AVX2__
        __m256 vr = _mm256_set1_ps(inv_rms);
        int i = 0;
        for (; i <= D - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(row + i);
            __m256 vg = _mm256_loadu_ps(gamma + i);
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_mul_ps(vx, vr), vg));
        }
        for (; i < D; ++i) dst[i] = row[i] * inv_rms * gamma[i];
#else
        for (int i = 0; i < D; ++i) dst[i] = row[i] * inv_rms * gamma[i];
#endif
    }
}

/* ─────────────────────────────────────────────────────────────────
 * 2.  SwiGLU fused
 *     out = silu(gate) * up
 *     gate and up are contiguous halves of a [N, 2*D] buffer.
 * ───────────────────────────────────────────────────────────────── */

static inline float silu_scalar(float x) {
    return x / (1.0f + fast_exp(-x));
}

extern "C"
void fused_swiglu_f32(
    const float* __restrict gate,  /* [N, D] */
    const float* __restrict up,    /* [N, D] */
    float*       __restrict out,   /* [N, D] */
    int N, int D)
{
    const int total = N * D;
#ifdef __AVX2__
    /* AVX2: approximate silu via silu(x) = x * sigmoid(x)
     * sigmoid(x) ≈ 0.5 + 0.25*x for |x| < 1 — but use proper formula here
     * via the fast_exp approximation. */
    int i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i <= total - 8; i += 8) {
        __m256 vg  = _mm256_loadu_ps(gate + i);
        __m256 vu  = _mm256_loadu_ps(up   + i);
        /* silu: x / (1 + exp(-x)) — computed scalar for accuracy */
        float tmp[8];
        _mm256_storeu_ps(tmp, vg);
        for (int j = 0; j < 8; ++j) tmp[j] = silu_scalar(tmp[j]);
        __m256 vs = _mm256_loadu_ps(tmp);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(vs, vu));
    }
    for (; i < total; ++i) out[i] = silu_scalar(gate[i]) * up[i];
#else
    for (int i = 0; i < total; ++i) out[i] = silu_scalar(gate[i]) * up[i];
#endif
}

/* ─────────────────────────────────────────────────────────────────
 * 3.  Diagonal SSM scan (element-wise A)
 *     h_t = a * h_{t-1} + b_t    where a is a [N] vector (diagonal)
 *
 *     This is the actual recurrence in xorzen's SSMPathway after
 *     discretisation: A_bar = exp(dt * A_diag).  We take a [N] diagonal
 *     instead of the full [N,N] matrix to avoid O(N^2) per-step cost.
 *
 *     Input layout:  B_seq [batch, T, state]  (contiguous)
 *                    A_diag [state]            (diagonal of A_bar)
 *     Output layout: states [batch, T, state]
 * ───────────────────────────────────────────────────────────────── */

extern "C"
void diagonal_ssm_scan_f32(
    const float* __restrict B_seq,   /* [batch, T, state] */
    const float* __restrict A_diag,  /* [state]           */
    float*       __restrict states,  /* [batch, T, state] output */
    int batch, int T, int state)
{
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; ++b) {
        const float* b_ptr = B_seq  + b * T * state;
        float*       s_ptr = states + b * T * state;

        /* h: current hidden state [state] — stack allocated if state is small */
        std::vector<float> h(state, 0.0f);

        for (int t = 0; t < T; ++t) {
            const float* bt = b_ptr + t * state;
            float*       st = s_ptr + t * state;

#ifdef __AVX2__
            int i = 0;
            for (; i <= state - 8; i += 8) {
                __m256 va = _mm256_loadu_ps(A_diag + i);
                __m256 vh = _mm256_loadu_ps(h.data() + i);
                __m256 vb = _mm256_loadu_ps(bt + i);
                __m256 vn = _mm256_fmadd_ps(va, vh, vb);   /* a*h + b */
                _mm256_storeu_ps(h.data() + i, vn);
                _mm256_storeu_ps(st + i, vn);
            }
            for (; i < state; ++i) {
                h[i] = A_diag[i] * h[i] + bt[i];
                st[i] = h[i];
            }
#else
            for (int i = 0; i < state; ++i) {
                h[i] = A_diag[i] * h[i] + bt[i];
                st[i] = h[i];
            }
#endif
        }
    }
}

/*
 * math_utils.cpp — Fused Elementwise Ops for xorzen
 * ==================================================
 * Fused kernels that collapse multiple PyTorch ops into one pass,
 * eliminating intermediate tensor allocations and extra memory writes.
 *
 *   fused_layernorm_gelu_f32  : LayerNorm then GELU in one pass
 *   fused_residual_add_f32    : dst += src  (AVX2)
 *   gelu_f32_inplace          : GELU in-place (AVX2)
 *   softmax_f32_inplace       : row-wise online softmax (AVX2)
 *
 * Compile: g++ -O3 -march=native -mavx2 -mfma -fopenmp -std=c++17
 */

#include "xorzen_kernels.h"


#include <cmath>
#include <cstring>
#include <limits>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef __AVX2__
  #include <immintrin.h>
#endif

/* ── GELU constants ──────────────────────────────────────────────── */
static const float SQRT2_OVER_PI_MU = 0.7978845608f;
static const float GELU_C           = 0.044715f;

static inline float _gelu(float x) {
    float x3  = x * x * x;
    float a   = SQRT2_OVER_PI_MU * (x + GELU_C * x3);
    /* tanh rational approx */
    float a2  = a * a;
    float num = a * (135135.0f + a2 * (17325.0f + a2 * (378.0f + a2)));
    float den = 135135.0f + a2 * (62370.0f + a2 * (3150.0f + a2 * 28.0f));
    float t   = num / den;
    return 0.5f * x * (1.0f + t);
}

/* ── fused LayerNorm + GELU ──────────────────────────────────────── */

extern "C"
void fused_layernorm_gelu_f32(
    float* RESTRICT x,
    const float* RESTRICT gamma,
    const float* RESTRICT beta,
    int N, int D, float eps)
{
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        float* row = x + n * D;

        /* ─ Pass 1: compute mean ─ */
        float mean = 0.0f;
#ifdef __AVX2__
        {
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i <= D - 8; i += 8)
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(row + i));
            __m128 lo  = _mm256_castps256_ps128(acc);
            __m128 hi  = _mm256_extractf128_ps(acc, 1);
            __m128 s   = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            mean = _mm_cvtss_f32(s);
            for (; i < D; ++i) mean += row[i];
        }
#else
        for (int i = 0; i < D; ++i) mean += row[i];
#endif
        mean /= D;

        /* ─ Pass 2: compute variance ─ */
        float var = 0.0f;
#ifdef __AVX2__
        {
            __m256 vm  = _mm256_set1_ps(mean);
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i <= D - 8; i += 8) {
                __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + i), vm);
                acc = _mm256_fmadd_ps(d, d, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s  = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            var = _mm_cvtss_f32(s);
            for (; i < D; ++i) { float d = row[i]-mean; var += d*d; }
        }
#else
        for (int i = 0; i < D; ++i) { float d = row[i]-mean; var += d*d; }
#endif
        var /= D;
        float inv_std = 1.0f / std::sqrt(var + eps);

        /* ─ Pass 3: normalise, scale/shift, apply GELU in one pass ─ */
        for (int i = 0; i < D; ++i) {
            float norm = (row[i] - mean) * inv_std;
            float y    = gamma[i] * norm + beta[i];
            row[i]     = _gelu(y);   /* GELU fused here */
        }
    }
}

/* ── fused residual add ──────────────────────────────────────────── */

extern "C"
void fused_residual_add_f32(
    float*       RESTRICT dst,
    const float* RESTRICT src,
    int n_elements)
{
#ifdef __AVX2__
    int i = 0;
    for (; i <= n_elements - 8; i += 8) {
        __m256 vd = _mm256_loadu_ps(dst + i);
        __m256 vs = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(vd, vs));
    }
    for (; i < n_elements; ++i) dst[i] += src[i];
#else
    for (int i = 0; i < n_elements; ++i) dst[i] += src[i];
#endif
}

/* ── GELU in-place ───────────────────────────────────────────────── */

extern "C"
void gelu_f32_inplace(float* RESTRICT x, int n_elements) {
    /* Scalar path for correctness; AVX2 path in router_ops.cpp is reusable
     * but lives in a different TU — just call scalar here for simplicity. */
    for (int i = 0; i < n_elements; ++i) x[i] = _gelu(x[i]);
}

/* ── online softmax ──────────────────────────────────────────────── */

extern "C"
void softmax_f32_inplace(float* RESTRICT x, int rows, int cols) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float* row = x + r * cols;

        /* find max */
        float mx = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c)
            if (row[c] > mx) mx = row[c];

        /* exp and sum */
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            row[c] = std::exp(row[c] - mx);
            sum   += row[c];
        }

        /* normalise */
        float inv = (sum > 1e-12f) ? 1.0f / sum : 0.0f;
#ifdef __AVX2__
        __m256 vi = _mm256_set1_ps(inv);
        int c = 0;
        for (; c <= cols - 8; c += 8)
            _mm256_storeu_ps(row + c,
                _mm256_mul_ps(_mm256_loadu_ps(row + c), vi));
        for (; c < cols; ++c) row[c] *= inv;
#else
        for (int c = 0; c < cols; ++c) row[c] *= inv;
#endif
    }
}

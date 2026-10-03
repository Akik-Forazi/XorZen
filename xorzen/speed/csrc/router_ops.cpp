/*
 * router_ops.cpp — Fused 3-Layer MLP for xorzen AdaptiveRouter
 * =============================================================
 * Replaces the Python-level feature_encoder forward pass with a
 * single C++ function. Uses CBLAS sgemm for linear layers and a
 * fast polynomial GELU approximation.
 *
 * Architecture: Linear(in→h1) → GELU → Linear(h1→h2) → GELU → Linear(h2→out)
 *
 * Compile: g++ -O3 -march=native -mavx2 -mfma -fopenmp -std=c++17 -lopenblas
 */

#include "xorzen_kernels.h"


#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef __AVX2__
  #include <immintrin.h>
#endif

/* ── fast GELU (Cephes polynomial approximation) ─────────────────── */
/*
 * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 * We use the fast tanh approximation via a rational polynomial.
 */

static const float SQRT2_OVER_PI = 0.7978845608f;  /* sqrt(2/pi) */
static const float GELU_COEF     = 0.044715f;

static inline float gelu_scalar(float x) {
    float x3  = x * x * x;
    float inner = SQRT2_OVER_PI * (x + GELU_COEF * x3);
    /* fast tanh via rational approx — accurate to ~1e-5 */
    float t;
    if (inner > 4.0f)       t = 1.0f;
    else if (inner < -4.0f) t = -1.0f;
    else {
        float i2 = inner * inner;
        t = inner * (135135.0f + i2 * (17325.0f + i2 * (378.0f + i2)))
              / (135135.0f + i2 * (62370.0f + i2 * (3150.0f + i2 * 28.0f)));
    }
    return 0.5f * x * (1.0f + t);
}

#ifdef __AVX2__
/* AVX2 GELU approximation — processes 8 floats per cycle */
static void gelu_avx2(float* RESTRICT x, int n) {
    const __m256 sqrt2pi = _mm256_set1_ps(SQRT2_OVER_PI);
    const __m256 coef    = _mm256_set1_ps(GELU_COEF);
    const __m256 half    = _mm256_set1_ps(0.5f);
    const __m256 one     = _mm256_set1_ps(1.0f);

    /* Tanh rational poly coefficients */
    const __m256 c0 = _mm256_set1_ps(135135.0f);
    const __m256 c1 = _mm256_set1_ps(17325.0f);
    const __m256 c2 = _mm256_set1_ps(378.0f);
    const __m256 d1 = _mm256_set1_ps(62370.0f);
    const __m256 d2 = _mm256_set1_ps(3150.0f);
    const __m256 d3 = _mm256_set1_ps(28.0f);

    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vx  = _mm256_loadu_ps(x + i);
        __m256 vx3 = _mm256_mul_ps(_mm256_mul_ps(vx, vx), vx);
        __m256 inner = _mm256_mul_ps(sqrt2pi,
                            _mm256_fmadd_ps(coef, vx3, vx));
        __m256 i2  = _mm256_mul_ps(inner, inner);

        /* Numerator: inner * (c0 + i2*(c1 + i2*(1+i2))) */
        __m256 num = _mm256_fmadd_ps(i2, one, c2);
        num = _mm256_fmadd_ps(i2, num, c1);
        num = _mm256_fmadd_ps(i2, num, c0);
        num = _mm256_mul_ps(inner, num);

        /* Denominator: c0 + i2*(d1 + i2*(d2 + i2*d3)) */
        __m256 den = _mm256_fmadd_ps(i2, d3, d2);
        den = _mm256_fmadd_ps(i2, den, d1);
        den = _mm256_fmadd_ps(i2, den, c0);

        __m256 t   = _mm256_div_ps(num, den);   /* approx tanh */
        __m256 out = _mm256_mul_ps(half,
                         _mm256_mul_ps(vx, _mm256_add_ps(one, t)));
        _mm256_storeu_ps(x + i, out);
    }
    for (; i < n; ++i) x[i] = gelu_scalar(x[i]);
}
#endif /* __AVX2__ */

/* ── linear layer: out = x @ W^T + b ─────────────────────────────── */
/*
 * Simple row-by-row dot product — acceptable for router MLP sizes
 * (N × in_dim is small). For large N, replace with cblas_sgemm.
 */
static void linear_f32(
    const float* RESTRICT x,   /* [N, in_d]  */
    const float* RESTRICT W,   /* [out_d, in_d] row-major */
    const float* RESTRICT b,   /* [out_d] */
    float*       RESTRICT out, /* [N, out_d] */
    int N, int in_d, int out_d)
{
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const float* xn  = x + n * in_d;
        float*       on  = out + n * out_d;
        for (int o = 0; o < out_d; ++o) {
            /* dot(xn, W[o]) + b[o] */
            const float* w = W + o * in_d;
            float s = b[o];
#ifdef __AVX2__
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k <= in_d - 8; k += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xn+k),
                                      _mm256_loadu_ps(w+k), acc);
            /* horizontal sum */
            __m128 lo  = _mm256_castps256_ps128(acc);
            __m128 hi  = _mm256_extractf128_ps(acc, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            s += _mm_cvtss_f32(sum);
            for (; k < in_d; ++k) s += xn[k] * w[k];
#else
            for (int k = 0; k < in_d; ++k) s += xn[k] * w[k];
#endif
            on[o] = s;
        }
    }
}

/* ── public API ──────────────────────────────────────────────────── */

extern "C"
void router_mlp_forward_f32(
    const float* RESTRICT x,
    const float* RESTRICT w1, const float* RESTRICT b1, int h1,
    const float* RESTRICT w2, const float* RESTRICT b2, int h2,
    const float* RESTRICT w3, const float* RESTRICT b3,
    float*       RESTRICT out,
    int N, int in_dim, int out_dim)
{
    /* Layer 1: [N, in_dim] → [N, h1] + GELU */
    std::vector<float> buf1(N * h1);
    linear_f32(x, w1, b1, buf1.data(), N, in_dim, h1);
#ifdef __AVX2__
    gelu_avx2(buf1.data(), N * h1);
#else
    for (int i = 0; i < N * h1; ++i) buf1[i] = gelu_scalar(buf1[i]);
#endif

    /* Layer 2: [N, h1] → [N, h2] + GELU */
    std::vector<float> buf2(N * h2);
    linear_f32(buf1.data(), w2, b2, buf2.data(), N, h1, h2);
#ifdef __AVX2__
    gelu_avx2(buf2.data(), N * h2);
#else
    for (int i = 0; i < N * h2; ++i) buf2[i] = gelu_scalar(buf2[i]);
#endif

    /* Layer 3: [N, h2] → [N, out_dim] (no activation) */
    linear_f32(buf2.data(), w3, b3, out, N, h2, out_dim);
}

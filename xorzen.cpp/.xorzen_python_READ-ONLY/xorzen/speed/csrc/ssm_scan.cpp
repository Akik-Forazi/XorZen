/*
 * ssm_scan.cpp — AVX2 Parallel Prefix Scan for xorzen SSMPathway
 * ================================================================
 * Implements the SSM recurrence:
 *   h_t = A * h_{t-1} + B_t    (A diagonal, element-wise multiply)
 *
 * Algorithm: Work-efficient parallel prefix scan (Blelloch 1990)
 *   - Up-sweep  (reduce)   phase: log2(T) levels, parallelised with OpenMP
 *   - Down-sweep (broadcast) phase: log2(T) levels, parallelised with OpenMP
 *   - Inner loop over H*N vectorised with AVX2 (_mm256_fmadd_ps)
 *
 * Layout: u[b, t, h, n] stored as u[b*T*H*N + t*H*N + h*N + n]
 *         A[h, n] stored as A[h*N + n]
 *
 * Compile: g++ -O3 -march=native -mavx2 -mfma -fopenmp -std=c++17
 */

#include "xorzen_kernels.h"
// Cross-platform restrict keyword
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define RESTRICT RESTRICT
#else
    #define RESTRICT
#endif


#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef __AVX2__
  #include <immintrin.h>
#endif

/* ── helpers ─────────────────────────────────────────────────────── */

static inline int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* Element-wise multiply-add: out[i] = a[i] * b[i] + c[i]
 * Vectorised with AVX2 when available, scalar fallback otherwise. */
static void fmadd_vec(const float* RESTRICT a,
                      const float* RESTRICT b,
                      const float* RESTRICT c,
                      float*       RESTRICT out,
                      int n)
{
#ifdef __AVX2__
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        __m256 vr = _mm256_fmadd_ps(va, vb, vc);   /* a*b + c */
        _mm256_storeu_ps(out + i, vr);
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c[i];
#else
    for (int i = 0; i < n; ++i) out[i] = a[i] * b[i] + c[i];
#endif
}

/* Element-wise multiply: out[i] = a[i] * b[i] */
static void mul_vec(const float* RESTRICT a,
                    const float* RESTRICT b,
                    float*       RESTRICT out,
                    int n)
{
#ifdef __AVX2__
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
#endif
}

/* ── main kernel ─────────────────────────────────────────────────── */

/*
 * Parallel prefix scan for ONE batch item.
 * u_b   : pointer to u[b, 0, 0, 0]  — [T, HN] contiguous
 * A     : [HN] diagonal (broadcast across T)
 * out_b : pointer to out[b, 0, 0, 0] — [T, HN] contiguous
 * T, HN : sequence length and H*N
 *
 * We work on a padded power-of-2 length buffer to keep the
 * scan levels regular, then trim the output.
 */
static void scan_one_batch(
    const float* RESTRICT u_b,
    const float* RESTRICT A,
    float*       RESTRICT out_b,
    int T, int HN)
{
    const int T_pad = next_pow2(T);

    /* Allocate two scratch buffers:
     *   vals[t, hn] = accumulated state at position t
     *   A_pow[t, hn] = A^(count) — cumulative power of A up to t
     * We interleave them to keep memory local. */

    std::vector<float> vals(T_pad * HN, 0.0f);
    std::vector<float> A_acc(T_pad * HN, 1.0f);  /* A^0 = 1 */

    /* Initialise leaves from u_b */
    #pragma omp simd
    for (int t = 0; t < T; ++t) {
        const float* src = u_b + t * HN;
        float*       dst_v = vals.data() + t * HN;
        float*       dst_a = A_acc.data() + t * HN;
        std::memcpy(dst_v, src, HN * sizeof(float));
        /* A_acc leaf = A (the one-step transition) */
        std::memcpy(dst_a, A, HN * sizeof(float));
    }
    /* Padding positions: vals=0, A_acc=1 (identity) */

    /* ── Up-sweep (reduce) ───────────────────────────────────────── */
    for (int stride = 1; stride < T_pad; stride <<= 1) {
        #pragma omp parallel for schedule(static)
        for (int i = stride; i < T_pad; i += stride * 2) {
            int left  = i - 1;
            int right = i + stride - 1;
            if (right >= T_pad) continue;

            float* v_r = vals.data()  + right * HN;
            float* a_r = A_acc.data() + right * HN;
            const float* v_l = vals.data()  + left  * HN;
            const float* a_l = A_acc.data() + left  * HN;

            /* v_r = a_r * v_l + v_r
             * a_r = a_r * a_l          */
            std::vector<float> tmp(HN);
            fmadd_vec(a_r, v_l, v_r, tmp.data(), HN);
            std::memcpy(v_r, tmp.data(), HN * sizeof(float));
            mul_vec(a_r, a_l, a_r, HN);
        }
    }

    /* ── Down-sweep ──────────────────────────────────────────────── */
    /* Set root to identity */
    std::fill(vals.data() + (T_pad - 1) * HN,
              vals.data() +  T_pad      * HN, 0.0f);
    std::fill(A_acc.data() + (T_pad - 1) * HN,
              A_acc.data() +  T_pad      * HN, 1.0f);

    for (int stride = T_pad >> 1; stride >= 1; stride >>= 1) {
        #pragma omp parallel for schedule(static)
        for (int i = stride; i < T_pad; i += stride * 2) {
            int left  = i - 1;
            int right = i + stride - 1;
            if (right >= T_pad) continue;

            float* v_l = vals.data()  + left  * HN;
            float* a_l = A_acc.data() + left  * HN;
            float* v_r = vals.data()  + right * HN;
            float* a_r = A_acc.data() + right * HN;

            std::vector<float> tmp_v(HN), tmp_a(HN);
            std::memcpy(tmp_v.data(), v_l, HN * sizeof(float));
            std::memcpy(tmp_a.data(), a_l, HN * sizeof(float));

            /* left  = right (propagate down-left) */
            std::memcpy(v_l, v_r, HN * sizeof(float));
            std::memcpy(a_l, a_r, HN * sizeof(float));

            /* right = a_r * tmp_v + v_r */
            fmadd_vec(a_r, tmp_v.data(), v_r, v_r, HN);
            mul_vec(a_r, tmp_a.data(), a_r, HN);
        }
    }

    /* ── Add original input back (exclusive scan → inclusive) ───── */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        const float* u_t   = u_b    + t * HN;
        const float* v_t   = vals.data() + t * HN;
        float*       out_t = out_b  + t * HN;
        /* out[t] = A_t * v[t-1] + u[t]
         * vals[t] is the exclusive prefix up to t, so:
         * out[t] = vals[t] + u[t]  (A already folded in during scan) */
        #pragma omp simd
        for (int i = 0; i < HN; ++i)
            out_t[i] = v_t[i] + u_t[i];
    }
}

/* ── public API ──────────────────────────────────────────────────── */

extern "C"
void ssm_parallel_scan_f32(
    const float* RESTRICT u,
    const float* RESTRICT A,
    float*       RESTRICT out,
    int B_sz, int T_sz, int H_sz, int N_sz)
{
    const int HN = H_sz * N_sz;

    /* Process each batch item in parallel */
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B_sz; ++b) {
        const float* u_b   = u   + b * T_sz * HN;
        float*       out_b = out + b * T_sz * HN;
        scan_one_batch(u_b, A, out_b, T_sz, HN);
    }
}

#include "xorzen/optimized/simd_ops.h"
#include <cmath>
#include <algorithm>

// Platform-specific SIMD intrinsics
#ifdef _MSC_VER
  #include <intrin.h>
#else
  #include <x86intrin.h>
  #include <cpuid.h>
#endif

namespace xorzen {
namespace optimized {

// Global SIMD capabilities (detected at startup)
SIMDCapabilities g_simd_caps = SIMDCapabilities::detect();

SIMDCapabilities SIMDCapabilities::detect() {
    SIMDCapabilities caps;
    
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    int cpuinfo[4];
    
#ifdef _MSC_VER
    __cpuid(cpuinfo, 1);
#else
    __cpuid(1, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
#endif
    
    caps.has_avx2 = (cpuinfo[1] & (1 << 5)) != 0;      // EBX bit 5
    caps.has_fma = (cpuinfo[2] & (1 << 12)) != 0;      // ECX bit 12
    caps.has_f16c = (cpuinfo[2] & (1 << 29)) != 0;     // ECX bit 29
    
#ifdef _MSC_VER
    __cpuidex(cpuinfo, 7, 0);
#else
    __cpuid_count(7, 0, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
#endif
    
    caps.has_avx512 = (cpuinfo[1] & (1 << 16)) != 0;   // EBX bit 16 (AVX-512F)
#endif
    
    return caps;
}

//==============================================================================
// RMSNorm SIMD
//==============================================================================

torch::Tensor rmsnorm_simd(const torch::Tensor& x, const torch::Tensor& weight, float eps) {
    TORCH_CHECK(x.is_contiguous(), "rmsnorm_simd: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "rmsnorm_simd: only FP32 supported");
    TORCH_CHECK(x.dim() == 3, "rmsnorm_simd: expected [B, T, D]");
    TORCH_CHECK(weight.dim() == 1, "rmsnorm_simd: weight must be 1D");
    
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t D = x.size(2);
    
    auto out = torch::empty_like(x);
    auto w = weight.contiguous();
    
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
#if defined(__AVX2__)
    if (g_simd_caps.has_avx2) {
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                const float* row = x_ptr + (b * T + t) * D;
                float* out_row = out_ptr + (b * T + t) * D;
                
                // Compute mean of squares (AVX2: 8 floats at once)
                __m256 sum_sq = _mm256_setzero_ps();
                int64_t d;
                for (d = 0; d + 8 <= D; d += 8) {
                    __m256 v = _mm256_loadu_ps(&row[d]);
                    sum_sq = _mm256_fmadd_ps(v, v, sum_sq);  // sum_sq += v * v
                }
                
                // Horizontal reduction
                float sum_sq_scalar = 0.0f;
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, sum_sq);
                for (int i = 0; i < 8; ++i) sum_sq_scalar += tmp[i];
                
                // Remaining elements (scalar)
                for (; d < D; ++d) {
                    sum_sq_scalar += row[d] * row[d];
                }
                
                // RMS
                float rms = std::sqrt(sum_sq_scalar / static_cast<float>(D) + eps);
                float inv_rms = 1.0f / rms;
                __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);
                
                // Normalize + scale by weight (AVX2)
                for (d = 0; d + 8 <= D; d += 8) {
                    __m256 v = _mm256_loadu_ps(&row[d]);
                    __m256 w_v = _mm256_loadu_ps(&w_ptr[d]);
                    __m256 normed = _mm256_mul_ps(v, inv_rms_vec);
                    __m256 scaled = _mm256_mul_ps(normed, w_v);
                    _mm256_storeu_ps(&out_row[d], scaled);
                }
                
                // Remaining elements
                for (; d < D; ++d) {
                    out_row[d] = (row[d] * inv_rms) * w_ptr[d];
                }
            }
        }
        return out;
    }
#endif
    
    // Fallback: use LibTorch native
    return torch::rms_norm(x, {weight.numel()}, weight, eps);
}

//==============================================================================
// SiLU SIMD: x * sigmoid(x) = x / (1 + exp(-x))
//==============================================================================

torch::Tensor silu_simd(const torch::Tensor& x) {
    if (!x.is_contiguous() || x.dtype() != torch::kFloat32) {
        return torch::silu(x);  // Fallback
    }
    
    auto out = torch::empty_like(x);
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = x.numel();
    
#if defined(__AVX2__)
    if (g_simd_caps.has_avx2) {
        #pragma omp parallel for
        for (int64_t i = 0; i < n; i += 8) {
            if (i + 8 <= n) {
                __m256 v = _mm256_loadu_ps(&x_ptr[i]);
                // Fast approximation: for production use libmvec or similar
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, v);
                for (int j = 0; j < 8; ++j) {
                    tmp[j] = tmp[j] / (1.0f + std::exp(-tmp[j]));
                }
                __m256 result = _mm256_load_ps(tmp);
                _mm256_storeu_ps(&out_ptr[i], result);
            } else {
                // Scalar fallback for remaining
                for (int64_t j = i; j < n; ++j) {
                    out_ptr[j] = x_ptr[j] / (1.0f + std::exp(-x_ptr[j]));
                }
            }
        }
        return out;
    }
#endif
    
    return torch::silu(x);
}

//==============================================================================
// SIMD Helpers: Fast Math
//==============================================================================

inline __m256 exp_ps_avx2(__m256 x) {
    // Basic polynomial approximation for exp(x)
    // exp(x) = lim (1 + x/n)^n, or use Taylor: 1 + x + x^2/2 + x^3/6 + ...
    // Here we use a more stable minimax approximation
    const __m256 l2e = _mm256_set1_ps(1.4426950408889634074f);
    const __m256 c1 = _mm256_set1_ps(0.693147180559945309417f);
    const __m256 c2 = _mm256_set1_ps(-0.00000000023190468138462991f);
    
    __m256 t = _mm256_mul_ps(x, l2e);
    __m256 fx = _mm256_floor_ps(_mm256_add_ps(t, _mm256_set1_ps(0.5f)));
    __m256 x_minus_fx_c1 = _mm256_sub_ps(x, _mm256_mul_ps(fx, c1));
    x = _mm256_sub_ps(x_minus_fx_c1, _mm256_mul_ps(fx, c2));
    
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 p = _mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.0001987569121f)), _mm256_set1_ps(0.0013981999507f));
    p = _mm256_add_ps(_mm256_mul_ps(p, x), _mm256_set1_ps(0.0083334519073f));
    p = _mm256_add_ps(_mm256_mul_ps(p, x), _mm256_set1_ps(0.0416657958941f));
    p = _mm256_add_ps(_mm256_mul_ps(p, x), _mm256_set1_ps(0.1666667905943f));
    p = _mm256_add_ps(_mm256_mul_ps(p, x), _mm256_set1_ps(0.5f));
    p = _mm256_add_ps(_mm256_mul_ps(p, x2), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    
    // 2^fx
    __m256i emm0 = _mm256_cvtps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2fx = _mm256_castsi256_ps(emm0);
    
    return _mm256_mul_ps(p, pow2fx);
}

inline __m256 tanh_ps_avx2(__m256 x) {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    __m256 x2 = _mm256_add_ps(x, x);
    __m256 e2x = exp_ps_avx2(x2);
    __m256 num = _mm256_sub_ps(e2x, _mm256_set1_ps(1.0f));
    __m256 den = _mm256_add_ps(e2x, _mm256_set1_ps(1.0f));
    return _mm256_div_ps(num, den);
}

//==============================================================================
// GELU SIMD: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//==============================================================================

torch::Tensor gelu_simd(const torch::Tensor& x) {
    if (!x.is_contiguous() || x.dtype() != torch::kFloat32 || !g_simd_caps.has_avx2) {
        return torch::gelu(x);
    }
    
    auto out = torch::empty_like(x);
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = x.numel();
    
    const __m256 k0 = _mm256_set1_ps(0.7978845608f); // sqrt(2/pi)
    const __m256 k1 = _mm256_set1_ps(0.044715f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 vx = _mm256_loadu_ps(&x_ptr[i]);
            // x + 0.044715 * x^3
            __m256 vx3 = _mm256_mul_ps(vx, _mm256_mul_ps(vx, vx));
            __m256 inner = _mm256_add_ps(vx, _mm256_mul_ps(k1, vx3));
            // tanh(sqrt(2/pi) * inner)
            __m256 t = tanh_ps_avx2(_mm256_mul_ps(k0, inner));
            // 0.5 * x * (1 + t)
            __m256 res = _mm256_mul_ps(half, _mm256_mul_ps(vx, _mm256_add_ps(one, t)));
            _mm256_storeu_ps(&out_ptr[i], res);
        } else {
            for (int64_t j = i; j < n; ++j) {
                float v = x_ptr[j];
                out_ptr[j] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
            }
        }
    }
    return out;
}

//==============================================================================
// Softmax SIMD
//==============================================================================

torch::Tensor softmax_simd(const torch::Tensor& x, int64_t dim) {
    if (!x.is_contiguous() || x.dtype() != torch::kFloat32 || !g_simd_caps.has_avx2 || dim != -1) {
        return torch::softmax(x, dim);
    }
    
    const int64_t D = x.size(-1);
    const int64_t N = x.numel() / D;
    
    auto out = torch::empty_like(x);
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        const float* row = x_ptr + i * D;
        float* out_row = out_ptr + i * D;
        
        // 1. Find max
        float max_val = -std::numeric_limits<float>::infinity();
        int64_t d = 0;
        __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        for (; d + 8 <= D; d += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(&row[d]));
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, vmax);
        for (int j = 0; j < 8; ++j) max_val = std::max(max_val, tmp[j]);
        for (; d < D; ++d) max_val = std::max(max_val, row[d]);
        
        // 2. Compute exp(x - max) and sum
        __m256 vmax_vec = _mm256_set1_ps(max_val);
        __m256 vsum = _mm256_setzero_ps();
        for (d = 0; d + 8 <= D; d += 8) {
            __m256 v = _mm256_loadu_ps(&row[d]);
            __m256 ve = exp_ps_avx2(_mm256_sub_ps(v, vmax_vec));
            _mm256_storeu_ps(&out_row[d], ve);
            vsum = _mm256_add_ps(vsum, ve);
        }
        float sum_val = 0.0f;
        _mm256_store_ps(tmp, vsum);
        for (int j = 0; j < 8; ++j) sum_val += tmp[j];
        for (; d < D; ++d) {
            float ve = std::exp(row[d] - max_val);
            out_row[d] = ve;
            sum_val += ve;
        }
        
        // 3. Normalize
        float inv_sum = 1.0f / sum_val;
        __m256 v_inv_sum = _mm256_set1_ps(inv_sum);
        for (d = 0; d + 8 <= D; d += 8) {
            _mm256_storeu_ps(&out_row[d], _mm256_mul_ps(_mm256_loadu_ps(&out_row[d]), v_inv_sum));
        }
        for (; d < D; ++d) out_row[d] *= inv_sum;
    }
    return out;
}

//==============================================================================
// Matrix Multiplication SIMD (small matrices only)
//==============================================================================

torch::Tensor matmul_simd(const torch::Tensor& a, const torch::Tensor& b) {
    if (!a.is_contiguous() || !b.is_contiguous() || 
        a.dtype() != torch::kFloat32 || b.dtype() != torch::kFloat32 ||
        !g_simd_caps.has_avx2) {
        return torch::matmul(a, b);
    }

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b.size(1);

    // Only optimize small matrices
    if (M > 128 || K > 128 || N > 128) {
        return torch::matmul(a, b);
    }

    auto out = torch::zeros({M, N}, torch::kFloat32);
    const float* p_a = a.data_ptr<float>();
    const float* p_b = b.data_ptr<float>();
    float* p_out = out.data_ptr<float>();

    #pragma omp parallel for collapse(2)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            __m256 vsum = _mm256_setzero_ps();
            int64_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 va = _mm256_loadu_ps(&p_a[m * K + k]);
                // This is not optimal because b is not transposed
                // For a small matmul, we'd ideally want b to be in column-major or transposed
                // But let's do a simple version for now.
                float b_vals[8];
                for (int j = 0; j < 8; ++j) b_vals[j] = p_b[(k + j) * N + n];
                __m256 vb = _mm256_loadu_ps(b_vals);
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, vsum);
            float sum = 0;
            for (int i = 0; i < 8; ++i) sum += tmp[i];
            for (; k < K; ++k) {
                sum += p_a[m * K + k] * p_b[k * N + n];
            }
            p_out[m * N + n] = sum;
        }
    }
    return out;
}

//==============================================================================
// Fused SwiGLU: silu(gate) * up
//==============================================================================

torch::Tensor fused_swiglu_simd(const torch::Tensor& gate, const torch::Tensor& up) {
    if (!gate.is_contiguous() || !up.is_contiguous() || 
        gate.dtype() != torch::kFloat32 || up.dtype() != torch::kFloat32 ||
        !g_simd_caps.has_avx2) {
        return torch::silu(gate) * up;
    }

    auto out = torch::empty_like(gate);
    const float* p_gate = gate.data_ptr<float>();
    const float* p_up = up.data_ptr<float>();
    float* p_out = out.data_ptr<float>();
    const int64_t n = gate.numel();

    const __m256 one = _mm256_set1_ps(1.0f);

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m256 vg = _mm256_loadu_ps(&p_gate[i]);
            __m256 vu = _mm256_loadu_ps(&p_up[i]);
            
            // silu(g) = g / (1 + exp(-g))
            __m256 vg_neg = _mm256_sub_ps(_mm256_setzero_ps(), vg);
            __m256 e = exp_ps_avx2(vg_neg);
            __m256 den = _mm256_add_ps(one, e);
            __m256 sig = _mm256_div_ps(one, den);
            __m256 silu_g = _mm256_mul_ps(vg, sig);
            
            _mm256_storeu_ps(&p_out[i], _mm256_mul_ps(silu_g, vu));
        } else {
            for (int64_t j = i; j < n; ++j) {
                float g = p_gate[j];
                p_out[j] = (g / (1.0f + std::exp(-g))) * p_up[j];
            }
        }
    }
    return out;
}

//==============================================================================
// Fused LayerNorm + GELU
//==============================================================================

torch::Tensor fused_layernorm_gelu_simd(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float eps
) {
    if (!x.is_contiguous() || x.dtype() != torch::kFloat32 || !g_simd_caps.has_avx2) {
        return torch::gelu(torch::layer_norm(x, {gamma.size(0)}, gamma, beta, eps));
    }

    const int64_t D = gamma.size(0);
    const int64_t N = x.numel() / D;
    
    auto out = torch::empty_like(x);
    const float* p_x = x.data_ptr<float>();
    const float* p_g = gamma.data_ptr<float>();
    const float* p_b = beta.data_ptr<float>();
    float* p_out = out.data_ptr<float>();

    const __m256 k0 = _mm256_set1_ps(0.7978845608f);
    const __m256 k1 = _mm256_set1_ps(0.044715f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);

    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        const float* row = p_x + i * D;
        float* out_row = p_out + i * D;
        
        // 1. Mean
        __m256 vsum = _mm256_setzero_ps();
        int64_t d = 0;
        for (; d + 8 <= D; d += 8) {
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(&row[d]));
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, vsum);
        float sum_val = 0;
        for (int j = 0; j < 8; ++j) sum_val += tmp[j];
        for (; d < D; ++d) sum_val += row[d];
        float mean = sum_val / D;
        
        // 2. Variance
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vsum_sq = _mm256_setzero_ps();
        for (d = 0; d + 8 <= D; d += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(&row[d]), vmean);
            vsum_sq = _mm256_fmadd_ps(v, v, vsum_sq);
        }
        _mm256_store_ps(tmp, vsum_sq);
        float sum_sq = 0;
        for (int j = 0; j < 8; ++j) sum_sq += tmp[j];
        for (; d < D; ++d) {
            float diff = row[d] - mean;
            sum_sq += diff * diff;
        }
        float inv_std = 1.0f / std::sqrt(sum_sq / D + eps);
        __m256 v_inv_std = _mm256_set1_ps(inv_std);
        
        // 3. Normalize + Gamma/Beta + GELU
        for (d = 0; d + 8 <= D; d += 8) {
            __m256 vx = _mm256_loadu_ps(&row[d]);
            __m256 vg = _mm256_loadu_ps(&p_g[d]);
            __m256 vb = _mm256_loadu_ps(&p_b[d]);
            
            // Normalize & Scale
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(vx, vmean), v_inv_std);
            __m256 res = _mm256_add_ps(_mm256_mul_ps(norm, vg), vb);
            
            // GELU
            __m256 res3 = _mm256_mul_ps(res, _mm256_mul_ps(res, res));
            __m256 inner = _mm256_add_ps(res, _mm256_mul_ps(k1, res3));
            __m256 t = tanh_ps_avx2(_mm256_mul_ps(k0, inner));
            __m256 final_res = _mm256_mul_ps(half, _mm256_mul_ps(res, _mm256_add_ps(one, t)));
            
            _mm256_storeu_ps(&out_row[d], final_res);
        }
        for (; d < D; ++d) {
            float res = (row[d] - mean) * inv_std * p_g[d] + p_b[d];
            out_row[d] = 0.5f * res * (1.0f + std::tanh(0.7978845608f * (res + 0.044715f * res * res * res)));
        }
    }
    return out;
}

} // namespace optimized
} // namespace xorzen

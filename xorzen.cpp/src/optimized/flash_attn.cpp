// ============================================================
//  xorzen.cpp — src/optimized/flash_attn.cpp
//  Flash Attention for CPU (optimized tiling)
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/optimized/flash_attn.h"
#include "xorzen/optimized/thread_pool.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <immintrin.h>

namespace xorzen {
namespace optimized {

#ifdef XORZEN_ENABLE_FLASH_ATTN

torch::Tensor flash_attention_cpu(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& attn_mask,
    bool is_causal,
    double scale_factor) 
{
    TORCH_CHECK(q.dim() == 4, "q must be [B, H, T, D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B, H, T, D]");
    TORCH_CHECK(v.dim() == 4, "v must be [B, H, T, D]");
    
    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t Tq = q.size(2);
    const int64_t Tk = k.size(2);
    const int64_t D = q.size(3);
    
    if (scale_factor == 0.0) {
        scale_factor = 1.0 / std::sqrt(static_cast<double>(D));
    }

    (void)attn_mask;

    auto out = torch::empty_like(q);
    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto vc = v.contiguous();

    const float* q_ptr = qc.data_ptr<float>();
    const float* k_ptr = kc.data_ptr<float>();
    const float* v_ptr = vc.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    #pragma omp parallel for collapse(3)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t tq = 0; tq < Tq; ++tq) {
                const float* q_vec = q_ptr + ((b * H + h) * Tq + tq) * D;
                float* out_vec = out_ptr + ((b * H + h) * Tq + tq) * D;

                std::vector<float> scores(static_cast<size_t>(Tk));
                float max_score = -std::numeric_limits<float>::infinity();

                for (int64_t tk = 0; tk < Tk; ++tk) {
                    const float* k_vec = k_ptr + ((b * H + h) * Tk + tk) * D;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        dot += q_vec[d] * k_vec[d];
                    }
                    if (is_causal && tk > tq) {
                        scores[static_cast<size_t>(tk)] = -std::numeric_limits<float>::infinity();
                    } else {
                        scores[static_cast<size_t>(tk)] = dot * static_cast<float>(scale_factor);
                    }
                    max_score = std::max(max_score, scores[static_cast<size_t>(tk)]);
                }

                float sum_exp = 0.0f;
                for (int64_t tk = 0; tk < Tk; ++tk) {
                    auto& s = scores[static_cast<size_t>(tk)];
                    if (std::isinf(s) && s < 0.0f) {
                        s = 0.0f;
                        continue;
                    }
                    s = std::exp(s - max_score);
                    sum_exp += s;
                }
                const float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
                std::fill(out_vec, out_vec + D, 0.0f);

                for (int64_t tk = 0; tk < Tk; ++tk) {
                    const float p = scores[static_cast<size_t>(tk)] * inv_sum;
                    if (p == 0.0f) continue;
                    const float* v_vec = v_ptr + ((b * H + h) * Tk + tk) * D;
                    for (int64_t d = 0; d < D; ++d) {
                        out_vec[d] += p * v_vec[d];
                    }
                }
            }
        }
    }

    return out;
}

torch::Tensor flash_decoding_cpu(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double scale_factor) 
{
    // Decoding: q is [B, H, 1, D]
    // k, v are [B, H, T_past, D]
    
    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t D = q.size(3);
    const int64_t T = k.size(2);
    
    if (scale_factor == 0.0) {
        scale_factor = 1.0 / std::sqrt(static_cast<double>(D));
    }

    auto out = torch::empty_like(q);
    
    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto vc = v.contiguous();
    
    const float* q_ptr = qc.data_ptr<float>();
    const float* k_ptr = kc.data_ptr<float>();
    const float* v_ptr = vc.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const float* q_vec = q_ptr + (b * H + h) * D;
            float* out_vec = out_ptr + (b * H + h) * D;
            
            // 1. Compute scores: s = q @ K.T * scale
            std::vector<float> scores(T);
            float max_score = -std::numeric_limits<float>::infinity();
            
            for (int64_t t = 0; t < T; ++t) {
                const float* k_vec = k_ptr + ((b * H + h) * T + t) * D;
                float dot = 0.0f;
                
                // SIMD dot product
                int64_t d = 0;
                #if defined(__AVX2__)
                __m256 vsum = _mm256_setzero_ps();
                for (; d + 8 <= D; d += 8) {
                    __m256 vq = _mm256_loadu_ps(&q_vec[d]);
                    __m256 vk = _mm256_loadu_ps(&k_vec[d]);
                    vsum = _mm256_fmadd_ps(vq, vk, vsum);
                }
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, vsum);
                for (int i = 0; i < 8; ++i) dot += tmp[i];
                #endif
                
                for (; d < D; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                
                scores[t] = dot * static_cast<float>(scale_factor);
                if (scores[t] > max_score) max_score = scores[t];
            }
            
            // 2. Softmax: p = exp(s - max_s) / sum(exp(s - max_s))
            float sum_exp = 0.0f;
            for (int64_t t = 0; t < T; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int64_t t = 0; t < T; ++t) {
                scores[t] *= inv_sum;
            }
            
            // 3. Output: o = p @ V
            std::fill(out_vec, out_vec + D, 0.0f);
            for (int64_t t = 0; t < T; ++t) {
                const float* v_vec = v_ptr + ((b * H + h) * T + t) * D;
                float p = scores[t];
                
                int64_t d = 0;
                #if defined(__AVX2__)
                __m256 vp = _mm256_set1_ps(p);
                for (; d + 8 <= D; d += 8) {
                    __m256 vv = _mm256_loadu_ps(&v_vec[d]);
                    __m256 vo = _mm256_loadu_ps(&out_vec[d]);
                    vo = _mm256_fmadd_ps(vp, vv, vo);
                    _mm256_storeu_ps(&out_vec[d], vo);
                }
                #endif
                
                for (; d < D; ++d) {
                    out_vec[d] += p * v_vec[d];
                }
            }
        }
    }
    
    return out;
}

torch::Tensor flash_gqa_cpu(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool is_causal,
    double scale_factor) 
{
    const int64_t Hq = q.size(1);
    const int64_t Hkv = k.size(1);
    
    if (Hq == Hkv) {
        return flash_attention_cpu(q, k, v, {}, is_causal, scale_factor);
    }
    
    // GQA: Expand K, V heads to match Q
    const int64_t group_size = Hq / Hkv;
    
    auto ke = k.repeat_interleave(group_size, 1);
    auto ve = v.repeat_interleave(group_size, 1);
    
    return flash_attention_cpu(q, ke, ve, {}, is_causal, scale_factor);
}

#endif // XORZEN_ENABLE_FLASH_ATTN

} // namespace optimized
} // namespace xorzen

#include "xorzen/ops.h"
#include "xorzen/optimized/simd_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xorzen {

torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& weight, double eps) {
    return optimized::rmsnorm_simd(x, weight, static_cast<float>(eps));
}

torch::Tensor silu(const torch::Tensor& x) {
    return optimized::silu_simd(x);
}

torch::Tensor top_k_filtering(const torch::Tensor& logits, int64_t top_k) {
    if (top_k <= 0 || top_k >= logits.size(-1)) return logits;
    auto values = std::get<0>(torch::topk(logits, top_k, -1));
    auto threshold = values.select(-1, top_k - 1).unsqueeze(-1);
    return torch::where(logits < threshold,
                        torch::full_like(logits, -std::numeric_limits<float>::infinity()),
                        logits);
}

torch::Tensor top_p_filtering(const torch::Tensor& logits, double top_p) {
    if (top_p <= 0.0 || top_p >= 1.0) return logits;
    auto sorted = torch::sort(logits, -1, true);
    auto sorted_logits = std::get<0>(sorted);
    auto sorted_indices = std::get<1>(sorted);
    auto cumulative = optimized::softmax_simd(sorted_logits, -1).cumsum(-1);
    auto remove = cumulative > top_p;
    remove.index_put_({"...", 0}, false);
    auto filtered_sorted = torch::where(remove,
        torch::full_like(sorted_logits, -std::numeric_limits<float>::infinity()),
        sorted_logits);
    auto result = torch::empty_like(logits);
    return result.scatter(-1, sorted_indices, filtered_sorted);
}

torch::Tensor sample_next_token(const torch::Tensor& logits, const GenerationConfig& config) {
    auto next_logits = logits / std::max(config.temperature, 1e-6f);
    next_logits = top_k_filtering(next_logits, config.top_k);
    next_logits = top_p_filtering(next_logits, config.top_p);
    if (!config.do_sample) {
        return std::get<1>(next_logits.max(-1, false));
    }
    auto probs = optimized::softmax_simd(next_logits, -1);
    return torch::multinomial(probs, 1).squeeze(-1);
}

torch::Tensor make_local_causal_mask(int64_t seq_len, int64_t window, torch::Device device) {
    auto positions = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kLong).device(device));
    auto i = positions.unsqueeze(0);
    auto j = positions.unsqueeze(1);
    auto causal = i <= j;
    if (window > 0) {
        auto dist = (i - j).abs();
        causal = causal.logical_and(dist <= window);
    }
    return causal;
}

RMSNormImpl::RMSNormImpl(int64_t dim, double eps) : eps(eps) {
    weight = register_parameter("weight", torch::ones({dim}));
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& x) {
    return rms_norm(x, weight, eps);
}

void xavier_linear(torch::nn::Linear& linear, double gain) {
    torch::nn::init::xavier_uniform_(linear->weight, gain);
    if (linear->bias.defined()) {
        torch::nn::init::zeros_(linear->bias);
    }
}

void ModelConfig::print() const {
    std::cout << "XORZEN.CPP config: vocab=" << vocab_size
              << " hidden=" << hidden_size
              << " layers=" << num_layers
              << " heads=" << num_attention_heads
              << " experts=" << expert_count
              << " top_k=" << top_k_experts
              << " context=" << context_length
              << " device=" << device << std::endl;
}

} // namespace xorzen

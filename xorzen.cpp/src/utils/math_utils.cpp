// ============================================================
//  xorzen.cpp — src/utils/math_utils.cpp
//  Full implementation of mathematical utilities
//  Ported from xorzen/utils/math_utils.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/math_utils.h"

#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_svd.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace xorzen {

// ============================================================
//  TensorStability
// ============================================================

torch::Tensor TensorStability::safe_softmax(const torch::Tensor& x,
                                            int64_t dim,
                                            double eps) {
    auto x_max = std::get<0>(x.max(dim, /*keepdim=*/true));
    auto x_stable = x - x_max;
    auto exp_x = torch::exp(x_stable);
    auto sum_exp = exp_x.sum(dim, /*keepdim=*/true).clamp_min(eps);
    return exp_x / sum_exp;
}

torch::Tensor TensorStability::safe_log_softmax(const torch::Tensor& x,
                                                int64_t dim,
                                                double eps) {
    auto x_max = std::get<0>(x.max(dim, /*keepdim=*/true));
    auto x_stable = x - x_max;
    auto log_sum = torch::log(torch::exp(x_stable).sum(dim, /*keepdim=*/true).clamp_min(eps));
    return x_stable - log_sum;
}

torch::Tensor TensorStability::logsumexp(const torch::Tensor& x,
                                         int64_t dim,
                                         bool keepdim) {
    auto x_max = std::get<0>(x.max(dim, /*keepdim=*/true));
    auto shifted = x - x_max;
    auto result = x_max + torch::log(torch::exp(shifted).sum(dim, keepdim).clamp_min(1e-12));
    if (!keepdim) {
        // x_max was keepdim=true, result already matches since we add then squeeze when needed
    }
    return result;
}

std::pair<torch::Tensor, torch::Tensor>
TensorStability::complex_softmax(const torch::Tensor& x_real,
                                 const torch::Tensor& x_imag,
                                 int64_t dim) {
    // Born-rule: softmax on magnitudes, then re-apply phases
    auto magnitude = torch::sqrt(x_real.pow(2) + x_imag.pow(2) + 1e-12f);
    auto mag_probs = torch::softmax(magnitude, dim);
    auto phases = torch::atan2(x_imag, x_real);
    auto real_prob = mag_probs * torch::cos(phases);
    auto imag_prob = mag_probs * torch::sin(phases);
    return {real_prob, imag_prob};
}

torch::Tensor TensorStability::unitary_project(const torch::Tensor& x) {
    // Nearest unitary via SVD: U @ Vh
    auto [U, S, Vh] = at::linalg_svd(x, /*full_matrices=*/true);
    return torch::matmul(U, Vh);
}

torch::Tensor TensorStability::symmetric_expm(const torch::Tensor& x) {
    // Symmetrise, eigen-decompose, exponentiate eigenvalues, reconstruct
    auto x_sym = (x + x.t()) / 2.0;
    auto [eigvals, eigvecs] = at::linalg_eigh(x_sym);
    auto exp_eigvals = torch::exp(eigvals);
    return torch::matmul(eigvecs, torch::matmul(torch::diag(exp_eigvals), eigvecs.t()));
}

torch::Tensor TensorStability::hadamard_product_normalized(const torch::Tensor& a,
                                                           const torch::Tensor& b,
                                                           double eps) {
    auto product = a * b;
    auto norm = torch::norm(product, 2, -1, /*keepdim=*/true).clamp_min(eps);
    return product / norm;
}

// ============================================================
//  InformationTheory
// ============================================================

torch::Tensor InformationTheory::shannon_entropy(const torch::Tensor& probs,
                                                  double eps) {
    auto p = probs / probs.sum(-1, /*keepdim=*/true).clamp_min(eps);
    auto log_p = torch::log(p.clamp_min(eps));
    return -(p * log_p).sum(-1);
}

torch::Tensor InformationTheory::kl_divergence(const torch::Tensor& p,
                                                const torch::Tensor& q,
                                                double eps) {
    auto p_n = p / p.sum(-1, true).clamp_min(eps);
    auto q_n = q / q.sum(-1, true).clamp_min(eps);
    return (p_n * torch::log(p_n.clamp_min(eps) / q_n.clamp_min(eps))).sum(-1);
}

torch::Tensor InformationTheory::mutual_information(const torch::Tensor& joint,
                                                    double eps) {
    auto p_x = joint.sum(1).clamp_min(eps);       // marginal over Y
    auto p_y = joint.sum(0).clamp_min(eps);       // marginal over X
    auto p_xy = p_x.unsqueeze(1) * p_y.unsqueeze(0); // outer product
    return (joint * torch::log(joint.clamp_min(eps) / p_xy.clamp_min(eps))).sum();
}

double InformationTheory::compression_ratio(double original_bits, double compressed_bits) {
    if (compressed_bits <= 0.0) return 0.0;
    double ratio = original_bits / compressed_bits;
    // Theoretical Shannon bound
    double theoretical_max = original_bits / (original_bits * std::log2(std::exp(1.0)));
    return std::min(ratio, theoretical_max);
}

torch::Tensor InformationTheory::token_complexity_score(const torch::Tensor& embeddings,
                                                        const std::string& method) {
    // embeddings: [B, T, D]
    const int64_t B = embeddings.size(0);
    const int64_t T = embeddings.size(1);
    const int64_t D = embeddings.size(2);

    if (method == "entropy") {
        auto flat = embeddings.view({B * T, D});
        auto probs = torch::softmax(flat, -1);
        auto entropy = -(probs * torch::log(probs + 1e-12f)).sum(-1);
        return entropy.view({B, T});
    } else if (method == "variance") {
        return embeddings.var(-1); // [B, T]
    } else if (method == "norm") {
        return torch::norm(embeddings, 2, -1); // [B, T]
    } else {
        throw std::invalid_argument("Unknown complexity method: " + method);
    }
}

double InformationTheory::compute_effective_params(int64_t total_params,
                                                   double active_ratio,
                                                   double specialization_factor,
                                                   double data_quality_factor) {
    double active = static_cast<double>(total_params) * active_ratio;
    double spec_gain  = 1.0 + std::log(std::max(specialization_factor, 1.0));
    double data_gain  = std::sqrt(std::max(data_quality_factor, 1.0));
    double total_gain = spec_gain * data_gain;
    double effective  = active * total_gain;
    // Shannon upper bound
    double shannon_bound = static_cast<double>(total_params) *
        std::log(static_cast<double>(total_params)) / std::log(2.0);
    return std::min(effective, shannon_bound);
}

// ============================================================
//  QuantizationMathematics
// ============================================================

std::pair<double, torch::Tensor>
QuantizationMathematics::compute_quantization_error(const torch::Tensor& weights,
                                                    int bits,
                                                    bool symmetric) {
    torch::Tensor quantized;
    if (symmetric) {
        auto abs_max = weights.abs().max();
        auto scale   = abs_max / static_cast<double>((1 << (bits - 1)) - 1);
        quantized    = torch::round(weights / scale) * scale;
    } else {
        auto w_min = weights.min();
        auto w_max = weights.max();
        auto scale = (w_max - w_min) / static_cast<double>((1 << bits) - 1);
        quantized   = torch::round((weights - w_min) / scale) * scale + w_min;
    }
    double mse = weights.sub(quantized).pow(2).mean().item<double>();
    return {mse, quantized};
}

double QuantizationMathematics::compute_parameter_stability(
    const std::vector<torch::Tensor>& weight_history,
    int64_t window_size) {
    if (weight_history.size() < 2) return 0.0;
    int64_t n = static_cast<int64_t>(weight_history.size());
    int64_t start = std::max<int64_t>(0, n - window_size);
    std::vector<torch::Tensor> recent(weight_history.begin() + start, weight_history.end());
    auto stacked     = torch::stack(recent, 0);           // [W, ...]
    auto variance    = stacked.var(0).mean();
    auto avg_mag     = stacked.abs().mean();
    auto stability   = 1.0 / (1.0 + variance / (avg_mag + 1e-12f));
    return stability.item<double>();
}

int QuantizationMathematics::optimal_quantization_schedule(int64_t step,
                                                           int64_t total_steps,
                                                           int initial_bits,
                                                           int final_bits,
                                                           const std::string& method) {
    if (total_steps <= 0) return final_bits;
    double progress = std::min(1.0, static_cast<double>(step) / static_cast<double>(total_steps));
    double t = 0.0;
    if (method == "cosine") {
        t = 0.5 * (1.0 - std::cos(progress * M_PI));
    } else if (method == "linear") {
        t = progress;
    } else if (method == "step") {
        t = (progress >= 0.5) ? 1.0 : 0.0;
    } else {
        throw std::invalid_argument("Unknown quantization schedule: " + method);
    }
    double bits = initial_bits + t * (final_bits - initial_bits);
    return std::max(final_bits, std::min(initial_bits, static_cast<int>(std::round(bits))));
}

torch::Tensor QuantizationMathematics::compute_qat_scale(const torch::Tensor& weights,
                                                         int bits) {
    auto abs_max = weights.abs().max();
    auto scale   = abs_max / static_cast<double>((1 << (bits - 1)) - 1);
    return scale;
}

std::vector<int> QuantizationMathematics::assign_mixed_precision(
    const std::vector<double>& layer_sensitivities,
    int min_bits,
    int max_bits) {
    if (layer_sensitivities.empty()) return {};
    double s_min = *std::min_element(layer_sensitivities.begin(), layer_sensitivities.end());
    double s_max = *std::max_element(layer_sensitivities.begin(), layer_sensitivities.end());
    double s_range = s_max - s_min;
    std::vector<int> bit_widths;
    bit_widths.reserve(layer_sensitivities.size());
    for (double s : layer_sensitivities) {
        double t = (s_range > 1e-8) ? (s - s_min) / s_range : 0.5;
        // High sensitivity → more bits; low sensitivity → fewer bits
        int bits = static_cast<int>(std::round(min_bits + t * (max_bits - min_bits)));
        bits = std::clamp(bits, min_bits, max_bits);
        bit_widths.push_back(bits);
    }
    return bit_widths;
}

// ============================================================
//  RoutingMathematics
// ============================================================

std::pair<double, double>
RoutingMathematics::compute_router_accuracy_bounds(int64_t hidden_dim,
                                                   int64_t num_classes,
                                                   int64_t training_tokens) {
    if (training_tokens <= 0 || num_classes <= 0) return {0.0, 1.0};
    // VC dimension approximation for MLP router
    double vc_dim   = static_cast<double>(hidden_dim * num_classes) *
                      std::log(static_cast<double>(hidden_dim));
    double gen_err  = std::sqrt((vc_dim * std::log(static_cast<double>(training_tokens))) /
                                static_cast<double>(training_tokens));
    double bayes    = 1.0 - 1.0 / static_cast<double>(num_classes);
    double random_a = 1.0 / static_cast<double>(num_classes);
    double expected = bayes - gen_err;
    return {std::max(random_a, expected), bayes};
}

torch::Tensor RoutingMathematics::load_balancing_loss(const torch::Tensor& expert_gates,
                                                       double importance_weight) {
    // expert_gates: [B,T,E] or [N,E]
    torch::Tensor gates;
    if (expert_gates.dim() == 3) {
        gates = expert_gates.view({-1, expert_gates.size(2)});
    } else if (expert_gates.dim() == 2) {
        gates = expert_gates;
    } else {
        throw std::invalid_argument("expert_gates must be 2D or 3D");
    }
    auto load       = gates.sum(0);                      // [E]
    auto importance = gates.pow(2).sum(0);               // [E]
    auto load_mean  = load.mean();
    auto load_std   = load.std();
    auto cv_loss    = load_std / (load_mean + 1e-12f);
    auto imp_mean   = importance.mean();
    auto imp_std    = importance.std();
    auto imp_loss   = imp_std / (imp_mean + 1e-12f);
    return cv_loss + static_cast<float>(importance_weight) * imp_loss;
}

std::unordered_map<std::string, double>
RoutingMathematics::compute_routing_efficiency(const torch::Tensor& depth_mask,
                                               const torch::Tensor& width_mask,
                                               double theoretical_min) {
    // depth_mask: [B, T, max_depth]
    // width_mask: [B, T]
    const int64_t B   = depth_mask.size(0);
    const int64_t T   = depth_mask.size(1);
    const int64_t D   = depth_mask.size(2);
    auto active_layers = depth_mask.sum(-1).to(torch::kFloat).mean().item<double>();
    auto avg_width     = width_mask.to(torch::kFloat).mean().item<double>();
    auto max_width     = width_mask.to(torch::kFloat).max().item<double>();
    double width_eff   = (max_width > 1e-8) ? avg_width / max_width : 0.0;
    double total_comp  = depth_mask.to(torch::kFloat).sum().item<double>() * avg_width;
    double max_comp    = static_cast<double>(B * T * D) * max_width;
    double efficiency  = (max_comp > 1e-8) ? 1.0 - total_comp / max_comp : 0.0;
    double opt_gap     = std::max(0.0, efficiency - theoretical_min);
    return {
        {"active_layers_per_token", active_layers},
        {"width_efficiency",        width_eff},
        {"overall_efficiency",      efficiency},
        {"optimality_gap",          opt_gap},
        {"total_compute",           total_comp},
        {"max_compute",             max_comp},
    };
}

} // namespace xorzen

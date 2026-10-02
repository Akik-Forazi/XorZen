#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/math_utils.h
//  Mathematical utilities: TensorStability, InformationTheory,
//  QuantizationMathematics, RoutingMathematics
//  Ported from xorzen/utils/math_utils.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================

#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>

namespace xorzen {

// ============================================================
//  TensorStability
//  Numerically stable tensor operations
// ============================================================
struct TensorStability {
    // Numerically stable softmax: subtract max, clamp denominator
    static torch::Tensor safe_softmax(const torch::Tensor& x,
                                      int64_t dim = -1,
                                      double eps = 1e-12);

    // Numerically stable log-softmax
    static torch::Tensor safe_log_softmax(const torch::Tensor& x,
                                          int64_t dim = -1,
                                          double eps = 1e-12);

    // Stable log-sum-exp
    static torch::Tensor logsumexp(const torch::Tensor& x,
                                   int64_t dim = -1,
                                   bool keepdim = false);

    // Born-rule normalisation for complex-valued inputs
    // Returns (real_prob, imag_prob)
    static std::pair<torch::Tensor, torch::Tensor>
    complex_softmax(const torch::Tensor& x_real,
                    const torch::Tensor& x_imag,
                    int64_t dim = -1);

    // Project a square matrix to its nearest unitary via SVD
    static torch::Tensor unitary_project(const torch::Tensor& x);

    // Matrix exponential for symmetric matrices via eigen-decomposition
    static torch::Tensor symmetric_expm(const torch::Tensor& x);

    // Normalised element-wise product along last dimension
    static torch::Tensor hadamard_product_normalized(const torch::Tensor& a,
                                                     const torch::Tensor& b,
                                                     double eps = 1e-8);
};

// ============================================================
//  InformationTheory
// ============================================================
struct InformationTheory {
    // Shannon entropy: H(p) = -Σ p_i log p_i
    static torch::Tensor shannon_entropy(const torch::Tensor& probs,
                                         double eps = 1e-12);

    // KL divergence: D_KL(p || q)
    static torch::Tensor kl_divergence(const torch::Tensor& p,
                                       const torch::Tensor& q,
                                       double eps = 1e-12);

    // Mutual information given a joint distribution [X, Y]
    static torch::Tensor mutual_information(const torch::Tensor& joint,
                                            double eps = 1e-12);

    // Shannon source-coding compression ratio bound
    static double compression_ratio(double original_bits, double compressed_bits);

    // Token complexity score from embeddings [B, T, D]
    // method: "entropy" | "variance" | "norm"
    static torch::Tensor token_complexity_score(const torch::Tensor& embeddings,
                                                const std::string& method = "entropy");

    // Effective parameter count accounting for MoE specialisation
    static double compute_effective_params(int64_t total_params,
                                           double active_ratio,
                                           double specialization_factor = 1.0,
                                           double data_quality_factor = 1.0);
};

// ============================================================
//  QuantizationMathematics
// ============================================================
struct QuantizationMathematics {
    // Compute MSE error from symmetric/asymmetric quantisation
    // Returns {mse_error, quantized_weights}
    static std::pair<double, torch::Tensor>
    compute_quantization_error(const torch::Tensor& weights,
                               int bits,
                               bool symmetric = true);

    // Stability score over a history of weight tensors
    static double compute_parameter_stability(
        const std::vector<torch::Tensor>& weight_history,
        int64_t window_size = 100);

    // Bit-width schedule at a given training step
    // method: "cosine" | "linear" | "step"
    static int optimal_quantization_schedule(int64_t training_step,
                                             int64_t total_steps,
                                             int initial_bits = 32,
                                             int final_bits = 4,
                                             const std::string& method = "cosine");

    // Quantisation-aware training scale factor
    static torch::Tensor compute_qat_scale(const torch::Tensor& weights, int bits);

    // Mixed-precision layer assignments given sensitivity scores
    // Returns per-layer bit-widths
    static std::vector<int> assign_mixed_precision(
        const std::vector<double>& layer_sensitivities,
        int min_bits = 4,
        int max_bits = 16);
};

// ============================================================
//  RoutingMathematics
// ============================================================
struct RoutingMathematics {
    // PAC-learning router accuracy bounds
    // Returns {lower_bound, upper_bound}
    static std::pair<double, double>
    compute_router_accuracy_bounds(int64_t hidden_dim,
                                   int64_t num_classes,
                                   int64_t training_tokens);

    // MoE load-balancing loss (coefficient of variation)
    static torch::Tensor load_balancing_loss(const torch::Tensor& expert_gates,
                                             double importance_weight = 0.01);

    // Routing efficiency metrics given depth/width masks
    static std::unordered_map<std::string, double>
    compute_routing_efficiency(const torch::Tensor& depth_mask,
                               const torch::Tensor& width_mask,
                               double theoretical_min = 0.02);
};

} // namespace xorzen

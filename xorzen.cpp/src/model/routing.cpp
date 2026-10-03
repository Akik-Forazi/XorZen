#include "xorzen/routing.h"
#include "xorzen/ops.h"
#include "xorzen/optimized/simd_ops.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace xorzen {

torch::Tensor load_balance_loss(const torch::Tensor& expert_probs,
                                const torch::Tensor&,
                                int64_t num_experts) {
    auto importance = expert_probs.mean(0).mean(0);
    auto target = torch::full_like(importance, 1.0 / static_cast<double>(num_experts));
    return (importance - target).pow(2).sum();
}

torch::Tensor router_z_loss(const torch::Tensor& router_logits) {
    return torch::logsumexp(router_logits, -1).pow(2).mean();
}

torch::Tensor path_diversity_loss(const torch::Tensor& path_probs) {
    auto avg = path_probs.mean(0).mean(0);
    auto target = torch::full_like(avg, 1.0 / static_cast<double>(path_probs.size(-1)));
    return (avg - target).pow(2).sum();
}

AdaptiveRouterImpl::AdaptiveRouterImpl(ModelConfig cfg)
    : config(std::move(cfg)) {
    config.normalize();
    hidden_dim = config.hidden_size;
    cot_dim = config.cot_total_dim();
    input_dim = hidden_dim + cot_dim;
    max_depth = config.max_depth;
    num_widths = static_cast<int64_t>(config.width_choices.size());
    num_paths = 3;
    num_experts = config.expert_count;
    top_k = config.top_k_experts;
    temperature = config.router_temperature;
    temperature_annealing = config.router_temperature_annealing;
    build_network();
    init_weights();
}

void AdaptiveRouterImpl::build_network() {
    const int64_t h = config.router_hidden_dim;
    const int64_t enc1 = std::max<int64_t>(128, h * 4);
    const int64_t enc2 = std::max<int64_t>(64, h * 2);
    const int64_t enc3 = std::max<int64_t>(32, h);
    const int64_t head = std::max<int64_t>(32, h / 2);

    feature_encoder = register_module("feature_encoder", torch::nn::Sequential(
        torch::nn::Linear(input_dim, enc1), torch::nn::LayerNorm(torch::nn::LayerNormOptions({enc1})), torch::nn::GELU(),
        torch::nn::Dropout(config.router_dropout),
        torch::nn::Linear(enc1, enc2), torch::nn::LayerNorm(torch::nn::LayerNormOptions({enc2})), torch::nn::GELU(),
        torch::nn::Dropout(config.router_dropout),
        torch::nn::Linear(enc2, enc3), torch::nn::LayerNorm(torch::nn::LayerNormOptions({enc3})), torch::nn::GELU()));

    depth_router = register_module("depth_router", torch::nn::Sequential(
        torch::nn::Linear(enc3, head), torch::nn::LayerNorm(torch::nn::LayerNormOptions({head})), torch::nn::GELU(),
        torch::nn::Linear(head, max_depth)));
    width_router = register_module("width_router", torch::nn::Sequential(
        torch::nn::Linear(enc3, head), torch::nn::LayerNorm(torch::nn::LayerNormOptions({head})), torch::nn::GELU(),
        torch::nn::Linear(head, num_widths)));
    path_router = register_module("path_router", torch::nn::Sequential(
        torch::nn::Linear(enc3, head), torch::nn::LayerNorm(torch::nn::LayerNormOptions({head})), torch::nn::GELU(),
        torch::nn::Linear(head, num_paths)));
    expert_router = register_module("expert_router", torch::nn::Sequential(
        torch::nn::Linear(enc3, enc3), torch::nn::LayerNorm(torch::nn::LayerNormOptions({enc3})), torch::nn::GELU(),
        torch::nn::Linear(enc3, num_experts)));
    character_router = register_module("character_router", torch::nn::Sequential(
        torch::nn::Linear(enc3, head), torch::nn::LayerNorm(torch::nn::LayerNormOptions({head})), torch::nn::GELU(),
        torch::nn::Linear(head, config.max_characters),
        torch::nn::Sigmoid()));
    complexity_estimator = register_module("complexity_estimator", torch::nn::Sequential(
        torch::nn::Linear(enc3, head), torch::nn::LayerNorm(torch::nn::LayerNormOptions({head})), torch::nn::GELU(),
        torch::nn::Linear(head, 1), torch::nn::Sigmoid()));
    uncertainty_estimator = register_module("uncertainty_estimator", torch::nn::Sequential(
        torch::nn::Linear(enc3, head), torch::nn::LayerNorm(torch::nn::LayerNormOptions({head})), torch::nn::GELU(),
        torch::nn::Linear(head, 1), torch::nn::Sigmoid()));

    width_values = torch::empty({num_widths}, torch::kFloat32);
    for (int64_t i = 0; i < num_widths; ++i) {
        width_values[i] = static_cast<float>(config.width_choices[static_cast<size_t>(i)]);
    }
    width_values = register_buffer("width_values", width_values);
}

void AdaptiveRouterImpl::init_weights() {
    for (auto& module : modules(false)) {
        if (auto* linear = module->as<torch::nn::Linear>()) {
            torch::nn::init::xavier_uniform_(linear->weight, 0.5);
            if (linear->bias.defined()) torch::nn::init::zeros_(linear->bias);
        }
    }
}

void AdaptiveRouterImpl::load_weights(const std::string& weights_dir) {
    torch::NoGradGuard no_grad;
    for (auto& pair : named_parameters()) {
        std::string name = pair.key();
        auto& param = pair.value();
        
        std::string filename = name;
        std::replace(filename.begin(), filename.end(), '.', '_');
        std::string path = weights_dir + "/" + filename + ".bin";
        
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            auto t = torch::empty_like(param);
            file.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
            param.copy_(t);
        } else {
            std::cerr << "Warning: Parameter file " << path << " not found." << std::endl;
        }
    }
}

RoutingDecision AdaptiveRouterImpl::forward(const torch::Tensor& x,
                                            const torch::Tensor& cot_features,
                                            bool deterministic,
                                            int64_t expert_capacity) {
    if (!is_training()) deterministic = true;
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    auto router_input = torch::cat({x, cot_features}, -1);
    auto flat = router_input.view({B * T, input_dim});
    auto features = feature_encoder->forward(flat).view({B, T, -1});

    auto depth_logits = depth_router->forward(features);
    auto width_logits = width_router->forward(features);
    auto path_logits = path_router->forward(features);
    auto expert_logits = expert_router->forward(features);
    auto character_probs = character_router->forward(features);
    auto complexity = complexity_estimator->forward(features);
    auto uncertainty = uncertainty_estimator->forward(features);

    double temp = temperature;
    if (temperature_annealing && is_training()) {
        temp = std::max(0.1, temperature * std::pow(0.99, static_cast<double>(training_step) / 1000.0));
    }

    auto [depth_probs, depth_mask] = route_depth(depth_logits, complexity, temp, deterministic);
    auto [width_probs, width_idx, width_multiplier] = route_width(width_logits, complexity, temp, deterministic);
    auto path_probs = route_path(path_logits, temp, deterministic);
    auto [expert_probs, expert_indices, expert_weights] = route_experts(expert_logits, temp, deterministic, expert_capacity);

    RoutingDecision decision;
    decision.depth_logits = depth_logits;
    decision.depth_probs = depth_probs;
    decision.depth_mask = depth_mask;
    decision.width_logits = width_logits;
    decision.width_probs = width_probs;
    decision.width_idx = width_idx;
    decision.width_multiplier = width_multiplier;
    decision.path_logits = path_logits;
    decision.path_probs = path_probs;
    decision.expert_logits = expert_logits;
    decision.expert_probs = expert_probs;
    decision.expert_indices = expert_indices;
    decision.expert_weights = expert_weights;
    decision.character_probs = character_probs;
    decision.complexity = complexity;
    decision.uncertainty = uncertainty;
    decision.auxiliary["temperature"] = torch::tensor(temp, x.options());
    if (is_training() && !deterministic) {
        decision.auxiliary["load_balance_loss"] = 0.0001 * load_balance_loss(expert_probs, expert_indices, num_experts);
        decision.auxiliary["router_z_loss"] = 0.0001 * router_z_loss(expert_logits);
        decision.auxiliary["path_div_loss"] = 0.02 * path_diversity_loss(path_probs);
    }
    ++training_step;
    return decision;
}

std::pair<torch::Tensor, torch::Tensor> AdaptiveRouterImpl::route_depth(const torch::Tensor& logits,
                                                                        const torch::Tensor& complexity,
                                                                        double temp,
                                                                        bool deterministic) {
    auto depth_bias = torch::linspace(0, 1, max_depth, logits.options()).view({1, 1, max_depth});
    auto scaled = (logits + complexity * depth_bias * 2.0) / std::max(temp, 1e-8);
    torch::Tensor probs;
    if (!deterministic && is_training()) {
        auto g = -torch::log(-torch::log(torch::rand_like(scaled) + 1e-10) + 1e-10);
        probs = torch::sigmoid((scaled + g) / std::max(temp, 1e-8));
    } else {
        probs = torch::sigmoid(scaled);
    }
    auto hard = (probs > 0.5).to(probs.dtype());
    auto mask = (!deterministic && is_training()) ? hard - probs.detach() + probs : hard;
    if (config.min_depth > 0) {
        auto forced = torch::zeros_like(mask);
        forced.slice(-1, 0, config.min_depth).fill_(1.0);
        mask = torch::where(forced.to(torch::kBool), torch::ones_like(mask), mask);
    }
    return {probs, mask};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AdaptiveRouterImpl::route_width(
    const torch::Tensor& logits, const torch::Tensor& complexity, double temp, bool deterministic) {
    auto bias = torch::linspace(-1, 1, num_widths, logits.options()).view({1, 1, num_widths});
    auto scaled = (logits + complexity * bias * 3.0) / std::max(temp, 1e-8);
    auto probs = (!deterministic && is_training())
        ? torch::nn::functional::gumbel_softmax(scaled, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(temp).hard(false).dim(-1))
        : optimized::softmax_simd(scaled, -1);
    auto idx = std::get<1>(probs.detach().max(-1));
    auto multiplier = (probs * width_values.view({1, 1, num_widths})).sum(-1) / static_cast<double>(config.hidden_size);
    return {probs, idx, multiplier.unsqueeze(-1)};
}

torch::Tensor AdaptiveRouterImpl::route_path(const torch::Tensor& logits, double temp, bool deterministic) {
    auto scaled = logits / std::max(temp, 1e-8);
    if (deterministic || !is_training()) return optimized::softmax_simd(scaled, -1);
    auto raw = torch::nn::functional::gumbel_softmax(
        scaled, torch::nn::functional::GumbelSoftmaxFuncOptions().tau(std::max(temp * 2.0, 1.0)).hard(false).dim(-1));
    double prior_weight = std::max(0.05, 0.5 * std::pow(0.995, static_cast<double>(training_step)));
    auto uniform = torch::full_like(raw, 1.0 / static_cast<double>(num_paths));
    return (1.0 - prior_weight) * raw + prior_weight * uniform;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AdaptiveRouterImpl::route_experts(
    const torch::Tensor& logits, double temp, bool deterministic, int64_t expert_capacity) {
    const int64_t B = logits.size(0);
    const int64_t T = logits.size(1);
    const int64_t N = B * T;
    if (expert_capacity < 0) expert_capacity = std::max<int64_t>(1, static_cast<int64_t>(N * 1.25 / num_experts));
    auto flat = logits.view({N, num_experts});
    torch::Tensor probs_flat;
    if (!deterministic && is_training()) {
        auto g = -torch::log(-torch::log(torch::rand_like(flat) + 1e-10) + 1e-10);
        probs_flat = torch::softmax((flat + g) / std::max(temp, 1e-8), -1);
    } else {
        probs_flat = optimized::softmax_simd(flat / std::max(temp, 1e-8), -1);
    }
    auto top = torch::topk(probs_flat, top_k, -1);
    auto weights = std::get<0>(top);
    auto indices = std::get<1>(top);
    weights = weights / (weights.sum(-1, true) + 1e-12);
    if (!deterministic && is_training() && expert_capacity > 0) {
        weights = apply_capacity_constraint(weights, indices, expert_capacity);
    }
    auto expert_probs = optimized::softmax_simd(logits / std::max(temp, 1e-8), -1);
    return {expert_probs, indices.view({B, T, top_k}), weights.view({B, T, top_k})};
}

torch::Tensor AdaptiveRouterImpl::apply_capacity_constraint(const torch::Tensor& weights,
                                                            const torch::Tensor& indices,
                                                            int64_t capacity) {
    const int64_t num_tokens = weights.size(0);
    const int64_t top_k = weights.size(1);

    // Sort by weight descending
    auto flat_w = weights.view(-1);
    auto order = torch::argsort(flat_w, /*dim=*/0, /*descending=*/true);
    
    // Expert index for each flat weight
    auto flat_indices = indices.view(-1);
    auto e_idx = flat_indices.index_select(0, order);

    // Keep mask
    auto keep_mask = torch::zeros_like(flat_w, torch::kBool);
    auto expert_counts = torch::zeros({num_experts}, torch::kInt64).to(weights.device());

    // Vectorized per-expert capacity enforcement using scatter_add
    // This part requires careful implementation to avoid python-like loops.
    // For now, to keep it functional while moving away from CPU loop:
    auto expert_arrivals = torch::arange(0, num_tokens * top_k, torch::kInt64).to(weights.device());
    
    // Count arrivals per expert
    for (int64_t e = 0; e < num_experts; ++e) {
        auto mask_e = (e_idx == e);
        if (mask_e.any().item<bool>()) {
            auto positions = expert_arrivals.masked_select(mask_e);
            auto keep = positions.slice(0, 0, std::min(positions.size(0), capacity));
            keep_mask.index_put_({keep}, true);
        }
    }

    auto final_mask = keep_mask.view({num_tokens, top_k});
    auto masked_weights = weights * final_mask.to(weights.dtype());
    return masked_weights / (masked_weights.sum(-1, true) + 1e-12);
}

torch::Tensor RoutingRegularizerImpl::forward(const RoutingDecision& decision) {
    auto loss = decision.uncertainty.mean() * 0.0001;
    loss = loss + path_diversity_loss(decision.path_probs) * 0.02;
    for (const auto& kv : decision.auxiliary) {
        if (kv.first.find("loss") != std::string::npos && kv.second.defined()) {
            loss = loss + kv.second.to(loss.device()).to(loss.dtype());
        }
    }
    return loss;
}

} // namespace xorzen

#include "xorzen/routing.h"
#include "xorzen/ops.h"
#include "xorzen/optimized/simd_ops.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace xorzen {

// ============================================================
// Auxiliary loss functions — mirror Python (routing.py:38-76, load_balance.py:151-177)
// ============================================================

// Switch Transformer load-balance loss (load_balance.py:151-177).
//   L = N * sum_e (f_e * p_e)
// where f_e = fraction of (token, slot) pairs dispatched to expert e (from one-hot indices),
//       p_e = mean router prob for expert e.
// Range [1, N]; 1 = perfectly balanced, N = complete collapse.
torch::Tensor load_balance_loss(const torch::Tensor& expert_probs,    // [B, T, E]
                                const torch::Tensor& expert_indices,  // [B, T, K]
                                int64_t num_experts) {
    auto probs_flat = expert_probs.reshape({-1, num_experts});      // [N, E]
    auto idx_flat = expert_indices.reshape({-1, expert_indices.size(-1)});  // [N, K]
    // f_e: one-hot the indices, sum over (N, K), normalize by N*K
    auto one_hot = torch::one_hot(idx_flat, /*num_classes=*/num_experts).to(probs_flat.dtype());  // [N, K, E]
    auto counts = one_hot.sum(/*dim=*/0, /*keepdim=*/false).sum(/*dim=*/0, /*keepdim=*/false);    // [E] (sum over K then N)
    // Note: sum over dim 0 then dim 0 again because one_hot is [N, K, E] — first sum collapses N, second collapses K.
    // Actually torch::one_hot returns [N, K, E]; .sum(0) gives [K, E]; .sum(0) gives [E]. Correct.
    auto f = counts / (static_cast<double>(idx_flat.numel()) + 1e-12);  // [E], sums to 1
    auto p = probs_flat.mean(0);                                        // [E]
    return static_cast<double>(num_experts) * (f * p).sum();
}

// router_z_loss (routing.py:38-44): (logsumexp(logits))^2.mean()
torch::Tensor router_z_loss(const torch::Tensor& router_logits) {
    return torch::logsumexp(router_logits, -1).pow(2).mean();
}

// path_diversity_loss (routing.py:47-58): -mean_entropy
//   entropy = -sum(p * log(p)); loss = -mean(entropy) → minimized by max entropy.
torch::Tensor path_diversity_loss(const torch::Tensor& path_probs) {
    auto entropy = -(path_probs * torch::log(path_probs + 1e-12)).sum(-1);  // [B, T]
    return -entropy.mean();
}

// width_diversity_loss (routing.py:61-76): -mean_entropy (same formula, different input)
torch::Tensor width_diversity_loss(const torch::Tensor& width_probs) {
    auto entropy = -(width_probs * torch::log(width_probs + 1e-12)).sum(-1);
    return -entropy.mean();
}

// ============================================================
// AdaptiveRouter
// ============================================================

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
    // Cost-aware routing (Python routing.py:546-548)
    cost_aware_routing = bool(getattr_cost_aware(config));
    compute_budget = getattr_compute_budget(config);
    // Eval routing noise (Python routing.py:465, config.eval_routing_noise default 0.15)
    eval_routing_noise = config.eval_routing_noise;
    // Auxiliary loss weights (Python routing.py:649-658)
    lb_loss_weight = config.lb_loss_weight;
    z_loss_weight = config.z_loss_weight;
    path_div_weight = config.path_div_weight;
    width_div_weight = config.width_div_weight;
    build_network();
    init_weights();
}

bool AdaptiveRouterImpl::getattr_cost_aware(const ModelConfig& c) const {
    // Python: getattr(self.config, 'cost_aware_routing', True)
    return c.cost_aware_routing;
}

double AdaptiveRouterImpl::getattr_compute_budget(const ModelConfig& c) const {
    // Python: getattr(self.config, 'compute_budget', 1.0)
    return c.compute_budget;
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
    // NOTE: character_router REMOVED — Python does not have it (routing.cpp:72-75 in prior version).
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
    // Python routing.py:424-439:
    //   feature_encoder Linears: gain=0.5
    //   depth/width/path/expert_router Linears: gain=0.1
    //   complexity/uncertainty Linears: gain=0.5
    auto init_linear_impl = [](torch::nn::LinearImpl* lin, double gain) {
        if (lin) {
            torch::nn::init::xavier_uniform_(lin->weight, gain);
            if (lin->bias.defined()) torch::nn::init::zeros_(lin->bias);
        }
    };
    for (auto& m : feature_encoder->modules(false)) {
        if (auto* lin = m->as<torch::nn::Linear>()) init_linear_impl(lin, 0.5);
    }
    for (auto& seq : {depth_router, width_router, path_router, expert_router}) {
        for (auto& m : seq->modules(false)) {
            if (auto* lin = m->as<torch::nn::Linear>()) init_linear_impl(lin, 0.1);
        }
    }
    for (auto& seq : {complexity_estimator, uncertainty_estimator}) {
        for (auto& m : seq->modules(false)) {
            if (auto* lin = m->as<torch::nn::Linear>()) init_linear_impl(lin, 0.5);
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

// Eval Gumbel noise (Python routing.py:442-475).
// Deterministic per-axis: seed = 1337 + axis_id, generated on CPU, moved to device.
torch::Tensor AdaptiveRouterImpl::eval_gumbel_noise( torch::IntArrayRef shape,
                                                     const torch::TensorOptions& opts,
                                                     int64_t axis_id) const {
    if (eval_routing_noise <= 0.0) {
        return torch::zeros(shape, opts);
    }
    // Per-axis fixed seed → reproducible, uncorrelated noise per axis.
    at::Generator gen = at::detail::createCPUGenerator(1337 + axis_id);
    auto u = torch::rand(shape, gen).to(opts.dtype());
    // Standard Gumbel(0,1): -log(-log(U))
    auto g = -torch::log(-torch::log(u.clamp_min(1e-10)) + 1e-10);
    return eval_routing_noise * g.to(opts);
}

RoutingDecision AdaptiveRouterImpl::forward(const torch::Tensor& x,
                                            const torch::Tensor& cot_features,
                                            bool deterministic,
                                            int64_t expert_capacity) {
    // Python routing.py:500-505: in eval mode, always deterministic
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
    auto complexity = complexity_estimator->forward(features);
    auto uncertainty = uncertainty_estimator->forward(features);

    // Temperature annealing (Python routing.py:527-529)
    double temp = temperature;
    if (temperature_annealing && is_training()) {
        temp = std::max(0.1, temperature * std::pow(0.99, static_cast<double>(training_step) / 1000.0));
    }

    // ===== v0.4 COST-AWARE ROUTING MODULATION (Python routing.py:546-588) =====
    // The C++ previously OMITTED this entire block. Now mirrored exactly.
    if (cost_aware_routing) {
        double budget = compute_budget;
        budget = std::max(0.05, std::min(1.0, budget));
        double sparsity_pressure = 1.0 - budget;
        // depth_shift: [B, T] — easier tokens (low complexity) get bigger negative shift
        auto depth_shift = -sparsity_pressure * 4.0 * (1.0 - complexity.squeeze(-1));  // [B, T]
        // width_bias_axis: [num_widths] linspace(+2*SP, -2*SP)
        auto width_bias_axis = torch::linspace(
            sparsity_pressure * 2.0, -sparsity_pressure * 2.0, num_widths, width_logits.options());
        width_logits = width_logits + width_bias_axis.view({1, 1, -1});
        // path_bias_axis: [num_paths] linspace(+1.5*SP, -0.5*SP)
        auto path_bias_axis = torch::linspace(
            sparsity_pressure * 1.5, -sparsity_pressure * 0.5, num_paths, path_logits.options());
        path_logits = path_logits + path_bias_axis.view({1, 1, -1});
        // depth_layer_bias: [max_depth] linspace(0, -3*SP)
        auto depth_layer_bias = torch::linspace(
            0.0, -sparsity_pressure * 3.0, max_depth, depth_logits.options());
        depth_logits = depth_logits + depth_layer_bias.view({1, 1, -1}) + depth_shift.unsqueeze(-1);
    }

    // Routing decisions
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
    decision.complexity = complexity;
    decision.uncertainty = uncertainty;
    decision.auxiliary["temperature"] = torch::tensor(temp, x.options());
    // NOTE: character_probs removed (character_router deleted)

    // Auxiliary losses (Python routing.py:641-658) — only during training
    if (is_training() && !deterministic) {
        decision.auxiliary["load_balance_loss"] = lb_loss_weight * load_balance_loss(expert_probs, expert_indices, num_experts);
        decision.auxiliary["router_z_loss"] = z_loss_weight * router_z_loss(expert_logits);
        decision.auxiliary["path_div_loss"] = path_div_weight * path_diversity_loss(path_probs);
        if (num_widths >= 2 && width_div_weight > 0) {
            decision.auxiliary["width_div_loss"] = width_div_weight * width_diversity_loss(width_probs);
        }
    }
    ++training_step;
    return decision;
}

// route_depth (Python routing.py:662-725)
std::pair<torch::Tensor, torch::Tensor> AdaptiveRouterImpl::route_depth(const torch::Tensor& logits,
                                                                        const torch::Tensor& complexity,
                                                                        double temp,
                                                                        bool deterministic) {
    auto depth_bias = torch::linspace(0, 1, max_depth, logits.options()).view({1, 1, max_depth});
    auto adjusted = logits + complexity * depth_bias * 2.0;
    auto scaled = adjusted / std::max(temp, 1e-8);

    torch::Tensor probs, mask;
    if (deterministic) {
        // v0.5 eval Gumbel noise (axis_id=0)
        auto noise = eval_gumbel_noise(scaled.sizes(), scaled.options(), 0);
        probs = torch::sigmoid(scaled + noise);
        mask = (probs > 0.5).to(probs.dtype());
    } else if (is_training()) {
        auto g = -torch::log(-torch::log(torch::rand_like(scaled) + 1e-10) + 1e-10);
        auto noisy = (scaled + g) / std::max(temp, 1e-8);
        probs = torch::sigmoid(noisy);
        auto hard = (probs > 0.5).to(probs.dtype());
        mask = hard - probs.detach() + probs;  // STE
    } else {
        probs = torch::sigmoid(scaled);
        mask = (probs > 0.5).to(probs.dtype());
    }

    // min_depth override (Python routing.py:713-723)
    if (config.min_depth > 0) {
        auto forced = torch::zeros_like(mask);
        forced.slice(-1, 0, config.min_depth).fill_(1.0);
        mask = torch::where(forced.to(torch::kBool), torch::ones_like(mask), mask);
    }
    return {probs, mask};
}

// route_width (Python routing.py:727-777)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AdaptiveRouterImpl::route_width(
    const torch::Tensor& logits, const torch::Tensor& complexity, double temp, bool deterministic) {
    auto bias = torch::linspace(-1, 1, num_widths, logits.options()).view({1, 1, num_widths});
    auto adjusted = logits + complexity * bias * 3.0;
    auto scaled = adjusted / std::max(temp, 1e-8);

    torch::Tensor probs;
    if (deterministic) {
        auto noise = eval_gumbel_noise(scaled.sizes(), scaled.options(), 1);
        probs = torch::softmax(scaled + noise, -1);
    } else if (is_training()) {
        probs = torch::nn::functional::gumbel_softmax(scaled,
            torch::nn::functional::GumbelSoftmaxFuncOptions().tau(temp).hard(false).dim(-1));
    } else {
        probs = torch::softmax(scaled, -1);
    }
    auto idx = std::get<1>(probs.detach().max(-1));
    auto width_normalizer = static_cast<double>(config.hidden_size);
    auto multiplier = (probs * width_values.view({1, 1, num_widths})).sum(-1) / width_normalizer;
    return {probs, idx, multiplier.unsqueeze(-1)};
}

// route_path (Python routing.py:779-830)
torch::Tensor AdaptiveRouterImpl::route_path(const torch::Tensor& logits, double temp, bool deterministic) {
    auto scaled = logits / std::max(temp, 1e-8);
    if (deterministic) {
        auto noise = eval_gumbel_noise(scaled.sizes(), scaled.options(), 2);
        return torch::softmax(scaled + noise, -1);
    }
    torch::Tensor raw;
    if (is_training()) {
        double explore_tau = std::max(temp * 2.0, 1.0);
        raw = torch::nn::functional::gumbel_softmax(scaled,
            torch::nn::functional::GumbelSoftmaxFuncOptions().tau(explore_tau).hard(false).dim(-1));
    } else {
        raw = torch::softmax(scaled, -1);
    }
    if (is_training()) {
        double prior_weight = std::max(0.05, 0.5 * std::pow(0.995, static_cast<double>(training_step)));
        auto uniform = torch::full_like(raw, 1.0 / static_cast<double>(num_paths));
        return (1.0 - prior_weight) * raw + prior_weight * uniform;
    }
    return raw;
}

// route_experts (Python routing.py:832-902)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AdaptiveRouterImpl::route_experts(
    const torch::Tensor& logits, double temp, bool deterministic, int64_t expert_capacity) {
    const int64_t B = logits.size(0);
    const int64_t T = logits.size(1);
    const int64_t N = B * T;
    if (expert_capacity < 0) expert_capacity = std::max<int64_t>(1, static_cast<int64_t>(N * 1.25 / num_experts));
    auto flat = logits.view({N, num_experts});

    torch::Tensor weights, indices;
    if (deterministic) {
        auto noise = eval_gumbel_noise(flat.sizes(), flat.options(), 3);
        auto probs = torch::softmax(flat + noise, -1);
        auto top = torch::topk(probs, top_k, -1);
        weights = std::get<0>(top);
        indices = std::get<1>(top);
        weights = weights / (weights.sum(-1, true) + 1e-12);
    } else if (is_training()) {
        auto g = -torch::log(-torch::log(torch::rand_like(flat) + 1e-10) + 1e-10);
        auto noisy = (flat + g) / std::max(temp, 1e-8);
        auto probs = torch::softmax(noisy, -1);
        auto top = torch::topk(probs, top_k, -1);
        weights = std::get<0>(top);
        indices = std::get<1>(top);
        weights = weights / (weights.sum(-1, true) + 1e-12);
        if (expert_capacity > 0) {
            weights = apply_capacity_constraint(weights, indices, expert_capacity);
        }
    } else {
        auto probs = torch::softmax(flat / std::max(temp, 1e-8), -1);
        auto top = torch::topk(probs, top_k, -1);
        weights = std::get<0>(top);
        indices = std::get<1>(top);
        weights = weights / (weights.sum(-1, true) + 1e-12);
    }
    auto expert_probs = torch::softmax(logits / std::max(temp, 1e-8), -1);
    return {expert_probs, indices.view({B, T, top_k}), weights.view({B, T, top_k})};
}

torch::Tensor AdaptiveRouterImpl::apply_capacity_constraint(const torch::Tensor& weights,
                                                            const torch::Tensor& indices,
                                                            int64_t capacity) {
    const int64_t num_tokens = weights.size(0);
    const int64_t top_k_local = weights.size(1);

    auto flat_w = weights.view(-1);
    auto order = torch::argsort(flat_w, /*dim=*/0, /*descending=*/true);

    auto flat_indices = indices.view(-1);
    auto e_idx = flat_indices.index_select(0, order);

    auto keep_mask = torch::zeros_like(flat_w, torch::kBool);
    auto expert_arrivals = torch::arange(0, num_tokens * top_k_local, torch::kInt64).to(weights.device());

    for (int64_t e = 0; e < num_experts; ++e) {
        auto mask_e = (e_idx == e);
        if (mask_e.any().item<bool>()) {
            auto positions = expert_arrivals.masked_select(mask_e);
            auto keep = positions.slice(0, 0, std::min(positions.size(0), capacity));
            keep_mask.index_put_({keep}, true);
        }
    }

    auto final_mask = keep_mask.view({num_tokens, top_k_local});
    auto masked_weights = weights * final_mask.to(weights.dtype());
    return masked_weights / (masked_weights.sum(-1, true) + 1e-12);
}

// ============================================================
// RoutingRegularizer — mirror Python routing.py:1691-1694
//   Returns ONLY uncertainty_loss = uncertainty_weight * mean(uncertainty).
//   The path_div / load_balance / z_loss / width_div are summed separately
//   in the model's forward (NOT here). This avoids double-counting.
// ============================================================
torch::Tensor RoutingRegularizerImpl::forward(const RoutingDecision& decision) {
    double uncertainty_weight = config.routing_loss_weight;  // Python: 0.01 default
    return decision.uncertainty.mean() * uncertainty_weight;
}

} // namespace xorzen

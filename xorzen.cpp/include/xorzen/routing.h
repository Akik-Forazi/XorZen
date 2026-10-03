#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <string>
#include "xorzen/types.h"

namespace xorzen {

struct RoutingDecision {
    torch::Tensor depth_logits;
    torch::Tensor depth_probs;
    torch::Tensor depth_mask;
    torch::Tensor width_logits;
    torch::Tensor width_probs;
    torch::Tensor width_idx;
    torch::Tensor width_multiplier;
    torch::Tensor path_logits;
    torch::Tensor path_probs;
    torch::Tensor expert_logits;
    torch::Tensor expert_probs;
    torch::Tensor expert_indices;
    torch::Tensor expert_weights;
    torch::Tensor complexity;
    torch::Tensor uncertainty;
    // NOTE: character_probs removed — Python AdaptiveRouter does not have character_router.
    std::unordered_map<std::string, torch::Tensor> auxiliary;
};

// Auxiliary loss functions — mirror Python (routing.py:38-76, load_balance.py:151-177)
torch::Tensor load_balance_loss(const torch::Tensor& expert_probs,
                                const torch::Tensor& expert_indices,
                                int64_t num_experts);
torch::Tensor router_z_loss(const torch::Tensor& router_logits);
torch::Tensor path_diversity_loss(const torch::Tensor& path_probs);
torch::Tensor width_diversity_loss(const torch::Tensor& width_probs);

struct AdaptiveRouterImpl : torch::nn::Module {
    ModelConfig config;
    int64_t hidden_dim;
    int64_t cot_dim;
    int64_t input_dim;
    int64_t max_depth;
    int64_t num_widths;
    int64_t num_paths;
    int64_t num_experts;
    int64_t top_k;
    double temperature;
    bool temperature_annealing;
    int64_t training_step = 0;

    // Cost-aware routing (Python routing.py:546-588)
    bool cost_aware_routing;
    double compute_budget;
    double eval_routing_noise;
    // Auxiliary loss weights (Python routing.py:649-658, 1689)
    double lb_loss_weight;
    double z_loss_weight;
    double path_div_weight;
    double width_div_weight;

    torch::nn::Sequential feature_encoder{nullptr};
    torch::nn::Sequential depth_router{nullptr};
    torch::nn::Sequential width_router{nullptr};
    torch::nn::Sequential path_router{nullptr};
    torch::nn::Sequential expert_router{nullptr};
    // NOTE: character_router removed — Python does not have it.
    torch::nn::Sequential complexity_estimator{nullptr};
    torch::nn::Sequential uncertainty_estimator{nullptr};
    torch::Tensor width_values;

    explicit AdaptiveRouterImpl(ModelConfig config);
    void load_weights(const std::string& path);
    RoutingDecision forward(const torch::Tensor& x,
                            const torch::Tensor& cot_features,
                            bool deterministic = false,
                            int64_t expert_capacity = -1);

private:
    void build_network();
    void init_weights();
    bool getattr_cost_aware(const ModelConfig& c) const;
    double getattr_compute_budget(const ModelConfig& c) const;
    torch::Tensor eval_gumbel_noise(torch::IntArrayRef shape,
                                    const torch::TensorOptions& opts,
                                    int64_t axis_id) const;
    std::pair<torch::Tensor, torch::Tensor> route_depth(const torch::Tensor& logits,
                                                        const torch::Tensor& complexity,
                                                        double temp,
                                                        bool deterministic);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> route_width(const torch::Tensor& logits,
                                                                        const torch::Tensor& complexity,
                                                                        double temp,
                                                                        bool deterministic);
    torch::Tensor route_path(const torch::Tensor& logits, double temp, bool deterministic);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> route_experts(const torch::Tensor& logits,
                                                                          double temp,
                                                                          bool deterministic,
                                                                          int64_t expert_capacity);
    torch::Tensor apply_capacity_constraint(const torch::Tensor& weights,
                                            const torch::Tensor& indices,
                                            int64_t capacity);
};
TORCH_MODULE(AdaptiveRouter);

struct RoutingRegularizerImpl : torch::nn::Module {
    ModelConfig config;
    explicit RoutingRegularizerImpl(ModelConfig config) : config(std::move(config)) {}
    // Mirror Python routing.py:1691-1694: returns ONLY uncertainty_loss.
    // The path_div / load_balance / z_loss / width_div are summed separately
    // in the model's forward (NOT here). This avoids double-counting.
    torch::Tensor forward(const RoutingDecision& decision);
};
TORCH_MODULE(RoutingRegularizer);

} // namespace xorzen

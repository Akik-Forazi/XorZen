// xorzen_greed.h — Greed Model (C++ implementation)
//
// Mirrors xorzen/models/greed/model.py exactly.
// GreedModel inherits from zeroModel and:
//   1. Replaces token embedding with continuous feature projection
//   2. Uses unfrozen CoT (trained end-to-end)
//   3. Adds Greedy Gated CoT Fusion before pooling
//   4. Uses mean-pooled classification head instead of LM head
#pragma once

#include <torch/torch.h>
#include "xorzen/model.h"
#include "xorzen/variants.h"

namespace xorzen {

struct GreedModelImpl : XorzenModelImpl {
    int64_t input_dim;
    int64_t num_classes;
    torch::nn::Sequential feature_proj{nullptr};
    torch::nn::Sequential greedy_cot_gate{nullptr};
    torch::nn::Linear greedy_fusion_proj{nullptr};
    torch::nn::Sequential classification_head{nullptr};

    GreedModelImpl(ModelConfig cfg, int64_t input_dim_ = 24, int64_t num_classes_ = 5,
                   bool test_mode = false)
        : XorzenModelImpl(cfg, test_mode), input_dim(input_dim_), num_classes(num_classes_) {
        // Feature projection (replaces token embedding)
        feature_proj = register_module("feature_proj", torch::nn::Sequential(
            torch::nn::Linear(input_dim, cfg.hidden_size),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.hidden_size}))));

        int64_t cot_total = cfg.cot_dim * cfg.cot_components;
        // Greedy Gated CoT Fusion
        greedy_cot_gate = register_module("greedy_cot_gate", torch::nn::Sequential(
            torch::nn::Linear(cfg.hidden_size + cot_total, cfg.hidden_size),
            torch::nn::SiLU(),
            torch::nn::Linear(cfg.hidden_size, 1),
            torch::nn::Sigmoid()));
        greedy_fusion_proj = register_module("greedy_fusion_proj",
            torch::nn::Linear(cfg.hidden_size + cot_total, cfg.hidden_size));

        // Classification head
        classification_head = register_module("classification_head", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.hidden_size})),
            torch::nn::Linear(cfg.hidden_size, num_classes)));

        // Enable CoT (Greed trains it end-to-end)
        enable_cot(true);
    }

    // Greed forward: takes continuous features [B, T, input_dim] instead of input_ids
    torch::Tensor forward_features(const torch::Tensor& features,
                                    torch::Tensor attention_mask = {}) {
        int64_t B = features.size(0), T = features.size(1);
        auto device = features.device();

        // Step 1: Project features
        auto hidden = feature_proj->forward(features);  // [B, T, H]
        auto pos_ids = torch::arange(T).unsqueeze(0).expand({B, T});
        hidden = hidden + position_embedding->forward(pos_ids);
        hidden = embedding_dropout->forward(hidden);

        if (!attention_mask.defined())
            attention_mask = torch::ones({B, T}, torch::kBool);

        // Step 2: Active CoT
        auto cot_out = cot->forward(hidden);
        auto cot_vector = cot_out.first;

        // Step 3: Routing
        auto decision = router->forward(hidden, cot_vector);

        // Step 4: HASS blocks
        for (int64_t i = 0; i < config.num_layers; ++i) {
            auto layer_mask = decision.depth_mask.select(-1, i);
            auto block = blocks[i]->as<HASSBlock>();
            torch::Tensor block_out;
            if (!is_training() && !layer_mask.any().item<bool>()) {
                continue;
            }
            block_out = block->forward(hidden, &decision, attention_mask);
            auto mask3d = layer_mask.unsqueeze(-1);
            hidden = block_out * mask3d + hidden * (1.0 - mask3d);
        }

        // Step 5: MoE
        auto hidden_flat = hidden.reshape({B * T, config.hidden_size});
        auto ei_flat = decision.expert_indices.reshape({B * T, -1});
        auto ew_flat = decision.expert_weights.reshape({B * T, -1});
        auto mask_flat = attention_mask.reshape({B * T});
        auto [moe_flat, moe_stats] = moe->forward(hidden_flat, ei_flat, ew_flat, mask_flat);
        auto moe_output = moe_flat.reshape({B, T, config.hidden_size});

        // Step 6: Merger
        auto merged = merger->forward(hidden, moe_output, cot_vector, attention_mask);
        hidden = final_norm->forward(merged);

        // Step 7: Greedy Gated CoT Fusion
        auto concat = torch::cat({hidden, cot_vector}, -1);
        auto gate = greedy_cot_gate->forward(concat);
        auto fused = greedy_fusion_proj->forward(concat);
        hidden = gate * fused + (1.0 - gate) * hidden;

        // Step 8: Pool + classify
        auto pooled = hidden.mean(1);  // [B, H]
        return classification_head->forward(pooled);  // [B, num_classes]
    }
};
TORCH_MODULE(GreedModel);

} // namespace xorzen

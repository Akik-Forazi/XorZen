#pragma once

#include <torch/torch.h>
#include "xorzen/types.h"
#include "xorzen/routing.h"
#include "xorzen/hass.h"
#include "xorzen/expert.h"
#include "xorzen/cot.h"
#include "xorzen/igris.h"
#include "xorzen/merger.h"

namespace xorzen {

struct XorzenModelImpl : torch::nn::Module {
    ModelConfig config;
    torch::nn::Embedding token_embedding{nullptr};
    torch::nn::Embedding position_embedding{nullptr};
    torch::nn::Dropout embedding_dropout{nullptr};
    AdaptiveRouter router{nullptr};
    RoutingRegularizer routing_regularizer{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    ShardedExpertFabric moe{nullptr};
    XorzenMergerGate merger{nullptr};
    RMSNorm final_norm{nullptr};
    torch::nn::Linear lm_head{nullptr};
    
    // IGRIS Agentic Bits
    torch::nn::Linear action_head{nullptr};
    torch::nn::Sequential critique_module{nullptr};
    torch::nn::Linear recursive_router{nullptr};

    // Latent Chain-of-Thought
    InternalLatentCoT cot{nullptr};
    CoTAuxiliaryLoss cot_loss_head{nullptr};
    
    int64_t step_count = 0;
    int64_t total_tokens_processed = 0;

    explicit XorzenModelImpl(ModelConfig config, bool test_mode = false);
    ModelOutput forward(const torch::Tensor& input_ids,
                        const torch::Tensor& attention_mask = {},
                        const torch::Tensor& position_ids = {},
                        const torch::Tensor& labels = {},
                        bool output_hidden_states = false,
                        bool output_routing_info = false);
    torch::Tensor generate(const torch::Tensor& prompt, const GenerationConfig& generation_config);
    int64_t count_parameters(bool only_trainable = false) const;
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    // Load from a per-tensor .bin directory (compatible with Python converter output).
    // Maps Python state_dict keys → C++ parameter names, copies weights in-place.
    // In test_mode, moe.experts.0.* → moe.dummy_expert.*.
    // Returns a struct with matched/missing/unexpected counts.
    struct LoadResult {
        int64_t matched = 0;
        int64_t missing = 0;      // C++ expects but Python doesn't have
        int64_t unexpected = 0;   // Python has but C++ doesn't expect
        int64_t shape_mismatches = 0;
        std::vector<std::string> missing_keys;
        std::vector<std::string> unexpected_keys;
        std::vector<std::string> shape_mismatch_keys;
    };
    LoadResult load_from_tensor_map(const std::unordered_map<std::string, torch::Tensor>& sd,
                                     bool strict = false);
    // Save to a per-tensor .bin directory (for checkpoint lifecycle tests).
    void save_to_tensor_map(const std::string& dir) const;

    void enable_cot(bool enabled = true);


private:
    void validate_config() const;
    void init_weights();
    torch::Tensor compute_load_balance_loss(const torch::Tensor& expert_indices,
                                            const torch::Tensor& expert_weights) const;
    int64_t estimate_active_params(const RoutingDecision& decision) const;
    double estimate_compute_cost(const RoutingDecision& decision, int64_t batch, int64_t seq) const;
};
TORCH_MODULE(XorzenModel);

} // namespace xorzen

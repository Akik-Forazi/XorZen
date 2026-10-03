#pragma once

#include <torch/torch.h>
#include "xorzen/types.h"
#include "xorzen/routing.h"
#include "xorzen/ops.h"

namespace xorzen {

struct LocalAttentionPathwayImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t num_heads;
    int64_t head_dim;
    int64_t window_size;
    bool causal;
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, out_proj{nullptr};
    torch::nn::LayerNorm ln_q{nullptr}, ln_k{nullptr};
    torch::nn::Dropout attn_dropout{nullptr}, resid_dropout{nullptr};

    LocalAttentionPathwayImpl(int64_t hidden_dim, int64_t num_heads, int64_t window_size,
                              double dropout = 0.0, bool causal = true);
    torch::Tensor forward(const torch::Tensor& x,
                          const torch::Tensor& attention_mask = {},
                          const torch::Tensor& position_bias = {});
};
TORCH_MODULE(LocalAttentionPathway);

struct LowRankGlobalPathwayImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t low_rank_dim;
    int64_t num_heads;
    torch::nn::Linear to_low_rank{nullptr}, from_low_rank{nullptr};
    torch::nn::LayerNorm ln_input{nullptr}, ln_low_rank{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::Tensor context_weights;

    LowRankGlobalPathwayImpl(int64_t hidden_dim, int64_t low_rank_dim, int64_t num_heads = 1, double dropout = 0.0);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(LowRankGlobalPathway);

struct SSMPathwayImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t state_dim;
    int64_t kernel_size;
    bool use_conv;
    torch::Tensor A_log;
    torch::nn::Linear dt_proj{nullptr}, B_proj{nullptr}, C_proj{nullptr}, D_proj{nullptr}, gate_proj{nullptr};
    torch::nn::Conv1d conv{nullptr};
    torch::nn::LayerNorm ln_input{nullptr}, ln_state{nullptr};
    torch::nn::Dropout dropout{nullptr};

    SSMPathwayImpl(int64_t hidden_dim, int64_t state_dim, int64_t kernel_size = 3,
                   double dropout = 0.0, bool use_conv = true);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(SSMPathway);

struct AdaptiveFFNImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t base_ffn_dim;
    std::string activation;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::LayerNorm ln_input{nullptr}, ln_hidden{nullptr};
    torch::nn::Dropout dropout{nullptr}, ffn_dropout{nullptr};

    AdaptiveFFNImpl(int64_t hidden_dim, double ffn_multiplier = 4.0,
                    std::string activation = "gelu", double dropout = 0.0);
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& width_multiplier = {});
};
TORCH_MODULE(AdaptiveFFN);

struct HASSBlockImpl : torch::nn::Module {
    ModelConfig config;
    int64_t layer_idx;
    LocalAttentionPathway local{nullptr};
    LowRankGlobalPathway low_rank{nullptr};
    SSMPathway ssm{nullptr};
    torch::nn::Sequential pathway_gate{nullptr};
    AdaptiveFFN ffn{nullptr};
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    HASSBlockImpl(ModelConfig config, int64_t layer_idx = 0);
    torch::Tensor forward(const torch::Tensor& x,
                          const RoutingDecision* routing_decision = nullptr,
                          const torch::Tensor& attention_mask = {},
                          bool compute_all_pathways = false);
};
TORCH_MODULE(HASSBlock);

// NOTE: The 2-gate GatedMergerImpl previously declared here has been REMOVED.
// The correct 3-gate GatedMergerImpl (matching Python xorzenMergerGate) lives
// in xorzen/merger.h and xorzen.cpp/src/model/merger.cpp. The main model
// (xorzen_model.cpp) now uses XorzenMergerGate from merger.h.

} // namespace xorzen

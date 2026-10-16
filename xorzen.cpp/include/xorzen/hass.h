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
    torch::nn::LayerNorm ln_input{nullptr};
    torch::nn::Dropout dropout{nullptr};
    // NOTE: ln_low_rank and context_weights REMOVED to match Python.
    // Python LowRankGlobalPathway (hass_block.py:261-368) has only ln_input,
    // to_low_rank, from_low_rank, dropout. No learned context query, no
    // second LayerNorm. Uses causal self-attention with GELU.

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

// ─────────────────────────────────────────────────────────────────────
// SlicedFFN — genuine per-token width slicing (matches Python SlicedFFN)
// ─────────────────────────────────────────────────────────────────────
// The weight matrices are allocated at max_width but only the first W_i
// columns/rows are used for tokens that selected width W_i. This gives
// genuine per-token compute reduction proportional to W_i / max_width.
//
// Key differences from AdaptiveFFN:
//   - Accepts per-token width_idx [B, T] long tensor (not a scalar multiplier)
//   - Groups tokens by selected width, processes each group with sliced matmul
//   - Skips ln_hidden when width < max_width (matches Python sliced_ffn.py:195)
//
// Checkpoint compatibility: uses the same fc1/fc2/ln_input/ln_hidden parameter
// names as AdaptiveFFN, so existing checkpoints load without migration.
struct SlicedFFNImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t max_width;
    std::string activation;
    std::vector<int64_t> width_choices;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::LayerNorm ln_input{nullptr}, ln_hidden{nullptr};
    torch::nn::Dropout dropout{nullptr}, ffn_dropout{nullptr};

    SlicedFFNImpl(int64_t hidden_dim, int64_t max_width,
                  std::string activation = "gelu", double dropout = 0.0,
                  std::vector<int64_t> width_choices = {});

    // forward with per-token width_idx [B, T] (inference)
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& width_idx = {});
    // forward with a single width for all tokens (fast path)
    torch::Tensor forward_single_width(const torch::Tensor& x_norm, int64_t width);
    // forward per-token width via grouping
    torch::Tensor forward_per_token_width(const torch::Tensor& x_norm, const torch::Tensor& width_idx);

private:
    torch::Tensor apply_activation(const torch::Tensor& h);
};
TORCH_MODULE(SlicedFFN);

struct HASSBlockImpl : torch::nn::Module {
    ModelConfig config;
    int64_t layer_idx;
    bool use_sliced_ffn;
    LocalAttentionPathway local{nullptr};
    LowRankGlobalPathway low_rank{nullptr};
    SSMPathway ssm{nullptr};
    // NOTE: pathway_gate REMOVED — Python removed it in v0.5 (hass_block.py:897-903).
    // When routing_decision is provided, path_probs are used directly.
    // ffn is either SlicedFFN (genuine per-token width, matches Python default)
    // or AdaptiveFFN (scalar width, legacy). Selected by config.use_sliced_ffn.
    AdaptiveFFN ffn_adaptive{nullptr};  // used when use_sliced_ffn=false
    SlicedFFN ffn_sliced{nullptr};     // used when use_sliced_ffn=true
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

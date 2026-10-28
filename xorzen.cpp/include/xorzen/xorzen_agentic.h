// xorzen_agentic.h — Jima (Agentic XORZEN) Model (C++ implementation)
//
// Jima = XorzenModel (HASS + SlicedFFN + AdaptiveRouter + MoE + MergerGate + CoT)
//        + Agentic extensions (ActionHead, CritiqueModule, RecursiveRouter,
//          memory_vault, recurrence)
//
// This uses the SAME XORZEN architecture as XorzenModel — HASS blocks with
// Local/LowRank/SSM pathways, SlicedFFN with per-token width, AdaptiveRouter
// with 4-axis routing, ShardedExpertFabric MoE, GatedMerger, InternalLatentCoT.
//
// Agentic extensions add:
//   - Memory vault associative retrieval (after embeddings)
//   - Recursive HASS re-processing (based on self-critique quality)
//   - Action head (parallel to LM head)
//   - Ponder cost regularization
//
// The previous ZeroAgenticModel (which used FlashSSM+GLA) is deprecated.
// This header now includes the full XORZEN components.
#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#include "xorzen/hass.h"        // HASSBlock, SlicedFFN, LocalAttentionPathway, etc.
#include "xorzen/routing.h"     // AdaptiveRouter, RoutingDecision
#include "xorzen/merger.h"      // XorzenMergerGate
#include "xorzen/cot.h"         // InternalLatentCoT
#include "xorzen/zmoe.h"        // ShardedExpertFabric
#include "xorzen/ops.h"         // RMSNorm
#include "xorzen/model.h"       // ModelConfig

namespace xorzen {

// ─── ZeroAgenticConfig ──────────────────────────────────────────
struct ZeroAgenticConfig {
    int64_t vocab_size = 10000;
    int64_t hidden_size = 256;
    int64_t num_layers = 6;
    int64_t num_attention_heads = 8;
    int64_t context_length = 2048;
    int64_t pad_token_id = 0;

    // XORZEN architecture
    bool use_sliced_ffn = true;
    int64_t max_depth = 6;
    int64_t min_depth = 1;
    int64_t expert_count = 4;
    int64_t top_k_experts = 2;
    int64_t pathway_top_k = 2;
    int64_t local_window_size = 128;
    int64_t low_rank_dim = 96;
    int64_t ssm_state_dim = 16;
    int64_t ssm_kernel_size = 4;
    std::string hidden_act = "gelu";
    double dropout = 0.0;
    double layer_norm_eps = 1e-5;
    bool tie_word_embeddings = true;
    bool gradient_checkpointing = false;

    // Routing
    bool cost_aware_routing = false;
    double compute_budget = 1.0;
    double eval_routing_noise = 0.0;
    int64_t router_hidden_dim = 32;

    // CoT
    int64_t cot_dim = 4;
    int64_t cot_components = 6;

    // Agentic capabilities
    bool use_latent_cot = true;
    int64_t latent_cot_dim = 128;
    int64_t num_action_slots = 16;
    bool use_self_critique = true;

    // Memory
    int64_t memory_slots = 64;
    int64_t memory_dim = 128;

    // Thinking
    int64_t recurrence_depth = 1;
    bool adaptive_depth = true;

    // Width choices (for SlicedFFN)
    std::vector<int64_t> width_choices = {};

    // Convert to ModelConfig for XORZEN components
    ModelConfig to_model_config() const {
        ModelConfig cfg;
        cfg.vocab_size = vocab_size;
        cfg.hidden_size = hidden_size;
        cfg.num_layers = num_layers;
        cfg.max_depth = max_depth;
        cfg.min_depth = min_depth;
        cfg.context_length = context_length;
        cfg.num_attention_heads = num_attention_heads;
        cfg.n_kv_heads = num_attention_heads;
        cfg.use_sliced_ffn = use_sliced_ffn;
        cfg.expert_count = expert_count;
        cfg.top_k_experts = top_k_experts;
        cfg.pathway_top_k = pathway_top_k;
        cfg.local_window_size = local_window_size;
        cfg.low_rank_dim = low_rank_dim;
        cfg.ssm_state_dim = ssm_state_dim;
        cfg.ssm_kernel_size = ssm_kernel_size;
        cfg.hidden_act = hidden_act;
        cfg.dropout = dropout;
        cfg.layer_norm_eps = layer_norm_eps;
        cfg.tie_word_embeddings = tie_word_embeddings;
        cfg.gradient_checkpointing = gradient_checkpointing;
        cfg.cost_aware_routing = cost_aware_routing;
        cfg.compute_budget = compute_budget;
        cfg.eval_routing_noise = eval_routing_noise;
        cfg.router_hidden_dim = router_hidden_dim;
        cfg.cot_dim = cot_dim;
        cfg.cot_components = cot_components;
        cfg.pad_token_id = pad_token_id;
        if (!width_choices.empty()) {
            cfg.width_choices = width_choices;
        } else {
            cfg.width_choices = {hidden_size / 2, hidden_size};
        }
        return cfg;
    }
};

// ─── Agentic Extensions (unchanged from previous version) ───────

struct ActionHeadImpl : torch::nn::Module {
    torch::nn::Linear proj{nullptr};
    ActionHeadImpl(int64_t d_model, int64_t num_actions)
        : proj(register_module("proj", torch::nn::Linear(d_model, num_actions))) {}
    torch::Tensor forward(const torch::Tensor& x) { return proj->forward(x); }
};
TORCH_MODULE(ActionHead);

struct CritiqueModuleImpl : torch::nn::Module {
    torch::nn::Sequential quality_eval{nullptr};
    CritiqueModuleImpl(int64_t d_model)
        : quality_eval(register_module("quality_eval", torch::nn::Sequential(
            torch::nn::Linear(d_model, std::max(int64_t(1), d_model / 2)),
            torch::nn::GELU(),
            torch::nn::Linear(std::max(int64_t(1), d_model / 2), 1),
            torch::nn::Sigmoid()))) {}
    torch::Tensor forward(const torch::Tensor& x) { return quality_eval->forward(x); }
};
TORCH_MODULE(CritiqueModule);

struct RecursiveRouterImpl : torch::nn::Module {
    torch::nn::Linear halt_proj{nullptr};
    RecursiveRouterImpl(int64_t d_model)
        : halt_proj(register_module("halt_proj", torch::nn::Linear(d_model, 1))) {}
    torch::Tensor forward(const torch::Tensor& x) { return torch::sigmoid(halt_proj->forward(x)); }
};
TORCH_MODULE(RecursiveRouter);

// ─── Jima Model — XORZEN core + agentic extensions ──────────────
//
// This replaces the old ZeroAgenticModel. It uses the SAME HASS blocks,
// SlicedFFN, AdaptiveRouter, ShardedExpertFabric, and GatedMerger as
// XorzenModelImpl, plus the agentic extensions.
//
// The FlashSSM and GatedLinearAttention from the old implementation are
// removed — the SSM pathway in HASSBlock already provides SSM, and the
// LocalAttentionPathway provides attention.

struct JimaModelImpl : torch::nn::Module {
    ZeroAgenticConfig agentic_config;
    ModelConfig model_config;  // Converted from agentic_config for XORZEN components
    bool test_mode = true;

    // XORZEN Core (identical to XorzenModelImpl)
    torch::nn::Embedding token_embedding{nullptr}, position_embedding{nullptr};
    torch::nn::Dropout embedding_dropout{nullptr};
    InternalLatentCoT cot{nullptr};
    AdaptiveRouter router{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    ShardedExpertFabric moe{nullptr};
    XorzenMergerGate merger{nullptr};
    RMSNorm final_norm{nullptr};
    torch::nn::Linear lm_head{nullptr};

    // Agentic Extensions
    ActionHead action_head{nullptr};
    CritiqueModule critique{nullptr};
    RecursiveRouter recursive_router{nullptr};
    torch::Tensor memory_vault;

    bool cot_enabled = false;

    JimaModelImpl(const ZeroAgenticConfig& cfg, bool test_mode_ = true)
        : agentic_config(cfg), model_config(cfg.to_model_config()), test_mode(test_mode_) {

        model_config.normalize();

        // ═══ XORZEN Core ═══
        token_embedding = register_module("token_embedding",
            torch::nn::Embedding(model_config.vocab_size, model_config.hidden_size));
        position_embedding = register_module("position_embedding",
            torch::nn::Embedding(model_config.context_length, model_config.hidden_size));
        embedding_dropout = register_module("embedding_dropout",
            torch::nn::Dropout(model_config.dropout));

        // CoT (frozen by default)
        cot = register_module("cot", InternalLatentCoT(model_config));
        _freeze_cot();

        // Adaptive Router
        router = register_module("router", AdaptiveRouter(model_config));

        // HASS Blocks (with SlicedFFN, Local/LowRank/SSM pathways)
        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int64_t i = 0; i < model_config.num_layers; ++i)
            blocks->push_back(HASSBlock(model_config, i));

        // MoE
        moe = register_module("moe", ShardedExpertFabric(model_config, test_mode));

        // Merger Gate
        merger = register_module("merger", XorzenMergerGate(model_config));

        // Output
        final_norm = register_module("final_norm", RMSNorm(
            torch::nn::LayerNormOptions({model_config.hidden_size}).eps(model_config.layer_norm_eps)));
        lm_head = register_module("lm_head",
            torch::nn::Linear(torch::nn::LinearOptions(model_config.hidden_size, model_config.vocab_size).bias(false)));
        lm_head->weight = token_embedding->weight;
        token_embedding->weight.data()[model_config.pad_token_id].zero_();

        // ═══ Agentic Extensions ═══
        memory_vault = register_parameter("memory_vault",
            torch::randn({1, agentic_config.memory_slots, model_config.hidden_size}) * 0.02);
        action_head = register_module("action_head",
            ActionHead(model_config.hidden_size, agentic_config.num_action_slots));
        critique = register_module("critique", CritiqueModule(model_config.hidden_size));
        recursive_router = register_module("recursive_router",
            RecursiveRouter(model_config.hidden_size));

        _init_weights();
        lm_head->weight = token_embedding->weight;
        token_embedding->weight.data()[model_config.pad_token_id].zero_();
    }

    void _freeze_cot() {
        for (auto& p : cot->parameters())
            p.set_requires_grad(false);
        cot_enabled = false;
    }

    void enable_cot() {
        for (auto& p : cot->parameters())
            p.set_requires_grad(true);
        cot_enabled = true;
    }

    void _init_weights() {
        torch::NoGradGuard ng;
        for (auto& m : modules(false)) {
            if (auto* lin = m->as<torch::nn::Linear>()) {
                lin->weight.data().normal_(0.0, 0.02);
                if (lin->bias.defined()) lin->bias.data().zero_();
            }
            if (auto* emb = m->as<torch::nn::Embedding>()) {
                emb->weight.data().normal_(0.0, 0.02);
            }
        }
    }

    struct Output {
        torch::Tensor logits;
        torch::Tensor loss;
        torch::Tensor actions;
        int64_t recurrence_steps = 0;
    };

    Output forward(const torch::Tensor& input_ids,
                   const torch::Tensor& labels = {},
                   const torch::Tensor& attention_mask = {},
                   const torch::Tensor& action_targets = {},
                   double action_loss_weight = 0.1) {
        int64_t B = input_ids.size(0), L = input_ids.size(1);
        auto device = input_ids.device();

        // ═══ STEP 1: EMBEDDINGS ═══
        auto pos_ids = torch::arange(L, torch::kLong).unsqueeze(0).to(device).expand({B, L});
        auto hidden = token_embedding->forward(input_ids) + position_embedding->forward(pos_ids);
        hidden = embedding_dropout->forward(hidden);

        // ═══ STEP 1.5: MEMORY VAULT (agentic extension) ═══
        auto vault = memory_vault.expand({B, -1, -1});
        auto mem_scores = torch::bmm(hidden, vault.transpose(1, 2)) /
                          std::sqrt(static_cast<double>(model_config.hidden_size));
        auto mem_weights = torch::softmax(mem_scores, -1);
        auto mem_retrieved = torch::bmm(mem_weights, vault);
        hidden = hidden + mem_retrieved;

        // ═══ STEP 2: CoT ═══
        int64_t cot_total_dim = model_config.cot_dim * model_config.cot_components;
        torch::Tensor cot_vector_seq;
        if (cot_enabled) {
            auto cot_result = cot->forward(hidden);
            cot_vector_seq = std::get<0>(cot_result);
        } else {
            cot_vector_seq = torch::zeros({B, L, cot_total_dim}, hidden.options());
        }

        // ═══ STEP 3: ADAPTIVE ROUTING ═══
        auto routing_decision = router->forward(hidden, cot_vector_seq);

        // ═══ STEP 4: HASS BLOCKS (with agentic recurrence) ═══
        int64_t total_recurrence_steps = 0;
        torch::Tensor ponder_cost = torch::tensor(0.0, hidden.options());

        for (int64_t layer_idx = 0; layer_idx < model_config.num_layers; ++layer_idx) {
            auto block = blocks[layer_idx]->as<HASSBlock>();
            auto layer_mask = routing_decision->depth_mask.index({torch::indexing::Slice(),
                torch::indexing::Slice(), layer_idx});  // [B, T]

            // HASS block forward (same as XorzenModelImpl)
            auto block_out = block->forward(hidden, &(*routing_decision), attention_mask);

            // STE mask blend
            auto mask_3d = layer_mask.unsqueeze(-1);
            hidden = block_out * mask_3d + hidden * (1.0 - mask_3d);

            // ═══ AGENTIC RECURRENCE ═══
            if (agentic_config.recurrence_depth > 1 && !this->is_training()) {
                for (int r = 0; r < agentic_config.recurrence_depth - 1; ++r) {
                    auto halt_prob = recursive_router->forward(hidden);
                    auto quality = critique->forward(hidden);
                    auto agentic_halt = halt_prob * quality;
                    ponder_cost = ponder_cost + agentic_halt.mean();

                    auto new_hidden = block->forward(hidden, &(*routing_decision), attention_mask);
                    hidden = agentic_halt * hidden + (1 - agentic_halt) * new_hidden;

                    if (agentic_halt.mean().item<double>() > 0.8) break;
                    total_recurrence_steps++;
                }
            }
        }

        // ═══ STEP 5: MoE ═══
        int64_t batch_seq = B * L;
        auto hidden_flat = hidden.reshape({batch_seq, -1});
        auto expert_indices_flat = routing_decision->expert_indices.reshape({batch_seq, -1});
        auto expert_weights_flat = routing_decision->expert_weights.reshape({batch_seq, -1});
        auto moe_output_flat = moe->forward(hidden_flat, expert_indices_flat, expert_weights_flat);
        auto moe_output = moe_output_flat.reshape({B, L, -1});

        // ═══ STEP 6: MERGER ═══
        auto merged = merger->forward(hidden, moe_output, cot_vector_seq, attention_mask);

        // ═══ STEP 7: OUTPUT ═══
        hidden = final_norm->forward(merged);
        auto logits = lm_head->forward(hidden);
        auto actions = action_head->forward(hidden);

        // ═══ STEP 8: LOSS ═══
        torch::Tensor loss;
        if (labels.defined()) {
            auto sl = logits.slice(1, 0, L - 1).contiguous().view({-1, model_config.vocab_size});
            auto sb = labels.slice(1, 1, L).contiguous().view({-1});
            loss = torch::nn::functional::cross_entropy(sl, sb);
            loss = loss - 0.01 * ponder_cost;
            if (action_targets.defined()) {
                loss = loss + action_loss_weight * torch::mse_loss(actions, action_targets);
            } else {
                auto ap = torch::softmax(actions, -1);
                auto alp = torch::log_softmax(actions, -1);
                auto entropy_reg = -(ap * alp).sum(-1).mean();
                loss = loss - 0.01 * entropy_reg;
            }
        }

        return {logits, loss, actions, total_recurrence_steps};
    }

    torch::Tensor generate(const torch::Tensor& prompt, int64_t max_length) {
        torch::NoGradGuard ng;
        auto tokens = prompt.clone();
        if (tokens.dim() == 1) tokens = tokens.unsqueeze(0);
        for (int64_t i = 0; i < max_length; ++i) {
            auto context = tokens;
            if (context.size(1) > model_config.context_length)
                context = context.slice(1, context.size(1) - model_config.context_length);
            auto out = forward(context);
            auto next = out.logits.select(1, -1).argmax(-1, false).unsqueeze(1);
            tokens = torch::cat({tokens, next}, 1);
        }
        return tokens;
    }
};
TORCH_MODULE(JimaModel);

// ─── Backward compatibility alias ────────────────────────────────
// Old code that references ZeroAgenticModel will still compile,
// but it now uses the full XORZEN architecture.
using ZeroAgenticModel = JimaModel;
using ZeroAgenticModelImpl = JimaModelImpl;

} // namespace xorzen

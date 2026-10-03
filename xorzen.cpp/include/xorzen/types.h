#pragma once

#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace xorzen {

// ============================================================================
// Core Data Structures
// ============================================================================

struct ModelOutput {
    torch::Tensor logits;                          // [batch, seq, vocab]
    torch::Tensor loss;                            // scalar
    torch::Tensor lm_loss;                         // scalar before auxiliary losses
    torch::Tensor cot_vector;                      // [batch, seq, cot_dim]
    torch::Tensor routing_loss;                    // scalar
    torch::Tensor load_balance_loss;              // scalar
    torch::Tensor cot_consistency_loss;           // scalar
    
    std::vector<torch::Tensor> layer_outputs;     // intermediate states
    std::unordered_map<std::string, float> expert_stats;  // expert usage stats
    torch::Tensor agentic_actions;                // [batch, seq, num_actions]
    
    int64_t active_params = 0;
    double compute_cost = 0.0;
    
    // Total loss as returned by model forward.
    // Forward already folds in LM, routing, load-balance, and auxiliary terms,
    // so this accessor should not add them again.
    torch::Tensor total_loss() const {
        return loss.clone();
    }
};

struct GenerationConfig {
    int64_t max_length = 128;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int64_t top_k = 50;
    float repetition_penalty = 1.0f;
    float length_penalty = 1.0f;
    int64_t num_beams = 1;
    bool early_stopping = true;
    bool do_sample = true;
    int64_t pad_token_id = 0;
    int64_t eos_token_id = 2;
    bool use_cache = true;
};

// ============================================================================
// Configuration Structures
// ============================================================================
struct ModelConfig {
    // Model architecture. Names follow the C++ API, while several aliases below
    // intentionally mirror the Python zeroModel config for straightforward porting.
    std::string model_name = "AniXO";
    int64_t vocab_size = 32000;
    int64_t d_model = 768;
    int64_t hidden_size = 768;
    int64_t n_layers = 12;
    int64_t num_layers = 12;
    int64_t max_depth = 12;
    int64_t min_depth = 1;
    int64_t n_heads = 12;
    int64_t num_attention_heads = 12;
    int64_t n_kv_heads = 4;  // Grouped-query attention
    int64_t d_ff = 3072;
    int64_t max_seq_len = 2048;
    int64_t context_length = 2048;
    bool tie_word_embeddings = true;

    // MoE configuration
    int64_t num_experts = 192;
    int64_t expert_count = 192;
    int64_t experts_per_token = 2;
    int64_t top_k_experts = 2;
    int64_t expert_dim = 512;
    float expert_hidden_multiplier = 4.0f;
    bool use_disk_cache = true;
    int64_t cache_size_mb = 512;
    int64_t max_expert_cache = 24;
    std::string expert_shard_dir = "experts_cpp";
    bool shard_experts = false;
    int64_t router_num_layers = 1;
    int64_t merger_num_layers = 1;
    bool use_sliced_ffn = true;
    bool causal = true;
    bool use_moe = true;
    bool test_mode = false;
    float target_active_ratio = 0.1f;
    bool unify_load_balance = true;
    float load_balancing_weight = 0.01f;

    // AniXO Chunk Memory
    int64_t chunk_size_frames = 96;     // 4 seconds at 24fps
    int64_t memory_slots = 4096;        // how many chunks to remember
    int64_t memory_key_dim = 256;
    int64_t memory_val_dim = 512;
    int64_t memory_top_k = 64;          // how many memory slots to attend per step
    bool use_chunk_memory = true;
    std::string memory_shard_dir = "memory_chunks";

    // AniXO Character Routing
    int64_t max_characters = 64;
    int64_t char_emb_dim = 256;
    int64_t experts_per_character = 3;

    // AniXO Multimodal
    int64_t visual_token_vocab = 8192;  // VQVAE codebook size
    int64_t audio_token_vocab = 1024;   // EnCodec codebook size
    int64_t patch_size = 16;            // visual patch size

    // IGRIS Agentic Capabilities
    bool use_recursion = false;
    int64_t max_recursion_depth = 1;
    int64_t num_action_slots = 16;
    bool use_self_critique = false;
    float ponder_loss_weight = 0.01f;

    // HASS configuration
    bool use_local_attention = true;
    bool use_low_rank_global = true;
    bool use_ssm = true;
    int64_t local_window = 256;
    int64_t local_window_size = 256;
    int64_t rank = 64;
    int64_t low_rank_dim = 64;
    std::vector<int64_t> width_choices = {384, 576, 768};
    std::string hidden_act = "gelu";
    
    // SSM configuration
    int64_t ssm_d_state = 16;
    int64_t ssm_state_dim = 16;
    int64_t ssm_d_conv = 4;
    int64_t ssm_kernel_size = 3;
    float ssm_dt_rank_ratio = 0.5f;
    
    // CoT configuration
    int64_t cot_latent_dim = 256;
    int64_t cot_dim = 256;
    int64_t cot_components = 6;
    int64_t num_cot_layers = 6;
    float cot_loss_weight = 0.1f;
    float cot_consistency_weight = 0.1f;
    std::string cot_update_method = "gru";
    int64_t cot_transformer_heads = 4;
    int64_t cot_transformer_layers = 1;
    float cot_diversity_weight = 0.01f;
    float cot_sparsity_weight = 0.001f;
    float cot_orthogonality_weight = 0.05f;


    // Merger configuration
    std::string merger_type = "gated";      // "gated" | "linear"
    float merger_hidden_multiplier = 2.0f;  // hidden multiplier for gate controller

    // Router configuration
    int64_t router_hidden_dim = 128;
    float router_temperature = 1.0f;
    bool router_temperature_annealing = true;
    float router_dropout = 0.1f;

    // Cost-aware routing (Python routing.py:546-588) — was missing in C++.
    bool cost_aware_routing = true;
    float compute_budget = 1.0f;
    // Eval-mode Gumbel noise (Python routing.py:442-475, default 0.15).
    float eval_routing_noise = 0.15f;

    // Auxiliary loss weights (Python routing.py:649-658, 1689)
    float routing_loss_weight = 0.01f;   // uncertainty weight (Python default 0.01)
    float lb_loss_weight = 0.0001f;       // load-balance (Switch formula) weight
    float z_loss_weight = 0.0001f;        // router z-loss weight
    float path_div_weight = 0.2f;         // path diversity (entropy) weight
    float width_div_weight = 0.1f;        // width diversity (entropy) weight
    
    // Training configuration
    float dropout = 0.1f;
    float layer_norm_eps = 1e-5f;
    bool use_rope = true;
    float rope_theta = 10000.0f;
    bool gradient_checkpointing = false;
    int64_t pad_token_id = 0;
    int64_t eos_token_id = 2;
    
    // Quantization
    bool use_quantization = false;
    std::string quant_type = "none";  // "none", "int8", "bf16"
    
    // Device
    std::string device = "cpu";  // "cpu", "cuda", "mps"
    
    // Keep aliases coherent after callers mutate either naming style.
    void normalize() {
        if (hidden_size != d_model) d_model = hidden_size;
        else hidden_size = d_model;
        if (num_layers != n_layers) n_layers = num_layers;
        else num_layers = n_layers;
        max_depth = num_layers;
        if (num_attention_heads != n_heads) n_heads = num_attention_heads;
        else num_attention_heads = n_heads;
        if (context_length != max_seq_len) max_seq_len = context_length;
        else context_length = max_seq_len;
        if (expert_count != num_experts) num_experts = expert_count;
        else expert_count = num_experts;
        if (top_k_experts != experts_per_token) experts_per_token = top_k_experts;
        else top_k_experts = experts_per_token;
        if (local_window_size != local_window) local_window = local_window_size;
        else local_window_size = local_window;
        if (low_rank_dim != rank) rank = low_rank_dim;
        else low_rank_dim = rank;
        if (ssm_state_dim != ssm_d_state) ssm_d_state = ssm_state_dim;
        else ssm_state_dim = ssm_d_state;
        if (cot_dim != cot_latent_dim) cot_latent_dim = cot_dim;
        else cot_dim = cot_latent_dim;
        if (width_choices.empty()) width_choices = {hidden_size / 2, hidden_size * 3 / 4, hidden_size};
        d_ff = static_cast<int64_t>(hidden_size * expert_hidden_multiplier);
        expert_dim = d_ff;
    }

    int64_t cot_total_dim() const {
        return cot_dim * cot_components;
    }

    void print() const;
};

struct ExpertInfo {
    int64_t expert_id;
    float weight;
    int64_t load_count = 0;
    int64_t hit_count = 0;
    double avg_activation = 0.0;
};

struct RoutingInfo {
    torch::Tensor expert_indices;   // [batch, seq, top_k]
    torch::Tensor expert_weights;   // [batch, seq, top_k]
    std::vector<ExpertInfo> expert_stats;
    float load_balance_loss = 0.0f;
};

// ============================================================================
// Utility Functions
// ============================================================================

// Check if CUDA is available
inline bool cuda_available() {
    return torch::cuda::is_available();
}

// Get optimal device
inline torch::Device get_device(const std::string& device_str = "auto") {
    if (device_str == "auto") {
        return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    }
    if (device_str == "cuda") return torch::kCUDA;
    if (device_str == "cpu") return torch::kCPU;
    return torch::kCPU;
}

// Print tensor stats (for debugging)
inline void print_tensor_stats(const torch::Tensor& t, const std::string& name) {
    if (!t.defined()) {
        std::cout << name << ": undefined" << std::endl;
        return;
    }
    auto mean = t.mean().item<float>();
    auto std = t.std().item<float>();
    auto min = t.min().item<float>();
    auto max = t.max().item<float>();
    std::cout << name << ": shape=" << t.sizes() 
              << " mean=" << mean << " std=" << std 
              << " min=" << min << " max=" << max << std::endl;
}

} // namespace xorzen

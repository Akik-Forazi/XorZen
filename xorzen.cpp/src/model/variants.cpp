// ============================================================
//  xorzen.cpp — src/model/variants.cpp
//  ConfigFactory: precision-tuned model size configs
//  Ported from xorzen/models/zero/variants.py + config.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/variants.h"
#include <stdexcept>

namespace xorzen {

// All configs follow ZARX architectural equations:
//   d_ff = hidden_size * 4  (approx via expert_hidden_multiplier)
//   cot_dim = 256, cot_components = 6  (always)
//   n_kv_heads = n_heads / 3  (GQA)
//   width_choices = {H/2, H*3/4, H}

ModelConfig ConfigFactory::get_config(ModelSize size) {
    ModelConfig c;

    switch (size) {

    // ── TINY 23K ──────────────────────────────────────────
    case ModelSize::TINY_23K:
        c.model_name           = "xorzen-tiny-23k";
        c.vocab_size           = 512;
        c.hidden_size          = 32;
        c.num_layers           = 2;
        c.num_attention_heads  = 4;
        c.n_kv_heads           = 1;
        c.num_experts          = 4;
        c.top_k_experts        = 1;
        c.max_expert_cache     = 4;
        c.expert_hidden_multiplier = 2.0f;
        c.cot_dim              = 16;
        c.cot_components       = 6;
        c.ssm_d_state          = 4;
        c.ssm_d_conv           = 2;
        c.context_length       = 128;
        c.router_hidden_dim    = 16;
        c.width_choices        = {16, 24, 32};
        c.cot_loss_weight      = 0.001f;
        c.load_balancing_weight = 0.0001f;
        break;

    // ── NANO 1M ───────────────────────────────────────────
    case ModelSize::NANO_1M:
        c.model_name           = "xorzen-nano-1m";
        c.vocab_size           = 8000;
        c.hidden_size          = 128;
        c.num_layers           = 4;
        c.num_attention_heads  = 4;
        c.n_kv_heads           = 2;
        c.num_experts          = 16;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 8;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 64;
        c.cot_components       = 6;
        c.ssm_d_state          = 8;
        c.ssm_d_conv           = 3;
        c.context_length       = 512;
        c.router_hidden_dim    = 64;
        c.width_choices        = {64, 96, 128};
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        break;

    // ── NANO 10M ──────────────────────────────────────────
    case ModelSize::NANO_10M:
        c.model_name           = "xorzen-nano-10m";
        c.vocab_size           = 16000;
        c.hidden_size          = 256;
        c.num_layers           = 6;
        c.num_attention_heads  = 8;
        c.n_kv_heads           = 2;
        c.num_experts          = 32;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 12;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 128;
        c.cot_components       = 6;
        c.ssm_d_state          = 12;
        c.ssm_d_conv           = 3;
        c.context_length       = 1024;
        c.router_hidden_dim    = 128;
        c.width_choices        = {128, 192, 256};
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        break;

    // ── MICRO 50M ─────────────────────────────────────────
    case ModelSize::MICRO_50M:
        c.model_name           = "xorzen-micro-50m";
        c.vocab_size           = 32000;
        c.hidden_size          = 512;
        c.num_layers           = 8;
        c.num_attention_heads  = 8;
        c.n_kv_heads           = 4;
        c.num_experts          = 64;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 16;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 192;
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 2048;
        c.router_hidden_dim    = 128;
        c.width_choices        = {256, 384, 512};
        c.cot_loss_weight      = 0.05f;
        c.load_balancing_weight = 0.005f;
        break;

    // ── MINI 277M (Flagship) ──────────────────────────────
    case ModelSize::MINI_277M:
        c.model_name           = "xorzen-mini-277m";
        c.vocab_size           = 32000;
        c.hidden_size          = 768;
        c.num_layers           = 16;
        c.num_attention_heads  = 12;
        c.n_kv_heads           = 4;
        c.num_experts          = 192;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 24;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 256;
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 4096;
        c.router_hidden_dim    = 256;
        c.width_choices        = {384, 576, 768};
        c.cot_loss_weight      = 0.1f;
        c.load_balancing_weight = 0.01f;
        c.ssm_kernel_size      = 3;
        break;

    // ── SMALL 500M ────────────────────────────────────────
    case ModelSize::SMALL_500M:
        c.model_name           = "xorzen-small-500m";
        c.vocab_size           = 32000;
        c.hidden_size          = 1024;
        c.num_layers           = 20;
        c.num_attention_heads  = 16;
        c.n_kv_heads           = 4;
        c.num_experts          = 256;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 32;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 256;
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 8192;
        c.router_hidden_dim    = 256;
        c.width_choices        = {512, 768, 1024};
        c.cot_loss_weight      = 0.1f;
        c.load_balancing_weight = 0.01f;
        break;

    // ── MEDIUM 1B ─────────────────────────────────────────
    case ModelSize::MEDIUM_1B:
        c.model_name           = "xorzen-medium-1b";
        c.vocab_size           = 32000;
        c.hidden_size          = 1536;
        c.num_layers           = 24;
        c.num_attention_heads  = 16;
        c.n_kv_heads           = 8;
        c.num_experts          = 384;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 48;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 256;
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 8192;
        c.router_hidden_dim    = 256;
        c.width_choices        = {768, 1152, 1536};
        c.cot_loss_weight      = 0.1f;
        c.load_balancing_weight = 0.01f;
        break;

    // ── XL 3B ─────────────────────────────────────────────
    case ModelSize::XL_3B:
        c.model_name           = "xorzen-xl-3b";
        c.vocab_size           = 32000;
        c.hidden_size          = 2560;
        c.num_layers           = 32;
        c.num_attention_heads  = 32;
        c.n_kv_heads           = 8;
        c.num_experts          = 512;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 64;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 256;
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 16384;
        c.router_hidden_dim    = 512;
        c.width_choices        = {1280, 1920, 2560};
        c.cot_loss_weight      = 0.1f;
        c.load_balancing_weight = 0.01f;
        break;

    // ── XL 7B ─────────────────────────────────────────────
    case ModelSize::XL_7B:
        c.model_name           = "xorzen-xl-7b";
        c.vocab_size           = 32000;
        c.hidden_size          = 4096;
        c.num_layers           = 32;
        c.num_attention_heads  = 32;
        c.n_kv_heads           = 8;
        c.num_experts          = 1024;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 128;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 256;
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 32768;
        c.router_hidden_dim    = 512;
        c.width_choices        = {2048, 3072, 4096};
        c.cot_loss_weight      = 0.1f;
        c.load_balancing_weight = 0.01f;
        c.use_rope             = true;
        break;

    // ── IGRIS NANO (~5M) ──────────────────────────────────
    case ModelSize::IGRIS_NANO:
        c.model_name           = "xorzen-igris-nano";
        c.vocab_size           = 10000;
        c.hidden_size          = 256;
        c.num_layers           = 6;
        c.num_attention_heads  = 8;
        c.n_kv_heads           = 2;
        c.num_experts          = 32;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 12;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 128;
        c.cot_components       = 6;
        c.ssm_d_state          = 12;
        c.ssm_d_conv           = 3;
        c.context_length       = 2048;
        c.router_hidden_dim    = 128;
        c.width_choices        = {128, 192, 256};
        
        // IGRIS Agentic Bits
        c.use_recursion        = true;
        c.max_recursion_depth  = 3;
        c.use_self_critique    = true;
        c.num_action_slots     = 16;
        break;

    // ── IGRIS MICRO (~50M) ────────────────────────────────
    case ModelSize::IGRIS_MICRO:
        c.model_name           = "xorzen-igris-micro";
        c.vocab_size           = 10000;
        c.hidden_size          = 512;
        c.num_layers           = 12;
        c.num_attention_heads  = 12;
        c.n_kv_heads           = 4;
        c.num_experts          = 64;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 16;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 192;
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 4096;
        c.router_hidden_dim    = 256;
        c.width_choices        = {256, 384, 512};
        
        // IGRIS Agentic Bits
        c.use_recursion        = true;
        c.max_recursion_depth  = 4;
        c.use_self_critique    = true;
        c.num_action_slots     = 32;
        break;

    default:
        throw std::invalid_argument("Unknown ModelSize");
    }

    // Common defaults
    c.num_cot_layers           = 6;
    c.dropout                  = 0.1f;
    c.layer_norm_eps           = 1e-5f;
    c.tie_word_embeddings      = true;
    c.gradient_checkpointing   = false;
    c.use_disk_cache           = true;
    c.expert_shard_dir         = "experts/" + c.model_name;
    c.merger_type              = "gated";
    c.merger_hidden_multiplier = 2.0f;
    c.use_ssm                  = true;
    c.use_rope                 = (size >= ModelSize::MINI_277M);

    // Sync all alias fields
    c.normalize();
    return c;
}

std::string ConfigFactory::size_name(ModelSize size) {
    switch (size) {
    case ModelSize::TINY_23K:   return "TINY_23K";
    case ModelSize::NANO_1M:    return "NANO_1M";
    case ModelSize::NANO_10M:   return "NANO_10M";
    case ModelSize::MICRO_50M:  return "MICRO_50M";
    case ModelSize::MINI_277M:  return "MINI_277M";
    case ModelSize::SMALL_500M: return "SMALL_500M";
    case ModelSize::MEDIUM_1B:  return "MEDIUM_1B";
    case ModelSize::XL_3B:      return "XL_3B";
    case ModelSize::XL_7B:      return "XL_7B";
    default: return "UNKNOWN";
    }
}

int64_t ConfigFactory::param_target(ModelSize size) {
    switch (size) {
    case ModelSize::TINY_23K:   return 37'824;
    case ModelSize::NANO_1M:    return 1'077'503;
    case ModelSize::NANO_10M:   return 10'970'548;
    case ModelSize::MICRO_50M:  return 50'399'371;
    case ModelSize::MINI_277M:  return 277'000'335;
    case ModelSize::SMALL_500M: return 500'000'083;
    case ModelSize::MEDIUM_1B:  return 1'000'000'886;
    case ModelSize::XL_3B:      return 3'000'000'822;
    case ModelSize::XL_7B:      return 7'000'000'466;
    default: return 0;
    }
}

} // namespace xorzen

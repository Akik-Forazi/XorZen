// ============================================================
//  xorzen.cpp — src/model/variants.cpp
//  ConfigFactory: EXACT Python parity for model size configs
//  Ported from xorzen/config.py — every value must match Python
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/variants.h"
#include <stdexcept>

namespace xorzen {

// All configs are EXACT copies of xorzen/config.py ConfigFactory.get_config().
// DO NOT modify these values without updating the Python source first.
// Python is the source of truth — if C++ and Python disagree, C++ is wrong.

ModelConfig ConfigFactory::get_config(ModelSize size) {
    ModelConfig c;

    switch (size) {

    // ── TINY 23K ──────────────────────────────────────────
    // Python: _h = 8, vocab=8, ctx=32, layers=1, heads=2, experts=1, top_k=1
    case ModelSize::TINY_23K:
        c.model_name           = "xorzen_tiny_23k";
        c.vocab_size           = 8;
        c.hidden_size          = 8;
        c.num_layers           = 1;
        c.num_attention_heads  = 2;
        c.n_kv_heads           = 2;  // no GQA in Python tiny
        c.num_experts          = 1;
        c.expert_count         = 1;
        c.top_k_experts        = 1;
        c.max_expert_cache     = 1;
        c.low_rank_dim         = 96;  // Python: low_rank_dim=96 for tiny_23k
        c.ssm_state_dim        = 16;  // Python: ssm_state_dim=16
        c.ssm_d_state          = 16;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 2;
        c.cot_components       = 6;
        c.ssm_d_conv           = 4;
        c.context_length       = 32;
        c.max_depth            = 1;
        c.min_depth            = 1;
        c.router_hidden_dim    = 1;
        c.router_num_layers   = 1;
        c.merger_num_layers   = 1;
        c.width_choices        = {8};  // Python: (_h,) = (8,)
        c.shard_experts        = false;
        c.cot_loss_weight      = 0.001f;
        c.load_balancing_weight = 0.0001f;
        c.pad_token_id         = 0;
        c.gradient_checkpointing = false;
        break;

    // ── NANO 1M ───────────────────────────────────────────
    // Python: _h = 64, vocab=6765, ctx=128, layers=3, heads=4, experts=2, top_k=1
    case ModelSize::NANO_1M:
        c.model_name           = "xorzen_nano_1m";
        c.vocab_size           = 6765;
        c.hidden_size          = 64;
        c.num_layers           = 3;
        c.num_attention_heads  = 4;
        c.n_kv_heads           = 4;  // no GQA
        c.num_experts          = 2;
        c.top_k_experts        = 1;
        c.max_expert_cache     = 2;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 16;  // Python: _h // 4 = 64 // 4 = 16
        c.cot_components       = 6;
        c.ssm_d_state          = 8;
        c.ssm_d_conv           = 4;
        c.context_length       = 128;
        c.max_depth            = 3;
        c.min_depth            = 1;
        c.router_hidden_dim    = 16;  // Python: _h // 4
        c.width_choices        = {64};  // Python: (_h,) = (64,)
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        c.pad_token_id         = 0;
        c.gradient_checkpointing = false;  // Python: False (no memory pressure)
        break;

    // ── NANO 10M ──────────────────────────────────────────
    // Python: _h = 192, vocab=10000, ctx=512, layers=6, heads=8, experts=8, top_k=2
    case ModelSize::NANO_10M:
        c.model_name           = "xorzen_nano_10m";
        c.vocab_size           = 10000;
        c.hidden_size          = 192;
        c.num_layers           = 6;
        c.num_attention_heads  = 8;
        c.n_kv_heads           = 8;
        c.num_experts          = 8;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 8;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 48;  // Python: _h // 4 = 192 // 4 = 48
        c.cot_components       = 6;
        c.ssm_d_state          = 12;
        c.ssm_d_conv           = 4;
        c.context_length       = 512;
        c.max_depth            = 6;
        c.min_depth            = 2;
        c.router_hidden_dim    = 48;  // Python: _h // 4
        c.width_choices        = {96, 192};  // Python: (_h // 2, _h) = (96, 192)
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        c.pad_token_id         = 0;
        c.gradient_checkpointing = true;  // Python default
        break;

    // ── MICRO 50M ─────────────────────────────────────────
    // Python: _h = 256, vocab=10000, ctx=1024, layers=10, heads=8, experts=43, top_k=2
    case ModelSize::MICRO_50M:
        c.model_name           = "xorzen_micro_50m";
        c.vocab_size           = 10000;
        c.hidden_size          = 256;
        c.num_layers           = 10;
        c.num_attention_heads  = 8;
        c.n_kv_heads           = 8;
        c.num_experts          = 43;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 24;  // Python: LRU capacity = min(24, num_experts)
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 64;  // Python: _h // 4 = 256 // 4 = 64
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 1024;
        c.max_depth            = 10;
        c.min_depth            = 3;
        c.router_hidden_dim    = 64;  // Python: _h // 4
        c.width_choices        = {128, 256};  // Python: (_h // 2, _h) = (128, 256)
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        c.pad_token_id         = 0;
        c.gradient_checkpointing = true;  // Python default
        break;

    // ── MINI 277M ─────────────────────────────────────────
    // Python: _h = 512, vocab=33898, ctx=1024, layers=13, heads=16, experts=64, top_k=2
    case ModelSize::MINI_277M:
        c.model_name           = "xorzen_zero_277m";
        c.vocab_size           = 33898;
        c.hidden_size          = 512;
        c.num_layers           = 13;
        c.num_attention_heads  = 16;
        c.n_kv_heads           = 16;
        c.num_experts          = 64;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 24;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 64;  // Python: _h // 8 = 512 // 8 = 64
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 1024;
        c.max_depth            = 13;
        c.min_depth            = 4;
        c.router_hidden_dim    = 64;  // Python: _h // 8
        c.width_choices        = {256, 512};  // Python: (_h // 2, _h)
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        c.pad_token_id         = 0;
        c.gradient_checkpointing = true;  // Python default
        break;

    // ── SMALL 500M ────────────────────────────────────────
    // Python: _h = 640, vocab=64563, ctx=8192, layers=16, heads=16, experts=69, top_k=2
    case ModelSize::SMALL_500M:
        c.model_name           = "xorzen_small_500m";
        c.vocab_size           = 64563;
        c.hidden_size          = 640;
        c.num_layers           = 16;
        c.num_attention_heads  = 16;
        c.n_kv_heads           = 16;
        c.num_experts          = 69;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 24;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 160;  // Python: _h // 4 = 640 // 4 = 160
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 8192;
        c.max_depth            = 16;
        c.min_depth            = 4;
        c.router_hidden_dim    = 160;  // Python: _h // 4
        c.width_choices        = {320, 640};  // Python: (_h // 2, _h)
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        c.pad_token_id         = 0;
        c.gradient_checkpointing = true;
        break;

    // ── MEDIUM 1B ─────────────────────────────────────────
    // Python: _h = 896, vocab=68003, ctx=8192, layers=24, heads=16, experts=64, top_k=2
    case ModelSize::MEDIUM_1B:
        c.model_name           = "xorzen_medium_1b";
        c.vocab_size           = 68003;
        c.hidden_size          = 896;
        c.num_layers           = 24;
        c.num_attention_heads  = 16;
        c.n_kv_heads           = 16;
        c.num_experts          = 64;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 24;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 224;  // Python: _h // 4 = 896 // 4 = 224
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 8192;
        c.max_depth            = 24;
        c.min_depth            = 4;
        c.router_hidden_dim    = 224;  // Python: _h // 4
        c.width_choices        = {448, 896};  // Python: (_h // 2, _h)
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        c.pad_token_id         = 0;
        c.gradient_checkpointing = true;
        break;

    // ── LARGE 3B ──────────────────────────────────────────
    // Python: _h = 1280, vocab=94689, ctx=8192, layers=32, heads=32, experts=104, top_k=2
    case ModelSize::LARGE_3B:
        c.model_name           = "xorzen_large_3b";
        c.vocab_size           = 94689;
        c.hidden_size          = 1280;
        c.num_layers           = 32;
        c.num_attention_heads  = 32;
        c.n_kv_heads           = 32;
        c.num_experts          = 104;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 24;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 320;  // Python: _h // 4 = 1280 // 4 = 320
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 8192;
        c.max_depth            = 32;
        c.min_depth            = 5;
        c.router_hidden_dim    = 320;  // Python: _h // 4
        c.width_choices        = {640, 1280};  // Python: (_h // 2, _h)
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        c.pad_token_id         = 0;
        c.gradient_checkpointing = true;
        break;

    // ── XL 7B ─────────────────────────────────────────────
    // Python: _h = 1792, vocab=98425, ctx=8192, layers=48, heads=32, experts=116, top_k=2
    case ModelSize::XL_7B:
        c.model_name           = "xorzen_xl_7b";
        c.vocab_size           = 98425;
        c.hidden_size          = 1792;
        c.num_layers           = 48;
        c.num_attention_heads  = 32;
        c.n_kv_heads           = 32;
        c.num_experts          = 116;
        c.top_k_experts        = 2;
        c.max_expert_cache     = 24;
        c.expert_hidden_multiplier = 4.0f;
        c.cot_dim              = 448;  // Python: _h // 4 = 1792 // 4 = 448
        c.cot_components       = 6;
        c.ssm_d_state          = 16;
        c.ssm_d_conv           = 4;
        c.context_length       = 8192;
        c.max_depth            = 48;
        c.min_depth            = 6;
        c.router_hidden_dim    = 448;  // Python: _h // 4
        c.width_choices        = {896, 1792};  // Python: (_h // 2, _h)
        c.cot_loss_weight      = 0.01f;
        c.load_balancing_weight = 0.001f;
        c.pad_token_id         = 0;
        c.gradient_checkpointing = true;
        break;

    default:
        throw std::runtime_error("Unknown ModelSize in ConfigFactory::get_config()");
    }

    // Common defaults matching Python's ModelConfig dataclass defaults
    c.tie_word_embeddings    = true;
    c.causal                = true;
    c.use_sliced_ffn        = true;
    c.use_moe               = true;
    c.test_mode             = false;
    c.local_window_size     = std::min<int64_t>(128, c.context_length);
    // Only set low_rank_dim if not already set by the case-specific config
    if (c.low_rank_dim == 64) {  // still at default
        c.low_rank_dim = c.hidden_size * 3 / 8;
    }
    if (c.ssm_state_dim == 16 && c.ssm_d_state == 16) {
        // ssm_state_dim already set by case or default
    }
    c.target_active_ratio   = 0.1f;  // Python default
    c.unify_load_balance    = true;  // Python default — zeros load_balance_loss

    return c;
}

int64_t ConfigFactory::param_target(ModelSize size) {
    // These MUST match Python's PARAM_COUNT in variants.py exactly.
    switch (size) {
    case ModelSize::TINY_23K:  return 37'824;
    case ModelSize::NANO_1M:   return 1'077'503;
    case ModelSize::NANO_10M:  return 10'970'548;
    case ModelSize::MICRO_50M: return 50'399'371;
    case ModelSize::MINI_277M: return 277'000'335;
    case ModelSize::SMALL_500M:return 500'000'083;
    case ModelSize::MEDIUM_1B: return 1'000'000'886;
    case ModelSize::LARGE_3B:  return 3'000'000'000;  // approximate
    case ModelSize::XL_7B:     return 7'000'000'466;
    default: return 0;
    }
}

} // namespace xorzen

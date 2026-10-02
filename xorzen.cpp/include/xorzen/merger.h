#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/merger.h
//  MergerGate: fuses HASS + MoE + CoT streams
//  Ported from xorzen/model/components/merger.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================

#include <torch/torch.h>
#include "xorzen/types.h"
#include <string>
#include <unordered_map>

namespace xorzen {

// ============================================================
//  LinearMerger — concat → single linear projection (baseline)
// ============================================================
struct LinearMergerImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t total_cot_dim;
    int64_t input_dim;

    torch::nn::Linear output_proj{nullptr};

    explicit LinearMergerImpl(const ModelConfig& cfg);

    // [B,T,H], [B,T,H], [B,T,C] → [B,T,H]
    torch::Tensor forward(const torch::Tensor& hass,
                          const torch::Tensor& moe,
                          const torch::Tensor& cot);

    std::unordered_map<std::string, int64_t> get_complexity() const;
};
TORCH_MODULE(LinearMerger);

// ============================================================
//  GatedMerger — learns per-token gates for HASS/MoE/CoT
// ============================================================
struct GatedMergerImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t total_cot_dim;
    int64_t input_dim;
    int64_t merger_hidden_dim;

    // Gate controller: concat → hidden → 3 scalars
    torch::nn::Sequential gate_controller{nullptr};

    // Project CoT from total_cot_dim → hidden_dim
    torch::nn::Linear cot_proj{nullptr};

    torch::nn::LayerNorm output_norm{nullptr};
    torch::nn::Dropout dropout_{nullptr};

    explicit GatedMergerImpl(const ModelConfig& cfg);

    torch::Tensor forward(const torch::Tensor& hass,
                          const torch::Tensor& moe,
                          const torch::Tensor& cot);

    std::unordered_map<std::string, int64_t> get_complexity() const;

private:
    void init_weights();
};
TORCH_MODULE(GatedMerger);

// ============================================================
//  XorzenMergerGate — configurable wrapper (default: gated)
//  This is what the main model uses
// ============================================================
struct XorzenMergerGateImpl : torch::nn::Module {
    std::string merger_type;

    // One of the following will be populated depending on merger_type
    LinearMerger linear_impl{nullptr};
    GatedMerger  gated_impl{nullptr};

    explicit XorzenMergerGateImpl(const ModelConfig& cfg);

    // [B,T,H], [B,T,H], [B,T,C] → [B,T,H]
    torch::Tensor forward(const torch::Tensor& hass,
                          const torch::Tensor& moe,
                          const torch::Tensor& cot,
                          const torch::Tensor& mask = {});

    std::unordered_map<std::string, int64_t> get_complexity() const;
};
TORCH_MODULE(XorzenMergerGate);

} // namespace xorzen

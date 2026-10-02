// ============================================================
//  xorzen.cpp — src/model/merger.cpp
//  Full merger gate implementation
//  Ported from xorzen/model/components/merger.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/merger.h"
#include "xorzen/optimized/simd_ops.h"

#include <stdexcept>
#include <string>

namespace xorzen {

// ============================================================
//  LinearMerger
// ============================================================

LinearMergerImpl::LinearMergerImpl(const ModelConfig& cfg)
    : hidden_dim(cfg.hidden_size),
      total_cot_dim(cfg.cot_dim * cfg.cot_components),
      input_dim(cfg.hidden_size * 2 + cfg.cot_dim * cfg.cot_components) {

    output_proj = register_module("output_proj",
        torch::nn::Linear(input_dim, hidden_dim));
    torch::nn::init::xavier_uniform_(output_proj->weight);
    torch::nn::init::zeros_(output_proj->bias);
}

torch::Tensor LinearMergerImpl::forward(const torch::Tensor& hass,
                                        const torch::Tensor& moe,
                                        const torch::Tensor& cot) {
    if (hass.size(-1) != hidden_dim)
        throw std::invalid_argument("LinearMerger: hass hidden_dim mismatch");
    if (moe.size(-1) != hidden_dim)
        throw std::invalid_argument("LinearMerger: moe hidden_dim mismatch");
    if (cot.size(-1) != total_cot_dim)
        throw std::invalid_argument("LinearMerger: cot total_cot_dim mismatch");

    auto merged = torch::cat({hass, moe, cot}, -1);  // [B, T, 2H+C]
    return output_proj->forward(merged);              // [B, T, H]
}

std::unordered_map<std::string, int64_t> LinearMergerImpl::get_complexity() const {
    int64_t params = 0;
    for (const auto& p : parameters()) params += p.numel();
    return {{"macs", input_dim * hidden_dim}, {"params", params}};
}

// ============================================================
//  GatedMerger
// ============================================================

GatedMergerImpl::GatedMergerImpl(const ModelConfig& cfg)
    : hidden_dim(cfg.hidden_size),
      total_cot_dim(cfg.cot_dim * cfg.cot_components),
      input_dim(cfg.hidden_size * 2 + cfg.cot_dim * cfg.cot_components),
      merger_hidden_dim(static_cast<int64_t>(cfg.hidden_size * cfg.merger_hidden_multiplier)) {

    // Gate controller: input → merger_hidden → 3 gate scalars
    gate_controller = register_module("gate_controller",
        torch::nn::Sequential(
            torch::nn::Linear(input_dim, merger_hidden_dim),
            torch::nn::SiLU(),
            torch::nn::Linear(merger_hidden_dim, 3)));

    // Project CoT → hidden_dim so it can be added to HASS/MoE
    cot_proj = register_module("cot_proj",
        torch::nn::Linear(total_cot_dim, hidden_dim));

    output_norm = register_module("output_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})
            .eps(cfg.layer_norm_eps)));

    dropout_ = register_module("dropout",
        torch::nn::Dropout(cfg.dropout));

    init_weights();
}

void GatedMergerImpl::init_weights() {
    for (auto& module : gate_controller->modules()) {
        if (auto* lin = module->as<torch::nn::Linear>()) {
            torch::nn::init::xavier_uniform_(lin->weight);
            if (lin->bias.defined())
                torch::nn::init::zeros_(lin->bias);
        }
    }
    torch::nn::init::xavier_uniform_(cot_proj->weight);
    torch::nn::init::zeros_(cot_proj->bias);
}

torch::Tensor GatedMergerImpl::forward(const torch::Tensor& hass,
                                       const torch::Tensor& moe,
                                       const torch::Tensor& cot) {
    if (hass.size(-1) != hidden_dim)
        throw std::invalid_argument("GatedMerger: hass hidden_dim mismatch");
    if (moe.size(-1) != hidden_dim)
        throw std::invalid_argument("GatedMerger: moe hidden_dim mismatch");
    if (cot.size(-1) != total_cot_dim)
        throw std::invalid_argument("GatedMerger: cot total_cot_dim mismatch");

    // Concatenate all streams for gate computation
    auto gate_input = torch::cat({hass, moe, cot}, -1);  // [B, T, 2H+C]

    // Gate values: [B, T, 3]
    auto gates = gate_controller->forward(gate_input);
    auto gate_weights = optimized::softmax_simd(gates, -1);        // softmax over 3 streams

    // Split into per-stream gates: each [B, T, 1]
    auto g = gate_weights.chunk(3, -1);
    auto g_hass = g[0];   // [B, T, 1]
    auto g_moe  = g[1];   // [B, T, 1]
    auto g_cot  = g[2];   // [B, T, 1]

    // Project CoT to hidden_dim
    auto cot_h = cot_proj->forward(cot);                  // [B, T, H]

    // Weighted fusion
    auto fused = g_hass * hass + g_moe * moe + g_cot * cot_h;  // [B, T, H]

    // Normalise + dropout
    return dropout_->forward(output_norm->forward(fused));
}

std::unordered_map<std::string, int64_t> GatedMergerImpl::get_complexity() const {
    int64_t params = 0;
    for (const auto& p : parameters()) params += p.numel();
    return {
        {"macs", input_dim * merger_hidden_dim + merger_hidden_dim * 3},
        {"params", params}
    };
}

// ============================================================
//  XorzenMergerGate — configurable wrapper
// ============================================================

XorzenMergerGateImpl::XorzenMergerGateImpl(const ModelConfig& cfg) {
    merger_type = cfg.merger_type.empty() ? "gated" : cfg.merger_type;

    if (merger_type == "linear") {
        linear_impl = register_module("merger_impl", LinearMerger(cfg));
    } else if (merger_type == "gated") {
        gated_impl  = register_module("merger_impl", GatedMerger(cfg));
    } else {
        throw std::invalid_argument("Unknown merger_type: " + merger_type);
    }
}

torch::Tensor XorzenMergerGateImpl::forward(const torch::Tensor& hass,
                                            const torch::Tensor& moe,
                                            const torch::Tensor& cot,
                                            const torch::Tensor& /*mask*/) {
    if (merger_type == "linear") return linear_impl->forward(hass, moe, cot);
    return gated_impl->forward(hass, moe, cot);
}

std::unordered_map<std::string, int64_t> XorzenMergerGateImpl::get_complexity() const {
    if (merger_type == "linear") return linear_impl->get_complexity();
    return gated_impl->get_complexity();
}

} // namespace xorzen

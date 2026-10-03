// ============================================================
//  xorzen.cpp — src/model/cot_vector.cpp
//  Internal Latent Chain-of-Thought — Full C++ Implementation
//  Ported from xorzen/model/components/cot_vector.py v2.0
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/cot.h"
#include "xorzen/ops.h"
#include "xorzen/optimized/simd_ops.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace xorzen {

// ============================================================
//  CoTGRUUpdater
// ============================================================

CoTGRUUpdaterImpl::CoTGRUUpdaterImpl(int64_t dim)
    : CoTUpdaterImpl(dim) {
    gru_cell = register_module("gru_cell",
        torch::nn::GRUCell(torch::nn::GRUCellOptions(dim, dim)));
}

torch::Tensor CoTGRUUpdaterImpl::forward(const torch::Tensor& new_components,
                                          const torch::Tensor& previous_cot) {
    // new_components: [N, total_cot_dim]
    // previous_cot:   [N, total_cot_dim]
    return gru_cell->forward(new_components, previous_cot);
}

// ============================================================
//  CoTTransformerUpdater
// ============================================================

CoTTransformerUpdaterImpl::CoTTransformerUpdaterImpl(int64_t dim,
                                                     int64_t num_heads,
                                                     int64_t num_layers,
                                                     double dropout) 
    : CoTUpdaterImpl(dim) {
    auto enc_layer = torch::nn::TransformerEncoderLayer(
        torch::nn::TransformerEncoderLayerOptions(dim, num_heads)
            .dim_feedforward(dim * 4)
            .dropout(dropout)
            .activation(torch::kGELU));
    transformer_encoder = register_module("transformer_encoder",
        torch::nn::TransformerEncoder(
            torch::nn::TransformerEncoderOptions(enc_layer, num_layers)));
    // Learnable residual gate, initialised small so it starts near identity
    gate = register_parameter("gate", torch::tensor(0.1f));
}

torch::Tensor CoTTransformerUpdaterImpl::forward(const torch::Tensor& new_components,
                                                  const torch::Tensor& previous_cot) {
    // Stack [prev, new] -> [N, 2, D] for transformer
    auto combined = torch::stack({previous_cot, new_components}, 0); // [2,N,D]
    auto out = transformer_encoder->forward(combined);
    auto updated = out.select(0, 1); // take the "new" output token
    auto g = torch::sigmoid(gate);
    return previous_cot * (1.0f - g) + updated * g;
}

// ============================================================
//  Helpers — create a single component projection
//  Mirrors Python: Linear(H, C*2) -> GLU -> LayerNorm(C) -> SiLU
// ============================================================

static torch::nn::Sequential make_component_projection(int64_t hidden_dim, int64_t cot_dim) {
    // GLU halves the last dimension: input [*, 2*C] -> output [*, C]
    return torch::nn::Sequential(
        torch::nn::Linear(hidden_dim, cot_dim * 2),
        torch::nn::GLU(torch::nn::GLUOptions(-1)),
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({cot_dim})),
        torch::nn::Functional(optimized::silu_simd));
}

// ============================================================
//  InternalLatentCoT
// ============================================================

InternalLatentCoTImpl::InternalLatentCoTImpl(ModelConfig cfg)
    : config(std::move(cfg)) {
    config.normalize();
    hidden_dim  = config.hidden_size;
    cot_dim     = config.cot_dim;
    num_components = config.cot_components;
    total_cot_dim  = config.cot_dim * config.cot_components;
    hidden_size    = config.hidden_size;

    // 6 component projections
    const std::vector<std::string> names = {
        "intention", "decomposition", "confidence",
        "contradiction", "direction", "summary"
    };
    component_projections = register_module("component_projections",
        torch::nn::ModuleDict());
    for (const auto& n : names) {
        component_projections->update(std::vector<std::pair<std::string, std::shared_ptr<torch::nn::Module>>>{
            {n, make_component_projection(hidden_dim, cot_dim).ptr()}
        });
    }

    // Recurrent updater
    if (config.cot_update_method == "gru") {
        updater = std::make_shared<CoTGRUUpdaterImpl>(total_cot_dim);
        register_module("updater", updater);
    } else if (config.cot_update_method == "transformer") {
        auto tr = std::make_shared<CoTTransformerUpdaterImpl>(
            total_cot_dim,
            config.cot_transformer_heads,
            config.cot_transformer_layers,
            config.dropout);
        updater = tr;
        register_module("updater", tr);
    } else {
        throw std::invalid_argument("Unknown CoT update method: " + config.cot_update_method);
    }

    // Layer norms
    component_norm = register_module("component_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({cot_dim})));
    cot_norm = register_module("cot_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({total_cot_dim})));

    // Output projection: CoT -> hidden influence
    output_proj = register_module("output_proj",
        torch::nn::Linear(total_cot_dim, hidden_dim));

    // Injection gate: learns how much CoT to blend in
    injection_gate = register_module("injection_gate",
        torch::nn::Linear(hidden_size, hidden_size));

    // Update gate: when should CoT be updated?
    update_gate = register_module("update_gate", torch::nn::Sequential(
        torch::nn::Linear(hidden_dim + total_cot_dim, 128),
        torch::nn::Functional(optimized::silu_simd),
        torch::nn::Linear(128, 1),
        torch::nn::Sigmoid()));

    init_weights();
}

void InternalLatentCoTImpl::init_weights() {
    // Component projections
    for (auto& kv : component_projections->named_modules("", false)) {
        if (auto* lin = kv.value()->as<torch::nn::Linear>()) {
            torch::nn::init::xavier_uniform_(lin->weight, 0.5);
            if (lin->bias.defined()) torch::nn::init::zeros_(lin->bias);
        }
    }
    // Output proj — small gain to avoid loud initial injection
    torch::nn::init::xavier_uniform_(output_proj->weight, 0.1);
    torch::nn::init::zeros_(output_proj->bias);
    // Injection gate — start closed
    torch::nn::init::zeros_(injection_gate->weight);
    torch::nn::init::constant_(injection_gate->bias, -2.0f);
    // Update gate
    for (auto& kv : update_gate->named_modules("", false)) {
        if (auto* lin = kv.value()->as<torch::nn::Linear>()) {
            torch::nn::init::xavier_uniform_(lin->weight, 0.5);
            if (lin->bias.defined()) torch::nn::init::zeros_(lin->bias);
        }
    }
}

void InternalLatentCoTImpl::enable_cot(bool enabled) {
    cot_enabled = enabled;
}

torch::Tensor InternalLatentCoTImpl::init_cot(int64_t batch_size, torch::Device device) {
    return torch::zeros({batch_size, total_cot_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

torch::Tensor InternalLatentCoTImpl::init_cot_sequence(int64_t batch_size,
                                                        int64_t seq_len,
                                                        torch::Device device) {
    return torch::zeros({batch_size, seq_len, total_cot_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

torch::Tensor InternalLatentCoTImpl::compute_components(const torch::Tensor& x) {
    // x: [B, T, H]  or  [N, H]
    const bool is_3d = (x.dim() == 3);
    const int64_t B = is_3d ? x.size(0) : 1;
    const int64_t T = is_3d ? x.size(1) : x.size(0);
    auto x_flat = is_3d ? x.view({B * T, hidden_dim}) : x; // [N, H]

    const std::vector<std::string> comp_names = {
        "intention", "decomposition", "confidence",
        "contradiction", "direction", "summary"
    };

    std::vector<torch::Tensor> parts;
    parts.reserve(num_components);
    for (const auto& name : comp_names) {
        auto proj = component_projections->at<torch::nn::SequentialImpl>(name);
        auto c = proj.forward(x_flat); // [N, C]
        c = component_norm->forward(c);

        if (name == "confidence" || name == "contradiction") {
            c = torch::sigmoid(c);
        } else if (name == "intention") {
            c = torch::tanh(c);
        } else if (name == "decomposition") {
            c = torch::nn::functional::softplus(c);
        } else if (name == "direction") {
            c = torch::nn::functional::normalize(c, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
        }
        // "summary" uses SiLU from the projection sequential — no extra transform
        parts.push_back(c);
    }

    auto cot_flat = torch::cat(parts, -1); // [N, total_cot_dim]
    cot_flat = cot_norm->forward(cot_flat);

    if (is_3d) {
        cot_flat = cot_flat.view({B, T, total_cot_dim});
    }
    return cot_flat;
}

std::pair<torch::Tensor, torch::Tensor> InternalLatentCoTImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& previous_cot_in,
    const torch::Tensor& update_mask) {
    // x: [B, T, H]
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    auto device = x.device();

    torch::Tensor prev_cot = previous_cot_in.defined()
        ? previous_cot_in
        : init_cot_sequence(B, T, device);

    auto new_components = compute_components(x); // [B, T, total_cot_dim]

    // Compute update gate
    torch::Tensor gate_val;
    if (!update_mask.defined()) {
        auto gate_input = torch::cat({x, prev_cot}, -1); // [B,T, H+D]
        gate_val = update_gate->forward(gate_input);      // [B,T,1]
    } else {
        gate_val = update_mask.to(torch::kFloat32).unsqueeze(-1); // [B,T,1]
    }

    // Flatten for GRU / Transformer update: [B*T, D]
    auto new_flat  = new_components.reshape({B * T, total_cot_dim});
    auto prev_flat = prev_cot.reshape({B * T, total_cot_dim});

    auto updated_flat = updater->forward(new_flat, prev_flat); // [B*T, D]

    // Gated blend: alpha * updated + (1-alpha) * prev
    auto alpha = gate_val.reshape({B * T, 1});
    auto gated_flat = alpha * updated_flat + (1.0f - alpha) * prev_flat;

    auto updated_cot = cot_norm->forward(gated_flat).reshape({B, T, total_cot_dim});

    // Output influence (for residual injection in model forward)
    auto output_influence = output_proj->forward(updated_cot); // [B, T, H]

    return {updated_cot, output_influence};
}

// Slice out a single named component from the concatenated CoT vector
torch::Tensor InternalLatentCoTImpl::get_component(const torch::Tensor& cot,
                                                    const std::string& name) const {
    const std::vector<std::string> comp_names = {
        "intention", "decomposition", "confidence",
        "contradiction", "direction", "summary"
    };
    auto it = std::find(comp_names.begin(), comp_names.end(), name);
    if (it == comp_names.end()) throw std::invalid_argument("Unknown CoT component: " + name);
    int64_t idx = static_cast<int64_t>(std::distance(comp_names.begin(), it));
    int64_t start = idx * cot_dim;
    int64_t end   = start + cot_dim;
    return cot.slice(-1, start, end);
}

torch::Tensor InternalLatentCoTImpl::compute_confidence(const torch::Tensor& cot) const {
    return get_component(cot, "confidence").mean(-1, true);
}

torch::Tensor InternalLatentCoTImpl::compute_contradiction(const torch::Tensor& cot) const {
    return get_component(cot, "contradiction").mean(-1, true);
}

std::unordered_map<std::string, torch::Tensor>
InternalLatentCoTImpl::analyze_cot(const torch::Tensor& cot) const {
    // cot: [B, T, total_cot_dim]
    const int64_t B = cot.size(0);
    const int64_t T = cot.size(1);
    auto cot_rs = cot.view({B, T, num_components, cot_dim});

    std::unordered_map<std::string, torch::Tensor> analysis;
    analysis["component_norms"]   = torch::norm(cot_rs, 2, -1);          // [B,T,C]
    analysis["confidence"]        = compute_confidence(cot);              // [B,T,1]
    analysis["contradiction"]     = compute_contradiction(cot);           // [B,T,1]
    analysis["update_magnitude"]  = torch::norm(cot, 2, -1, true);       // [B,T,1]

    const std::vector<std::string> comp_names = {
        "intention", "decomposition", "confidence",
        "contradiction", "direction", "summary"
    };
    for (int64_t i = 0; i < num_components; ++i) {
        auto comp = cot_rs.select(-2, i); // [B,T,D]
        analysis[comp_names[i] + "_norm"] = torch::norm(comp, 2, -1);
        analysis[comp_names[i] + "_mean"] = comp.mean(-1);
        analysis[comp_names[i] + "_std"]  = comp.std(-1);
    }
    return analysis;
}

torch::nn::Sequential InternalLatentCoTImpl::create_component_projection() {
    return make_component_projection(hidden_dim, cot_dim);
}

// ============================================================
//  CoTAuxiliaryLoss
// ============================================================

CoTAuxiliaryLossImpl::CoTAuxiliaryLossImpl(ModelConfig cfg)
    : config(std::move(cfg)),
      consistency_weight(cfg.cot_consistency_weight),
      diversity_weight(cfg.cot_diversity_weight),
      sparsity_weight(cfg.cot_sparsity_weight),
      orthogonality_weight(cfg.cot_orthogonality_weight) {
    const int64_t total_cot = config.cot_dim * config.cot_components;
    token_complexity_head = register_module("token_complexity_head",
        torch::nn::Sequential(
            torch::nn::Linear(total_cot, 128),
            torch::nn::Functional(optimized::silu_simd),
            torch::nn::Linear(128, 1)));
    next_token_head = register_module("next_token_head",
        torch::nn::Sequential(
            torch::nn::Linear(total_cot, config.hidden_size),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.hidden_size})),
            torch::nn::Linear(config.hidden_size, config.vocab_size)));
}

std::unordered_map<std::string, torch::Tensor>
CoTAuxiliaryLossImpl::forward(const torch::Tensor& cot_states,
                              const torch::Tensor& targets,
                              const torch::Tensor& token_complexity,
                              const torch::Tensor& mask_in) {
    // cot_states: [B, T, total_cot_dim]
    const int64_t B  = cot_states.size(0);
    const int64_t T  = cot_states.size(1);
    const int64_t C  = config.cot_components;
    const int64_t D  = config.cot_dim;

    auto mask = mask_in.defined()
        ? mask_in.to(cot_states.dtype())
        : torch::ones({B, T}, cot_states.options());

    std::unordered_map<std::string, torch::Tensor> losses;

    // 1. Consistency loss — penalise abrupt jumps over the sequence
    if (T > 1) {
        auto diff = cot_states.slice(1, 1, T) - cot_states.slice(1, 0, T - 1);
        losses["consistency"] = diff.pow(2).mean() * consistency_weight;
    } else {
        losses["consistency"] = torch::zeros({}, cot_states.options());
    }

    // Reshape for per-component operations: [B*T, C, D]
    auto cot_rs = cot_states.view({B * T, C, D});

    // 2. Diversity loss (pairwise cosine similarity, off-diagonal)
    auto sim = torch::nn::functional::cosine_similarity(
        cot_rs.unsqueeze(2), cot_rs.unsqueeze(1),
        torch::nn::functional::CosineSimilarityFuncOptions().dim(-1)); // [B*T, C, C]
    auto eye = torch::eye(C, cot_states.options());
    losses["diversity"] = ((sim * (1.0f - eye)).pow(2)).mean() * diversity_weight;

    // 3. Sparsity — L1 regularisation
    losses["sparsity"] = cot_states.abs().mean() * sparsity_weight;

    // 4. Orthogonality — Gram matrix on normalised components
    auto cot_norm = torch::nn::functional::normalize(cot_rs,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1)); // [B*T,C,D]
    auto gram = torch::matmul(cot_norm, cot_norm.transpose(1, 2));  // [B*T,C,C]
    losses["orthogonality"] = ((gram * (1.0f - eye)).pow(2)).mean() * orthogonality_weight;

    // 5. Token complexity prediction (auxiliary head)
    if (token_complexity.defined()) {
        auto pred = token_complexity_head->forward(cot_states).squeeze(-1); // [B,T]
        auto masked_pred = pred * mask;
        auto masked_tc   = token_complexity * mask;
        auto sum_mask = mask.sum().clamp_min(1.0f);
        losses["complexity_pred"] = torch::nn::functional::mse_loss(
            masked_pred, masked_tc,
            torch::nn::functional::MSELossFuncOptions().reduction(torch::kSum))
            / sum_mask * 0.1f;
    }

    // 6. Next-token prediction (auxiliary head, only if labels provided)
    if (targets.defined() && T > 1) {
        auto nt_logits = next_token_head->forward(cot_states.slice(1, 0, T - 1)); // [B,T-1,V]
        auto nt_labels = targets.slice(1, 1, T).contiguous().view({-1}); // [B*(T-1)]
        losses["next_token_pred"] = torch::nn::functional::cross_entropy(
            nt_logits.contiguous().view({-1, config.vocab_size}), nt_labels) * 0.05f;
    }

    // Sum all sub-losses
    torch::Tensor total = torch::zeros({}, cot_states.options());
    for (const auto& kv : losses) {
        if (kv.second.defined() && kv.second.numel() > 0) {
            total = total + kv.second.to(total.device()).to(total.dtype());
        }
    }
    losses["total_auxiliary"] = total;

    return losses;
}

} // namespace xorzen

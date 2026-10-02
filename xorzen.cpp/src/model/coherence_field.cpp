// ============================================================
//  xorzen.cpp — src/model/coherence_field.cpp
//  Coherence Field Theory — full C++ implementation
//  Ported from xorzen/model/components/kido/coherence_field.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/coherence_field.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace xorzen {

// ============================================================
//  Constructor
// ============================================================

CoherenceFieldTheoryImpl::CoherenceFieldTheoryImpl(const CoherenceFieldConfig& cfg)
    : config(cfg),
      num_chars(cfg.num_characters),
      num_attrs(cfg.num_attributes),
      hidden_size(cfg.hidden_size) {

    // Canonical attribute embeddings (trainable, but also manually settable)
    canonical = register_module("canonical",
        torch::nn::Embedding(num_chars, num_attrs));
    torch::nn::init::normal_(canonical->weight, 0.0, 0.02);

    // Per-attribute OU parameters (will be run through softplus to ensure +ve)
    coherence_strength = register_parameter("coherence_strength",
        torch::ones({num_attrs}) * static_cast<float>(cfg.initial_lambda));
    noise_scale = register_parameter("noise_scale",
        torch::ones({num_attrs}) * static_cast<float>(cfg.initial_sigma));

    // LSTM for permanent state transition detection
    state_transition_lstm = register_module("state_transition_lstm",
        torch::nn::LSTM(
            torch::nn::LSTMOptions(hidden_size, num_attrs)
                .num_layers(cfg.transition_lstm_layers)
                .batch_first(true)));

    // Attribute → hidden linear (replaces the random matrix in Python enforce())
    attr_to_hidden = register_module("attr_to_hidden",
        torch::nn::Linear(num_attrs, hidden_size));
    torch::nn::init::xavier_uniform_(attr_to_hidden->weight, 0.1);

    // Cross-attention projector: latent (Q) attends to attribute keys/values
    attribute_projector = register_module("attribute_projector",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(hidden_size, 8)));

    // Current OU state buffer (not a parameter — updated in-place)
    current_state = register_buffer("current_state",
        torch::zeros({num_chars, num_attrs}));
}

// ============================================================
//  initialize_character
// ============================================================

void CoherenceFieldTheoryImpl::initialize_character(
    int64_t char_id, const torch::Tensor& canonical_attrs) {
    if (char_id < 0 || char_id >= num_chars)
        throw std::out_of_range("char_id out of range");
    torch::NoGradGuard ng;
    canonical->weight[char_id] = canonical_attrs.to(canonical->weight.device());
    current_state[char_id]     = canonical_attrs.to(current_state.device()).clone();
}

// ============================================================
//  step — one OU timestep
// ============================================================

torch::Tensor CoherenceFieldTheoryImpl::step(
    const torch::Tensor& character_id,
    const torch::Tensor& frame_features,
    double dt) {
    // character_id: [B] int64
    // frame_features: [B, H]
    auto device = frame_features.device();
    auto char_ids = character_id.to(device);

    // Canonical and current state for these characters
    auto C0 = canonical->forward(char_ids);        // [B, A]
    auto C  = current_state.index({char_ids});     // [B, A]

    // Ensure positive OU parameters
    auto lam   = torch::nn::functional::softplus(coherence_strength);  // [A]
    auto sigma = torch::nn::functional::softplus(noise_scale);         // [A]

    // Discrete OU step:
    //   dC = -λ·(C - C₀)·dt + σ·√dt·noise
    auto noise = torch::randn_like(C);
    auto dC    = -lam * (C - C0) * static_cast<float>(dt)
                 + sigma * static_cast<float>(std::sqrt(dt)) * noise;
    auto C_new = C + dC;

    // Permanent state transition detection via LSTM
    // frame_features: [B, H] → unsqueeze → [B, 1, H]
    auto lstm_out = std::get<0>(
        state_transition_lstm->forward(frame_features.unsqueeze(1)));  // [B, 1, A]
    auto delta = lstm_out.squeeze(1);                                  // [B, A]

    // Only apply if magnitude exceeds threshold
    auto transition_mask = (delta.abs() > static_cast<float>(config.transition_threshold))
                           .to(torch::kFloat32);
    C_new = C_new + transition_mask * delta;

    // Update buffers in-place (no gradient needed)
    {
        torch::NoGradGuard ng;
        // Update canonical slightly for permanent changes
        for (int64_t i = 0; i < char_ids.size(0); ++i) {
            int64_t cid = char_ids[i].item<int64_t>();
            canonical->weight[cid] += transition_mask[i] * delta[i] * 0.1f;
            current_state[cid] = C_new[i].to(current_state.device());
        }
    }

    return C_new;  // [B, A]
}

// ============================================================
//  enforce — steer latent toward coherent attributes
// ============================================================

torch::Tensor CoherenceFieldTheoryImpl::enforce(
    const torch::Tensor& frame_latent,
    const torch::Tensor& target_attributes) {
    // frame_latent:    [B, T, H]
    // target_attributes: [B, A]

    // Project attributes to hidden space
    auto attr_proj = attr_to_hidden->forward(target_attributes) // [B, H]
                         .unsqueeze(1);                          // [B, 1, H]

    // Cross-attention: frame_latent (Q) attends to attr_proj (K, V)
    auto query = frame_latent.transpose(0, 1); // [T, B, H]
    auto key_value = attr_proj.transpose(0, 1); // [1, B, H]
    auto [corrected_seq, _] = attribute_projector->forward(query, key_value, key_value);
    auto corrected = corrected_seq.transpose(0, 1); // [B, T, H]

    // Soft blend: (1-α) * original + α * corrected
    float alpha = static_cast<float>(config.enforce_blend);
    return (1.0f - alpha) * frame_latent + alpha * corrected;
}

// ============================================================
//  compute_drift_penalty
// ============================================================

torch::Tensor CoherenceFieldTheoryImpl::compute_drift_penalty() const {
    auto canonical_all = canonical->weight;    // [C, A]
    auto current_all   = current_state;        // [C, A]
    auto lam           = torch::nn::functional::softplus(
                            coherence_strength); // [A]

    auto drift          = (current_all - canonical_all).pow(2);    // [C, A]
    auto weighted_drift = drift * lam.unsqueeze(0);                // [C, A]
    return weighted_drift.mean();
}

// ============================================================
//  reset_character_state
// ============================================================

void CoherenceFieldTheoryImpl::reset_character_state(int64_t char_id) {
    if (char_id < 0 || char_id >= num_chars)
        throw std::out_of_range("char_id out of range");
    torch::NoGradGuard ng;
    current_state[char_id] = canonical->weight[char_id].clone()
                                 .to(current_state.device());
}

// ============================================================
//  get_coherence_strengths — interpretability
// ============================================================

std::unordered_map<std::string, double>
CoherenceFieldTheoryImpl::get_coherence_strengths() const {
    auto lam = torch::nn::functional::softplus(coherence_strength)
               .detach().cpu();
    std::unordered_map<std::string, double> info;
    for (int64_t i = 0; i < num_attrs && i < static_cast<int64_t>(config.attribute_names.size()); ++i) {
        info[config.attribute_names[i]] = lam[i].item<double>();
    }
    return info;
}

} // namespace xorzen

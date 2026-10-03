#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/coherence_field.h
//  Coherence Field Theory (CFT) — Character Consistency
//  via Ornstein-Uhlenbeck Mean-Reverting Process
//  Ported from xorzen/model/components/kido/coherence_field.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
// Mathematical Foundation:
//   ∂C/∂t = −λ·(C − C₀) + σ·η(t)
//   Steady-state: C(t) ~ N(C₀, σ²/2λ)
//   High λ → tight (eye color), Low λ → loose (expression)
// ============================================================

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace xorzen {

// ============================================================
//  CoherenceFieldConfig
// ============================================================
struct CoherenceFieldConfig {
    int64_t num_characters  = 100;   // Max simultaneous characters
    int64_t num_attributes  = 16;    // Attributes per character
    int64_t hidden_size     = 256;   // XORZEN hidden dimension

    double  initial_lambda  = 2.0;   // Default coherence strength
    double  initial_sigma   = 0.1;   // Default noise scale
    double  transition_threshold = 0.5; // Min magnitude for state transition

    int64_t transition_lstm_layers = 2;
    double  enforce_blend   = 0.2;   // How much correction to blend (0=none, 1=full)

    // Default attribute names (matches the 16 attributes)
    std::vector<std::string> attribute_names = {
        "eye_color", "hair_color", "hair_length", "skin_tone",
        "costume_primary", "costume_secondary", "expression_base",
        "body_type", "height", "age_appearance",
        "accessory_1", "accessory_2", "weapon", "power_aura",
        "emotion_state", "injury_state"
    };
};

// ============================================================
//  CoherenceFieldTheory
//  Integrates with XORZEN HASS layers to maintain character
//  attribute consistency across frame generation
// ============================================================

struct CoherenceFieldTheoryImpl : torch::nn::Module {
    CoherenceFieldConfig config;
    int64_t num_chars;
    int64_t num_attrs;
    int64_t hidden_size;

    // Canonical attribute embeddings [num_chars, num_attrs]
    torch::nn::Embedding canonical{nullptr};

    // Per-attribute OU process parameters (learned)
    torch::Tensor coherence_strength;  // [num_attrs]   — λ
    torch::Tensor noise_scale;         // [num_attrs]   — σ

    // State transition detector (permanent changes: wounds, power-ups)
    torch::nn::LSTM state_transition_lstm{nullptr};

    // Attribute cross-attention projector
    torch::nn::MultiheadAttention attribute_projector{nullptr};

    // Learnable attribute → hidden projection
    torch::nn::Linear attr_to_hidden{nullptr};

    // Current OU field state (buffer, not parameter)
    // [num_characters, num_attributes]
    torch::Tensor current_state;

    explicit CoherenceFieldTheoryImpl(const CoherenceFieldConfig& cfg);

    // Set canonical attributes for a specific character
    void initialize_character(int64_t char_id,
                              const torch::Tensor& canonical_attrs);

    // Run one OU step for given characters
    // character_id: [B] int64
    // frame_features: [B, hidden_size]
    // dt: time step (typically 1/24 for 24fps)
    // Returns: [B, num_attributes] — target attributes for this frame
    torch::Tensor step(const torch::Tensor& character_id,
                       const torch::Tensor& frame_features,
                       double dt = 1.0);

    // Steer latent toward attribute-coherent space
    // frame_latent: [B, seq_len, hidden_size]
    // target_attributes: [B, num_attributes]
    // Returns: [B, seq_len, hidden_size]
    torch::Tensor enforce(const torch::Tensor& frame_latent,
                          const torch::Tensor& target_attributes);

    // Training loss: penalise drift from canonical weighted by λ
    torch::Tensor compute_drift_penalty() const;

    // Reset one character to canonical (new episode/scene)
    void reset_character_state(int64_t char_id);

    // Interpretability: returns {attr_name: {lambda, level}}
    std::unordered_map<std::string, double> get_coherence_strengths() const;
};
TORCH_MODULE(CoherenceFieldTheory);

} // namespace xorzen

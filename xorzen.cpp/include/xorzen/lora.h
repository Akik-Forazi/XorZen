#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/lora.h
//  LoRA Adapter Engine for Progressive Learning
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
// LoRA: Low-Rank Adaptation — reparameterises weight updates as:
//   W' = W + α/r * B @ A
// where A ∈ R^{r × d_in}, B ∈ R^{d_out × r}, both initialised small.
// Only A and B are trained; W is frozen.
// On merge: W ← W + α/r * B @ A, then drop A and B.
// ============================================================

#include <torch/torch.h>
#include "xorzen/types.h"
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace xorzen {

// ============================================================
//  LoRAConfig
// ============================================================
struct LoRAConfig {
    int64_t rank       = 16;      // LoRA rank r
    double  alpha      = 32.0;    // LoRA alpha (scale = alpha/rank)
    double  dropout    = 0.05;    // Dropout on LoRA path
    bool    bias       = false;   // Train bias?

    // Which module names to apply LoRA to (substring match)
    std::vector<std::string> target_modules = {
        "q_proj", "k_proj", "v_proj", "out_proj",
        "gate_proj", "up_proj", "down_proj"
    };

    double  ewc_lambda = 0.4;     // EWC regularisation strength
    double  confidence_threshold = 0.3; // Min improvement to accept adapter

    double scale() const { return alpha / static_cast<double>(rank); }
};

// ============================================================
//  LoRALinear — drop-in replacement for nn::Linear with LoRA
// ============================================================
struct LoRALinearImpl : torch::nn::Module {
    // Frozen base weight (not a parameter when adapter is active)
    torch::Tensor weight_frozen;  // [d_out, d_in]

    // Trainable LoRA matrices
    torch::nn::Linear lora_A{nullptr};  // [rank, d_in]
    torch::nn::Linear lora_B{nullptr};  // [d_out, rank]
    torch::nn::Dropout dropout_{nullptr};

    int64_t in_features, out_features, rank;
    double  scale;

    LoRALinearImpl(int64_t in_features, int64_t out_features,
                   int64_t rank = 16, double alpha = 32.0,
                   double dropout = 0.05, bool bias = false);

    // [*, in] → [*, out]
    torch::Tensor forward(const torch::Tensor& x);

    // Merge LoRA into frozen weight (makes adapter permanent)
    void merge_weights();

    // Reset LoRA matrices (start fresh adapter)
    void reset_lora();

    // Export LoRA delta tensors: {A: tensor, B: tensor}
    std::unordered_map<std::string, torch::Tensor> export_delta() const;

    // Import LoRA delta tensors
    void import_delta(const std::unordered_map<std::string, torch::Tensor>& delta);
};
TORCH_MODULE(LoRALinear);

// ============================================================
//  LoRAEngine — manages LoRA injection and adapter sessions
// ============================================================
class LoRAEngine {
public:
    LoRAEngine(torch::nn::Module& model, const LoRAConfig& cfg);

    // Inject LoRA into all target_modules in the model
    // Returns count of injected layers
    int inject(bool freeze_base = true);

    // Collect only LoRA parameters (for targeted optimisation)
    std::vector<torch::Tensor> lora_parameters() const;

    // Compute EWC penalty against reference (pre-training) fisher info
    // Call once before learning session to capture reference state
    void capture_ewc_reference();
    torch::Tensor ewc_penalty() const;

    // Export all LoRA deltas: {layer_name: {A, B}}
    std::unordered_map<std::string, std::unordered_map<std::string, torch::Tensor>>
    export_all_deltas() const;

    // Import deltas from a previous session and apply EMA merge
    void import_deltas(
        const std::unordered_map<std::string,
              std::unordered_map<std::string, torch::Tensor>>& deltas,
        double ema_decay = 0.9);

    // Permanently merge adapters into base weights and remove LoRA
    void merge_and_remove();

    // Reset all LoRA matrices (start new learning session)
    void reset_all();

    // Get adapter confidence score (based on gradient magnitude improvement)
    double adapter_confidence(const torch::Tensor& baseline_loss,
                              const torch::Tensor& current_loss) const;

    const LoRAConfig& config() const { return cfg_; }

private:
    torch::nn::Module& model_;
    LoRAConfig cfg_;

    // Registered LoRA layers: {qualified_name → LoRALinear}
    std::vector<std::pair<std::string, LoRALinear>> lora_layers_;

    // EWC: Fisher information (diagonal approx) for each param
    std::unordered_map<std::string, torch::Tensor> fisher_info_;
    std::unordered_map<std::string, torch::Tensor> ref_params_;

    bool is_target(const std::string& name) const;
};

// ============================================================
//  LoRATrainer — one-shot learning session on a topic
// ============================================================
struct LoRATrainerConfig {
    int64_t steps           = 100;
    int64_t batch_size      = 4;
    double  learning_rate   = 1e-3;
    double  weight_decay    = 0.01;
    double  grad_clip       = 1.0;
    int64_t warmup_steps    = 10;
    std::string topic;
};

class LoRATrainer {
public:
    LoRATrainer(torch::nn::Module& model,
                LoRAEngine& engine,
                const LoRATrainerConfig& cfg = {});

    // Train on (src, tgt) tensor pairs — next-token prediction
    // Returns {steps_trained, final_loss}
    std::pair<int64_t, double> train(
        const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches);

    // Export session deltas for saving to .xorzen project
    std::unordered_map<std::string, torch::Tensor> export_flat_deltas() const;

private:
    torch::nn::Module& model_;
    LoRAEngine& engine_;
    LoRATrainerConfig cfg_;

    double compute_loss(const torch::Tensor& src, const torch::Tensor& tgt);
};

} // namespace xorzen

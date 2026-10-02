#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "xorzen/types.h"

namespace xorzen {

// CoTUpdater interface and subclasses
struct CoTUpdaterImpl : torch::nn::Module {
    int64_t total_cot_dim;
    explicit CoTUpdaterImpl(int64_t total_cot_dim) : total_cot_dim(total_cot_dim) {}
    virtual torch::Tensor forward(const torch::Tensor& new_components, const torch::Tensor& previous_cot) = 0;
};

struct CoTGRUUpdaterImpl : CoTUpdaterImpl {
    torch::nn::GRUCell gru_cell{nullptr};
    
    explicit CoTGRUUpdaterImpl(int64_t total_cot_dim);
    torch::Tensor forward(const torch::Tensor& new_components, const torch::Tensor& previous_cot) override;
};
TORCH_MODULE(CoTGRUUpdater);

struct CoTTransformerUpdaterImpl : CoTUpdaterImpl {
    torch::nn::TransformerEncoder transformer_encoder{nullptr};
    torch::Tensor gate; // Learnable gate for residual connection
    
    CoTTransformerUpdaterImpl(int64_t total_cot_dim, int64_t num_heads, int64_t num_layers, double dropout);
    torch::Tensor forward(const torch::Tensor& new_components, const torch::Tensor& previous_cot) override;
};
TORCH_MODULE(CoTTransformerUpdater);

// Main InternalLatentCoT module
struct InternalLatentCoTImpl : torch::nn::Module {
    ModelConfig config;
    int64_t hidden_dim;
    int64_t cot_dim;
    int64_t num_components;
    int64_t total_cot_dim;
    int64_t hidden_size;
    
    torch::nn::ModuleDict component_projections{nullptr};
    std::shared_ptr<CoTUpdaterImpl> updater{nullptr};
    
    torch::nn::LayerNorm component_norm{nullptr};
    torch::nn::LayerNorm cot_norm{nullptr};
    torch::nn::Linear output_proj{nullptr};
    torch::nn::Linear injection_gate{nullptr};
    torch::nn::Sequential update_gate{nullptr};
    
    bool cot_enabled = false;

    explicit InternalLatentCoTImpl(ModelConfig config);
    
    void enable_cot(bool enabled = true);
    torch::Tensor init_cot(int64_t batch_size, torch::Device device);
    torch::Tensor init_cot_sequence(int64_t batch_size, int64_t seq_len, torch::Device device);
    torch::Tensor compute_components(const torch::Tensor& x);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x,
                                                    const torch::Tensor& previous_cot = {},
                                                    const torch::Tensor& update_mask = {});
                                                    
    torch::Tensor get_component(const torch::Tensor& cot, const std::string& component_name) const;
    torch::Tensor compute_confidence(const torch::Tensor& cot) const;
    torch::Tensor compute_contradiction(const torch::Tensor& cot) const;
    std::unordered_map<std::string, torch::Tensor> analyze_cot(const torch::Tensor& cot) const;
    
private:
    torch::nn::Sequential create_component_projection();
    void init_weights();
};
TORCH_MODULE(InternalLatentCoT);

// CoTAuxiliaryLoss module
struct CoTAuxiliaryLossImpl : torch::nn::Module {
    ModelConfig config;
    double consistency_weight;
    double diversity_weight;
    double sparsity_weight;
    double orthogonality_weight;
    
    torch::nn::Sequential token_complexity_head{nullptr};
    torch::nn::Sequential next_token_head{nullptr};
    
    explicit CoTAuxiliaryLossImpl(ModelConfig config);
    std::unordered_map<std::string, torch::Tensor> forward(const torch::Tensor& cot_states,
                                                           const torch::Tensor& targets = {},
                                                           const torch::Tensor& token_complexity = {},
                                                           const torch::Tensor& mask = {});
};
TORCH_MODULE(CoTAuxiliaryLoss);

} // namespace xorzen

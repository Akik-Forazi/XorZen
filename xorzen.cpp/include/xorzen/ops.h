#pragma once

#include <torch/torch.h>
#include "xorzen/types.h"

namespace xorzen {

torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& weight, double eps);
torch::Tensor silu(const torch::Tensor& x);
torch::Tensor top_k_filtering(const torch::Tensor& logits, int64_t top_k);
torch::Tensor top_p_filtering(const torch::Tensor& logits, double top_p);
torch::Tensor sample_next_token(const torch::Tensor& logits, const GenerationConfig& config);
torch::Tensor make_local_causal_mask(int64_t seq_len, int64_t window, torch::Device device);

struct RMSNormImpl : torch::nn::Module {
    torch::Tensor weight;
    double eps;

    explicit RMSNormImpl(int64_t dim, double eps = 1e-5);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(RMSNorm);

void xavier_linear(torch::nn::Linear& linear, double gain = 1.0);

} // namespace xorzen

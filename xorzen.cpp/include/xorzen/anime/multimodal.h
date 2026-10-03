#pragma once

#include <torch/torch.h>
#include "xorzen/types.h"

namespace xorzen {
namespace anime {

struct VisualTokenEncoderImpl : torch::nn::Module {
    int64_t vocab_size;
    int64_t hidden_dim;
    torch::nn::Embedding embedding{nullptr};

    VisualTokenEncoderImpl(int64_t vocab_size, int64_t hidden_dim);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(VisualTokenEncoder);

struct AudioTokenEncoderImpl : torch::nn::Module {
    int64_t vocab_size;
    int64_t hidden_dim;
    torch::nn::Embedding embedding{nullptr};

    AudioTokenEncoderImpl(int64_t vocab_size, int64_t hidden_dim);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(AudioTokenEncoder);

struct MultimodalEmbedderImpl : torch::nn::Module {
    ModelConfig config;
    VisualTokenEncoder visual_encoder{nullptr};
    AudioTokenEncoder audio_encoder{nullptr};

    explicit MultimodalEmbedderImpl(ModelConfig config);
    
    // Fuse text, visual, and audio into a single sequence
    torch::Tensor forward(const torch::Tensor& text_emb,
                          const torch::Tensor& visual_tokens,
                          const torch::Tensor& audio_tokens);
};
TORCH_MODULE(MultimodalEmbedder);

} // namespace anime
} // namespace xorzen

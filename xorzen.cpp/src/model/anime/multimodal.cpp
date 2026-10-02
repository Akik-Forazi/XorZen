// ============================================================
//  xorzen.cpp — src/model/anime/multimodal.cpp
//  Multimodal Embedders (Visual, Audio, Text) for AniXO
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/anime/multimodal.h"

namespace xorzen {
namespace anime {

VisualTokenEncoderImpl::VisualTokenEncoderImpl(int64_t vocab, int64_t hidden)
    : vocab_size(vocab), hidden_dim(hidden) {
    embedding = register_module("embedding", torch::nn::Embedding(vocab_size, hidden_dim));
}

torch::Tensor VisualTokenEncoderImpl::forward(const torch::Tensor& x) {
    return embedding->forward(x);
}

AudioTokenEncoderImpl::AudioTokenEncoderImpl(int64_t vocab, int64_t hidden)
    : vocab_size(vocab), hidden_dim(hidden) {
    embedding = register_module("embedding", torch::nn::Embedding(vocab_size, hidden_dim));
}

torch::Tensor AudioTokenEncoderImpl::forward(const torch::Tensor& x) {
    return embedding->forward(x);
}

MultimodalEmbedderImpl::MultimodalEmbedderImpl(ModelConfig cfg)
    : config(std::move(cfg)) {
    visual_encoder = register_module("visual_encoder",
        VisualTokenEncoder(config.visual_token_vocab, config.hidden_size));
    audio_encoder = register_module("audio_encoder",
        AudioTokenEncoder(config.audio_token_vocab, config.hidden_size));
}

torch::Tensor MultimodalEmbedderImpl::forward(const torch::Tensor& text_emb,
                                              const torch::Tensor& visual_tokens,
                                              const torch::Tensor& audio_tokens) {
    // text_emb: [B, T_text, D]
    // visual_tokens: [B, T_vis, D] (indices)
    // audio_tokens: [B, T_aud, D] (indices)
    
    std::vector<torch::Tensor> embeddings = { text_emb };
    
    if (visual_tokens.defined()) {
        embeddings.push_back(visual_encoder->forward(visual_tokens));
    }
    
    if (audio_tokens.defined()) {
        embeddings.push_back(audio_encoder->forward(audio_tokens));
    }
    
    // Concatenate along time dimension
    return torch::cat(embeddings, 1);
}

} // namespace anime
} // namespace xorzen

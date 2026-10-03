#pragma once

#include "xorzen/model.h"
#include "xorzen/anime/chunk_memory.h"
#include "xorzen/anime/multimodal.h"

namespace xorzen {
namespace anime {

struct AniXOModelImpl : XorzenModelImpl {
    MultimodalEmbedder multimodal_embedder{nullptr};
    ChunkMemoryBank memory_bank{nullptr};

    AniXOModelImpl(ModelConfig cfg, bool test_mode = false);

    // Multimodal forward pass
    ModelOutput forward_multimodal(
        const torch::Tensor& input_ids,
        const torch::Tensor& visual_tokens = {},
        const torch::Tensor& audio_tokens = {},
        const torch::Tensor& attention_mask = {},
        const torch::Tensor& position_ids = {},
        const torch::Tensor& labels = {}
    );
};
TORCH_MODULE(AniXOModel);

} // namespace anime
} // namespace xorzen

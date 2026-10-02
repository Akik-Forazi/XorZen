#pragma once

#include <torch/torch.h>
#include "xorzen/types.h"

namespace xorzen {

// Bit Upgrade: Maintains a hidden 'reasoning state' that evolves.
// This is already implemented as InternalLatentCoT.

// Bit Upgrade: Natively predicts agentic actions.
struct ActionHeadImpl : torch::nn::Module {
    torch::nn::Linear proj{nullptr};

    ActionHeadImpl(int64_t d_model, int64_t num_actions);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(ActionHead);

// Bit Upgrade: Identifies 'uncertainty' or 'errors' in the current latent state.
struct CritiqueModuleImpl : torch::nn::Module {
    torch::nn::Sequential quality_eval{nullptr};

    CritiqueModuleImpl(int64_t d_model);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(CritiqueModule);

// Recursive Router: Decides halt probability.
struct RecursiveRouterImpl : torch::nn::Module {
    torch::nn::Linear halt_proj{nullptr};

    RecursiveRouterImpl(int64_t d_model);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(RecursiveRouter);

} // namespace xorzen

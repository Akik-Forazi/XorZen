#include "xorzen/igris.h"

namespace xorzen {

ActionHeadImpl::ActionHeadImpl(int64_t d_model, int64_t num_actions) {
    proj = register_module("proj", torch::nn::Linear(d_model, num_actions));
}

torch::Tensor ActionHeadImpl::forward(const torch::Tensor& x) {
    return proj->forward(x);
}

CritiqueModuleImpl::CritiqueModuleImpl(int64_t d_model) {
    quality_eval = register_module("quality_eval", torch::nn::Sequential(
        torch::nn::Linear(d_model, d_model / 2),
        torch::nn::GELU(),
        torch::nn::Linear(d_model / 2, 1),
        torch::nn::Sigmoid()
    ));
}

torch::Tensor CritiqueModuleImpl::forward(const torch::Tensor& x) {
    return quality_eval->forward(x);
}

RecursiveRouterImpl::RecursiveRouterImpl(int64_t d_model) {
    halt_proj = register_module("halt_proj", torch::nn::Linear(d_model, 1));
}

torch::Tensor RecursiveRouterImpl::forward(const torch::Tensor& x) {
    return torch::sigmoid(halt_proj->forward(x));
}

} // namespace xorzen

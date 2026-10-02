// ============================================================
//  xorzen.cpp — src/training/lora.cpp
//  LoRA Adapter Engine implementation
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/lora.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace xorzen {

// ============================================================
//  LoRALinear
// ============================================================

LoRALinearImpl::LoRALinearImpl(int64_t in_f, int64_t out_f,
                               int64_t r, double alpha,
                               double drop, bool b)
    : in_features(in_f), out_features(out_f), rank(r), scale(alpha / r) {
    // LoRA A: initialized from N(0, 1/√r) — small but not zero
    lora_A = register_module("lora_A",
        torch::nn::Linear(torch::nn::LinearOptions(in_f, r).bias(false)));
    torch::nn::init::kaiming_normal_(lora_A->weight, /*a=*/std::sqrt(5.0));

    // LoRA B: initialized to zero so initial delta = 0
    lora_B = register_module("lora_B",
        torch::nn::Linear(torch::nn::LinearOptions(r, out_f).bias(b)));
    torch::nn::init::zeros_(lora_B->weight);
    if (b && lora_B->bias.defined()) torch::nn::init::zeros_(lora_B->bias);

    dropout_ = register_module("dropout", torch::nn::Dropout(drop));
}

torch::Tensor LoRALinearImpl::forward(const torch::Tensor& x) {
    // Base: W_frozen @ x^T
    auto base = torch::nn::functional::linear(x, weight_frozen);
    // LoRA delta: scale * B(A(dropout(x)))
    auto lora_out = lora_B->forward(lora_A->forward(dropout_->forward(x)));
    return base + scale * lora_out;
}

void LoRALinearImpl::merge_weights() {
    // W_merged = W_frozen + scale * B @ A
    // lora_B->weight: [out, rank], lora_A->weight: [rank, in]
    torch::NoGradGuard ng;
    auto delta = scale * torch::mm(lora_B->weight, lora_A->weight); // [out, in]
    weight_frozen = weight_frozen + delta;
    reset_lora();
}

void LoRALinearImpl::reset_lora() {
    torch::NoGradGuard ng;
    torch::nn::init::kaiming_normal_(lora_A->weight, std::sqrt(5.0));
    torch::nn::init::zeros_(lora_B->weight);
    if (lora_B->bias.defined()) torch::nn::init::zeros_(lora_B->bias);
}

std::unordered_map<std::string, torch::Tensor> LoRALinearImpl::export_delta() const {
    return {{"A", lora_A->weight.detach().clone()},
            {"B", lora_B->weight.detach().clone()}};
}

void LoRALinearImpl::import_delta(
    const std::unordered_map<std::string, torch::Tensor>& delta) {
    torch::NoGradGuard ng;
    if (delta.count("A")) lora_A->weight.copy_(delta.at("A"));
    if (delta.count("B")) lora_B->weight.copy_(delta.at("B"));
}

// ============================================================
//  LoRAEngine
// ============================================================

LoRAEngine::LoRAEngine(torch::nn::Module& model, const LoRAConfig& cfg)
    : model_(model), cfg_(cfg) {}

bool LoRAEngine::is_target(const std::string& name) const {
    for (const auto& t : cfg_.target_modules)
        if (name.find(t) != std::string::npos) return true;
    return false;
}

int LoRAEngine::inject(bool freeze_base) {
    int count = 0;
    // Traverse named modules looking for nn::Linear targets
    for (auto& [name, module] : model_.named_modules()) {
        if (!is_target(name)) continue;
        auto* lin = module->as<torch::nn::Linear>();
        if (!lin) continue;

        // Build LoRALinear and copy frozen weights
        auto lora = LoRALinear(
            lin->weight.size(1), lin->weight.size(0),
            cfg_.rank, cfg_.alpha, cfg_.dropout,
            lin->bias.defined());
        lora->weight_frozen = lin->weight.detach().clone();

        // Freeze base weights
        if (freeze_base) lin->weight.set_requires_grad(false);

        lora_layers_.emplace_back(name, lora);
        ++count;
    }
    std::cout << "[LoRA] Injected " << count << " LoRA layers\n";
    return count;
}

std::vector<torch::Tensor> LoRAEngine::lora_parameters() const {
    std::vector<torch::Tensor> params;
    for (auto& [name, layer] : lora_layers_) {
        for (auto& p : layer->parameters())
            params.push_back(p);
    }
    return params;
}

void LoRAEngine::capture_ewc_reference() {
    // Approximate diagonal Fisher via squared gradients over a dummy forward
    // This should be called after a few training steps on the reference data
    fisher_info_.clear();
    ref_params_.clear();
    for (auto& [name, p] : model_.named_parameters()) {
        if (p.grad().defined()) {
            fisher_info_[name] = p.grad().pow(2).detach().clone();
        }
        ref_params_[name] = p.detach().clone();
    }
    std::cout << "[LoRA] EWC reference captured (" << fisher_info_.size() << " params)\n";
}

torch::Tensor LoRAEngine::ewc_penalty() const {
    torch::Tensor penalty = torch::zeros({1});
    for (auto& [name, p] : model_.named_parameters()) {
        if (!fisher_info_.count(name) || !ref_params_.count(name)) continue;
        auto diff = p - ref_params_.at(name);
        penalty = penalty + (fisher_info_.at(name) * diff.pow(2)).sum();
    }
    return penalty * static_cast<float>(cfg_.ewc_lambda * 0.5);
}

std::unordered_map<std::string, std::unordered_map<std::string, torch::Tensor>>
LoRAEngine::export_all_deltas() const {
    std::unordered_map<std::string, std::unordered_map<std::string, torch::Tensor>> out;
    for (auto& [name, layer] : lora_layers_)
        out[name] = layer->export_delta();
    return out;
}

void LoRAEngine::import_deltas(
    const std::unordered_map<std::string,
          std::unordered_map<std::string, torch::Tensor>>& deltas,
    double ema_decay) {
    for (auto& [name, layer] : lora_layers_) {
        if (!deltas.count(name)) continue;
        auto& src = deltas.at(name);
        auto existing = layer->export_delta();
        // EMA merge: new = decay * old + (1-decay) * incoming
        std::unordered_map<std::string, torch::Tensor> merged;
        for (auto& [k, v] : src) {
            if (existing.count(k))
                merged[k] = ema_decay * existing[k] + (1.0 - ema_decay) * v;
            else
                merged[k] = v.clone();
        }
        layer->import_delta(merged);
    }
    std::cout << "[LoRA] EMA-merged " << deltas.size() << " adapter layers\n";
}

void LoRAEngine::merge_and_remove() {
    for (auto& [name, layer] : lora_layers_)
        layer->merge_weights();
    lora_layers_.clear();
    std::cout << "[LoRA] Merged all adapters into base weights.\n";
}

void LoRAEngine::reset_all() {
    for (auto& [name, layer] : lora_layers_)
        layer->reset_lora();
}

double LoRAEngine::adapter_confidence(const torch::Tensor& baseline_loss,
                                      const torch::Tensor& current_loss) const {
    double bl = baseline_loss.item<double>();
    double cl = current_loss.item<double>();
    if (bl < 1e-8) return 0.0;
    double improvement = (bl - cl) / bl;  // relative improvement
    return std::max(0.0, improvement);
}

// ============================================================
//  LoRATrainer
// ============================================================

LoRATrainer::LoRATrainer(torch::nn::Module& model,
                         LoRAEngine& engine,
                         const LoRATrainerConfig& cfg)
    : model_(model), engine_(engine), cfg_(cfg) {}

std::pair<int64_t, double> LoRATrainer::train(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches) {

    // Build optimizer over only LoRA params
    auto lora_params = engine_.lora_parameters();
    if (lora_params.empty())
        throw std::runtime_error("LoRATrainer: no LoRA params found. Call inject() first.");

    torch::optim::AdamW opt(lora_params,
        torch::optim::AdamWOptions(cfg_.learning_rate)
            .weight_decay(cfg_.weight_decay));

    model_.train();
    double total_loss = 0.0;
    int64_t step = 0;

    while (step < cfg_.steps) {
        auto& [src, tgt] = batches[step % batches.size()];

        // LR warmup
        double lr = cfg_.learning_rate;
        if (step < cfg_.warmup_steps)
            lr *= static_cast<double>(step + 1) / cfg_.warmup_steps;
        for (auto& pg : opt.param_groups())
            pg.options().set_lr(lr);

        opt.zero_grad();
        auto loss_val = compute_loss(src, tgt);

        // Add EWC penalty to prevent forgetting
        auto ewc = engine_.ewc_penalty();
        auto total = torch::tensor(loss_val) + ewc;
        total.backward();

        torch::nn::utils::clip_grad_norm_(lora_params, cfg_.grad_clip);
        opt.step();

        total_loss += loss_val;
        ++step;

        if (step % 20 == 0) {
            std::cout << "[LoRA] [" << cfg_.topic << "] step=" << step
                      << " loss=" << loss_val
                      << " ewc=" << ewc.item<double>() << "\n";
        }
    }
    return {step, total_loss / step};
}

double LoRATrainer::compute_loss(const torch::Tensor& src,
                                 const torch::Tensor& tgt) {
    auto output = model_.forward({src});
    torch::Tensor logits;
    if (output.isTensor()) logits = output.toTensor();
    else logits = output.toTuple()->elements()[0].toTensor();

    auto B = logits.size(0), T = logits.size(1), V = logits.size(2);
    return torch::nn::functional::cross_entropy(
        logits.reshape({B * T, V}), tgt.reshape({B * T}),
        torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100)
    ).item<double>();
}

std::unordered_map<std::string, torch::Tensor>
LoRATrainer::export_flat_deltas() const {
    auto all = engine_.export_all_deltas();
    std::unordered_map<std::string, torch::Tensor> flat;
    for (auto& [layer, kv] : all)
        for (auto& [k, v] : kv)
            flat[layer + "." + k] = v;
    return flat;
}

} // namespace xorzen

#include "xorzen/model.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace {

int arg_or(int argc, char** argv, int index, int fallback) {
    return argc > index ? std::atoi(argv[index]) : fallback;
}

} // namespace

int main(int argc, char** argv) {
    const int iterations = std::max(1, arg_or(argc, argv, 1, 16));
    const int batch = std::max(1, arg_or(argc, argv, 2, 2));
    const int seq = std::max(4, arg_or(argc, argv, 3, 32));

    xorzen::ModelConfig cfg;
    cfg.vocab_size = 1024;
    cfg.hidden_size = cfg.d_model = 128;
    cfg.num_layers = cfg.n_layers = 2;
    cfg.max_depth = 2;
    cfg.num_attention_heads = cfg.n_heads = 4;
    cfg.context_length = cfg.max_seq_len = std::max<int64_t>(64, seq);
    cfg.expert_count = cfg.num_experts = 8;
    cfg.top_k_experts = cfg.experts_per_token = 2;
    cfg.width_choices = {64, 96, 128};
    cfg.cot_dim = cfg.cot_latent_dim = 32;
    cfg.cot_components = 2;
    cfg.use_disk_cache = false;
    cfg.normalize();

    auto model = xorzen::XorzenModel(cfg, true);
    model->eval();

    auto input = torch::randint(0, cfg.vocab_size, {batch, seq}, torch::kLong);
    torch::NoGradGuard no_grad;

    for (int i = 0; i < 3; ++i) {
        (void)model->forward(input, {}, {}, input, false, true);
    }

    const auto start = std::chrono::steady_clock::now();
    torch::Tensor last_logits;
    for (int i = 0; i < iterations; ++i) {
        auto out = model->forward(input, {}, {}, input, false, true);
        last_logits = out.logits;
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    const double tokens = static_cast<double>(iterations) * batch * seq;
    std::cout << "iterations=" << iterations
              << " batch=" << batch
              << " seq=" << seq
              << " avg_ms=" << (elapsed * 1000.0 / iterations)
              << " tokens_per_sec=" << (tokens / elapsed)
              << " logits=" << last_logits.sizes()
              << std::endl;
    return 0;
}

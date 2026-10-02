#include "xorzen/model.h"
#include <iostream>

int main(int argc, char** argv) {
    xorzen::ModelConfig cfg;
    cfg.vocab_size = 1024;
    cfg.hidden_size = cfg.d_model = 128;
    cfg.num_layers = cfg.n_layers = 2;
    cfg.max_depth = 2;
    cfg.num_attention_heads = cfg.n_heads = 4;
    cfg.context_length = cfg.max_seq_len = 64;
    cfg.expert_count = cfg.num_experts = 8;
    cfg.top_k_experts = cfg.experts_per_token = 2;
    cfg.width_choices = {64, 96, 128};
    cfg.cot_dim = cfg.cot_latent_dim = 32;
    cfg.cot_components = 2;
    cfg.use_disk_cache = false;
    cfg.normalize();

    auto model = xorzen::XorzenModel(cfg, true);
    if (argc > 1) model->load_checkpoint(argv[1]);
    xorzen::GenerationConfig gen;
    gen.max_length = 8;
    auto prompt = torch::tensor({1, 5, 9}, torch::kLong);
    auto tokens = model->generate(prompt, gen);
    std::cout << tokens << std::endl;
    return 0;
}

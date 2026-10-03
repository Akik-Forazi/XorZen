#include "xorzen/model.h"
#include "xorzen/ops.h"

#include <stdexcept>

namespace xorzen {

XorzenModelImpl::XorzenModelImpl(ModelConfig cfg, bool test_mode) : config(std::move(cfg)) {
    config.normalize();
    validate_config();

    token_embedding = register_module("token_embedding",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(config.vocab_size, config.hidden_size).padding_idx(config.pad_token_id)));
    position_embedding = register_module("position_embedding",
        torch::nn::Embedding(config.context_length, config.hidden_size));
    embedding_dropout = register_module("embedding_dropout", torch::nn::Dropout(config.dropout));
    router = register_module("router", AdaptiveRouter(config));
    routing_regularizer = register_module("routing_regularizer", RoutingRegularizer(config));
    blocks = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < config.num_layers; ++i) {
        blocks->push_back(HASSBlock(config, i));
    }
    moe = register_module("moe", ShardedExpertFabric(config, test_mode));
    merger = register_module("merger", XorzenMergerGate(config));
    final_norm = register_module("final_norm", RMSNorm(config.hidden_size, config.layer_norm_eps));
    lm_head = register_module("lm_head",
        torch::nn::Linear(torch::nn::LinearOptions(config.hidden_size, config.vocab_size).bias(false)));
    
    // Latent Chain-of-Thought
    cot = register_module("cot", InternalLatentCoT(config));
    cot_loss_head = register_module("cot_loss_head", CoTAuxiliaryLoss(config));
    
    init_weights();
    if (config.tie_word_embeddings) {
        lm_head->weight = token_embedding->weight;
    }
}

void XorzenModelImpl::validate_config() const {
    if (config.vocab_size <= 0) throw std::invalid_argument("vocab_size must be positive");
    if (config.hidden_size <= 0) throw std::invalid_argument("hidden_size must be positive");
    if (config.num_layers <= 0) throw std::invalid_argument("num_layers must be positive");
    if (config.num_attention_heads <= 0) throw std::invalid_argument("num_attention_heads must be positive");
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::invalid_argument("hidden_size must be divisible by num_attention_heads");
    }
    if (config.top_k_experts <= 0 || config.top_k_experts > config.expert_count) {
        throw std::invalid_argument("top_k_experts must be in [1, expert_count]");
    }
}

void XorzenModelImpl::init_weights() {
    for (auto& module : modules(false)) {
        if (auto* linear = module->as<torch::nn::Linear>()) {
            torch::nn::init::normal_(linear->weight, 0.0, 0.02);
            if (linear->bias.defined()) torch::nn::init::zeros_(linear->bias);
        } else if (auto* embedding = module->as<torch::nn::Embedding>()) {
            torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
        }
    }
}

ModelOutput XorzenModelImpl::forward(const torch::Tensor& input_ids,
                                     const torch::Tensor& attention_mask_in,
                                     const torch::Tensor& position_ids_in,
                                     const torch::Tensor& labels,
                                     bool output_hidden_states,
                                     bool) {
    if (input_ids.dim() != 2) throw std::invalid_argument("input_ids must be [batch, seq]");
    const int64_t B = input_ids.size(0);
    const int64_t T = input_ids.size(1);
    if (T > config.context_length) throw std::invalid_argument("sequence length exceeds context_length");
    auto device = input_ids.device();

    auto hidden = token_embedding->forward(input_ids);
    torch::Tensor position_ids = position_ids_in;
    if (!position_ids.defined()) {
        position_ids = torch::arange(T, torch::TensorOptions().dtype(torch::kLong).device(device)).view({1, T}).expand({B, T});
    }
    hidden = embedding_dropout->forward(hidden + position_embedding->forward(position_ids));

    torch::Tensor attention_mask = attention_mask_in;
    if (!attention_mask.defined()) {
        attention_mask = torch::ones({B, T}, torch::TensorOptions().dtype(torch::kBool).device(device));
    }

    torch::Tensor cot_vector;
    torch::Tensor cot_influence;
    if (cot->cot_enabled) {
        auto cot_out = cot->forward(hidden);
        cot_vector = cot_out.first;
        cot_influence = cot_out.second;
        auto gate = torch::sigmoid(cot->injection_gate->forward(hidden));
        hidden = hidden + cot_influence * gate;
    } else {
        cot_vector = torch::zeros({B, T, config.cot_total_dim()}, hidden.options());
    }

    auto decision = router->forward(hidden, cot_vector);

    std::vector<torch::Tensor> layer_outputs;
    for (int64_t i = 0; i < config.num_layers; ++i) {
        auto layer_mask = decision.depth_mask.select(-1, i);
        auto block = blocks[i]->as<HASSBlock>();
        torch::Tensor block_out;
        if (!is_training() && !layer_mask.any().item<bool>()) {
            block_out = hidden;
        } else {
            block_out = block->forward(hidden, &decision, attention_mask);
        }
        hidden = block_out * layer_mask.unsqueeze(-1) + hidden * (1.0 - layer_mask.unsqueeze(-1));
        if (output_hidden_states) layer_outputs.push_back(hidden);
    }

    auto hidden_flat = hidden.reshape({B * T, config.hidden_size});
    auto expert_indices_flat = decision.expert_indices.reshape({B * T, config.top_k_experts});
    auto expert_weights_flat = decision.expert_weights.reshape({B * T, config.top_k_experts});
    auto mask_flat = attention_mask.reshape({B * T});
    auto [moe_flat, moe_stats] = moe->forward(hidden_flat, expert_indices_flat, expert_weights_flat, mask_flat);
    auto moe_output = moe_flat.reshape({B, T, config.hidden_size});
    auto merged = merger->forward(hidden, moe_output, cot_vector, attention_mask);
    hidden = final_norm->forward(merged);
    auto logits = lm_head->forward(hidden);

    torch::Tensor loss;
    torch::Tensor lm_loss;
    if (labels.defined()) {
        if (labels.sizes() != input_ids.sizes()) throw std::invalid_argument("labels shape must match input_ids");
        auto shift_logits = logits.slice(1, 0, T - 1).contiguous();
        auto shift_labels = labels.slice(1, 1, T).contiguous();
        lm_loss = torch::nn::functional::cross_entropy(
            shift_logits.view({-1, config.vocab_size}),
            shift_labels.view({-1}),
            torch::nn::functional::CrossEntropyFuncOptions().ignore_index(config.pad_token_id));
    }

    auto routing_loss = routing_regularizer->forward(decision);
    auto load_balance = compute_load_balance_loss(expert_indices_flat, expert_weights_flat);
    torch::Tensor cot_consistency = torch::zeros({}, hidden.options());
    if (lm_loss.defined()) loss = lm_loss + routing_loss + load_balance;
    else loss = routing_loss + load_balance;

    if (cot->cot_enabled) {
        auto cot_losses = cot_loss_head->forward(cot_vector, labels, {}, attention_mask);
        cot_consistency = cot_losses["consistency"];
        auto total_cot_aux = cot_losses["total_auxiliary"];
        loss = loss + total_cot_aux;
    }

    ModelOutput output;
    auto actions = torch::zeros({B, T, 1}, hidden.options());
    output.logits = logits;
    output.loss = loss;
    output.lm_loss = lm_loss;
    output.cot_vector = cot_vector;
    output.routing_loss = routing_loss;
    output.load_balance_loss = load_balance;
    output.cot_consistency_loss = cot_consistency;
    output.layer_outputs = std::move(layer_outputs);
    output.agentic_actions = actions;
    output.expert_stats = {
        {"experts_used", static_cast<float>(moe_stats.experts_used)},
        {"cache_hit_rate", static_cast<float>(moe_stats.cache_hit_rate)},
        {"routing_entropy", static_cast<float>(moe_stats.routing_entropy)}
    };
    output.active_params = estimate_active_params(decision);
    output.compute_cost = estimate_compute_cost(decision, B, T);
    ++step_count;
    total_tokens_processed += B * T;
    return output;
}

torch::Tensor XorzenModelImpl::generate(const torch::Tensor& prompt, const GenerationConfig& generation_config) {
    torch::NoGradGuard guard;
    auto tokens = prompt.clone();
    if (tokens.dim() == 1) tokens = tokens.unsqueeze(0);
    for (int64_t i = 0; i < generation_config.max_length; ++i) {
        auto context = tokens;
        if (context.size(1) > config.context_length) {
            context = context.slice(1, context.size(1) - config.context_length, context.size(1));
        }
        auto out = forward(context);
        auto next_logits = out.logits.select(1, out.logits.size(1) - 1);
        auto next = sample_next_token(next_logits, generation_config).view({tokens.size(0), 1});
        tokens = torch::cat({tokens, next}, 1);
        if ((next == generation_config.eos_token_id).all().item<bool>() && generation_config.early_stopping) break;
    }
    return tokens;
}

int64_t XorzenModelImpl::count_parameters(bool only_trainable) const {
    int64_t total = 0;
    for (const auto& p : parameters()) {
        if (!only_trainable || p.requires_grad()) total += p.numel();
    }
    return total;
}

void XorzenModelImpl::save_checkpoint(const std::string& path) {
    torch::serialize::OutputArchive archive;
    save(archive);
    archive.save_to(path);
}

void XorzenModelImpl::load_checkpoint(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    load(archive);
}

torch::Tensor XorzenModelImpl::compute_load_balance_loss(const torch::Tensor& expert_indices,
                                                         const torch::Tensor& expert_weights) const {
    auto usage = torch::zeros({config.expert_count}, expert_weights.options());
    for (int64_t k = 0; k < config.top_k_experts; ++k) {
        usage.scatter_add_(0, expert_indices.select(1, k), expert_weights.select(1, k));
    }
    auto probs = usage / (static_cast<double>(expert_indices.size(0) * config.top_k_experts) + 1e-8);
    return config.load_balancing_weight * (probs - 1.0 / static_cast<double>(config.expert_count)).pow(2).sum();
}

int64_t XorzenModelImpl::estimate_active_params(const RoutingDecision& decision) const {
    auto avg_depth = decision.depth_mask.sum(-1).to(torch::kFloat).mean().item<double>();
    auto active_experts = static_cast<double>(config.top_k_experts) / static_cast<double>(config.expert_count);
    return static_cast<int64_t>(
        config.vocab_size * config.hidden_size +
        avg_depth * config.hidden_size * config.hidden_size * 8.0 +
        active_experts * config.expert_count * config.hidden_size * config.expert_hidden_multiplier);
}

double XorzenModelImpl::estimate_compute_cost(const RoutingDecision& decision, int64_t batch, int64_t seq) const {
    auto avg_depth = decision.depth_mask.to(torch::kFloat).mean().item<double>() * config.num_layers;
    return static_cast<double>(batch * seq * config.hidden_size * config.hidden_size) * avg_depth;
}

void XorzenModelImpl::enable_cot(bool enabled) {
    if (cot) {
        cot->enable_cot(enabled);
    }
}

} // namespace xorzen

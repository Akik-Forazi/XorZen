#include "xorzen/model.h"
#include "xorzen/ops.h"

#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_map>

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
    // NOTE: cot_loss_head REMOVED — Python does not instantiate it.
    
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

    // NOTE: cot_loss_head removed — Python does not add CoT auxiliary losses
    // during pre-training. The cot_consistency loss is computed separately
    // in the Python model._compute_cot_consistency_loss but only when
    // _cot_enabled is True. For tiny_23k pre-training, CoT is frozen.

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

// ─── Load from per-tensor map (Python converter output) ─────────
XorzenModelImpl::LoadResult
XorzenModelImpl::load_from_tensor_map(
    const std::unordered_map<std::string, torch::Tensor>& sd, bool strict) {
    LoadResult result;
    torch::NoGradGuard ng;

    // Build the set of C++ parameter names we expect.
    std::set<std::string> cpp_param_names;
    for (const auto& nv : named_parameters()) cpp_param_names.insert(nv.key());
    for (const auto& nv : named_buffers()) cpp_param_names.insert(nv.key());

    // Track which Python keys we've consumed.
    std::set<std::string> consumed;

    // Helper: try to copy a Python tensor into a C++ parameter.
    // Tries the exact name, then a Python alias.
    auto try_copy = [&](torch::Tensor& param, const std::string& cpp_name,
                        const std::vector<std::string>& py_aliases) {
        for (const auto& alias : py_aliases) {
            auto it = sd.find(alias);
            if (it == sd.end()) continue;
            if (it->second.sizes() != param.sizes()) {
                result.shape_mismatches++;
                result.shape_mismatch_keys.push_back(alias);
                continue;
            }
            param.copy_(it->second);
            result.matched++;
            consumed.insert(alias);
            return true;
        }
        return false;
    };

    // Iterate over all C++ parameters and try to load each.
    for (auto& nv : named_parameters()) {
        const auto& name = nv.key();
        auto& param = nv.value();

        // Direct match
        { auto it = sd.find(name);
          if (it != sd.end() && it->second.sizes() == param.sizes()) {
              param.copy_(it->second); result.matched++; consumed.insert(name); continue;
          }
        }

        // Python alias: moe.dummy_expert.* ← moe.experts.0.*
        if (name.find("moe.dummy_expert.") == 0) {
            auto py_name = std::string("moe.experts.0.") + name.substr(strlen("moe.dummy_expert."));
            auto it = sd.find(py_name);
            if (it != sd.end() && it->second.sizes() == param.sizes()) {
                param.copy_(it->second); result.matched++; consumed.insert(py_name); continue;
            }
        }

        // Python alias: blocks.{i}.local.* ← blocks.{i}.pathways.local.*
        // C++ registers local/low_rank/ssm directly, Python uses a ModuleDict "pathways"
        if (name.find("blocks.") == 0) {
            // Find the second dot: "blocks.{i}." → the dot after {i}
            // blocks.0.local... → second dot at position 8
            // blocks.10.local... → second dot at position 9
            size_t first_dot = name.find('.');
            size_t second_dot = name.find('.', first_dot + 1);
            if (second_dot != std::string::npos) {
                auto rest = name.substr(second_dot + 1);
                if (rest.find("local.") == 0 || rest.find("low_rank.") == 0 || rest.find("ssm.") == 0) {
                    auto py_name = name.substr(0, second_dot + 1) + "pathways." + rest;
                    auto it = sd.find(py_name);
                    if (it != sd.end() && it->second.sizes() == param.sizes()) {
                        param.copy_(it->second); result.matched++; consumed.insert(py_name); continue;
                    }
                }
            }
        }

        // Python alias: blocks.{i}.pathways.* — C++ uses same name (pathways is in the ModuleDict)
        // No alias needed — the names should match.

        // Not found
        result.missing++;
        result.missing_keys.push_back(name);
    }

    // Also handle buffers (e.g. router.width_values)
    for (auto& nv : named_buffers()) {
        const auto& name = nv.key();
        auto& buf = nv.value();
        auto it = sd.find(name);
        if (it != sd.end() && it->second.sizes() == buf.sizes()) {
            buf.copy_(it->second); result.matched++; consumed.insert(name);
        }
    }

    // Tie lm_head to token_embedding if config says so
    if (config.tie_word_embeddings) {
        lm_head->weight = token_embedding->weight;
    }

    // Find unexpected keys (in Python but not consumed)
    for (const auto& [k, v] : sd) {
        if (consumed.find(k) == consumed.end()) {
            // Known C++-only keys that are OK to skip
            if (k.find("pathway_gate.") != std::string::npos ||
                k.find("low_rank.context_weights") != std::string::npos ||
                k.find("low_rank.ln_low_rank.") != std::string::npos ||
                k.find("character_router.") != std::string::npos ||
                k.find("cot_loss_head.") != std::string::npos ||
                k.find("moe.experts.") == 0) {
                // moe.experts.0.* are consumed by the moe.dummy_expert.* alias
                continue;
            }
            result.unexpected++;
            result.unexpected_keys.push_back(k);
        }
    }

    if (strict) {
        if (result.missing > 0 || result.unexpected > 0 || result.shape_mismatches > 0) {
            std::string msg = "load_from_tensor_map STRICT mode failed:\n";
            msg += "  missing: " + std::to_string(result.missing) + "\n";
            msg += "  unexpected: " + std::to_string(result.unexpected) + "\n";
            msg += "  shape_mismatches: " + std::to_string(result.shape_mismatches) + "\n";
            for (const auto& k : result.missing_keys) msg += "    MISSING: " + k + "\n";
            for (const auto& k : result.unexpected_keys) msg += "    UNEXPECTED: " + k + "\n";
            for (const auto& k : result.shape_mismatch_keys) msg += "    SHAPE: " + k + "\n";
            throw std::runtime_error(msg);
        }
    }

    return result;
}

// ─── Save to per-tensor .bin directory ──────────────────────────
void XorzenModelImpl::save_to_tensor_map(const std::string& dir) const {
    std::filesystem::create_directories(dir);
    std::ofstream manifest(dir + "/state_dict_manifest.txt");
    manifest << "state_dict\n";
    for (const auto& nv : named_parameters()) {
        const auto& name = nv.key();
        const auto& t = nv.value();
        auto tc = t.to(torch::kCPU).contiguous();
        std::string fname = "param_" + name + ".bin";
        std::ofstream f(dir + "/" + fname, std::ios::binary);
        f.write(reinterpret_cast<const char*>(tc.data_ptr()), tc.nbytes());
        std::string shape_csv;
        for (size_t i = 0; i < tc.sizes().size(); ++i) {
            if (i > 0) shape_csv += ",";
            shape_csv += std::to_string(tc.size(i));
        }
        manifest << "tensor " << name << " float32 " << fname << " " << shape_csv << "\n";
    }
    for (const auto& nv : named_buffers()) {
        const auto& name = nv.key();
        const auto& t = nv.value();
        auto tc = t.to(torch::kCPU).contiguous();
        std::string fname = "param_" + name + ".bin";
        std::ofstream f(dir + "/" + fname, std::ios::binary);
        f.write(reinterpret_cast<const char*>(tc.data_ptr()), tc.nbytes());
        std::string shape_csv;
        for (size_t i = 0; i < tc.sizes().size(); ++i) {
            if (i > 0) shape_csv += ",";
            shape_csv += std::to_string(tc.size(i));
        }
        manifest << "tensor " << name << " float32 " << fname << " " << shape_csv << "\n";
    }
    manifest << "end\n";
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

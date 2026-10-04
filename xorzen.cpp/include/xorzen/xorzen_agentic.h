// xorzen_agentic.h — Zero Agentic Model (C++ implementation)
//
// Mirrors xorzen/models/zero/agentic_model.py exactly.
// This is a SEPARATE architecture from XorzenModel — it uses:
//   - FlashSSM (not the HASS SSM pathway)
//   - GatedLinearAttention (not standard multi-head attention)
//   - ZeroAgenticBlock (not HASSBlock)
//   - Memory vault, ActionHead, CritiqueModule, RecursiveRouter
//   - Latent CoT (different from InternalLatentCoT in the Zero model)
#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace xorzen {

// ─── ZeroAgenticConfig ──────────────────────────────────────────
struct ZeroAgenticConfig {
    int64_t vocab_size = 10000;
    int64_t hidden_size = 256;
    int64_t num_layers = 6;
    int64_t num_attention_heads = 8;
    int64_t context_length = 2048;
    int64_t pad_token_id = 0;

    // Agentic capabilities
    bool use_latent_cot = true;
    int64_t latent_cot_dim = 128;
    int64_t num_action_slots = 16;
    bool use_self_critique = true;

    // Memory
    int64_t memory_slots = 64;
    int64_t memory_dim = 128;

    // Thinking
    int64_t recurrence_depth = 1;
    bool adaptive_depth = true;

    // SSM
    int64_t d_state = 16;
};

// ─── InternalLatentCoT (agentic_model.py:18-45) ─────────────────
struct InternalLatentCoTAgenticImpl : torch::nn::Module {
    int64_t cot_dim;
    torch::nn::Linear update_gate{nullptr};
    torch::nn::Sequential transform{nullptr};

    InternalLatentCoTAgenticImpl(int64_t d_model, int64_t cot_dim_)
        : cot_dim(cot_dim_) {
        update_gate = register_module("update_gate",
            torch::nn::Linear(d_model + cot_dim, cot_dim));
        transform = register_module("transform", torch::nn::Sequential(
            torch::nn::Linear(cot_dim, cot_dim),
            torch::nn::GELU(),
            torch::nn::Linear(cot_dim, cot_dim)));
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& prev_cot = {}) {
        int64_t B = x.size(0), L = x.size(1);
        torch::Tensor pc = prev_cot.defined() ? prev_cot :
            torch::zeros({B, L, cot_dim}, x.options());
        auto combined = torch::cat({x, pc}, -1);
        auto gate = torch::sigmoid(update_gate->forward(combined));
        auto thought = transform->forward(pc);
        return (1 - gate) * pc + gate * thought;
    }
};
TORCH_MODULE(InternalLatentCoTAgentic);

// ─── ActionHead (agentic_model.py:47-57) ────────────────────────
struct ActionHeadImpl : torch::nn::Module {
    torch::nn::Linear proj{nullptr};
    ActionHeadImpl(int64_t d_model, int64_t num_actions)
        : proj(register_module("proj", torch::nn::Linear(d_model, num_actions))) {}
    torch::Tensor forward(const torch::Tensor& x) { return proj->forward(x); }
};
TORCH_MODULE(ActionHead);

// ─── CritiqueModule (agentic_model.py:59-74) ────────────────────
struct CritiqueModuleImpl : torch::nn::Module {
    torch::nn::Sequential quality_eval{nullptr};
    CritiqueModuleImpl(int64_t d_model)
        : quality_eval(register_module("quality_eval", torch::nn::Sequential(
            torch::nn::Linear(d_model, d_model / 2),
            torch::nn::GELU(),
            torch::nn::Linear(d_model / 2, 1),
            torch::nn::Sigmoid()))) {}
    torch::Tensor forward(const torch::Tensor& x) { return quality_eval->forward(x); }
};
TORCH_MODULE(CritiqueModule);

// ─── FlashSSM (agentic_model.py:78-93) ──────────────────────────
struct FlashSSMImpl : torch::nn::Module {
    int64_t d_model;
    torch::nn::Linear in_proj{nullptr}, x_proj{nullptr}, out_proj{nullptr};
    torch::nn::Conv1d conv1d{nullptr};

    FlashSSMImpl(int64_t d_model_, int64_t d_state = 16)
        : d_model(d_model_) {
        in_proj = register_module("in_proj", torch::nn::Linear(d_model, d_model * 2));
        conv1d = register_module("conv1d", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(d_model, d_model, 4).groups(d_model).padding(3)));
        x_proj = register_module("x_proj", torch::nn::Linear(d_model, d_model));
        out_proj = register_module("out_proj", torch::nn::Linear(d_model, d_model));
    }

    torch::Tensor forward(const torch::Tensor& x) {
        int64_t B = x.size(0), L = x.size(1);
        auto u = in_proj->forward(x);
        auto chunks = u.chunk(2, -1);
        auto x_ssm = chunks[0];
        auto z = chunks[1];
        x_ssm = conv1d->forward(x_ssm.transpose(1, 2)).slice(2, 0, L).transpose(1, 2);
        auto y = x_ssm * torch::sigmoid(x_proj->forward(x_ssm));
        return out_proj->forward(y * torch::silu(z));
    }
};
TORCH_MODULE(FlashSSM);

// ─── GatedLinearAttention (agentic_model.py:95-115) ─────────────
struct GatedLinearAttentionImpl : torch::nn::Module {
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, g_proj{nullptr}, out_proj{nullptr};

    GatedLinearAttentionImpl(int64_t d_model, int64_t num_heads = 8) {
        q_proj = register_module("q_proj", torch::nn::Linear(d_model, d_model));
        k_proj = register_module("k_proj", torch::nn::Linear(d_model, d_model));
        v_proj = register_module("v_proj", torch::nn::Linear(d_model, d_model));
        g_proj = register_module("g_proj", torch::nn::Linear(d_model, d_model));
        out_proj = register_module("out_proj", torch::nn::Linear(d_model, d_model));
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& attention_mask = {}) {
        auto q = q_proj->forward(x);
        auto k = k_proj->forward(x);
        auto v = v_proj->forward(x);
        auto g = torch::sigmoid(g_proj->forward(x));
        if (attention_mask.defined()) {
            auto m = attention_mask.unsqueeze(-1).to(torch::kFloat32);
            k = k * m;
            v = v * m;
        }
        // Linear attention: kv = relu(k) ⊗ v, cumsum, y = relu(q) · kv_cum
        auto kv = torch::einsum("bld,ble->blde", {torch::relu(k), v});
        auto kv_cum = kv.cumsum(1);
        auto y = torch::einsum("bld,blde->ble", {torch::relu(q), kv_cum});
        return out_proj->forward(y * g);
    }
};
TORCH_MODULE(GatedLinearAttention);

// ─── RecursiveRouter (agentic_model.py:117-123) ─────────────────
struct RecursiveRouterImpl : torch::nn::Module {
    torch::nn::Linear halt_proj{nullptr};
    RecursiveRouterImpl(int64_t d_model)
        : halt_proj(register_module("halt_proj", torch::nn::Linear(d_model, 1))) {}
    torch::Tensor forward(const torch::Tensor& x) { return torch::sigmoid(halt_proj->forward(x)); }
};
TORCH_MODULE(RecursiveRouter);

// ─── ZeroAgenticBlock (agentic_model.py:125-151) ────────────────
struct ZeroAgenticBlockImpl : torch::nn::Module {
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    FlashSSM ssm{nullptr};
    GatedLinearAttention attn{nullptr};
    torch::nn::Sequential mlp{nullptr};
    torch::Tensor mix_gate;

    ZeroAgenticBlockImpl(const ZeroAgenticConfig& config) {
        ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.hidden_size})));
        ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.hidden_size})));
        ssm = register_module("ssm", FlashSSM(config.hidden_size, config.d_state));
        attn = register_module("attn", GatedLinearAttention(config.hidden_size, config.num_attention_heads));
        mlp = register_module("mlp", torch::nn::Sequential(
            torch::nn::Linear(config.hidden_size, config.hidden_size * 4),
            torch::nn::GELU(),
            torch::nn::Linear(config.hidden_size * 4, config.hidden_size)));
        mix_gate = register_parameter("mix_gate", torch::tensor({0.5}));
    }

    torch::Tensor forward(const torch::Tensor& x_in, const torch::Tensor& attention_mask = {}) {
        torch::Tensor x = x_in;
        auto residual = x;
        auto xn = ln1->forward(x);
        auto mg = mix_gate.unsqueeze(0).unsqueeze(0);
        auto x_mix = ssm->forward(xn) * mg + attn->forward(xn, attention_mask) * (1 - mg);
        x = residual + x_mix;
        residual = x;
        x = ln2->forward(x);
        return residual + mlp->forward(x);
    }
};
TORCH_MODULE(ZeroAgenticBlock);

// ─── ZeroAgenticModel (agentic_model.py:155-301) ────────────────
struct ZeroAgenticModelImpl : torch::nn::Module {
    ZeroAgenticConfig config;
    torch::nn::Embedding token_embedding{nullptr}, pos_embedding{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    InternalLatentCoTAgentic latent_cot{nullptr};
    torch::nn::Linear cot_to_hidden{nullptr};
    ActionHead action_head{nullptr};
    CritiqueModule critique{nullptr};
    RecursiveRouter router{nullptr};
    torch::Tensor memory_vault;
    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};

    ZeroAgenticModelImpl(const ZeroAgenticConfig& cfg) : config(cfg) {
        token_embedding = register_module("token_embedding",
            torch::nn::Embedding(cfg.vocab_size, cfg.hidden_size));
        pos_embedding = register_module("pos_embedding",
            torch::nn::Embedding(cfg.context_length, cfg.hidden_size));
        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int64_t i = 0; i < cfg.num_layers; ++i)
            blocks->push_back(ZeroAgenticBlock(cfg));
        if (cfg.use_latent_cot) {
            latent_cot = register_module("latent_cot",
                InternalLatentCoTAgentic(cfg.hidden_size, cfg.latent_cot_dim));
            cot_to_hidden = register_module("cot_to_hidden",
                torch::nn::Linear(cfg.latent_cot_dim, cfg.hidden_size));
        }
        action_head = register_module("action_head", ActionHead(cfg.hidden_size, cfg.num_action_slots));
        critique = register_module("critique", CritiqueModule(cfg.hidden_size));
        router = register_module("router", RecursiveRouter(cfg.hidden_size));
        memory_vault = register_parameter("memory_vault",
            torch::randn({1, cfg.memory_slots, cfg.hidden_size}));
        ln_f = register_module("ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.hidden_size})));
        lm_head = register_module("lm_head",
            torch::nn::Linear(torch::nn::LinearOptions(cfg.hidden_size, cfg.vocab_size).bias(false)));
        // Tie weights
        lm_head->weight = token_embedding->weight;
        // Init weights
        _init_weights();
    }

    void _init_weights() {
        torch::NoGradGuard ng;
        for (auto& m : modules(false)) {
            if (auto* lin = m->as<torch::nn::Linear>()) {
                lin->weight.data().normal_(0.0, 0.02);
                if (lin->bias.defined()) lin->bias.data().zero_();
            }
            if (auto* emb = m->as<torch::nn::Embedding>()) {
                emb->weight.data().normal_(0.0, 0.02);
            }
            if (auto* ln = m->as<torch::nn::LayerNorm>()) {
                ln->weight.data().fill_(1.0);
                if (ln->bias.defined()) ln->bias.data().zero_();
            }
        }
        lm_head->weight = token_embedding->weight;
    }

    struct Output {
        torch::Tensor logits;
        torch::Tensor loss;
        torch::Tensor actions;
        int64_t recurrence_steps = 0;
    };

    Output forward(const torch::Tensor& input_ids,
                   const torch::Tensor& labels = {},
                   const torch::Tensor& attention_mask = {},
                   const torch::Tensor& action_targets = {},
                   double action_loss_weight = 0.1) {
        int64_t B = input_ids.size(0), L = input_ids.size(1);
        auto pos_ids = torch::arange(L).unsqueeze(0).expand({B, L});
        auto x = token_embedding->forward(input_ids) + pos_embedding->forward(pos_ids);

        // Memory vault retrieval
        auto vault = memory_vault.expand({B, -1, -1});
        auto mem_scores = torch::bmm(x, vault.transpose(1, 2)) /
                          std::sqrt(static_cast<double>(config.hidden_size));
        auto mem_weights = torch::softmax(mem_scores, -1);
        auto mem_retrieved = torch::bmm(mem_weights, vault);
        x = x + mem_retrieved;

        // Latent CoT
        torch::Tensor latent_state;
        if (config.use_latent_cot) {
            latent_state = latent_cot->forward(x);
            x = x + cot_to_hidden->forward(latent_state);
        }

        int64_t total_recurrence_steps = 0;
        torch::Tensor ponder_cost = torch::tensor(0.0, x.options());

        for (auto& block_mod : *blocks) {
            auto block = block_mod->as<ZeroAgenticBlock>();
            x = block->forward(x, attention_mask);

            if (config.recurrence_depth > 1) {
                for (int r = 0; r < config.recurrence_depth - 1; ++r) {
                    auto halt_prob = router->forward(x);
                    auto quality = critique->forward(x);
                    auto agentic_halt = halt_prob * quality;
                    ponder_cost = ponder_cost + agentic_halt.mean();
                    auto new_x = block->forward(x, attention_mask);
                    x = agentic_halt * x + (1 - agentic_halt) * new_x;
                    if (config.use_latent_cot) {
                        latent_state = latent_cot->forward(x, latent_state);
                        x = x + 0.1 * cot_to_hidden->forward(latent_state);
                    }
                    if (agentic_halt.mean().item<double>() > 0.8) break;
                    total_recurrence_steps++;
                }
            }
        }

        auto actions = action_head->forward(x);
        x = ln_f->forward(x);
        auto logits = lm_head->forward(x);

        torch::Tensor loss;
        if (labels.defined()) {
            auto sl = logits.slice(1, 0, L - 1).contiguous().view({-1, config.vocab_size});
            auto sb = labels.slice(1, 1, L).contiguous().view({-1});
            loss = torch::nn::functional::cross_entropy(sl, sb);
            loss = loss - 0.01 * ponder_cost;
            if (action_targets.defined()) {
                loss = loss + action_loss_weight * torch::mse_loss(actions, action_targets);
            } else {
                auto ap = torch::softmax(actions, -1);
                auto alp = torch::log_softmax(actions, -1);
                auto entropy_reg = -(ap * alp).sum(-1).mean();
                loss = loss - 0.01 * entropy_reg;
            }
        }

        return {logits, loss, actions, total_recurrence_steps};
    }

    torch::Tensor generate(const torch::Tensor& prompt, int64_t max_length) {
        torch::NoGradGuard ng;
        auto tokens = prompt.clone();
        if (tokens.dim() == 1) tokens = tokens.unsqueeze(0);
        for (int64_t i = 0; i < max_length; ++i) {
            auto context = tokens;
            if (context.size(1) > config.context_length)
                context = context.slice(1, context.size(1) - config.context_length);
            auto out = forward(context);
            auto next = out.logits.select(1, -1).argmax(-1, false).unsqueeze(1);
            tokens = torch::cat({tokens, next}, 1);
        }
        return tokens;
    }

    void load_from_tensor_map(const std::unordered_map<std::string, torch::Tensor>& sd) {
        torch::NoGradGuard ng;
        for (auto& nv : named_parameters()) {
            auto it = sd.find(nv.key());
            if (it != sd.end() && it->second.sizes() == nv.value().sizes()) {
                nv.value().copy_(it->second);
            }
        }
        if (config.vocab_size > 0)
            lm_head->weight = token_embedding->weight;
    }

    void save_to_tensor_map(const std::string& dir) const {
        std::filesystem::create_directories(dir);
        for (auto& nv : named_parameters()) {
            auto tc = nv.value().to(torch::kCPU).contiguous();
            std::ofstream f(dir + "/param_" + nv.key() + ".bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(tc.data_ptr()), tc.nbytes());
        }
    }
};
TORCH_MODULE(ZeroAgenticModel);

} // namespace xorzen

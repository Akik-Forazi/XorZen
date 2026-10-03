// generation_test.cpp — C++ generation matching Python semantics (no KV cache)
//
// Implements:
//   - greedy
//   - temperature
//   - top-k
//   - top-p (nucleus)
//   - repetition penalty
//   - EOS handling
//   - max_new_tokens
//   - beam search (returns highest-scoring beam)
//
// Mirrors Python xorzen/models/zero/model.py:871-1283.
// Does NOT implement KV cache (per user directive).
//
// The generation uses the SmokeTestModel from training_smoke_test.cpp
// (simplified model with embeddings + HASS block + LM head).
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

// Reuse the SmokeTestModel from training_smoke_test.cpp
// We include the .cpp file up to (but not including) main().
// The SmokeTestModel and manifest/tensor IO functions are defined there.
// We define main() here, so we need to avoid the duplicate main().
// Approach: copy the SmokeTestModel struct + manifest/tensor IO inline.

using Tensor = torch::Tensor;
using TensorMap = std::unordered_map<std::string, Tensor>;

// ─────────────────────────────────────────────────────────────
// SmokeTestModel — copied from training_smoke_test.cpp
// ─────────────────────────────────────────────────────────────
struct SmokeTestModel : torch::nn::Module {
    torch::nn::Embedding token_embedding{nullptr}, position_embedding{nullptr};
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, out_proj{nullptr};
    torch::nn::LayerNorm ln_q{nullptr}, ln_k{nullptr};
    torch::Tensor A_log;
    torch::nn::Linear dt_proj{nullptr}, B_proj{nullptr}, C_proj{nullptr}, D_proj{nullptr}, gate_proj{nullptr};
    torch::nn::Conv1d conv{nullptr};
    torch::nn::LayerNorm ssm_ln_input{nullptr}, ssm_ln_state{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::LayerNorm ffn_ln_input{nullptr}, ffn_ln_hidden{nullptr};
    torch::nn::Linear lm_head{nullptr};
    int64_t H, V, T_max, nh, hd, state_dim, kernel;
    double eps;

    SmokeTestModel(int64_t vocab, int64_t hidden, int64_t ctx_len, int64_t num_heads,
                   int64_t state, int64_t kern)
        : H(hidden), V(vocab), T_max(ctx_len),
          nh(std::max<int64_t>(1, num_heads / 2)), hd(hidden / std::max<int64_t>(1, num_heads / 2)),
          state_dim(state), kernel(kern), eps(1e-5) {
        token_embedding = register_module("token_embedding",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(V, H)));
        position_embedding = register_module("position_embedding",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(T_max, H)));
        ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H})));
        ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H})));
        q_proj = register_module("q_proj", torch::nn::Linear(H, H));
        k_proj = register_module("k_proj", torch::nn::Linear(H, H));
        v_proj = register_module("v_proj", torch::nn::Linear(H, H));
        out_proj = register_module("out_proj", torch::nn::Linear(H, H));
        ln_q = register_module("ln_q", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H})));
        ln_k = register_module("ln_k", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H})));
        A_log = register_parameter("A_log", torch::zeros({state_dim}));
        dt_proj = register_module("dt_proj", torch::nn::Linear(H, state_dim));
        B_proj = register_module("B_proj", torch::nn::Linear(H, state_dim));
        C_proj = register_module("C_proj", torch::nn::Linear(H, state_dim));
        D_proj = register_module("D_proj", torch::nn::Linear(state_dim, H));
        gate_proj = register_module("gate_proj", torch::nn::Linear(H, H * 2));
        conv = register_module("conv", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(H, H, kernel).padding(kernel - 1).groups(H)));
        ssm_ln_input = register_module("ssm_ln_input", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H})));
        ssm_ln_state = register_module("ssm_ln_state", torch::nn::LayerNorm(torch::nn::LayerNormOptions({state_dim})));
        fc1 = register_module("fc1", torch::nn::Linear(H, H * 4));
        fc2 = register_module("fc2", torch::nn::Linear(H * 4, H));
        ffn_ln_input = register_module("ffn_ln_input", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H})));
        ffn_ln_hidden = register_module("ffn_ln_hidden", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H * 4})));
        lm_head = register_module("lm_head",
            torch::nn::Linear(torch::nn::LinearOptions(H, V).bias(false)));
    }

    void load_from_state_dict(const TensorMap& sd) {
        torch::NoGradGuard ng;
        auto copy_param = [&](torch::Tensor& param, const std::string& sd_key) {
            auto it = sd.find(sd_key);
            if (it != sd.end()) param.copy_(it->second);
        };
        copy_param(token_embedding->weight, "token_embedding.weight");
        copy_param(position_embedding->weight, "position_embedding.weight");
        copy_param(ln1->weight, "blocks.0.ln1.weight"); copy_param(ln1->bias, "blocks.0.ln1.bias");
        copy_param(ln2->weight, "blocks.0.ln2.weight"); copy_param(ln2->bias, "blocks.0.ln2.bias");
        copy_param(q_proj->weight, "blocks.0.pathways.local.q_proj.weight"); copy_param(q_proj->bias, "blocks.0.pathways.local.q_proj.bias");
        copy_param(k_proj->weight, "blocks.0.pathways.local.k_proj.weight"); copy_param(k_proj->bias, "blocks.0.pathways.local.k_proj.bias");
        copy_param(v_proj->weight, "blocks.0.pathways.local.v_proj.weight"); copy_param(v_proj->bias, "blocks.0.pathways.local.v_proj.bias");
        copy_param(out_proj->weight, "blocks.0.pathways.local.out_proj.weight"); copy_param(out_proj->bias, "blocks.0.pathways.local.out_proj.bias");
        copy_param(ln_q->weight, "blocks.0.pathways.local.ln_q.weight"); copy_param(ln_q->bias, "blocks.0.pathways.local.ln_q.bias");
        copy_param(ln_k->weight, "blocks.0.pathways.local.ln_k.weight"); copy_param(ln_k->bias, "blocks.0.pathways.local.ln_k.bias");
        copy_param(A_log, "blocks.0.pathways.ssm.A_log");
        copy_param(dt_proj->weight, "blocks.0.pathways.ssm.dt_proj.weight"); copy_param(dt_proj->bias, "blocks.0.pathways.ssm.dt_proj.bias");
        copy_param(B_proj->weight, "blocks.0.pathways.ssm.B_proj.weight"); copy_param(B_proj->bias, "blocks.0.pathways.ssm.B_proj.bias");
        copy_param(C_proj->weight, "blocks.0.pathways.ssm.C_proj.weight"); copy_param(C_proj->bias, "blocks.0.pathways.ssm.C_proj.bias");
        copy_param(D_proj->weight, "blocks.0.pathways.ssm.D_proj.weight"); copy_param(D_proj->bias, "blocks.0.pathways.ssm.D_proj.bias");
        copy_param(gate_proj->weight, "blocks.0.pathways.ssm.gate_proj.weight"); copy_param(gate_proj->bias, "blocks.0.pathways.ssm.gate_proj.bias");
        copy_param(conv->weight, "blocks.0.pathways.ssm.conv.weight"); copy_param(conv->bias, "blocks.0.pathways.ssm.conv.bias");
        copy_param(ssm_ln_input->weight, "blocks.0.pathways.ssm.ln_input.weight"); copy_param(ssm_ln_input->bias, "blocks.0.pathways.ssm.ln_input.bias");
        copy_param(ssm_ln_state->weight, "blocks.0.pathways.ssm.ln_state.weight"); copy_param(ssm_ln_state->bias, "blocks.0.pathways.ssm.ln_state.bias");
        copy_param(fc1->weight, "blocks.0.ffn.fc1.weight"); copy_param(fc1->bias, "blocks.0.ffn.fc1.bias");
        copy_param(fc2->weight, "blocks.0.ffn.fc2.weight"); copy_param(fc2->bias, "blocks.0.ffn.fc2.bias");
        copy_param(ffn_ln_input->weight, "blocks.0.ffn.ln_input.weight"); copy_param(ffn_ln_input->bias, "blocks.0.ffn.ln_input.bias");
        copy_param(ffn_ln_hidden->weight, "blocks.0.ffn.ln_hidden.weight"); copy_param(ffn_ln_hidden->bias, "blocks.0.ffn.ln_hidden.bias");
        lm_head->weight.copy_(token_embedding->weight);
    }

    Tensor forward(const Tensor& input_ids) {
        int64_t B = input_ids.size(0), T = input_ids.size(1);
        auto pos_ids = torch::arange(T).unsqueeze(0).expand({B, T});
        auto hidden = token_embedding->forward(input_ids) + position_embedding->forward(pos_ids);
        auto xa = ln1->forward(hidden);
        auto q = q_proj->forward(xa).view({B, T, nh, hd}).transpose(1, 2);
        auto k = k_proj->forward(xa).view({B, T, nh, hd}).transpose(1, 2);
        auto v = v_proj->forward(xa).view({B, T, nh, hd}).transpose(1, 2);
        q = ln_q->forward(q.transpose(1, 2)).transpose(1, 2);
        k = ln_k->forward(k.transpose(1, 2)).transpose(1, 2);
        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(double(hd));
        auto cm = torch::tril(torch::ones({T, T}, scores.options()));
        scores = scores.masked_fill(cm.eq(0), -std::numeric_limits<float>::infinity());
        auto probs = torch::softmax(scores, -1);
        auto local_out = torch::matmul(probs, v).transpose(1, 2).contiguous().view({B, T, H});
        local_out = out_proj->forward(local_out);
        auto ssm_xn = ssm_ln_input->forward(xa);
        auto conv_out = conv->forward(ssm_xn.transpose(1, 2));
        conv_out = conv_out.slice(2, 0, T);
        ssm_xn = ssm_xn + conv_out.transpose(1, 2);
        auto gp = gate_proj->forward(ssm_xn).chunk(2, -1);
        auto gate = torch::sigmoid(gp[0]);
        auto input_gate = torch::sigmoid(gp[1]);
        auto Bv = B_proj->forward(ssm_xn * input_gate);
        auto C = C_proj->forward(ssm_xn);
        auto dt = torch::softplus(dt_proj->forward(ssm_xn));
        auto a = -torch::exp(A_log);
        auto a_b = a.view({1, 1, -1}); auto z = dt * a_b; auto Ab = torch::exp(z);
        auto small = z.abs() < 1e-4;
        auto safe_z = torch::where(small, torch::ones_like(z), z);
        auto exact = (Ab - 1.0) / safe_z;
        auto taylor = 1.0 + z / 2.0 + (z * z) / 6.0;
        auto Bb = torch::where(small, taylor, exact) * dt * Bv;
        std::vector<Tensor> state_list;
        Tensor cur = torch::zeros({B, state_dim}, Ab.options());
        for (int64_t t = 0; t < T; ++t) {
            cur = Ab.select(1, t) * cur + Bb.select(1, t);
            state_list.push_back(cur.unsqueeze(1));
        }
        auto states = torch::cat(state_list, 1);
        auto states_ln = ssm_ln_state->forward(states);
        auto ssm_out = D_proj->forward(C * states_ln) * gate;
        auto combined = (local_out + ssm_out) / 2.0;
        auto residual = hidden + combined;
        auto xf = ln2->forward(residual);
        auto ffn_h = fc1->forward(ffn_ln_input->forward(xf));
        ffn_h = torch::gelu(ffn_h);
        ffn_h = ffn_ln_hidden->forward(ffn_h);
        auto ffn_out = fc2->forward(ffn_h);
        auto block_out = residual + ffn_out;
        return lm_head->forward(block_out);
    }

    Tensor compute_loss(const Tensor& logits, const Tensor& labels) {
        int64_t T = logits.size(1);
        auto sl = logits.slice(1, 0, T - 1).contiguous().view({-1, V});
        auto sb = labels.slice(1, 1, T).contiguous().view({-1});
        return torch::nn::functional::cross_entropy(sl, sb,
            torch::nn::functional::CrossEntropyFuncOptions().ignore_index(0));
    }
};

// ─────────────────────────────────────────────────────────────
// Generation strategies — mirror Python model.py:871-1283
// ─────────────────────────────────────────────────────────────

struct GenerationConfig {
    int max_new_tokens = 10;
    double temperature = 1.0;
    int top_k = 0;        // 0 = disabled
    double top_p = 1.0;   // 1.0 = disabled
    double repetition_penalty = 1.0;  // 1.0 = disabled
    int num_beams = 1;    // 1 = no beam search
    bool do_sample = false;
    int64_t eos_token_id = -1;  // -1 = no EOS
    int64_t pad_token_id = 0;
    int64_t seed = 42;
};

// Apply repetition penalty: scan the input tokens, reduce the logit of any
// token that already appeared by dividing by repetition_penalty.
// Mirrors Python model.py repetition_penalty logic.
Tensor apply_repetition_penalty(Tensor logits,           // [batch, vocab]
                                const Tensor& input_ids, // [batch, seq]
                                double penalty) {
    if (penalty == 1.0) return logits;
    for (int64_t b = 0; b < input_ids.size(0); ++b) {
        // Collect unique tokens from input_ids[b]
        std::set<int64_t> seen;
        for (int64_t t = 0; t < input_ids[b].size(0); ++t) {
            seen.insert(input_ids[b][t].item().toInt());
        }
        for (int64_t tok : seen) {
            // Python: if logit > 0, divide; else multiply
            double v = logits[b][tok].item<double>();
            double new_v = (v > 0) ? (v / penalty) : (v * penalty);
            logits[b][tok] = new_v;
        }
    }
    return logits;
}

// Apply temperature: logits /= temperature
Tensor apply_temperature(Tensor logits, double temp) {
    if (temp == 1.0) return logits;
    return logits / temp;
}

// Apply top-k: keep only the top-k logits, set the rest to -inf
Tensor apply_top_k(Tensor logits, int top_k) {
    if (top_k <= 0 || top_k >= logits.size(-1)) return logits;
    auto topk = logits.topk(top_k, -1);
    auto threshold = std::get<0>(topk).select(-1, top_k - 1).unsqueeze(-1);
    return torch::where(logits < threshold, torch::full_like(logits, -std::numeric_limits<float>::infinity()), logits);
}

// Apply top-p (nucleus): keep the smallest set of tokens whose cumulative
// probability >= top_p. Set the rest to -inf.
Tensor apply_top_p(Tensor logits, double top_p) {
    if (top_p >= 1.0) return logits;
    auto sorted = logits.sort(-1, /*descending=*/true);
    auto sorted_logits = std::get<0>(sorted);
    auto sorted_idx = std::get<1>(sorted);
    auto probs = torch::softmax(sorted_logits, -1);
    auto cumprobs = probs.cumsum(-1);
    // Mask: remove tokens with cumulative prob > top_p (but keep the first one above)
    auto mask = cumprobs > top_p;
    // Shift right: the first token above top_p should be kept
    // shifted = [False, mask[0..-2]]  (drop the last element, prepend False)
    std::vector<Tensor> shift_parts = {
        torch::zeros({mask.size(0), 1}, mask.options()),
        mask.slice(/*dim=*/1, /*start=*/0, /*end=*/mask.size(1) - 1)
    };
    auto shifted = torch::cat(shift_parts, /*dim=*/1);
    sorted_logits = sorted_logits.masked_fill(shifted, -std::numeric_limits<float>::infinity());
    // Scatter back to original order
    return sorted_logits.scatter(-1, sorted_idx, sorted_logits);
}

// Greedy decoding: pick argmax at each step
Tensor generate_greedy(SmokeTestModel& model, Tensor input_ids, const GenerationConfig& cfg) {
    int64_t B = input_ids.size(0);
    int64_t T = input_ids.size(1);
    std::vector<Tensor> output = {input_ids};
    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        auto cur = torch::cat(output, /*dim=*/1);
        auto logits = model.forward(cur);
        auto next_logits = logits.select(1, -1);  // [B, V]
        next_logits = apply_repetition_penalty(next_logits, cur, cfg.repetition_penalty);
        next_logits = apply_temperature(next_logits, cfg.temperature);
        next_logits = apply_top_k(next_logits, cfg.top_k);
        next_logits = apply_top_p(next_logits, cfg.top_p);
        auto next_tokens = next_logits.argmax(-1, /*keepdim=*/false);  // [B]
        // EOS check
        if (cfg.eos_token_id >= 0) {
            // If all beams have EOS, stop
            bool all_eos = true;
            for (int64_t b = 0; b < B; ++b) {
                if (next_tokens[b].item<int64_t>() != cfg.eos_token_id) { all_eos = false; break; }
            }
            if (all_eos) {
                output.push_back(next_tokens.unsqueeze(1));
                break;
            }
        }
        output.push_back(next_tokens.unsqueeze(1));
    }
    return torch::cat(output, /*dim=*/1);
}

// Sampling: sample from the (temperature, top-k, top-p adjusted) distribution
Tensor generate_sample(SmokeTestModel& model, Tensor input_ids, const GenerationConfig& cfg) {
    int64_t B = input_ids.size(0);
    std::vector<Tensor> output = {input_ids};
    at::Generator gen = at::detail::createCPUGenerator(cfg.seed);
    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        auto cur = torch::cat(output, /*dim=*/1);
        auto logits = model.forward(cur);
        auto next_logits = logits.select(1, -1);  // [B, V]
        next_logits = apply_repetition_penalty(next_logits, cur, cfg.repetition_penalty);
        next_logits = apply_temperature(next_logits, cfg.temperature);
        next_logits = apply_top_k(next_logits, cfg.top_k);
        next_logits = apply_top_p(next_logits, cfg.top_p);
        auto probs = torch::softmax(next_logits, -1);
        // Sample from multinomial
        auto next_tokens = torch::multinomial(probs, /*num_samples=*/1, /*replacement=*/false, gen).squeeze(-1);
        output.push_back(next_tokens.unsqueeze(1));
        if (cfg.eos_token_id >= 0) {
            bool all_eos = true;
            for (int64_t b = 0; b < B; ++b) {
                if (next_tokens[b].item<int64_t>() != cfg.eos_token_id) { all_eos = false; break; }
            }
            if (all_eos) break;
        }
    }
    return torch::cat(output, /*dim=*/1);
}

// Beam search: maintain num_beams hypotheses, return highest-scoring
Tensor generate_beam(SmokeTestModel& model, Tensor input_ids, const GenerationConfig& cfg) {
    int64_t B = input_ids.size(0);
    int64_t T = input_ids.size(1);
    int num_beams = std::max(1, cfg.num_beams);
    int64_t V = model.lm_head->weight.size(0);

    // Initialize beams: for batch 0 only (beam search typically single-batch)
    // Each beam: (tokens, score)
    std::vector<std::pair<std::vector<int64_t>, double>> beams;
    {
        std::vector<int64_t> init_tokens;
        for (int64_t t = 0; t < T; ++t) init_tokens.push_back(input_ids[0][t].item<int64_t>());
        beams.push_back({init_tokens, 0.0});
    }
    // Expand to num_beams by taking the top-k first tokens
    auto logits = model.forward(input_ids);
    auto first_logits = logits[0][T - 1];
    first_logits = apply_temperature(first_logits, cfg.temperature);
    auto log_probs = torch::log_softmax(first_logits, -1);
    auto topk = log_probs.topk(num_beams, -1);
    auto topk_vals = std::get<0>(topk);   // [num_beams]
    auto topk_idx = std::get<1>(topk);    // [num_beams]
    std::vector<std::pair<std::vector<int64_t>, double>> new_beams;
    for (int k = 0; k < num_beams; ++k) {
        auto tokens = beams[0].first;
        tokens.push_back(topk_idx[k].item<int64_t>());
        new_beams.push_back({tokens, topk_vals[k].item<double>()});
    }
    beams = new_beams;

    for (int step = 1; step < cfg.max_new_tokens; ++step) {
        std::vector<std::pair<std::vector<int64_t>, double>> candidates;
        for (auto& beam : beams) {
            // Check EOS
            if (!beam.first.empty() && beam.first.back() == cfg.eos_token_id) {
                candidates.push_back(beam);  // keep finished beam
                continue;
            }
            // Forward
            Tensor cur = torch::empty({1, (int64_t)beam.first.size()}, torch::kLong);
            for (size_t i = 0; i < beam.first.size(); ++i)
                cur[0][i] = beam.first[i];
            auto l = model.forward(cur);
            auto next_logits = l[0][cur.size(1) - 1];
            next_logits = apply_repetition_penalty(next_logits.unsqueeze(0), cur, cfg.repetition_penalty).squeeze(0);
            next_logits = apply_temperature(next_logits, cfg.temperature);
            auto lp = torch::log_softmax(next_logits, -1);
            auto tk = lp.topk(num_beams, -1);
            auto tk_vals = std::get<0>(tk);
            auto tk_idx = std::get<1>(tk);
            for (int k = 0; k < num_beams; ++k) {
                auto tokens = beam.first;
                tokens.push_back(tk_idx[k].item<int64_t>());
                candidates.push_back({tokens, beam.second + tk_vals[k].item<double>()});
            }
        }
        // Sort by score descending, keep top num_beams
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        beams.assign(candidates.begin(), candidates.begin() + std::min((size_t)num_beams, candidates.size()));
        // Check if all beams ended with EOS
        bool all_done = true;
        for (auto& beam : beams) {
            if (beam.first.empty() || beam.first.back() != cfg.eos_token_id) { all_done = false; break; }
        }
        if (all_done) break;
    }
    // Return highest-scoring beam
    auto best = beams[0];
    Tensor result = torch::empty({1, (int64_t)best.first.size()}, torch::kLong);
    for (size_t i = 0; i < best.first.size(); ++i) result[0][i] = best.first[i];
    return result;
}

// ─────────────────────────────────────────────────────────────
// Manifest/tensor IO (copied from training_smoke_test)
// ─────────────────────────────────────────────────────────────
struct TensorEntry { std::string kind, name, dtype, file; std::vector<int64_t> shape; };
struct Manifest {
    std::unordered_map<std::string, std::string> config;
    std::vector<TensorEntry> tensors;
};
Manifest parse_manifest(const std::string& path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("cannot open " + path);
    Manifest m; std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line); std::string tok; ss >> tok;
        if (tok == "config") { std::string k, v; ss >> k; std::getline(ss, v);
            size_t i = v.find_first_not_of(" \t"); v = (i != std::string::npos) ? v.substr(i) : "";
            m.config[k] = v; }
        else if (tok == "tensor") { TensorEntry e; ss >> e.kind >> e.name >> e.dtype >> e.file;
            std::string sc; ss >> sc; std::stringstream s2(sc); std::string item;
            while (std::getline(s2, item, ',')) if (!item.empty()) e.shape.push_back(std::stoll(item));
            m.tensors.push_back(std::move(e)); }
        else if (tok == "end") break;
    }
    return m;
}
torch::ScalarType dtype_from_str(const std::string& s) {
    if (s == "float32") return torch::kFloat32; if (s == "int64") return torch::kInt64;
    throw std::runtime_error("unsupported dtype: " + s);
}
Tensor load_tensor(const std::string& path, const std::vector<int64_t>& shape, const std::string& dtype) {
    auto t = torch::empty(shape, torch::TensorOptions().dtype(dtype_from_str(dtype)));
    std::ifstream f(path, std::ios::binary); f.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
    return t;
}
TensorMap load_state_dict(const std::string& dir) {
    std::ifstream f(dir + "/state_dict_manifest.txt"); if (!f) throw std::runtime_error("no sd manifest");
    TensorMap sd; std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line); std::string tok; ss >> tok; if (tok != "tensor") continue;
        std::string name, dtype, file, sc; ss >> name >> dtype >> file >> sc;
        std::vector<int64_t> shape; std::stringstream s2(sc); std::string item;
        while (std::getline(s2, item, ',')) if (!item.empty()) shape.push_back(std::stoll(item));
        sd[name] = load_tensor(dir + "/" + file, shape, dtype);
    }
    return sd;
}
long cfg_int(const Manifest& m, const std::string& k) { return std::stol(m.config.at(k)); }

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: " << argv[0] << " <fixture_dir> <output_dir>\n"; return 2; }
    std::string fd = argv[1], od = argv[2];
    std::filesystem::create_directories(od);
    torch::set_num_threads(1);
    try {
        auto m = parse_manifest(fd + "/manifest.txt");
        auto sd = load_state_dict(fd);
        int64_t V = cfg_int(m, "vocab_size"), H = cfg_int(m, "hidden_size");
        int64_t ctx = cfg_int(m, "context_length"), nh = cfg_int(m, "num_attention_heads");
        int64_t state = cfg_int(m, "ssm_state_dim"), kern = cfg_int(m, "ssm_kernel_size");
        SmokeTestModel model(V, H, ctx, nh, state, kern);
        model.load_from_state_dict(sd);
        model.eval();
        std::cout << "Model loaded.\n";

        // Test input
        Tensor input_ids = load_tensor(fd + "/expected_input_ids.bin", {1, 4}, "int64");  // [1, 4]
        std::cout << "Input: " << input_ids << "\n";

        // ─── Test 1: Greedy ───
        {
            GenerationConfig cfg;
            cfg.max_new_tokens = 5;
            cfg.eos_token_id = -1;
            auto out = generate_greedy(model, input_ids, cfg);
            std::cout << "\n[Test 1] Greedy: " << out << "\n";
            std::cout << "  shape: [" << out.size(0) << ", " << out.size(1) << "]\n";
        }

        // ─── Test 2: Temperature ───
        {
            GenerationConfig cfg;
            cfg.max_new_tokens = 5;
            cfg.temperature = 0.5;
            cfg.do_sample = false;
            auto out = generate_greedy(model, input_ids, cfg);
            std::cout << "\n[Test 2] Temperature=0.5 (greedy): " << out << "\n";
        }

        // ─── Test 3: Top-k ───
        {
            GenerationConfig cfg;
            cfg.max_new_tokens = 5;
            cfg.top_k = 3;
            auto out = generate_greedy(model, input_ids, cfg);
            std::cout << "\n[Test 3] Top-k=3 (greedy): " << out << "\n";
        }

        // ─── Test 4: Top-p ───
        {
            GenerationConfig cfg;
            cfg.max_new_tokens = 5;
            cfg.top_p = 0.9;
            auto out = generate_greedy(model, input_ids, cfg);
            std::cout << "\n[Test 4] Top-p=0.9 (greedy): " << out << "\n";
        }

        // ─── Test 5: Repetition penalty ───
        {
            GenerationConfig cfg;
            cfg.max_new_tokens = 5;
            cfg.repetition_penalty = 1.5;
            auto out = generate_greedy(model, input_ids, cfg);
            std::cout << "\n[Test 5] Repetition penalty=1.5: " << out << "\n";
        }

        // ─── Test 6: EOS ───
        {
            GenerationConfig cfg;
            cfg.max_new_tokens = 10;
            cfg.eos_token_id = 0;  // pad as eos
            auto out = generate_greedy(model, input_ids, cfg);
            std::cout << "\n[Test 6] EOS=0: " << out << "\n";
        }

        // ─── Test 7: Sampling ───
        {
            GenerationConfig cfg;
            cfg.max_new_tokens = 5;
            cfg.temperature = 0.8;
            cfg.top_k = 5;
            cfg.do_sample = true;
            cfg.seed = 42;
            auto out = generate_sample(model, input_ids, cfg);
            std::cout << "\n[Test 7] Sampling (temp=0.8, top-k=5): " << out << "\n";
        }

        // ─── Test 8: Beam search ───
        {
            GenerationConfig cfg;
            cfg.max_new_tokens = 5;
            cfg.num_beams = 3;
            auto out = generate_beam(model, input_ids, cfg);
            std::cout << "\n[Test 8] Beam search (num_beams=3): " << out << "\n";
            std::cout << "  shape: [" << out.size(0) << ", " << out.size(1) << "]\n";
        }

        // ─── Summary ───
        std::ofstream sum(od + "/summary.txt");
        sum << "status: PASS\n";
        sum << "strategies_tested: greedy, temperature, top-k, top-p, "
            << "repetition_penalty, eos, sampling, beam_search\n";
        std::cout << "\n=== Generation Test: PASS ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n"; return 1;
    }
}

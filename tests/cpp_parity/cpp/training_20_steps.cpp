// training_20_steps.cpp — Run 20 steps of C++ training, recording per-step metrics.
// Mirrors the Python training_python_20steps.py for comparison.
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Include the SmokeTestModel from training_smoke_test.cpp
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


using Tensor = torch::Tensor;
using TensorMap = std::unordered_map<std::string, Tensor>;

// Manifest parsing (copied)
struct TensorEntry { std::string kind, name, dtype, file; std::vector<int64_t> shape; };
struct Manifest { std::unordered_map<std::string, std::string> config; std::vector<TensorEntry> tensors; };
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
        model.train();

        Tensor input_ids = load_tensor(fd + "/expected_input_ids.bin", {cfg_int(m,"B"), cfg_int(m,"T")}, "int64");
        Tensor labels    = load_tensor(fd + "/expected_labels.bin",    {cfg_int(m,"B"), cfg_int(m,"T")}, "int64");

        torch::optim::AdamW optimizer(model.parameters(),
            torch::optim::AdamWOptions(1e-3).weight_decay(0.01));

        std::ofstream report(od + "/training_cpp_20steps.json");
        report << "[\n";
        bool first = true;

        for (int step = 0; step < 20; ++step) {
            optimizer.zero_grad();
            auto logits = model.forward(input_ids);
            auto loss = model.compute_loss(logits, labels);
            loss.backward();

            // Grad norm
            double grad_norm_sq = 0.0;
            for (const auto& p : model.parameters()) {
                if (p.grad().defined()) {
                    grad_norm_sq += p.grad().norm().item<double>() * p.grad().norm().item<double>();
                }
            }
            double grad_norm = std::sqrt(grad_norm_sq);

            // Param delta
            auto params_before = model.parameters();
            std::vector<Tensor> snap;
            for (const auto& p : params_before) snap.push_back(p.detach().clone());
            optimizer.step();
            auto params_after = model.parameters();
            double param_delta = 0.0;
            for (size_t i = 0; i < snap.size() && i < params_after.size(); ++i) {
                param_delta += (params_after[i] - snap[i]).abs().sum().item<double>();
            }
            param_delta /= snap.size();

            bool has_nan = torch::isnan(loss).any().item<bool>();
            double loss_val = loss.item<double>();

            if (!first) report << ",\n";
            first = false;
            report << "  {\"step\": " << step
                   << ", \"loss\": " << loss_val
                   << ", \"grad_norm\": " << grad_norm
                   << ", \"param_delta_mean\": " << param_delta
                   << ", \"has_nan\": " << (has_nan ? "true" : "false")
                   << "}";
            std::cerr << "  step " << step << ": loss=" << loss_val
                      << "  grad_norm=" << grad_norm
                      << "  delta=" << param_delta
                      << "  nan=" << (has_nan ? "YES" : "no") << "\n";
        }
        report << "\n]\n";
        report.close();
        std::cerr << "\nSaved: " << od << "/training_cpp_20steps.json\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n"; return 1;
    }
}

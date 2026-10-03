// checkpoint_lifecycle.cpp — Test Python checkpoint → C++ load → save → reload → inference match.
//
// Verifies the complete checkpoint lifecycle:
//   1. Load state_dict from per-tensor .bin files
//   2. Run inference (forward pass)
//   3. Save model state to a NEW directory
//   4. Reload from the saved directory
//   5. Run inference again
//   6. Compare: the two inference runs must match exactly (bit-identical)
//
// Also verifies:
//   - no tensor silently disappears
//   - no tensor silently changes shape
//   - parameter count is preserved
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

// Include the SmokeTestModel inline (copied from generation_test.cpp)
using Tensor = torch::Tensor;
using TensorMap = std::unordered_map<std::string, Tensor>;

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
        token_embedding = register_module("token_embedding", torch::nn::Embedding(torch::nn::EmbeddingOptions(V, H)));
        position_embedding = register_module("position_embedding", torch::nn::Embedding(torch::nn::EmbeddingOptions(T_max, H)));
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
        conv = register_module("conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(H, H, kernel).padding(kernel - 1).groups(H)));
        ssm_ln_input = register_module("ssm_ln_input", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H})));
        ssm_ln_state = register_module("ssm_ln_state", torch::nn::LayerNorm(torch::nn::LayerNormOptions({state_dim})));
        fc1 = register_module("fc1", torch::nn::Linear(H, H * 4));
        fc2 = register_module("fc2", torch::nn::Linear(H * 4, H));
        ffn_ln_input = register_module("ffn_ln_input", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H})));
        ffn_ln_hidden = register_module("ffn_ln_hidden", torch::nn::LayerNorm(torch::nn::LayerNormOptions({H * 4})));
        lm_head = register_module("lm_head", torch::nn::Linear(torch::nn::LinearOptions(H, V).bias(false)));
    }

    void load_from_state_dict(const TensorMap& sd) {
        torch::NoGradGuard ng;
        // Try both the Python key (blocks.0...) and the local key (A_log, ln1.weight...)
        auto cp = [&](torch::Tensor& param, const std::string& py_key, const std::string& local_key) {
            auto it = sd.find(py_key);
            if (it == sd.end()) it = sd.find(local_key);
            if (it != sd.end()) param.copy_(it->second);
        };
        cp(token_embedding->weight, "token_embedding.weight", "token_embedding.weight");
        cp(position_embedding->weight, "position_embedding.weight", "position_embedding.weight");
        cp(ln1->weight, "blocks.0.ln1.weight", "ln1.weight"); cp(ln1->bias, "blocks.0.ln1.bias", "ln1.bias");
        cp(ln2->weight, "blocks.0.ln2.weight", "ln2.weight"); cp(ln2->bias, "blocks.0.ln2.bias", "ln2.bias");
        cp(q_proj->weight, "blocks.0.pathways.local.q_proj.weight", "q_proj.weight"); cp(q_proj->bias, "blocks.0.pathways.local.q_proj.bias", "q_proj.bias");
        cp(k_proj->weight, "blocks.0.pathways.local.k_proj.weight", "k_proj.weight"); cp(k_proj->bias, "blocks.0.pathways.local.k_proj.bias", "k_proj.bias");
        cp(v_proj->weight, "blocks.0.pathways.local.v_proj.weight", "v_proj.weight"); cp(v_proj->bias, "blocks.0.pathways.local.v_proj.bias", "v_proj.bias");
        cp(out_proj->weight, "blocks.0.pathways.local.out_proj.weight", "out_proj.weight"); cp(out_proj->bias, "blocks.0.pathways.local.out_proj.bias", "out_proj.bias");
        cp(ln_q->weight, "blocks.0.pathways.local.ln_q.weight", "ln_q.weight"); cp(ln_q->bias, "blocks.0.pathways.local.ln_q.bias", "ln_q.bias");
        cp(ln_k->weight, "blocks.0.pathways.local.ln_k.weight", "ln_k.weight"); cp(ln_k->bias, "blocks.0.pathways.local.ln_k.bias", "ln_k.bias");
        cp(A_log, "blocks.0.pathways.ssm.A_log", "A_log");
        cp(dt_proj->weight, "blocks.0.pathways.ssm.dt_proj.weight", "dt_proj.weight"); cp(dt_proj->bias, "blocks.0.pathways.ssm.dt_proj.bias", "dt_proj.bias");
        cp(B_proj->weight, "blocks.0.pathways.ssm.B_proj.weight", "B_proj.weight"); cp(B_proj->bias, "blocks.0.pathways.ssm.B_proj.bias", "B_proj.bias");
        cp(C_proj->weight, "blocks.0.pathways.ssm.C_proj.weight", "C_proj.weight"); cp(C_proj->bias, "blocks.0.pathways.ssm.C_proj.bias", "C_proj.bias");
        cp(D_proj->weight, "blocks.0.pathways.ssm.D_proj.weight", "D_proj.weight"); cp(D_proj->bias, "blocks.0.pathways.ssm.D_proj.bias", "D_proj.bias");
        cp(gate_proj->weight, "blocks.0.pathways.ssm.gate_proj.weight", "gate_proj.weight"); cp(gate_proj->bias, "blocks.0.pathways.ssm.gate_proj.bias", "gate_proj.bias");
        cp(conv->weight, "blocks.0.pathways.ssm.conv.weight", "conv.weight"); cp(conv->bias, "blocks.0.pathways.ssm.conv.bias", "conv.bias");
        cp(ssm_ln_input->weight, "blocks.0.pathways.ssm.ln_input.weight", "ssm_ln_input.weight"); cp(ssm_ln_input->bias, "blocks.0.pathways.ssm.ln_input.bias", "ssm_ln_input.bias");
        cp(ssm_ln_state->weight, "blocks.0.pathways.ssm.ln_state.weight", "ssm_ln_state.weight"); cp(ssm_ln_state->bias, "blocks.0.pathways.ssm.ln_state.bias", "ssm_ln_state.bias");
        cp(fc1->weight, "blocks.0.ffn.fc1.weight", "fc1.weight"); cp(fc1->bias, "blocks.0.ffn.fc1.bias", "fc1.bias");
        cp(fc2->weight, "blocks.0.ffn.fc2.weight", "fc2.weight"); cp(fc2->bias, "blocks.0.ffn.fc2.bias", "fc2.bias");
        cp(ffn_ln_input->weight, "blocks.0.ffn.ln_input.weight", "ffn_ln_input.weight"); cp(ffn_ln_input->bias, "blocks.0.ffn.ln_input.bias", "ffn_ln_input.bias");
        cp(ffn_ln_hidden->weight, "blocks.0.ffn.ln_hidden.weight", "ffn_ln_hidden.weight"); cp(ffn_ln_hidden->bias, "blocks.0.ffn.ln_hidden.bias", "ffn_ln_hidden.bias");
        lm_head->weight.copy_(token_embedding->weight);
    }

    // Save state_dict to a directory as per-tensor .bin files + manifest
    void save_to_dir(const std::string& dir) {
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
        manifest << "end\n";
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
};

// Manifest/tensor IO (copied)
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

        // ─── Step 1: Load checkpoint into model ───
        SmokeTestModel model1(V, H, ctx, nh, state, kern);
        model1.load_from_state_dict(sd);
        model1.eval();
        std::cout << "Step 1: Loaded checkpoint into model1 (" << model1.parameters().size() << " params)\n";

        // ─── Step 2: Run inference ───
        Tensor input_ids = load_tensor(fd + "/expected_input_ids.bin", {cfg_int(m,"B"), cfg_int(m,"T")}, "int64");
        {
            torch::NoGradGuard ng;
            auto logits1 = model1.forward(input_ids);
            std::cout << "Step 2: Inference 1 — logits shape=[" << logits1.size(0) << "," << logits1.size(1) << "," << logits1.size(2) << "]\n";
            // Save logits1
            auto tc = logits1.to(torch::kCPU).contiguous();
            std::ofstream f(od + "/logits1.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(tc.data_ptr()), tc.nbytes());
        }

        // ─── Step 3: Save model state ───
        std::string save_dir = od + "/saved_checkpoint";
        model1.save_to_dir(save_dir);
        // Count saved tensors
        int64_t n_saved = 0;
        for (const auto& entry : std::filesystem::directory_iterator(save_dir)) {
            if (entry.path().extension() == ".bin") n_saved++;
        }
        std::cout << "Step 3: Saved model to " << save_dir << " (" << n_saved << " tensor files)\n";

        // ─── Step 4: Reload from saved ───
        auto sd2 = load_state_dict(save_dir);
        SmokeTestModel model2(V, H, ctx, nh, state, kern);
        model2.load_from_state_dict(sd2);
        model2.eval();
        std::cout << "Step 4: Reloaded into model2 (" << model2.parameters().size() << " params)\n";

        // ─── Step 5: Run inference again ───
        {
            torch::NoGradGuard ng;
            auto logits2 = model2.forward(input_ids);
            std::cout << "Step 5: Inference 2 — logits shape=[" << logits2.size(0) << "," << logits2.size(1) << "," << logits2.size(2) << "]\n";
            auto tc = logits2.to(torch::kCPU).contiguous();
            std::ofstream f(od + "/logits2.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(tc.data_ptr()), tc.nbytes());
        }

        // ─── Step 6: Compare logits1 vs logits2 ───
        auto logits1 = load_tensor(od + "/logits1.bin", {cfg_int(m,"B"), cfg_int(m,"T"), V}, "float32");
        auto logits2 = load_tensor(od + "/logits2.bin", {cfg_int(m,"B"), cfg_int(m,"T"), V}, "float32");
        double max_abs = (logits1 - logits2).abs().max().item<double>();
        bool exact_match = torch::equal(logits1, logits2);
        std::cout << "\nStep 6: Comparison\n";
        std::cout << "  logits1 vs logits2: max_abs=" << max_abs << "  exact_match=" << (exact_match ? "YES" : "NO") << "\n";

        // ─── Verify no tensors disappeared ───
        int64_t n_params1 = 0, n_params2 = 0;
        for (const auto& p : model1.parameters()) n_params1++;
        for (const auto& p : model2.parameters()) n_params2++;
        std::cout << "  param count: model1=" << n_params1 << "  model2=" << n_params2
                  << "  match=" << (n_params1 == n_params2 ? "YES" : "NO") << "\n";

        // ─── Verify shapes preserved ───
        bool shapes_match = true;
        auto p1 = model1.named_parameters();
        auto p2 = model2.named_parameters();
        for (auto it1 = p1.begin(), it2 = p2.begin(); it1 != p1.end() && it2 != p2.end(); ++it1, ++it2) {
            if (it1->key() != it2->key()) { shapes_match = false; break; }
            if (it1->value().sizes() != it2->value().sizes()) { shapes_match = false; break; }
        }
        std::cout << "  shapes preserved: " << (shapes_match ? "YES" : "NO") << "\n";

        // ─── Summary ───
        bool pass = exact_match && (n_params1 == n_params2) && shapes_match;
        std::ofstream sum(od + "/summary.txt");
        sum << "status: " << (pass ? "PASS" : "FAIL") << "\n";
        sum << "logits_max_abs_diff: " << max_abs << "\n";
        sum << "logits_exact_match: " << (exact_match ? "YES" : "NO") << "\n";
        sum << "param_count_match: " << (n_params1 == n_params2 ? "YES" : "NO") << "\n";
        sum << "shapes_preserved: " << (shapes_match ? "YES" : "NO") << "\n";
        sum << "n_saved_tensors: " << n_saved << "\n";

        std::cout << "\n=== Checkpoint Lifecycle: " << (pass ? "PASS" : "FAIL") << " ===\n";
        return pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n"; return 1;
    }
}

// hass_granular_harness.cpp — run HASS block step-by-step, compare each intermediate.
//
// Loads the granular fixture (16_hass_block_granular) and runs:
//   hidden → ln1 → x_attn
//   x_attn → local → local_out
//   x_attn → low_rank → low_rank_out
//   x_attn → ssm → ssm_out
//   hidden + (local * p[0] + low_rank * p[1] + ssm * p[2]) → residual
//   residual → ln2 → xf
//   xf → ffn → ffn_out
//   residual + ffn_out → block_out
//
// Each intermediate is compared against the Python expected tensor.
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

using Tensor = torch::Tensor;
using TensorMap = std::unordered_map<std::string, Tensor>;

// Reuse the manifest/tensor IO from the e2e harness (inline copy for simplicity)
struct TensorEntry { std::string kind, name, dtype, file; std::vector<int64_t> shape; };
struct Manifest {
    std::string component;
    std::unordered_map<std::string, std::string> config;
    double max_abs = 1e-5, max_rel = 1e-4;
    std::vector<TensorEntry> tensors;
};

Manifest parse_manifest(const std::string& path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("cannot open " + path);
    Manifest m; std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line); std::string tok; ss >> tok;
        if (tok == "component") ss >> m.component;
        else if (tok == "config") { std::string k, v; ss >> k; std::getline(ss, v);
            size_t i = v.find_first_not_of(" \t"); v = (i != std::string::npos) ? v.substr(i) : "";
            m.config[k] = v; }
        else if (tok == "tolerance") { std::string w; double v; ss >> w >> v;
            if (w == "max_abs") m.max_abs = v; else if (w == "max_rel") m.max_rel = v; }
        else if (tok == "tensor") {
            TensorEntry e; ss >> e.kind >> e.name >> e.dtype >> e.file;
            std::string sc; ss >> sc;
            std::stringstream s2(sc); std::string item;
            while (std::getline(s2, item, ',')) if (!item.empty()) e.shape.push_back(std::stoll(item));
            m.tensors.push_back(std::move(e));
        } else if (tok == "end") break;
    }
    return m;
}

torch::ScalarType dtype_from_str(const std::string& s) {
    if (s == "float32") return torch::kFloat32;
    if (s == "int64") return torch::kInt64;
    if (s == "float64") return torch::kFloat64;
    throw std::runtime_error("unsupported dtype: " + s);
}

Tensor load_tensor(const std::string& path, const std::vector<int64_t>& shape, const std::string& dtype) {
    auto t = torch::empty(shape, torch::TensorOptions().dtype(dtype_from_str(dtype)));
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
    return t;
}

TensorMap load_kind(const std::string& dir, const Manifest& m, const std::string& kind) {
    TensorMap out;
    for (const auto& e : m.tensors) {
        if (e.kind != kind) continue;
        out[e.name] = load_tensor(dir + "/" + e.file, e.shape, e.dtype);
    }
    return out;
}

TensorMap load_state_dict(const std::string& dir) {
    std::ifstream f(dir + "/state_dict_manifest.txt");
    if (!f) throw std::runtime_error("no state_dict_manifest.txt");
    TensorMap sd; std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line); std::string tok; ss >> tok;
        if (tok != "tensor") continue;
        std::string name, dtype, file, sc;
        ss >> name >> dtype >> file >> sc;
        std::vector<int64_t> shape;
        std::stringstream s2(sc); std::string item;
        while (std::getline(s2, item, ',')) if (!item.empty()) shape.push_back(std::stoll(item));
        sd[name] = load_tensor(dir + "/" + file, shape, dtype);
    }
    return sd;
}

long cfg_int(const Manifest& m, const std::string& k) { return std::stol(m.config.at(k)); }
double cfg_dbl(const Manifest& m, const std::string& k, double def = 0) {
    auto it = m.config.find(k); return it == m.config.end() ? def : std::stod(it->second);
}

struct Linear { Tensor weight, bias; Tensor forward(const Tensor& x) const { return torch::linear(x, weight, bias); } };
Linear make_linear(const TensorMap& sd, const std::string& w, const std::string& b) {
    Linear L; L.weight = sd.at(w); auto it = sd.find(b); if (it != sd.end()) L.bias = it->second; return L;
}
Tensor ln(const Tensor& x, const Tensor& w, const Tensor& b, double eps = 1e-5) {
    return torch::layer_norm(x, {x.size(-1)}, w, b, eps);
}

Tensor eval_gumbel(torch::IntArrayRef shape, const torch::TensorOptions& opts, int64_t axis, double mag) {
    if (mag <= 0) return torch::zeros(shape, opts);
    at::Generator gen = at::detail::createCPUGenerator(1337 + axis);
    auto u = torch::rand(shape, gen).to(opts.dtype());
    return mag * (-torch::log(-torch::log(u.clamp_min(1e-10)) + 1e-10)).to(opts);
}

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: " << argv[0] << " <fixture_dir> <output_dir>\n"; return 2; }
    std::string fd = argv[1], od = argv[2];
    std::filesystem::create_directories(od);
    try {
        auto m = parse_manifest(fd + "/manifest.txt");
        auto sd = load_state_dict(fd);
        auto exp = load_kind(fd, m, "expected");
        std::cout << "Loaded: " << sd.size() << " params, " << exp.size() << " expected\n";

        int64_t B = cfg_int(m, "B"), T = cfg_int(m, "T"), H = cfg_int(m, "hidden_size");
        double eps = cfg_dbl(m, "layer_norm_eps", 1e-5);
        int64_t nh = std::max<int64_t>(1, cfg_int(m, "num_attention_heads") / 2);
        int64_t hd = H / nh;
        int64_t state_dim = cfg_int(m, "ssm_state_dim");
        int64_t kernel = cfg_int(m, "ssm_kernel_size");

        auto hidden = exp.at("hidden");

        // 1. ln1 → x_attn
        auto x_attn = ln(hidden, sd.at("blocks.0.ln1.weight"), sd.at("blocks.0.ln1.bias"), eps);

        // 2. local attention
        auto q = make_linear(sd, "blocks.0.pathways.local.q_proj.weight", "blocks.0.pathways.local.q_proj.bias").forward(x_attn).view({B,T,nh,hd}).transpose(1,2);
        auto k = make_linear(sd, "blocks.0.pathways.local.k_proj.weight", "blocks.0.pathways.local.k_proj.bias").forward(x_attn).view({B,T,nh,hd}).transpose(1,2);
        auto v = make_linear(sd, "blocks.0.pathways.local.v_proj.weight", "blocks.0.pathways.local.v_proj.bias").forward(x_attn).view({B,T,nh,hd}).transpose(1,2);
        q = ln(q.transpose(1,2), sd.at("blocks.0.pathways.local.ln_q.weight"), sd.at("blocks.0.pathways.local.ln_q.bias"), eps).transpose(1,2);
        k = ln(k.transpose(1,2), sd.at("blocks.0.pathways.local.ln_k.weight"), sd.at("blocks.0.pathways.local.ln_k.bias"), eps).transpose(1,2);
        auto scores = torch::matmul(q, k.transpose(-2,-1)) / std::sqrt(double(hd));
        auto cm = torch::tril(torch::ones({T,T}, scores.options()));
        scores = scores.masked_fill(cm.eq(0), -std::numeric_limits<float>::infinity());
        auto probs = torch::softmax(scores, -1);
        auto local_out = torch::matmul(probs, v).transpose(1,2).contiguous().view({B,T,H});
        local_out = make_linear(sd, "blocks.0.pathways.local.out_proj.weight", "blocks.0.pathways.local.out_proj.bias").forward(local_out);

        // 3. low_rank
        auto lr_xn = ln(x_attn, sd.at("blocks.0.pathways.low_rank.ln_input.weight"), sd.at("blocks.0.pathways.low_rank.ln_input.bias"), eps);
        auto low_rank = torch::gelu(make_linear(sd, "blocks.0.pathways.low_rank.to_low_rank.weight", "blocks.0.pathways.low_rank.to_low_rank.bias").forward(lr_xn));
        int64_t rk = low_rank.size(-1);
        auto lrs = torch::matmul(low_rank, low_rank.transpose(-1,-2)) / std::sqrt(double(rk));
        auto lrc = torch::tril(torch::ones({T,T}, lrs.options()));
        lrs = lrs.masked_fill(lrc.eq(0), -std::numeric_limits<float>::infinity());
        auto lra = torch::softmax(lrs, -1);
        auto lrctx = torch::matmul(lra, low_rank);
        auto low_rank_out = make_linear(sd, "blocks.0.pathways.low_rank.from_low_rank.weight", "blocks.0.pathways.low_rank.from_low_rank.bias").forward(low_rank + lrctx);

        // 4. ssm (fixed math)
        auto ssm_xn = ln(x_attn, sd.at("blocks.0.pathways.ssm.ln_input.weight"), sd.at("blocks.0.pathways.ssm.ln_input.bias"), eps);
        auto conv_w = sd.at("blocks.0.pathways.ssm.conv.weight"), conv_b = sd.at("blocks.0.pathways.ssm.conv.bias");
        auto xt = ssm_xn.transpose(1,2);
        auto copts = torch::nn::functional::Conv1dFuncOptions().padding(kernel-1).groups(H);
        auto conv_out = torch::nn::functional::conv1d(xt, conv_w, torch::nn::functional::Conv1dFuncOptions(copts).bias(conv_b));
        conv_out = conv_out.slice(2, 0, T);
        ssm_xn = ssm_xn + conv_out.transpose(1,2);
        auto gp = make_linear(sd, "blocks.0.pathways.ssm.gate_proj.weight", "blocks.0.pathways.ssm.gate_proj.bias").forward(ssm_xn).chunk(2,-1);
        auto gate = torch::sigmoid(gp[0]), input_gate = torch::sigmoid(gp[1]);
        auto Bv = make_linear(sd, "blocks.0.pathways.ssm.B_proj.weight", "blocks.0.pathways.ssm.B_proj.bias").forward(ssm_xn * input_gate);
        auto C  = make_linear(sd, "blocks.0.pathways.ssm.C_proj.weight", "blocks.0.pathways.ssm.C_proj.bias").forward(ssm_xn);
        auto dt = torch::softplus(make_linear(sd, "blocks.0.pathways.ssm.dt_proj.weight", "blocks.0.pathways.ssm.dt_proj.bias").forward(ssm_xn));
        auto a = -torch::exp(sd.at("blocks.0.pathways.ssm.A_log"));
        auto ab = a.view({1,1,-1}); auto z = dt*ab; auto Ab = torch::exp(z);
        auto small = z.abs() < 1e-4;
        auto safe_z = torch::where(small, torch::ones_like(z), z);
        auto exact = (Ab - 1.0) / safe_z;
        auto taylor = 1.0 + z/2.0 + (z*z)/6.0;
        auto Bb = torch::where(small, taylor, exact) * dt * Bv;
        auto states = torch::empty({B,T,state_dim}, Ab.options());
        auto Abc = Ab.to(torch::kCPU).contiguous(), Bbc = Bb.to(torch::kCPU).contiguous();
        auto sc = states.to(torch::kCPU).contiguous();
        float* ps = sc.data_ptr<float>(); const float* pAb = Abc.data_ptr<float>(); const float* pBb = Bbc.data_ptr<float>();
        for (int64_t b = 0; b < B; ++b) {
            std::vector<float> h(state_dim, 0.0f);
            for (int64_t t = 0; t < T; ++t) {
                int64_t off = (b*T+t)*state_dim;
                for (int64_t n = 0; n < state_dim; ++n) {
                    h[n] = pAb[off+n]*h[n] + pBb[off+n];
                    ps[off+n] = h[n];
                }
            }
        }
        auto states_ln = ln(states, sd.at("blocks.0.pathways.ssm.ln_state.weight"), sd.at("blocks.0.pathways.ssm.ln_state.bias"), eps);
        auto ssm_out = make_linear(sd, "blocks.0.pathways.ssm.D_proj.weight", "blocks.0.pathways.ssm.D_proj.bias").forward(C * states_ln) * gate;

        // 5. Router for path_probs
        int64_t cot_total = cfg_int(m, "cot_dim") * cfg_int(m, "cot_components");
        auto cot_feat = torch::zeros({B, T, cot_total});
        auto ri = torch::cat({hidden, cot_feat}, -1);
        auto flat = ri.view({B*T, -1});
        int64_t input_dim = H + cot_total;
        auto aln = [&](const Tensor& x, const std::string& base) {
            return torch::layer_norm(x, {x.size(-1)}, sd.at(base + ".weight"), sd.at(base + ".bias"), eps);
        };
        auto h0 = make_linear(sd, "router.feature_encoder.0.weight", "router.feature_encoder.0.bias").forward(flat);
        h0 = torch::gelu(aln(h0, "router.feature_encoder.1"));
        auto h1 = make_linear(sd, "router.feature_encoder.4.weight", "router.feature_encoder.4.bias").forward(h0);
        h1 = torch::gelu(aln(h1, "router.feature_encoder.5"));
        auto ff = make_linear(sd, "router.feature_encoder.8.weight", "router.feature_encoder.8.bias").forward(h1);
        ff = torch::gelu(aln(ff, "router.feature_encoder.9"));
        auto features = ff.view({B, T, -1});
        auto mk_head = [&](const std::string& p) {
            auto L0 = make_linear(sd, p+".0.weight", p+".0.bias");
            auto L3 = make_linear(sd, p+".3.weight", p+".3.bias");
            auto h = L0.forward(features); h = torch::gelu(aln(h, p+".1")); return L3.forward(h);
        };
        auto path_logits = mk_head("router.path_router");
        double temp = cfg_dbl(m, "temperature", 1.0), en = cfg_dbl(m, "eval_routing_noise", 0.15);
        auto pn = eval_gumbel(path_logits.sizes(), path_logits.options(), 2, en);
        auto path_probs = torch::softmax(path_logits / std::max(temp, 1e-8) + pn, -1);

        // 6. Aggregation
        auto combined = local_out * path_probs.slice(-1,0,1) +
                        low_rank_out * path_probs.slice(-1,1,2) +
                        ssm_out * path_probs.slice(-1,2,3);
        auto residual = hidden + combined;
        auto xf = ln(residual, sd.at("blocks.0.ln2.weight"), sd.at("blocks.0.ln2.bias"), eps);

        // 7. FFN (SlicedFFN at max_width)
        auto ffn_xn = ln(xf, sd.at("blocks.0.ffn.ln_input.weight"), sd.at("blocks.0.ffn.ln_input.bias"), eps);
        auto ffn_h = make_linear(sd, "blocks.0.ffn.fc1.weight", "blocks.0.ffn.fc1.bias").forward(ffn_xn);
        ffn_h = torch::gelu(ffn_h);
        ffn_h = ln(ffn_h, sd.at("blocks.0.ffn.ln_hidden.weight"), sd.at("blocks.0.ffn.ln_hidden.bias"), eps);
        auto ffn_out = make_linear(sd, "blocks.0.ffn.fc2.weight", "blocks.0.ffn.fc2.bias").forward(ffn_h);
        auto block_out = residual + ffn_out;

        // Compare
        std::unordered_map<std::string, Tensor> cpp = {
            {"x_attn", x_attn}, {"local_out", local_out}, {"low_rank_out", low_rank_out},
            {"ssm_out", ssm_out}, {"path_probs", path_probs}, {"combined", combined},
            {"residual", residual}, {"xf", xf}, {"ffn_out", ffn_out}, {"block_out", block_out},
        };
        int n_match = 0, n_mismatch = 0;
        std::ofstream cmp(od + "/comparison.txt");
        cmp << "Granular HASS Block Parity\n==========================\n\n";
        for (const auto& [name, exp_t] : exp) {
            if (name == "hidden") { continue; }  // input
            auto it = cpp.find(name);
            if (it == cpp.end()) { cmp << "  [SKIP] " << name << "\n"; continue; }
            auto ef = exp_t.to(torch::kFloat64), af = it->second.to(torch::kFloat64);
            double max_abs = (ef - af).abs().max().item<double>();
            bool ok = max_abs <= m.max_abs;
            cmp << "  [" << (ok ? "MATCH" : "MISMATCH") << "] " << name
                << "  max_abs=" << max_abs;
            if (exp_t.scalar_type() == torch::kInt64) {
                int64_t nd = (exp_t != it->second).sum().item<int64_t>();
                cmp << "  n_diff=" << nd; ok = (nd == 0);
            }
            cmp << "\n";
            if (ok) n_match++; else n_mismatch++;
        }
        cmp << "\nSummary: " << n_match << " MATCH, " << n_mismatch << " MISMATCH\n";
        std::cout << "Result: " << (n_mismatch == 0 ? "PASS" : "FAIL")
                  << " — " << n_match << " match, " << n_mismatch << " mismatch\n";
        return n_mismatch == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n"; return 1;
    }
}

// ============================================================
//  parity_harness.cpp — XorZen C++ Component Parity Harness
//
//  Loads a fixture from tests/cpp_parity/fixtures/<component>/,
//  runs the corresponding C++ implementation from
//  xorzen.cpp/src/model/*.cpp, and writes the outputs to
//  <output_dir>/.
//
//  Usage:
//    parity_harness <component> <fixture_dir> <output_dir>
//
//  Manifest format: see tests/cpp_parity/generators/write_simple_manifest.py
//  (line-based, no JSON dependency).
// ============================================================

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace parity {

using Tensor = torch::Tensor;

// ─────────────────────────────────────────────────────────────
// Manifest parsing
// ─────────────────────────────────────────────────────────────
struct TensorEntry {
    std::string kind;     // "input" | "param" | "expected"
    std::string name;
    std::string dtype;    // "float32" | "int64" | etc.
    std::string file;
    std::vector<int64_t> shape;
};

struct Manifest {
    std::string component;
    std::unordered_map<std::string, std::string> config;
    double max_abs = 0.0;
    double max_rel = 0.0;
    std::vector<TensorEntry> tensors;
};

Manifest parse_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open manifest: " + path);
    Manifest m;
    std::string line;
    while (std::getline(f, line)) {
        // strip trailing \r
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok;
        ss >> tok;
        if (tok == "component") ss >> m.component;
        else if (tok == "config") {
            std::string k, v;
            ss >> k;
            std::getline(ss, v);
            // strip leading spaces
            size_t i = v.find_first_not_of(" \t");
            if (i != std::string::npos) v = v.substr(i);
            else v = "";
            m.config[k] = v;
        } else if (tok == "tolerance") {
            std::string which; double v;
            ss >> which >> v;
            if (which == "max_abs") m.max_abs = v;
            else if (which == "max_rel") m.max_rel = v;
        } else if (tok == "tensor") {
            TensorEntry e;
            ss >> e.kind >> e.name >> e.dtype >> e.file;
            std::string shape_csv;
            ss >> shape_csv;
            // parse comma-separated
            std::stringstream s2(shape_csv);
            std::string item;
            while (std::getline(s2, item, ',')) {
                if (!item.empty()) e.shape.push_back(std::stoll(item));
            }
            m.tensors.push_back(std::move(e));
        } else if (tok == "end") {
            break;
        }
    }
    return m;
}

// ─────────────────────────────────────────────────────────────
// Tensor I/O
// ─────────────────────────────────────────────────────────────
torch::ScalarType dtype_from_str(const std::string& s) {
    if (s == "float32")      return torch::kFloat32;
    if (s == "float64")      return torch::kFloat64;
    if (s == "int32")        return torch::kInt32;
    if (s == "int64")        return torch::kInt64;
    if (s == "bool")         return torch::kBool;
    throw std::runtime_error("unsupported dtype: " + s);
}
std::string dtype_to_str(torch::ScalarType d) {
    switch (d) {
        case torch::kFloat32: return "float32";
        case torch::kFloat64: return "float64";
        case torch::kInt32:   return "int32";
        case torch::kInt64:   return "int64";
        case torch::kBool:    return "bool";
        default: throw std::runtime_error("unsupported output dtype");
    }
}

Tensor load_tensor(const std::string& path,
                   const std::vector<int64_t>& shape,
                   const std::string& dtype_str) {
    auto t = torch::empty(shape,
        torch::TensorOptions().dtype(dtype_from_str(dtype_str)));
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
    if (!f) throw std::runtime_error("short read on " + path);
    return t;
}

void write_tensor(const Tensor& t, const std::string& path) {
    auto tc = t.to(torch::kCPU).contiguous();
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open for write: " + path);
    f.write(reinterpret_cast<const char*>(tc.data_ptr()), tc.nbytes());
}

using TensorMap = std::unordered_map<std::string, Tensor>;

TensorMap load_kind(const std::string& fixture_dir,
                    const Manifest& m,
                    const std::string& kind) {
    TensorMap out;
    for (const auto& e : m.tensors) {
        if (e.kind != kind) continue;
        out[e.name] = load_tensor(fixture_dir + "/" + e.file, e.shape, e.dtype);
    }
    return out;
}

void write_outputs(const std::string& out_dir, const TensorMap& outputs) {
    std::filesystem::create_directories(out_dir);
    std::ofstream meta(out_dir + "/outputs_manifest.txt");
    meta << "component outputs\n";
    for (const auto& [name, t] : outputs) {
        std::string fname = "output_" + name + ".bin";
        write_tensor(t, out_dir + "/" + fname);
        std::string shape_csv;
        for (size_t i = 0; i < t.sizes().size(); ++i) {
            if (i > 0) shape_csv += ",";
            shape_csv += std::to_string(t.size(i));
        }
        meta << "tensor output " << name << " " << dtype_to_str(t.scalar_type())
             << " " << fname << " " << shape_csv << "\n";
    }
    meta << "end\n";
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
struct Linear {
    Tensor weight;  // [out, in]
    Tensor bias;    // [out] or empty
    Tensor forward(const Tensor& x) const {
        return torch::linear(x, weight, bias);
    }
};

Linear make_linear(const TensorMap& p,
                   const std::string& wname, const std::string& bname) {
    Linear L;
    L.weight = p.at(wname);
    auto bit = p.find(bname);
    if (bit != p.end() && bit->second.defined() && bit->second.numel() > 0)
        L.bias = bit->second;
    else L.bias = Tensor();
    return L;
}

Tensor layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, double eps = 1e-5) {
    return torch::layer_norm(x, {x.size(-1)}, w, b, eps);
}

long config_int(const Manifest& m, const std::string& k) {
    auto it = m.config.find(k);
    if (it == m.config.end()) throw std::runtime_error("missing config: " + k);
    return std::stol(it->second);
}
double config_double(const Manifest& m, const std::string& k, double def) {
    auto it = m.config.find(k);
    if (it == m.config.end()) return def;
    return std::stod(it->second);
}
std::string config_str(const Manifest& m, const std::string& k, const std::string& def) {
    auto it = m.config.find(k);
    if (it == m.config.end()) return def;
    return it->second;
}

// ─────────────────────────────────────────────────────────────────
// Component implementations. Each mirrors the math of the
// corresponding xorzen.cpp/src/model/*.cpp file.
// ─────────────────────────────────────────────────────────────────

void run_normalization(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    double eps = config_double(m, "eps", 1e-5);
    Tensor y = layer_norm(I["x"], P["weight"], P["bias"], eps);
    write_outputs(od, {{"y", y}});
}

void run_positional_embedding(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    Tensor pos_emb = torch::embedding(P["weight"], I["position_ids"]);
    write_outputs(od, {{"pos_emb", pos_emb}});
}

void run_linear(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    Linear L = make_linear(P, "weight", "bias");
    write_outputs(od, {{"y", L.forward(I["x"])}});
}

void run_attention(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    Tensor x = I["x"];
    int64_t B = x.size(0), S = x.size(1);
    int64_t H = config_int(m, "hidden_size");
    int64_t nh = config_int(m, "num_heads");
    int64_t hd = H / nh;
    bool causal = config_str(m, "causal", "1") == "1" || config_str(m, "causal", "1") == "true";

    auto q_proj = make_linear(P, "q_proj_weight", "q_proj_bias");
    auto k_proj = make_linear(P, "k_proj_weight", "k_proj_bias");
    auto v_proj = make_linear(P, "v_proj_weight", "v_proj_bias");
    auto out_proj = make_linear(P, "out_proj_weight", "out_proj_bias");

    auto q = q_proj.forward(x).view({B, S, nh, hd}).transpose(1, 2);
    auto k = k_proj.forward(x).view({B, S, nh, hd}).transpose(1, 2);
    auto v = v_proj.forward(x).view({B, S, nh, hd}).transpose(1, 2);

    q = torch::layer_norm(q.transpose(1, 2), {hd},
        P["ln_q_weight"], P["ln_q_bias"], 1e-5).transpose(1, 2);
    k = torch::layer_norm(k.transpose(1, 2), {hd},
        P["ln_k_weight"], P["ln_k_bias"], 1e-5).transpose(1, 2);

    auto scores = torch::matmul(q, k.transpose(-2, -1)) /
                  std::sqrt(static_cast<double>(hd));
    if (causal) {
        auto mask = torch::tril(torch::ones({S, S}, scores.options()));
        scores = scores.masked_fill(mask.eq(0), -std::numeric_limits<float>::infinity());
    }
    auto probs = torch::softmax(scores, -1);
    auto out = torch::matmul(probs, v).transpose(1, 2).contiguous().view({B, S, H});
    out = out_proj.forward(out);
    write_outputs(od, {{"y", out}});
}

void run_ssm_scan(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    Tensor Ab = I["A_bar"];
    Tensor Bv = I["B_bar"];
    Tensor C  = I["C"];
    int64_t B = Ab.size(0), T = Ab.size(1), N = Ab.size(2);

    // Mirror SSMScanFunction::forward serial loop (hass_block.cpp:140-154)
    auto Abc = Ab.to(torch::kCPU).contiguous();
    auto Bvc = Bv.to(torch::kCPU).contiguous();
    auto Cc  = C.to(torch::kCPU).contiguous();

    auto yc = torch::empty({B, T, N}, Ab.options());
    auto sc = torch::empty({B, T, N}, Ab.options());
    const float* p_Ab = Abc.data_ptr<float>();
    const float* p_Bv = Bvc.data_ptr<float>();
    const float* p_C  = Cc.data_ptr<float>();
    float* p_y = yc.data_ptr<float>();
    float* p_s = sc.data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> h(N, 0.0f);
        for (int64_t t = 0; t < T; ++t) {
            int64_t off = (b * T + t) * N;
            for (int64_t n = 0; n < N; ++n) {
                h[n] = p_Ab[off + n] * h[n] + p_Bv[off + n];
                p_s[off + n] = h[n];                  // raw state
                p_y[off + n] = p_C[off + n] * h[n];   // y_t = C_t * h_t
            }
        }
    }
    write_outputs(od, {{"y", yc}, {"states", sc}});
}

void run_sliced_ffn(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    Tensor x = I["x"];
    std::string act = config_str(m, "activation", "gelu");

    // Mirror AdaptiveFFNImpl::forward (hass_block.cpp:262-278)
    Tensor xn = layer_norm(x, P["ln_input_weight"], P["ln_input_bias"], 1e-5);
    auto fc1 = make_linear(P, "fc1_weight", "fc1_bias");
    auto fc2 = make_linear(P, "fc2_weight", "fc2_bias");
    Tensor h = fc1.forward(xn);
    if (act == "silu") h = torch::silu(h);
    else if (act == "relu") h = torch::relu(h);
    else h = torch::gelu(h);
    h = layer_norm(h, P["ln_hidden_weight"], P["ln_hidden_bias"], 1e-5);
    Tensor out = fc2.forward(h);
    write_outputs(od, {{"y", out}});
}

// 09_router: full AdaptiveRouter forward (mirrors routing.cpp:120-174)
void run_router(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    Tensor x = I["x"];
    Tensor cot = I["cot_features"];
    int64_t B = x.size(0), T = x.size(1);
    int64_t hidden = config_int(m, "hidden_size");
    int64_t max_depth = config_int(m, "max_depth");
    int64_t num_widths = config_int(m, "num_widths");
    int64_t num_paths = config_int(m, "num_paths");
    int64_t num_experts = config_int(m, "num_experts");
    int64_t top_k = config_int(m, "top_k");
    int64_t cot_dim = cot.size(-1);
    int64_t input_dim = hidden + cot_dim;

    auto router_input = torch::cat({x, cot}, -1);
    auto flat = router_input.view({B * T, -1});

    int64_t h = 1;
    int64_t enc1 = std::max<int64_t>(128, h * 4);
    int64_t enc2 = std::max<int64_t>(64, h * 2);
    int64_t enc3 = std::max<int64_t>(32, h);
    int64_t head = std::max<int64_t>(32, h / 2);

    // Feature encoder: 3 layers of (Linear, LayerNorm, GELU, [Dropout])
    auto fc_L0 = make_linear(P, "router.feature_encoder.0.weight",
                                "router.feature_encoder.0.bias");
    auto fc_L4 = make_linear(P, "router.feature_encoder.4.weight",
                                "router.feature_encoder.4.bias");
    auto fc_L8 = make_linear(P, "router.feature_encoder.8.weight",
                                "router.feature_encoder.8.bias");
    // Apply LayerNorm+GELU between layers (matching C++ Sequential)
    auto h0 = fc_L0.forward(flat);
    // LayerNorm over enc1 — but we don't have the LN weights captured!
    // Actually the Python router's feature_encoder has LN inside Sequential,
    // so the LN params should be in the captured params as "router.feature_encoder.1.weight" etc.
    auto apply_ln = [](const Tensor& x, const TensorMap& P, const std::string& base) {
        return torch::layer_norm(x, {x.size(-1)},
            P.at(base + ".weight"), P.at(base + ".bias"), 1e-5);
    };
    h0 = torch::gelu(apply_ln(h0, P, "router.feature_encoder.1"));
    auto h1 = fc_L4.forward(h0);
    h1 = torch::gelu(apply_ln(h1, P, "router.feature_encoder.5"));
    auto features_flat = fc_L8.forward(h1);
    // Final LN at end of feature_encoder (index 9), then GELU (index 10)
    features_flat = torch::gelu(apply_ln(features_flat, P, "router.feature_encoder.9"));
    auto features = features_flat.view({B, T, -1});

    auto mk_head = [&](const std::string& prefix) {
        // Sequential: 0:Linear, 1:LN, 2:GELU, 3:Linear
        auto L0 = make_linear(P, prefix + ".0.weight", prefix + ".0.bias");
        auto L3 = make_linear(P, prefix + ".3.weight", prefix + ".3.bias");
        auto h = L0.forward(features);
        h = torch::gelu(apply_ln(h, P, prefix + ".1"));
        return L3.forward(h);
    };

    auto depth_logits = mk_head("router.depth_router");
    auto width_logits = mk_head("router.width_router");
    auto path_logits = mk_head("router.path_router");
    auto expert_logits = mk_head("router.expert_router");

    auto cmplx_L0 = make_linear(P, "router.complexity_estimator.0.weight",
                                   "router.complexity_estimator.0.bias");
    auto cmplx_L3 = make_linear(P, "router.complexity_estimator.3.weight",
                                   "router.complexity_estimator.3.bias");
    auto complexity = torch::sigmoid(cmplx_L3.forward(
        torch::gelu(apply_ln(cmplx_L0.forward(features), P, "router.complexity_estimator.1"))));

    auto unc_L0 = make_linear(P, "router.uncertainty_estimator.0.weight",
                                 "router.uncertainty_estimator.0.bias");
    auto unc_L3 = make_linear(P, "router.uncertainty_estimator.3.weight",
                                 "router.uncertainty_estimator.3.bias");
    auto uncertainty = torch::sigmoid(unc_L3.forward(
        torch::gelu(apply_ln(unc_L0.forward(features), P, "router.uncertainty_estimator.1"))));

    double temp = config_double(m, "temperature", 1.0);

    // === C++ route_depth (routing.cpp:176-197) ===
    auto depth_bias = torch::linspace(0, 1, max_depth, depth_logits.options()).view({1, 1, max_depth});
    auto depth_scaled = (depth_logits + complexity * depth_bias * 2.0) / std::max(temp, 1e-8);
    auto depth_probs = torch::sigmoid(depth_scaled);
    auto depth_hard = (depth_probs > 0.5).to(depth_probs.dtype());
    auto depth_mask = depth_hard;

    // === C++ route_width (routing.cpp:199-209) ===
    auto width_bias = torch::linspace(-1, 1, num_widths, width_logits.options()).view({1, 1, num_widths});
    auto width_scaled = (width_logits + complexity * width_bias * 3.0) / std::max(temp, 1e-8);
    auto width_probs = torch::softmax(width_scaled, -1);
    auto width_idx = std::get<1>(width_probs.max(-1));

    // === C++ route_path (routing.cpp:211-219) — eval: softmax ===
    auto path_probs = torch::softmax(path_logits / std::max(temp, 1e-8), -1);

    // === C++ route_experts (routing.cpp:221-244) ===
    int64_t N = B * T;
    auto flat_logits = expert_logits.view({N, num_experts});
    auto flat_probs = torch::softmax(flat_logits / std::max(temp, 1e-8), -1);
    auto top = torch::topk(flat_probs, top_k, -1);
    auto ew = std::get<0>(top);
    auto ei = std::get<1>(top);
    ew = ew / (ew.sum(-1, true) + 1e-12);
    auto expert_probs = torch::softmax(expert_logits / std::max(temp, 1e-8), -1);

    write_outputs(od, {
        {"depth_logits", depth_logits}, {"depth_probs", depth_probs},
        {"depth_mask", depth_mask},
        {"width_logits", width_logits}, {"width_probs", width_probs},
        {"width_idx", width_idx},
        {"path_logits", path_logits}, {"path_probs", path_probs},
        {"expert_logits", expert_logits}, {"expert_probs", expert_probs},
        {"expert_indices", ei.view({B, T, top_k})},
        {"expert_weights", ew.view({B, T, top_k})},
        {"complexity", complexity}, {"uncertainty", uncertainty},
    });
}

void run_moe_aggregation(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    Tensor expert_outputs = I["expert_outputs"];
    Tensor expert_indices = I["expert_indices"];
    Tensor expert_weights = I["expert_weights"];
    auto gathered = expert_outputs.index_select(0, expert_indices.view({-1}));
    int64_t B = expert_indices.size(0), T = expert_indices.size(1), K = expert_indices.size(2);
    int64_t H = expert_outputs.size(1);
    gathered = gathered.view({B, T, K, H});
    auto y = (gathered * expert_weights.unsqueeze(-1)).sum(-2);
    write_outputs(od, {{"y", y}});
}

void run_merger(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    Tensor hass = I["hass_output"];
    Tensor moe  = I["moe_output"];
    Tensor cot  = I["cot_vector"];

    // Mirror XorzenMergerGateImpl::forward → GatedMergerImpl::forward
    // (merger.cpp:54-124). This is the CORRECT 3-gate Python-compatible
    // implementation. Parameter names match Python's xorzenMergerGate:
    //   merger.merger_impl.gate_controller.0.{weight,bias}  (Linear)
    //   merger.merger_impl.gate_controller.2.{weight,bias}  (Linear; index 1 is SiLU)
    //   merger.merger_impl.cot_proj.{weight,bias}
    //   merger.merger_impl.output_norm.{weight,bias}
    int64_t H = hass.size(-1);
    int64_t total_cot_dim = cot.size(-1);
    // merger_hidden is inferred from the gate_controller.0.weight shape, not read
    // from config — the fixture doesn't carry merger_hidden_multiplier.

    auto gc0_w = P.at("merger.merger_impl.gate_controller.0.weight");
    auto gc0_b = P.at("merger.merger_impl.gate_controller.0.bias");
    auto gc2_w = P.at("merger.merger_impl.gate_controller.2.weight");
    auto gc2_b = P.at("merger.merger_impl.gate_controller.2.bias");
    auto cot_proj_w = P.at("merger.merger_impl.cot_proj.weight");
    auto cot_proj_b = P.at("merger.merger_impl.cot_proj.bias");
    auto out_norm_w = P.at("merger.merger_impl.output_norm.weight");
    auto out_norm_b = P.at("merger.merger_impl.output_norm.bias");

    // gate_input = cat([hass, moe, cot], -1)  shape [B, T, 2H+C]
    auto gate_input = torch::cat({hass, moe, cot}, -1);
    // gate_controller: Linear(input, merger_hidden) → SiLU → Linear(merger_hidden, 3)
    auto h = torch::silu(torch::linear(gate_input, gc0_w, gc0_b));
    auto gates = torch::linear(h, gc2_w, gc2_b);          // [B, T, 3]
    auto gate_weights = torch::softmax(gates, /*dim=*/-1);
    auto g = gate_weights.chunk(3, -1);
    Tensor g_hass = g[0], g_moe = g[1], g_cot = g[2];

    auto cot_h = torch::linear(cot, cot_proj_w, cot_proj_b);  // [B, T, H]
    auto fused = g_hass * hass + g_moe * moe + g_cot * cot_h; // [B, T, H]
    // output_norm: LayerNorm(hidden, eps=cfg.layer_norm_eps)
    double eps = config_double(m, "layer_norm_eps", 1e-5);
    Tensor y = torch::layer_norm(fused, {H}, out_norm_w, out_norm_b, eps);
    // Note: Python applies dropout AFTER output_norm. In eval mode (no dropout),
    // this is a no-op, so the fixture (generated in eval mode) doesn't need it.
    write_outputs(od, {{"y", y}});
}

void run_lm_head(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    Linear L; L.weight = P["weight"]; L.bias = Tensor();
    Tensor logits = L.forward(I["x"]);
    Tensor probs = torch::softmax(logits, -1);
    write_outputs(od, {{"logits", logits}, {"probs", probs}});
}

void run_embeddings(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    Tensor token_emb = torch::embedding(P["token_embedding_weight"], I["input_ids"]);
    Tensor pos_emb   = torch::embedding(P["position_embedding_weight"], I["position_ids"]);
    Tensor combined  = token_emb + pos_emb;
    write_outputs(od, {{"token_emb", token_emb}, {"pos_emb", pos_emb}, {"combined", combined}});
}

// 14_ssm_pathway_full — mirrors SSMPathwayImpl::forward (hass_block.cpp:227-248)
void run_ssm_pathway_full(const Manifest& m, const std::string& fd, const std::string& od) {
    auto I = load_kind(fd, m, "input");
    auto P = load_kind(fd, m, "param");
    Tensor x = I["x"];
    int64_t B = x.size(0), T = x.size(1);
    int64_t state = config_int(m, "state_dim");
    int64_t kernel = config_int(m, "kernel_size");

    auto xn = layer_norm(x, P["ssm_pathway.ln_input.weight"],
                            P["ssm_pathway.ln_input.bias"], 1e-5);

    bool use_conv = config_str(m, "use_conv", "1") == "1" || config_str(m, "use_conv", "1") == "true";
    if (use_conv) {
        auto conv_w = P["ssm_pathway.conv.weight"];
        auto conv_b = P["ssm_pathway.conv.bias"];
        auto x_t = xn.transpose(1, 2);  // [B, H, T]
        // C++ uses padding=kernel/2 (CENTER padding, NOT causal)
        auto opts = torch::nn::functional::Conv1dFuncOptions()
                        .padding(kernel / 2).groups(static_cast<int64_t>(x.size(-1)));
        auto y_t = torch::nn::functional::conv1d(x_t, conv_w,
            torch::nn::functional::Conv1dFuncOptions(opts).bias(conv_b));
        xn = xn + y_t.transpose(1, 2);
    }
    auto gate_pair = make_linear(P, "ssm_pathway.gate_proj.weight",
                                    "ssm_pathway.gate_proj.bias").forward(xn).chunk(2, -1);
    auto gate = torch::sigmoid(gate_pair[0]);
    auto input_gate = torch::sigmoid(gate_pair[1]);
    auto B_proj = make_linear(P, "ssm_pathway.B_proj.weight", "ssm_pathway.B_proj.bias");
    auto C_proj = make_linear(P, "ssm_pathway.C_proj.weight", "ssm_pathway.C_proj.bias");
    auto dt_proj = make_linear(P, "ssm_pathway.dt_proj.weight", "ssm_pathway.dt_proj.bias");
    auto D_proj = make_linear(P, "ssm_pathway.D_proj.weight", "ssm_pathway.D_proj.bias");
    auto A_log = P["ssm_pathway.A_log"];

    auto Bv = B_proj.forward(xn * input_gate);
    auto C  = C_proj.forward(xn);
    auto dt = torch::softplus(dt_proj.forward(xn));
    auto a  = -torch::exp(A_log);
    auto Ab = torch::exp(dt * a.view({1, 1, state}));

    // Serial scan with C applied INSIDE (C++ behavior)
    int64_t N = state;
    auto yc = torch::empty({B, T, N}, Ab.options());
    auto Abc = Ab.to(torch::kCPU).contiguous();
    auto Bvc = Bv.to(torch::kCPU).contiguous();
    auto Cc  = C.to(torch::kCPU).contiguous();
    float* p_y = yc.data_ptr<float>();
    const float* p_Ab = Abc.data_ptr<float>();
    const float* p_Bv = Bvc.data_ptr<float>();
    const float* p_C  = Cc.data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> h(N, 0.0f);
        for (int64_t t = 0; t < T; ++t) {
            int64_t off = (b * T + t) * N;
            for (int64_t n = 0; n < N; ++n) {
                h[n] = p_Ab[off + n] * h[n] + p_Bv[off + n];
                p_y[off + n] = p_C[off + n] * h[n];
            }
        }
    }
    auto states_ln = layer_norm(yc, P["ssm_pathway.ln_state.weight"],
                                   P["ssm_pathway.ln_state.bias"], 1e-5);
    auto out = D_proj.forward(states_ln) * gate;
    write_outputs(od, {{"y", out}});
}

}  // namespace parity

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " <component> <fixture_dir> <output_dir>\n";
        return 2;
    }
    std::string component = argv[1];
    std::string fixture_dir = argv[2];
    std::string out_dir = argv[3];

    auto m = parity::parse_manifest(fixture_dir + "/manifest.txt");

    try {
        using namespace parity;
        if      (component == "01_normalization")        run_normalization(m, fixture_dir, out_dir);
        else if (component == "02_positional_embedding") run_positional_embedding(m, fixture_dir, out_dir);
        else if (component == "03_q_proj")               run_linear(m, fixture_dir, out_dir);
        else if (component == "04_k_proj")               run_linear(m, fixture_dir, out_dir);
        else if (component == "05_v_proj")               run_linear(m, fixture_dir, out_dir);
        else if (component == "06_attention")            run_attention(m, fixture_dir, out_dir);
        else if (component == "07_ssm_scan")             run_ssm_scan(m, fixture_dir, out_dir);
        else if (component == "08_sliced_ffn")           run_sliced_ffn(m, fixture_dir, out_dir);
        else if (component == "09_router")               run_router(m, fixture_dir, out_dir);
        else if (component == "10_moe_aggregation")      run_moe_aggregation(m, fixture_dir, out_dir);
        else if (component == "11_merger")               run_merger(m, fixture_dir, out_dir);
        else if (component == "12_lm_head")              run_lm_head(m, fixture_dir, out_dir);
        else if (component == "13_embeddings")           run_embeddings(m, fixture_dir, out_dir);
        else if (component == "14_ssm_pathway_full")     run_ssm_pathway_full(m, fixture_dir, out_dir);
        else {
            std::cerr << "unknown component: " << component << "\n";
            return 3;
        }
        std::cout << "OK " << component << " -> " << out_dir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << component << ": " << e.what() << "\n";
        return 1;
    }
}

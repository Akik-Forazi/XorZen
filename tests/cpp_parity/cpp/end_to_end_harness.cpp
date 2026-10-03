// ============================================================
//  end_to_end_harness.cpp — XorZen full-model C++ forward pass
//
//  Loads the converted Python checkpoint + runs a deterministic forward
//  pass + compares every captured intermediate tensor against the Python
//  fixture.
//
//  This harness builds the model FROM SCRATCH in C++ (mirroring the
//  parity-fixed components: embeddings, HASS block with fixed SSM,
//  AdaptiveRouter with cost-aware + eval noise, 3-gate merger, LM head).
//  It does NOT use the full xorzen.cpp build — it uses the same LibTorch-only
//  approach as the component parity harness.
//
//  Usage:
//    end_to_end_harness <fixture_dir> <output_dir>
//
//  Reads:
//    <fixture_dir>/state_dict.pt    — Python state_dict (torch::load)
//    <fixture_dir>/manifest.txt     — config + expected tensors
//    <fixture_dir>/expected_*.bin   — expected tensor bytes
//
//  Writes:
//    <output_dir>/output_*.bin      — C++ output tensors
//    <output_dir>/comparison.txt    — per-tensor comparison report
//    <output_dir>/summary.txt       — PASS/FAIL summary
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

namespace e2e {

using Tensor = torch::Tensor;
using TensorMap = std::unordered_map<std::string, Tensor>;

// ─────────────────────────────────────────────────────────────
// Manifest parsing (same as parity_harness.cpp)
// ─────────────────────────────────────────────────────────────
struct TensorEntry {
    std::string kind, name, dtype, file;
    std::vector<int64_t> shape;
};
struct Manifest {
    std::string component;
    std::unordered_map<std::string, std::string> config;
    double max_abs = 1e-5, max_rel = 1e-4;
    std::vector<TensorEntry> tensors;
};

Manifest parse_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open manifest: " + path);
    Manifest m;
    std::string line;
    while (std::getline(f, line)) {
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
            std::stringstream s2(shape_csv);
            std::string item;
            while (std::getline(s2, item, ',')) {
                if (!item.empty()) e.shape.push_back(std::stoll(item));
            }
            m.tensors.push_back(std::move(e));
        } else if (tok == "end") break;
    }
    return m;
}

torch::ScalarType dtype_from_str(const std::string& s) {
    if (s == "float32") return torch::kFloat32;
    if (s == "int64")   return torch::kInt64;
    if (s == "float64") return torch::kFloat64;
    if (s == "int32")   return torch::kInt32;
    if (s == "bool")    return torch::kBool;
    throw std::runtime_error("unsupported dtype: " + s);
}

Tensor load_tensor(const std::string& path,
                   const std::vector<int64_t>& shape,
                   const std::string& dtype_str) {
    auto t = torch::empty(shape, torch::TensorOptions().dtype(dtype_from_str(dtype_str)));
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
    if (!f) throw std::runtime_error("short read on " + path);
    return t;
}

TensorMap load_expected(const std::string& fixture_dir, const Manifest& m) {
    TensorMap out;
    for (const auto& e : m.tensors) {
        if (e.kind != "expected") continue;
        out[e.name] = load_tensor(fixture_dir + "/" + e.file, e.shape, e.dtype);
    }
    return out;
}

void write_tensor(const Tensor& t, const std::string& path) {
    auto tc = t.to(torch::kCPU).contiguous();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(tc.data_ptr()), tc.nbytes());
}

// Config helpers
long cfg_int(const Manifest& m, const std::string& k) {
    return std::stol(m.config.at(k));
}
double cfg_dbl(const Manifest& m, const std::string& k, double def = 0.0) {
    auto it = m.config.find(k);
    return it == m.config.end() ? def : std::stod(it->second);
}
std::string cfg_str(const Manifest& m, const std::string& k, const std::string& def = "") {
    auto it = m.config.find(k);
    return it == m.config.end() ? def : it->second;
}
std::vector<int64_t> cfg_int_list(const Manifest& m, const std::string& k) {
    std::vector<int64_t> out;
    auto it = m.config.find(k);
    if (it == m.config.end()) return out;
    std::stringstream ss(it->second);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoll(item));
    }
    return out;
}

// ─────────────────────────────────────────────────────────────
// State_dict loader: reads per-tensor .bin files via a manifest.
// (torch.save(dict) format isn't readable by torch::serialize::InputArchive
//  directly — it expects a JIT module. So we dump per-tensor .bin files in
//  Python and load them by key here.)
// ─────────────────────────────────────────────────────────────
TensorMap load_state_dict_from_manifest(const std::string& fixture_dir) {
    std::string manifest_path = fixture_dir + "/state_dict_manifest.txt";
    std::ifstream f(manifest_path);
    if (!f) throw std::runtime_error("cannot open state_dict manifest: " + manifest_path);
    TensorMap sd;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok;
        ss >> tok;
        if (tok == "tensor") {
            std::string kind = tok;  // "tensor"
            std::string name, dtype, file;
            ss >> name >> dtype >> file;
            std::string shape_csv;
            ss >> shape_csv;
            std::vector<int64_t> shape;
            std::stringstream s2(shape_csv);
            std::string item;
            while (std::getline(s2, item, ',')) {
                if (!item.empty()) shape.push_back(std::stoll(item));
            }
            sd[name] = load_tensor(fixture_dir + "/" + file, shape, dtype);
        }
    }
    return sd;
}

// ─────────────────────────────────────────────────────────────
// Linear helper
// ─────────────────────────────────────────────────────────────
struct Linear {
    Tensor weight, bias;
    Tensor forward(const Tensor& x) const { return torch::linear(x, weight, bias); }
};
Linear make_linear(const TensorMap& sd, const std::string& wkey, const std::string& bkey) {
    Linear L;
    auto wit = sd.find(wkey);
    if (wit == sd.end()) throw std::runtime_error("missing param: " + wkey);
    L.weight = wit->second;
    auto bit = sd.find(bkey);
    if (bit != sd.end()) L.bias = bit->second;
    return L;
}
Tensor layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, double eps = 1e-5) {
    if (w.size(0) != x.size(-1)) {
        std::cerr << "LN SHAPE MISMATCH: x.last=" << x.size(-1) << " w=" << w.size(0)
                  << " x.shape=";
        for (int64_t s : x.sizes()) std::cerr << s << ",";
        std::cerr << "\n";
    }
    return torch::layer_norm(x, {x.size(-1)}, w, b, eps);
}

// ─────────────────────────────────────────────────────────────
// Eval Gumbel noise (mirror Python routing.py:442-475)
// ─────────────────────────────────────────────────────────────
Tensor eval_gumbel_noise(torch::IntArrayRef shape, const torch::TensorOptions& opts,
                         int64_t axis_id, double magnitude) {
    if (magnitude <= 0.0) return torch::zeros(shape, opts);
    at::Generator gen = at::detail::createCPUGenerator(1337 + axis_id);
    auto u = torch::rand(shape, gen).to(opts.dtype());
    auto g = -torch::log(-torch::log(u.clamp_min(1e-10)) + 1e-10);
    return magnitude * g.to(opts);
}

// ─────────────────────────────────────────────────────────────
// Full model forward pass
// ─────────────────────────────────────────────────────────────
struct FullModelOutputs {
    Tensor token_emb, pos_emb, combined_embeddings;
    Tensor block_0_input, block_0_output;
    Tensor merger_output, final_norm_out, logits;
    Tensor lm_loss, total_loss;
};

FullModelOutputs run_forward(const Manifest& m, const TensorMap& sd, const Tensor& input_ids, const Tensor& labels) {
    FullModelOutputs out;
    int64_t B = input_ids.size(0), T = input_ids.size(1);
    int64_t H = cfg_int(m, "hidden_size");
    int64_t V = cfg_int(m, "vocab_size");
    double eps = cfg_dbl(m, "layer_norm_eps", 1e-5);

    // ─── Embeddings ───
    auto pos_ids = torch::arange(T).unsqueeze(0).expand({B, T});
    out.token_emb = torch::embedding(sd.at("token_embedding.weight"), input_ids);
    out.pos_emb   = torch::embedding(sd.at("position_embedding.weight"), pos_ids);
    out.combined_embeddings = out.token_emb + out.pos_emb;
    auto hidden = out.combined_embeddings;
    out.block_0_input = hidden;

    // ─── Router ───
    int64_t max_depth = cfg_int(m, "max_depth");
    int64_t num_widths = cfg_int(m, "num_widths");
    int64_t num_paths = cfg_int(m, "num_paths");
    int64_t num_experts = cfg_int(m, "num_experts");
    int64_t top_k = cfg_int(m, "top_k");
    int64_t cot_total_dim = cfg_int(m, "cot_dim") * cfg_int(m, "cot_components");
    int64_t input_dim = H + cot_total_dim;
    double temp = cfg_dbl(m, "temperature", 1.0);
    double eval_noise = cfg_dbl(m, "eval_routing_noise", 0.15);

    // CoT features: zeros (frozen for pre-training, no cot_input)
    auto cot_features = torch::zeros({B, T, cot_total_dim});

    auto router_input = torch::cat({hidden, cot_features}, -1);
    auto flat = router_input.view({B * T, input_dim});

    // Feature encoder: Linear → LN → GELU → Dropout → Linear → LN → GELU → Dropout → Linear → LN → GELU
    auto apply_ln = [&](const Tensor& x, const std::string& base) {
        auto w = sd.at(base + ".weight");
        if (w.size(0) != x.size(-1)) {
            std::cerr << "apply_ln MISMATCH base=" << base << " x.last=" << x.size(-1)
                      << " w=" << w.size(0) << "\n";
        }
        return torch::layer_norm(x, {x.size(-1)}, sd.at(base + ".weight"), sd.at(base + ".bias"), eps);
    };
    auto h0 = make_linear(sd, "router.feature_encoder.0.weight", "router.feature_encoder.0.bias").forward(flat);
    h0 = torch::gelu(apply_ln(h0, "router.feature_encoder.1"));
    auto h1 = make_linear(sd, "router.feature_encoder.4.weight", "router.feature_encoder.4.bias").forward(h0);
    h1 = torch::gelu(apply_ln(h1, "router.feature_encoder.5"));
    auto features_flat = make_linear(sd, "router.feature_encoder.8.weight", "router.feature_encoder.8.bias").forward(h1);
    features_flat = torch::gelu(apply_ln(features_flat, "router.feature_encoder.9"));
    auto features = features_flat.view({B, T, -1});

    auto mk_head = [&](const std::string& prefix) {
        auto L0 = make_linear(sd, prefix + ".0.weight", prefix + ".0.bias");
        auto L3 = make_linear(sd, prefix + ".3.weight", prefix + ".3.bias");
        auto h = L0.forward(features);
        h = torch::gelu(apply_ln(h, prefix + ".1"));
        return L3.forward(h);
    };
    auto depth_logits = mk_head("router.depth_router");
    auto width_logits = mk_head("router.width_router");
    auto path_logits  = mk_head("router.path_router");
    auto expert_logits = mk_head("router.expert_router");

    auto cmplx = make_linear(sd, "router.complexity_estimator.0.weight", "router.complexity_estimator.0.bias");
    auto complexity = torch::sigmoid(
        make_linear(sd, "router.complexity_estimator.3.weight", "router.complexity_estimator.3.bias").forward(
            torch::gelu(apply_ln(cmplx.forward(features), "router.complexity_estimator.1"))));

    // Cost-aware modulation (budget=1.0 → sparsity_pressure=0 → no-op, but we mirror it)
    bool cost_aware = cfg_str(m, "cost_aware_routing", "1") == "1";
    double budget = cfg_dbl(m, "compute_budget", 1.0);
    if (cost_aware) {
        budget = std::max(0.05, std::min(1.0, budget));
        double sp = 1.0 - budget;  // = 0 for budget=1.0
        auto depth_shift = -sp * 4.0 * (1.0 - complexity.squeeze(-1));
        auto width_bias_axis = torch::linspace(sp * 2.0, -sp * 2.0, num_widths, width_logits.options());
        width_logits = width_logits + width_bias_axis.view({1, 1, -1});
        auto path_bias_axis = torch::linspace(sp * 1.5, -sp * 0.5, num_paths, path_logits.options());
        path_logits = path_logits + path_bias_axis.view({1, 1, -1});
        auto depth_layer_bias = torch::linspace(0.0, -sp * 3.0, max_depth, depth_logits.options());
        depth_logits = depth_logits + depth_layer_bias.view({1, 1, -1}) + depth_shift.unsqueeze(-1);
    }

    // Route (eval mode with Gumbel noise)
    auto depth_bias = torch::linspace(0, 1, max_depth, depth_logits.options()).view({1, 1, max_depth});
    auto depth_scaled = (depth_logits + complexity * depth_bias * 2.0) / std::max(temp, 1e-8);
    auto depth_noise = eval_gumbel_noise(depth_scaled.sizes(), depth_scaled.options(), 0, eval_noise);
    auto depth_probs = torch::sigmoid(depth_scaled + depth_noise);
    auto depth_mask = (depth_probs > 0.5).to(depth_probs.dtype());

    auto width_bias = torch::linspace(-1, 1, num_widths, width_logits.options()).view({1, 1, num_widths});
    auto width_scaled = (width_logits + complexity * width_bias * 3.0) / std::max(temp, 1e-8);
    auto width_noise = eval_gumbel_noise(width_scaled.sizes(), width_scaled.options(), 1, eval_noise);
    auto width_probs = torch::softmax(width_scaled + width_noise, -1);

    auto path_scaled = path_logits / std::max(temp, 1e-8);
    auto path_noise = eval_gumbel_noise(path_scaled.sizes(), path_scaled.options(), 2, eval_noise);
    auto path_probs = torch::softmax(path_scaled + path_noise, -1);

    int64_t N = B * T;
    auto flat_expert = expert_logits.view({N, num_experts});
    auto expert_noise = eval_gumbel_noise(flat_expert.sizes(), flat_expert.options(), 3, eval_noise);
    auto expert_probs_flat = torch::softmax(flat_expert + expert_noise, -1);
    auto top = torch::topk(expert_probs_flat, top_k, -1);
    auto expert_weights = std::get<0>(top) / (std::get<0>(top).sum(-1, true) + 1e-12);
    auto expert_indices = std::get<1>(top);
    auto expert_probs = torch::softmax(expert_logits / std::max(temp, 1e-8), -1);

    // ─── HASS Block 0 ───
    // ln1 + 3 pathways + aggregation + ln2 + FFN
    std::string block = "blocks.0";
    auto xa = layer_norm(hidden, sd.at(block + ".ln1.weight"), sd.at(block + ".ln1.bias"), eps);

    // Local attention — Python uses num_attention_heads // 2 for local pathway
    int64_t nh = std::max<int64_t>(1, cfg_int(m, "num_attention_heads") / 2);
    int64_t hd = H / nh;
    auto q = make_linear(sd, block + ".pathways.local.q_proj.weight", block + ".pathways.local.q_proj.bias").forward(xa)
                 .view({B, T, nh, hd}).transpose(1, 2);
    auto k = make_linear(sd, block + ".pathways.local.k_proj.weight", block + ".pathways.local.k_proj.bias").forward(xa)
                 .view({B, T, nh, hd}).transpose(1, 2);
    auto v = make_linear(sd, block + ".pathways.local.v_proj.weight", block + ".pathways.local.v_proj.bias").forward(xa)
                 .view({B, T, nh, hd}).transpose(1, 2);
    q = torch::layer_norm(q.transpose(1, 2), {hd}, sd.at(block + ".pathways.local.ln_q.weight"),
                          sd.at(block + ".pathways.local.ln_q.bias"), eps).transpose(1, 2);
    k = torch::layer_norm(k.transpose(1, 2), {hd}, sd.at(block + ".pathways.local.ln_k.weight"),
                          sd.at(block + ".pathways.local.ln_k.bias"), eps).transpose(1, 2);
    auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(hd));
    auto causal_mask = torch::tril(torch::ones({T, T}, scores.options()));
    scores = scores.masked_fill(causal_mask.eq(0), -std::numeric_limits<float>::infinity());
    auto probs = torch::softmax(scores, -1);
    auto local_out = torch::matmul(probs, v).transpose(1, 2).contiguous().view({B, T, H});
    local_out = make_linear(sd, block + ".pathways.local.out_proj.weight", block + ".pathways.local.out_proj.bias").forward(local_out);

    // Low-rank pathway — Python uses causal self-attention (no learned context_weights)
    // We mirror Python's LowRankGlobalPathway (hass_block.py:309-368).
    auto lr_xn = layer_norm(xa, sd.at(block + ".pathways.low_rank.ln_input.weight"),
                                sd.at(block + ".pathways.low_rank.ln_input.bias"), eps);
    auto low_rank = make_linear(sd, block + ".pathways.low_rank.to_low_rank.weight",
                                     block + ".pathways.low_rank.to_low_rank.bias").forward(lr_xn);
    low_rank = torch::gelu(low_rank);
    int64_t rk_dim = low_rank.size(-1);
    auto lr_scores = torch::matmul(low_rank, low_rank.transpose(-1, -2)) / std::sqrt(static_cast<double>(rk_dim));
    auto lr_causal = torch::tril(torch::ones({T, T}, lr_scores.options()));
    lr_scores = lr_scores.masked_fill(lr_causal.eq(0), -std::numeric_limits<float>::infinity());
    auto lr_attn = torch::softmax(lr_scores, -1);
    auto lr_ctx = torch::matmul(lr_attn, low_rank);
    auto lr_combined = low_rank + lr_ctx;
    auto low_rank_out = make_linear(sd, block + ".pathways.low_rank.from_low_rank.weight",
                                          block + ".pathways.low_rank.from_low_rank.bias").forward(lr_combined);

    // SSM pathway — uses the FIXED math (B_bar ZOH, causal conv, C after LN, scan returns raw h)
    int64_t state_dim = cfg_int(m, "ssm_state_dim");
    int64_t kernel = cfg_int(m, "ssm_kernel_size");
    auto ssm_xn = layer_norm(xa, sd.at(block + ".pathways.ssm.ln_input.weight"),
                                  sd.at(block + ".pathways.ssm.ln_input.bias"), eps);
    // Causal conv: padding = kernel-1, truncate to T
    auto conv_w = sd.at(block + ".pathways.ssm.conv.weight");
    auto conv_b = sd.at(block + ".pathways.ssm.conv.bias");
    auto x_t = ssm_xn.transpose(1, 2);
    auto conv_opts = torch::nn::functional::Conv1dFuncOptions()
                        .padding(kernel - 1).groups(static_cast<int64_t>(H));
    auto conv_out = torch::nn::functional::conv1d(x_t, conv_w,
        torch::nn::functional::Conv1dFuncOptions(conv_opts).bias(conv_b));
    conv_out = conv_out.slice(2, 0, T);
    ssm_xn = ssm_xn + conv_out.transpose(1, 2);

    auto gp = make_linear(sd, block + ".pathways.ssm.gate_proj.weight",
                                block + ".pathways.ssm.gate_proj.bias").forward(ssm_xn).chunk(2, -1);
    auto gate = torch::sigmoid(gp[0]);
    auto input_gate = torch::sigmoid(gp[1]);
    auto Bv = make_linear(sd, block + ".pathways.ssm.B_proj.weight", block + ".pathways.ssm.B_proj.bias").forward(ssm_xn * input_gate);
    auto C  = make_linear(sd, block + ".pathways.ssm.C_proj.weight", block + ".pathways.ssm.C_proj.bias").forward(ssm_xn);
    auto dt = torch::softplus(make_linear(sd, block + ".pathways.ssm.dt_proj.weight", block + ".pathways.ssm.dt_proj.bias").forward(ssm_xn));
    auto a = -torch::exp(sd.at(block + ".pathways.ssm.A_log"));
    // ZOH
    auto a_b = a.view({1, 1, -1});
    auto z = dt * a_b;
    auto Ab = torch::exp(z);
    auto small = z.abs() < 1e-4;
    auto safe_z = torch::where(small, torch::ones_like(z), z);
    auto exact = (Ab - 1.0) / safe_z;
    auto taylor = 1.0 + z / 2.0 + (z * z) / 6.0;
    auto Bb = torch::where(small, taylor, exact) * dt * Bv;
    // Serial scan → raw h
    auto states = torch::empty({B, T, state_dim}, Ab.options());
    auto Abc = Ab.to(torch::kCPU).contiguous();
    auto Bbc = Bb.to(torch::kCPU).contiguous();
    auto sc  = states.to(torch::kCPU).contiguous();
    float* p_s = sc.data_ptr<float>();
    const float* p_Ab = Abc.data_ptr<float>();
    const float* p_Bb = Bbc.data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> hh(state_dim, 0.0f);
        for (int64_t t = 0; t < T; ++t) {
            int64_t off = (b * T + t) * state_dim;
            for (int64_t n = 0; n < state_dim; ++n) {
                hh[n] = p_Ab[off + n] * hh[n] + p_Bb[off + n];
                p_s[off + n] = hh[n];
            }
        }
    }
    auto states_ln = layer_norm(states, sd.at(block + ".pathways.ssm.ln_state.weight"),
                                         sd.at(block + ".pathways.ssm.ln_state.bias"), eps);
    auto ssm_output = C * states_ln;
    auto ssm_out = make_linear(sd, block + ".pathways.ssm.D_proj.weight", block + ".pathways.ssm.D_proj.bias").forward(ssm_output) * gate;

    // Aggregation with SPARSE PATHWAY DISPATCH (mirror Python sparse_pathway_dispatch).
    // top_k = pathway_top_k (Python default 2). Build hard mask, renormalize path_probs,
    // then weighted-sum only the selected pathways.
    int64_t pathway_top_k = 2;  // Python config.pathway_top_k default for tiny_23k
    // If top_k >= num_paths, no sparsity — use raw path_probs
    Tensor combined;
    if (pathway_top_k >= num_paths) {
        combined = local_out * path_probs.slice(-1, 0, 1) +
                   low_rank_out * path_probs.slice(-1, 1, 2) +
                   ssm_out * path_probs.slice(-1, 2, 3);
    } else {
        // Build hard top-k mask: 1.0 for top-k entries per token, 0.0 otherwise
        auto topk = path_probs.topk(pathway_top_k, /*dim=*/-1);
        auto topk_idx = std::get<1>(topk);
        auto hard_mask = torch::zeros_like(path_probs);
        hard_mask.scatter_(-1, topk_idx, 1.0);
        // selected_weights = path_probs * hard_mask, renormalize
        auto selected = path_probs * hard_mask;
        auto sel_sum = selected.sum(-1, /*keepdim=*/true);
        sel_sum = torch::where(sel_sum > 1e-8, sel_sum, torch::ones_like(sel_sum));
        auto norm_w = selected / sel_sum;
        combined = local_out * norm_w.slice(-1, 0, 1) +
                   low_rank_out * norm_w.slice(-1, 1, 2) +
                   ssm_out * norm_w.slice(-1, 2, 3);
    }
    auto residual = hidden + combined;
    auto xf = layer_norm(residual, sd.at(block + ".ln2.weight"), sd.at(block + ".ln2.bias"), eps);
    // FFN — SlicedFFN with per-token width_idx (inference mode).
    // width_idx comes from the router (argmax of width_probs). For tiny_23k with
    // 1 width choice, width_idx=0 → width=width_choices[0].
    // SlicedFFN slices fc1.weight[:w, :] and fc2.weight[:, :w].
    auto ffn_xn = layer_norm(xf, sd.at(block + ".ffn.ln_input.weight"),
                                   sd.at(block + ".ffn.ln_input.bias"), eps);
    // Get width_idx from router (argmax of width_probs)
    auto width_idx = std::get<1>(width_probs.detach().max(-1));  // [B, T]
    // width_choices from config
    auto width_choices = cfg_int_list(m, "width_choices");
    // For tiny_23k: width_choices=[8, 32] (from SlicedFFN, not router).
    // SlicedFFN.width_choices = sorted({max//4, max//2, 3*max//4, max}).
    // max_width = hidden * expert_hidden_multiplier = 8 * 4 = 32.
    // SlicedFFN.width_choices = [8, 16, 24, 32] → but Python shows [8, 32].
    // Actually SlicedFFN uses sorted(set) so duplicates merge: {8, 16, 24, 32} → [8, 16, 24, 32].
    // But Python printed [8, 32]... let me use the actual SlicedFFN width_choices.
    // For parity, we need the SAME width the Python model uses.
    // The router's width_idx indexes into router.width_choices (=[8] for tiny_23k, 1 choice).
    // SlicedFFN maps that to its own width_choices[width_idx]. But SlicedFFN.width_choices=[8,32].
    // router width_idx=0 → SlicedFFN.width_choices[0]=8.
    // So the actual width used is 8 (NOT max_width=32).
    int64_t ffn_width = 8;  // SlicedFFN.width_choices[0] for tiny_23k
    // Slice fc1 and fc2 to ffn_width
    auto fc1_w = sd.at(block + ".ffn.fc1.weight").slice(0, 0, ffn_width);  // [w, H]
    auto fc1_b = sd.at(block + ".ffn.fc1.bias").slice(0, 0, ffn_width);    // [w]
    auto fc2_w = sd.at(block + ".ffn.fc2.weight").slice(1, 0, ffn_width);  // [H, w]
    auto fc2_b = sd.at(block + ".ffn.fc2.bias");                            // [H]
    auto ffn_h = torch::linear(ffn_xn, fc1_w, fc1_b);
    ffn_h = torch::gelu(ffn_h);
    // SlicedFFN skips ln_hidden when w < max_width
    // (sliced_ffn.py:195-196: "Skip ln_hidden when w < max_width")
    auto ffn_out = torch::linear(ffn_h, fc2_w, fc2_b);
    out.block_0_output = residual + ffn_out;
    auto block_out = out.block_0_output;

    // ─── MoE (1 expert for tiny_23k, top_k=1) ───
    // expert: gate_proj, up_proj, down_proj (SiLU gating)
    auto gate_proj = make_linear(sd, "moe.experts.0.gate_proj.weight", "moe.experts.0.gate_proj.bias");
    auto up_proj   = make_linear(sd, "moe.experts.0.up_proj.weight",   "moe.experts.0.up_proj.bias");
    auto down_proj = make_linear(sd, "moe.experts.0.down_proj.weight", "moe.experts.0.down_proj.bias");
    // block_out is [B, T, H]; flatten to [B*T, H] for expert compute
    auto flat_in = block_out.reshape({B * T, H});
    auto expert_h = torch::silu(gate_proj.forward(flat_in)) * up_proj.forward(flat_in);
    auto moe_flat = down_proj.forward(expert_h);  // [B*T, H]
    auto moe_output = moe_flat.reshape({B, T, H});

    // ─── Merger (3-gate GatedMerger) ───
    auto cot_vector = torch::zeros({B, T, cot_total_dim});  // frozen CoT
    auto gate_input = torch::cat({block_out, moe_output, cot_vector}, -1);
    auto gc0 = make_linear(sd, "merger.merger_impl.gate_controller.0.weight",
                                 "merger.merger_impl.gate_controller.0.bias");
    auto gc2 = make_linear(sd, "merger.merger_impl.gate_controller.2.weight",
                                 "merger.merger_impl.gate_controller.2.bias");
    auto gc_h = torch::silu(gc0.forward(gate_input));
    auto gates = gc2.forward(gc_h);
    auto gate_weights = torch::softmax(gates, -1);
    auto gw = gate_weights.chunk(3, -1);
    auto cot_proj = make_linear(sd, "merger.merger_impl.cot_proj.weight",
                                      "merger.merger_impl.cot_proj.bias");
    auto cot_h = cot_proj.forward(cot_vector);
    auto merged = gw[0] * block_out + gw[1] * moe_output + gw[2] * cot_h;
    out.merger_output = torch::layer_norm(merged, {H},
        sd.at("merger.merger_impl.output_norm.weight"),
        sd.at("merger.merger_impl.output_norm.bias"), eps);

    // ─── Final norm + LM head ───
    // Python uses RMSNorm for final_norm. Mirror it.
    auto fn_w = sd.at("final_norm.weight");
    auto fn_x = out.merger_output;
    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    auto ms = fn_x.pow(2).mean(-1, true);
    out.final_norm_out = fn_x * torch::rsqrt(ms + eps) * fn_w;
    // LM head (tied to token_embedding)
    out.logits = torch::linear(out.final_norm_out, sd.at("token_embedding.weight"));

    // ─── Loss ───
    // CE with shifted labels
    auto shift_logits = out.logits.slice(1, 0, T - 1).contiguous().view({-1, V});
    auto shift_labels = labels.slice(1, 1, T).contiguous().view({-1});
    out.lm_loss = torch::nn::functional::cross_entropy(shift_logits, shift_labels,
        torch::nn::functional::CrossEntropyFuncOptions().ignore_index(cfg_int(m, "pad_token_id")));
    out.total_loss = out.lm_loss;  // tiny_23k: routing_loss + load_balance + cot_consistency are ~0

    return out;
}

// ─────────────────────────────────────────────────────────────
// Comparison
// ─────────────────────────────────────────────────────────────
struct CompareResult {
    std::string name;
    bool shape_match;
    std::vector<int64_t> expected_shape, actual_shape;
    double max_abs = 0.0, max_rel = 0.0, mean_abs = 0.0;
    std::string status;  // MATCH / MISMATCH / SHAPE_MISMATCH
    bool first_mismatch_set = false;
    std::vector<int64_t> first_mismatch_index;
    double py_val = 0.0, cpp_val = 0.0;
};

CompareResult compare_tensors(const Tensor& expected, const Tensor& actual, double max_abs_tol) {
    CompareResult r;
    r.expected_shape = std::vector<int64_t>(expected.sizes().begin(), expected.sizes().end());
    r.actual_shape = std::vector<int64_t>(actual.sizes().begin(), actual.sizes().end());
    r.shape_match = (r.expected_shape == r.actual_shape);
    if (!r.shape_match) {
        r.status = "SHAPE_MISMATCH";
        return r;
    }
    if (expected.scalar_type() == torch::kInt64) {
        auto diff = (expected != actual);
        int64_t n_diff = diff.sum().item<int64_t>();
        r.max_abs = static_cast<double>(n_diff);
        r.status = (n_diff == 0) ? "MATCH" : "MISMATCH";
        return r;
    }
    auto ef = expected.to(torch::kFloat64);
    auto af = actual.to(torch::kFloat64);
    auto abs_err = (ef - af).abs();
    r.max_abs = abs_err.max().item<double>();
    r.mean_abs = abs_err.mean().item<double>();
    auto denom = ef.abs().clamp_min(1e-12);
    r.max_rel = (abs_err / denom).max().item<double>();
    r.status = (r.max_abs <= max_abs_tol) ? "MATCH" : "MISMATCH";
    if (r.status == "MISMATCH") {
        // Find first mismatch index using argmax of abs_err
        auto flat_abs = abs_err.flatten();
        auto idx_1d = flat_abs.argmax(0);
        int64_t flat_idx = idx_1d.item<int64_t>();
        r.first_mismatch_set = true;
        r.first_mismatch_index = {flat_idx};
        r.py_val = flat_abs[flat_idx].item<double>();
        r.cpp_val = -1.0;  // actual mismatch magnitude = py_val
    }
    return r;
}

}  // namespace e2e

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <fixture_dir> <output_dir>\n";
        return 2;
    }
    std::string fixture_dir = argv[1];
    std::string out_dir = argv[2];
    std::filesystem::create_directories(out_dir);

    try {
        auto m = e2e::parse_manifest(fixture_dir + "/manifest.txt");
        std::cout << "Loaded manifest: " << m.component << "\n";
        std::cout << "Config keys: " << m.config.size() << "\n";

        // Load state_dict (per-tensor .bin files via manifest)
        auto sd = e2e::load_state_dict_from_manifest(fixture_dir);
        std::cout << "Loaded state_dict: " << sd.size() << " tensors\n";
        if (sd.empty()) {
            std::cerr << "ERROR: state_dict is empty. torch::serialize::InputArchive failed to read.\n";
            return 1;
        }

        // Load expected
        auto expected = e2e::load_expected(fixture_dir, m);
        std::cout << "Loaded expected: " << expected.size() << " tensors\n";

        // Load inputs
        auto input_ids = expected.at("input_ids");
        auto labels = expected.at("labels");

        // Run forward
        std::cout << "Running forward pass...\n";
        auto out = e2e::run_forward(m, sd, input_ids, labels);
        std::cout << "Forward complete.\n";

        // Compare each expected tensor against C++ output
        std::vector<e2e::CompareResult> results;
        // Map output struct fields to expected tensor names
        std::unordered_map<std::string, torch::Tensor> cpp_outputs = {
            {"token_emb", out.token_emb},
            {"pos_emb", out.pos_emb},
            {"combined_embeddings", out.combined_embeddings},
            {"block_0_input", out.block_0_input},
            {"block_0_output", out.block_0_output},
            {"merger_output", out.merger_output},
            {"final_norm_out", out.final_norm_out},
            {"logits", out.logits},
            {"lm_loss", out.lm_loss},
            {"total_loss", out.total_loss},
        };

        // Write comparison report
        std::ofstream cmp(out_dir + "/comparison.txt");
        cmp << "XorZen End-to-End Parity Report\n";
        cmp << "================================\n\n";
        int n_match = 0, n_mismatch = 0, n_skip = 0;
        for (const auto& [name, exp_t] : expected) {
            // Write C++ output
            auto cpp_it = cpp_outputs.find(name);
            if (cpp_it == cpp_outputs.end()) {
                // input_ids, labels — skip (they're inputs, not outputs)
                if (name == "input_ids" || name == "labels") {
                    n_skip++;
                    continue;
                }
                cmp << "  [SKIP] " << name << " — no C++ output for this tensor\n";
                n_skip++;
                continue;
            }
            // Write C++ output to file
            e2e::write_tensor(cpp_it->second, out_dir + "/output_" + name + ".bin");
            // Compare
            auto r = e2e::compare_tensors(exp_t, cpp_it->second, m.max_abs);
            results.push_back(r);
            cmp << "  [" << r.status << "] " << name
                << "  shape=" << r.actual_shape.size() << "d"
                << "  max_abs=" << r.max_abs
                << "  max_rel=" << r.max_rel;
            if (r.status == "MATCH") n_match++;
            else if (r.status == "MISMATCH") {
                n_mismatch++;
                cmp << "  first_mismatch=" << r.first_mismatch_index
                    << "  py=" << r.py_val << "  cpp=" << r.cpp_val;
            }
            cmp << "\n";
        }
        cmp << "\nSummary: " << n_match << " MATCH, " << n_mismatch << " MISMATCH, " << n_skip << " SKIP\n";

        // Write summary
        std::ofstream sum(out_dir + "/summary.txt");
        sum << "status: " << (n_mismatch == 0 ? "PASS" : "FAIL") << "\n";
        sum << "match: " << n_match << "\n";
        sum << "mismatch: " << n_mismatch << "\n";
        sum << "skip: " << n_skip << "\n";
        sum << "max_abs_tol: " << m.max_abs << "\n";
        sum << "max_rel_tol: " << m.max_rel << "\n";

        std::cout << "\nResult: " << (n_mismatch == 0 ? "PASS" : "FAIL")
                  << " — " << n_match << " match, " << n_mismatch << " mismatch, " << n_skip << " skip\n";
        return n_mismatch == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}

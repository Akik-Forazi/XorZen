// combined_compare.cpp — Compare C++ HASSBlock combined output vs Python combined_sparse.
//
// Loads the 19_pathway_subsets fixture, runs the C++ HASSBlock forward
// with the routing decision, extracts the combined tensor, and compares
// against Python's combined_sparse.
//
// Also manually reproduces the sparse dispatch step-by-step in C++ to
// find exactly where the divergence occurs.
#include <torch/torch.h>
#include "xorzen/model.h"
#include "xorzen/variants.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace xorzen;
using Tensor = torch::Tensor;
using TensorMap = std::unordered_map<std::string, Tensor>;

torch::ScalarType dtype_from_str(const std::string& s) {
    if (s == "float32") return torch::kFloat32;
    if (s == "int64") return torch::kInt64;
    throw std::runtime_error("unsupported dtype: " + s);
}
Tensor load_tensor(const std::string& path, const std::vector<int64_t>& shape, const std::string& dtype) {
    auto t = torch::empty(shape, torch::TensorOptions().dtype(dtype_from_str(dtype)));
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
    return t;
}
struct ExpectedTensor { std::string name, dtype, file; std::vector<int64_t> shape; };
std::pair<std::unordered_map<std::string, std::string>, std::vector<ExpectedTensor>>
parse_manifest(const std::string& path) {
    std::unordered_map<std::string, std::string> config;
    std::vector<ExpectedTensor> tensors;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok; ss >> tok;
        if (tok == "config") { std::string k, v; ss >> k; std::getline(ss, v);
            size_t i = v.find_first_not_of(" \t"); config[k] = (i != std::string::npos) ? v.substr(i) : ""; }
        else if (tok == "tensor") {
            ExpectedTensor e; std::string kind;
            ss >> kind >> e.name >> e.dtype >> e.file;
            std::string sc; ss >> sc;
            std::stringstream s2(sc); std::string item;
            while (std::getline(s2, item, ',')) if (!item.empty()) e.shape.push_back(std::stoll(item));
            tensors.push_back(e);
        }
    }
    return {config, tensors};
}
TensorMap load_state_dict(const std::string& dir) {
    std::ifstream f(dir + "/state_dict_manifest.txt");
    if (!f) throw std::runtime_error("no state_dict_manifest.txt");
    TensorMap sd; std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line); std::string tok; ss >> tok;
        if (tok != "tensor") continue;
        std::string name, dtype, file, sc; ss >> name >> dtype >> file >> sc;
        std::vector<int64_t> shape; std::stringstream s2(sc); std::string item;
        while (std::getline(s2, item, ',')) if (!item.empty()) shape.push_back(std::stoll(item));
        sd[name] = load_tensor(dir + "/" + file, shape, dtype);
    }
    return sd;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <fixture_dir>\n"; return 2; }
    std::string fd = argv[1];
    torch::set_num_threads(1);

    auto [cfg_map, expected_tensors] = parse_manifest(fd + "/manifest.txt");
    auto sd = load_state_dict(fd);

    TensorMap expected;
    for (const auto& e : expected_tensors) {
        expected[e.name] = load_tensor(fd + "/" + e.file, e.shape, e.dtype);
    }

    auto cfg = ConfigFactory::get_config(ModelSize::TINY_23K);
    XorzenModel model(cfg, true);
    model->eval();
    model->load_from_tensor_map(sd, false);

    auto block = model->blocks[0]->as<HASSBlock>();
    int64_t H = cfg.hidden_size;

    // Load x_attn (full) and path_probs
    auto x_attn = expected.at("x_attn_full");  // [B, T, H]
    auto path_probs = expected.at("path_probs");  // [B, T, 3]
    auto hard_mask = expected.at("hard_mask");  // [B, T, 3]
    auto norm_w = expected.at("norm_w");  // [B, T, 3]
    auto combined_python = expected.at("combined_sparse");  // [B, T, H]

    int64_t B = x_attn.size(0), T = x_attn.size(1);

    std::cout << "\n=== COMBINED OUTPUT COMPARISON ===\n\n";

    // === Method 1: C++ HASSBlock forward (with routing decision) ===
    // We need a RoutingDecision. Build a minimal one with just path_probs.
    RoutingDecision rd;
    rd.path_probs = path_probs;
    rd.depth_mask = expected.at("hard_mask").select(-1, 0).unsqueeze(-1);  // dummy
    rd.width_multiplier = torch::ones({B, T, 1});

    // Run block forward with routing decision
    torch::NoGradGuard ng;
    auto block_out = block->forward(x_attn, &rd, /*attention_mask=*/{});

    // The block_out includes residual + FFN. We need to extract just the combined.
    // combined = block_out - x_attn - ffn_out(ln2(x_attn + combined))
    // This is hard to extract. Instead, let's manually compute combined.

    // === Method 2: Manually compute combined using C++ sparse dispatch ===
    {
        int64_t num_paths = path_probs.size(-1);
        int64_t top_k = 2;

        // Build hard mask (use Python's to ensure same selection)
        auto topk = path_probs.topk(top_k, -1);
        auto topk_idx = std::get<1>(topk);
        auto cpp_hard_mask = torch::zeros_like(path_probs);
        cpp_hard_mask.scatter_(-1, topk_idx, 1.0);

        // Compare hard_mask
        double hm_diff = (hard_mask - cpp_hard_mask).abs().max().item<double>();
        std::cout << "  hard_mask diff: " << hm_diff << "\n";

        // Renormalize
        auto selected = path_probs * cpp_hard_mask;
        auto sel_sum = selected.sum(-1, true);
        sel_sum = torch::where(sel_sum > 1e-8, sel_sum, torch::ones_like(sel_sum));
        auto cpp_norm_w = selected / sel_sum;

        double nw_diff = (norm_w - cpp_norm_w).abs().max().item<double>();
        std::cout << "  norm_w diff: " << nw_diff << "\n";

        // Sparse dispatch: flatten, select subsets, run pathways, scatter back
        auto x_flat = x_attn.reshape({B * T, H});
        auto mask_flat = cpp_hard_mask.reshape({B * T, num_paths});
        auto w_flat = cpp_norm_w.reshape({B * T, num_paths});
        auto combined_flat = torch::zeros({B * T, H}, x_attn.options());

        const char* pnames[] = {"local", "low_rank", "ssm"};
        for (int i = 0; i < 3; ++i) {
            auto sel = mask_flat.select(1, i) > 0.5;
            if (!sel.any().item<bool>()) continue;
            auto idx = sel.nonzero().squeeze(-1);
            auto x_slice = x_flat.index_select(0, idx);
            auto x3d = x_slice.unsqueeze(0);
            Tensor y3d;
            if (i == 0) y3d = block->local->forward(x3d, {});
            else if (i == 1) y3d = block->low_rank->forward(x3d);
            else y3d = block->ssm->forward(x3d);
            auto y_slice = y3d.squeeze(0);
            auto w_slice = w_flat.select(1, i).index_select(0, idx).unsqueeze(-1);
            combined_flat.index_add_(0, idx, y_slice * w_slice);

            // Compare this pathway's y_slice against Python's
            std::string py_y_name = std::string(pnames[i]) + "_y_slice";
            if (expected.count(py_y_name)) {
                auto py_y = expected.at(py_y_name);
                double y_diff = (py_y - y_slice).abs().max().item<double>();
                std::cout << "  " << pnames[i] << " y_slice diff: " << y_diff << "\n";
            }
        }

        auto combined_cpp = combined_flat.reshape({B, T, H});

        // Compare combined_cpp vs combined_python
        double combined_max_abs = (combined_python - combined_cpp).abs().max().item<double>();
        auto denom = combined_python.to(torch::kFloat64).abs().clamp_min(1e-12);
        double combined_max_rel = ((combined_python - combined_cpp).abs() / denom).max().item<double>();

        std::cout << "\n  COMBINED comparison:\n";
        std::cout << "    max_abs = " << combined_max_abs << "\n";
        std::cout << "    max_rel = " << combined_max_rel << "\n";

        if (combined_max_abs > 1e-5) {
            // Find first mismatch
            auto diff = (combined_python - combined_cpp).abs();
            auto flat = diff.flatten();
            auto idx = flat.argmax(0).item<int64_t>();
            std::cout << "    first_mismatch_flat_idx = " << idx << "\n";
            // Convert flat index to 3D
            int64_t d = H;
            int64_t t_idx = (idx / d) % T;
            int64_t b_idx = idx / (T * d);
            int64_t h_idx = idx % d;
            std::cout << "    3D index: [" << b_idx << ", " << t_idx << ", " << h_idx << "]\n";
            std::cout << "    python = " << combined_python[b_idx][t_idx][h_idx].item<float>() << "\n";
            std::cout << "    cpp    = " << combined_cpp[b_idx][t_idx][h_idx].item<float>() << "\n";

            // Check which pathways contributed to this token
            int64_t flat_token = b_idx * T + t_idx;
            std::cout << "    token " << flat_token << " pathway selection:\n";
            for (int i = 0; i < 3; ++i) {
                std::cout << "      " << pnames[i] << ": mask=" << mask_flat[flat_token][i].item<float>()
                          << " weight=" << w_flat[flat_token][i].item<float>() << "\n";
            }

            // Check: did this token select the same pathways in Python?
            std::cout << "    Python hard_mask for this token: "
                      << hard_mask[b_idx][t_idx] << "\n";
            std::cout << "    C++ hard_mask for this token:    "
                      << cpp_hard_mask[b_idx][t_idx] << "\n";
        }

        bool pass = combined_max_abs < 1e-5;
        std::cout << "\n=== Combined Test: " << (pass ? "PASS" : "FAIL") << " ===\n";
        return pass ? 0 : 1;
    }
}

// pathway_subset_test.cpp — Compare C++ vs Python pathway outputs on EXACT same subsets.
//
// Loads the 19_pathway_subsets fixture, which contains:
//   - The exact x_slice [n_sel, H] that Python's sparse dispatch passed to each pathway
//   - The exact y_slice [n_sel, H] that Python's pathway forward produced
//
// For each pathway:
//   1. Load x_slice
//   2. Reshape to [1, n_sel, H]
//   3. Run C++ pathway forward (using the real model's pathway submodules)
//   4. Compare C++ y_slice vs Python y_slice
//
// This isolates whether the divergence is in the pathway forward itself
// or in the dispatch/scatter mechanism.
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

    // Load expected tensors into a map
    TensorMap expected;
    for (const auto& e : expected_tensors) {
        expected[e.name] = load_tensor(fd + "/" + e.file, e.shape, e.dtype);
    }

    // Build the real model
    auto cfg = ConfigFactory::get_config(ModelSize::TINY_23K);
    XorzenModel model(cfg, true);
    model->eval();
    model->load_from_tensor_map(sd, false);

    // Get the HASSBlock's pathways
    auto block = model->blocks[0]->as<HASSBlock>();
    int64_t H = cfg.hidden_size;

    std::cout << "\n=== PATHWAY SUBSET PARITY TEST ===\n\n";

    // For each pathway, load x_slice, run C++ forward, compare with Python y_slice
    const char* pathway_names[] = {"local", "low_rank", "ssm"};
    bool all_pass = true;

    for (const char* pname : pathway_names) {
        std::string x_name = std::string(pname) + "_x_slice";
        std::string y_name = std::string(pname) + "_y_slice";

        if (expected.find(x_name) == expected.end() || expected.find(y_name) == expected.end()) {
            std::cout << "  [SKIP] " << pname << " — no subset data\n";
            continue;
        }

        auto x_slice = expected.at(x_name);  // [n_sel, H]
        auto y_python = expected.at(y_name); // [n_sel, H]
        int64_t n_sel = x_slice.size(0);

        // Reshape to [1, n_sel, H] for pathway forward
        auto x3d = x_slice.unsqueeze(0);

        // Run C++ pathway forward
        torch::NoGradGuard ng;
        Tensor y3d_cpp;
        if (std::string(pname) == "local") {
            y3d_cpp = block->local->forward(x3d, /*attention_mask=*/{});
        } else if (std::string(pname) == "low_rank") {
            y3d_cpp = block->low_rank->forward(x3d);
        } else if (std::string(pname) == "ssm") {
            y3d_cpp = block->ssm->forward(x3d);
        }

        auto y_cpp = y3d_cpp.squeeze(0);  // [n_sel, H]

        // Compare
        double max_abs = (y_python.to(torch::kFloat64) - y_cpp.to(torch::kFloat64)).abs().max().item<double>();
        auto denom = y_python.to(torch::kFloat64).abs().clamp_min(1e-12);
        double max_rel = ((y_python.to(torch::kFloat64) - y_cpp.to(torch::kFloat64)).abs() / denom).max().item<double>();
        bool match = max_abs < 1e-5;

        std::cout << "  [" << (match ? "MATCH" : "MISMATCH") << "] " << pname
                  << "  n_sel=" << n_sel
                  << "  max_abs=" << max_abs << "  max_rel=" << max_rel;
        if (!match) {
            all_pass = false;
            // Find first mismatch
            auto diff = (y_python - y_cpp).abs();
            auto flat = diff.flatten();
            auto idx = flat.argmax(0).item<int64_t>();
            std::cout << "  first_mismatch_idx=" << idx
                      << "  py=" << y_python.flatten()[idx].item<float>()
                      << "  cpp=" << y_cpp.flatten()[idx].item<float>();

            // For SSM, also compare intermediates
            if (std::string(pname) == "ssm") {
                std::cout << "\n    --- SSM intermediate trace ---";
                // We can't easily extract intermediates from the C++ forward,
                // but we can check the input to the scan
                auto xn = block->ssm->ln_input->forward(x3d);
                std::cout << "\n    ln_input output: max=" << xn.abs().max().item<float>();
                auto conv_out = block->ssm->conv->forward(xn.transpose(1, 2));
                conv_out = conv_out.slice(2, 0, n_sel);
                xn = xn + conv_out.transpose(1, 2);
                std::cout << "\n    after conv+residual: max=" << xn.abs().max().item<float>();
                auto gp = block->ssm->gate_proj->forward(xn).chunk(2, -1);
                std::cout << "\n    gate: max=" << torch::sigmoid(gp[0]).abs().max().item<float>();
                auto Bv = block->ssm->B_proj->forward(xn * torch::sigmoid(gp[1]));
                auto C = block->ssm->C_proj->forward(xn);
                auto dt = torch::softplus(block->ssm->dt_proj->forward(xn));
                auto a = -torch::exp(block->ssm->A_log);
                std::cout << "\n    A_log: " << block->ssm->A_log.squeeze();
                std::cout << "\n    dt[0,0]: " << dt[0, 0];
                std::cout << "\n    Bv[0,0]: " << Bv[0, 0];
                std::cout << "\n    C[0,0]: " << C[0, 0];
            }
        }
        std::cout << "\n";
    }

    // Also compare combined_sparse
    if (expected.count("combined_sparse")) {
        std::cout << "\n  --- Combined comparison ---\n";
        // We can't easily reproduce the combined without running the full block forward,
        // but we saved the Python combined_sparse for reference.
        std::cout << "  (combined_sparse is in the fixture for reference)\n";
    }

    std::cout << "\n=== Pathway Subset Test: " << (all_pass ? "PASS" : "FAIL") << " ===\n";
    return all_pass ? 0 : 1;
}

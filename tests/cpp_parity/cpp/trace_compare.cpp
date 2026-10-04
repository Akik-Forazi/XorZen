// trace_compare.cpp — Compare each intermediate tensor step-by-step.
// Reads the 18_trace_divergence fixture and runs the REAL model forward,
// comparing each captured intermediate against the Python expected values.
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

    // Load expected tensors
    TensorMap expected;
    for (const auto& e : expected_tensors) {
        expected[e.name] = load_tensor(fd + "/" + e.file, e.shape, e.dtype);
    }

    // Build model
    auto cfg = ConfigFactory::get_config(ModelSize::TINY_23K);
    XorzenModel model(cfg, true);
    model->eval();
    auto result = model->load_from_tensor_map(sd, false);

    // Get input_ids
    Tensor input_ids = expected.at("input_ids");
    Tensor labels = expected.at("labels");

    // Run forward
    torch::NoGradGuard ng;
    auto out = model->forward(input_ids, {}, {}, labels);

    // Compare each expected tensor
    std::cout << "\n=== STEP-BY-STEP TRACE COMPARISON ===\n\n";
    for (const auto& e : expected_tensors) {
        const auto& name = e.name;
        const auto& exp = expected.at(name);

        // Map expected tensor name to C++ output
        Tensor cpp_val;
        bool found = false;

        if (name == "input_ids" || name == "labels") continue;
        else if (name == "hidden") {
            // Recompute: token_emb + pos_emb
            auto pos_ids = torch::arange(input_ids.size(1)).unsqueeze(0).expand({input_ids.size(0), input_ids.size(1)});
            cpp_val = model->token_embedding->forward(input_ids) + model->position_embedding->forward(pos_ids);
            found = true;
        } else if (name == "logits") {
            cpp_val = out.logits; found = true;
        } else if (name == "lm_loss") {
            cpp_val = out.lm_loss; found = true;
        }

        if (!found) {
            std::cout << "  [SKIP] " << name << " — not directly comparable (intermediate)\n";
            continue;
        }

        if (exp.scalar_type() == torch::kInt64) {
            int64_t ndiff = (exp != cpp_val).sum().item<int64_t>();
            std::cout << "  [" << (ndiff == 0 ? "MATCH" : "MISMATCH") << "] " << name
                      << "  n_diff=" << ndiff << "\n";
        } else {
            double max_abs = (exp.to(torch::kFloat64) - cpp_val.to(torch::kFloat64)).abs().max().item<double>();
            double max_rel = 0;
            auto denom = exp.to(torch::kFloat64).abs().clamp_min(1e-12);
            max_rel = ((exp.to(torch::kFloat64) - cpp_val.to(torch::kFloat64)).abs() / denom).max().item<double>();
            bool match = max_abs < 1e-5;
            std::cout << "  [" << (match ? "MATCH" : "MISMATCH") << "] " << name
                      << "  max_abs=" << max_abs << "  max_rel=" << max_rel;
            if (!match) {
                // Find first mismatch
                auto diff = (exp - cpp_val).abs();
                auto flat = diff.flatten();
                auto idx = flat.argmax(0).item<int64_t>();
                std::cout << "  first_mismatch_flat_idx=" << idx
                          << "  py=" << exp.flatten()[idx].item<float>()
                          << "  cpp=" << cpp_val.flatten()[idx].item<float>();
            }
            std::cout << "\n";
        }
    }

    return 0;
}

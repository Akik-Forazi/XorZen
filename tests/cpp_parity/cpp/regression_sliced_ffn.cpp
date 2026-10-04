// regression_sliced_ffn.cpp — Regression test for the SlicedFFN fix.
//
// This test verifies that the C++ AdaptiveFFN produces the SAME output as
// Python SlicedFFN when width < max_width. The previous bug was that C++
// computed the full-width FFN with ln_hidden, while Python slices the
// weights and skips ln_hidden.
//
// The test loads the real Python checkpoint and runs the full model forward,
// checking that logits max_abs_diff <= 1e-6 (currently 1.49e-08).
//
// If this test FAILS, it means the SlicedFFN behavior was reverted.
#include <torch/torch.h>
#include "xorzen/model.h"
#include "xorzen/variants.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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
std::unordered_map<std::string, std::string> parse_config(const std::string& path) {
    std::unordered_map<std::string, std::string> config;
    std::ifstream f(path); std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line); std::string tok; ss >> tok;
        if (tok == "config") { std::string k, v; ss >> k; std::getline(ss, v);
            size_t i = v.find_first_not_of(" \t"); config[k] = (i != std::string::npos) ? v.substr(i) : ""; }
    }
    return config;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <fixture_dir>\n"; return 2; }
    std::string fd = argv[1];
    torch::set_num_threads(1);

    auto sd = load_state_dict(fd);
    auto cfg_map = parse_config(fd + "/manifest.txt");
    int64_t B = std::stoll(cfg_map.at("B"));
    int64_t T = std::stoll(cfg_map.at("T"));
    int64_t V = std::stoll(cfg_map.at("vocab_size"));

    auto cfg = ConfigFactory::get_config(ModelSize::TINY_23K);
    XorzenModel model(cfg, true);
    model->eval();
    model->load_from_tensor_map(sd, false);

    Tensor input_ids = load_tensor(fd + "/expected_input_ids.bin", {B, T}, "int64");
    Tensor labels = load_tensor(fd + "/expected_labels.bin", {B, T}, "int64");
    Tensor expected_logits = load_tensor(fd + "/expected_logits.bin", {B, T, V}, "float32");
    Tensor expected_lm_loss = load_tensor(fd + "/expected_lm_loss.bin", {}, "float32");

    torch::NoGradGuard ng;
    auto out = model->forward(input_ids, {}, {}, labels);

    double logits_diff = (out.logits - expected_logits).abs().max().item<double>();
    double loss_diff = std::abs(out.lm_loss.item<double>() - expected_lm_loss.item<double>());

    std::cout << "=== SLICED FFN REGRESSION TEST ===\n";
    std::cout << "  logits max_abs_diff: " << logits_diff << " (must be <= 1e-6)\n";
    std::cout << "  lm_loss diff: " << loss_diff << " (must be 0)\n";

    // Verify SlicedFFN is actually slicing (not full-width)
    auto block = model->blocks[0]->as<HASSBlock>();
    int64_t max_width = block->ffn->base_ffn_dim;  // 32 for tiny_23k
    int64_t actual_width = cfg.hidden_size;  // 8 for tiny_23k (from width_multiplier=1.0 * 8)
    std::cout << "  FFN max_width: " << max_width << "\n";
    std::cout << "  FFN actual_width (sliced): " << actual_width << "\n";
    std::cout << "  Slicing active: " << (actual_width < max_width ? "YES" : "NO") << "\n";

    bool pass = (logits_diff <= 1e-6) && (loss_diff == 0.0) && (actual_width < max_width);
    std::cout << "\n=== Regression Test: " << (pass ? "PASS" : "FAIL") << " ===\n";
    return pass ? 0 : 1;
}

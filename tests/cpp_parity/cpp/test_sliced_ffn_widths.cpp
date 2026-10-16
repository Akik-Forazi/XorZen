// tests/cpp_parity/cpp/test_sliced_ffn_widths.cpp
// C++ parity test for SlicedFFN at multiple widths (narrow, intermediate, max, mixed).
//
// This test is SELF-CONTAINED — it implements SlicedFFN directly (matching
// the Python xorzen.model.components.sliced_ffn.SlicedFFN) so it can be
// compiled without the full C++ model library. The implementation matches
// xorzen.cpp/src/model/hass_block.cpp SlicedFFNImpl.
//
// Loads the Python-generated fixtures from tests/cpp_parity/fixtures/08_sliced_ffn_*
// and verifies that the C++ SlicedFFN produces matching outputs.
//
// The mixed-width fixture is the critical test: it would FAIL if the C++
// implementation silently uses max width for all tokens.
//
// Usage:
//   ./test_sliced_ffn_widths <fixtures_dir> <output_dir>

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <set>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────
// SlicedFFN — self-contained implementation matching Python SlicedFFN
// ─────────────────────────────────────────────────────────────────────
// This mirrors xorzen.cpp/src/model/hass_block.cpp SlicedFFNImpl but is
// self-contained so it can be compiled without the full model library.
struct SlicedFFN : torch::nn::Module {
    int64_t hidden_dim;
    int64_t max_width;
    std::string activation;
    std::vector<int64_t> width_choices;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::LayerNorm ln_input{nullptr}, ln_hidden{nullptr};

    SlicedFFN(int64_t hidden, int64_t max_w, std::string act,
              std::vector<int64_t> choices)
        : hidden_dim(hidden), max_width(max_w), activation(std::move(act)) {
        if (choices.empty()) {
            choices = {std::max(int64_t(1), max_w / 4),
                       std::max(int64_t(1), max_w / 2),
                       std::max(int64_t(1), 3 * max_w / 4),
                       max_w};
            std::sort(choices.begin(), choices.end());
            choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
        }
        for (auto& w : choices) w = std::max(int64_t(1), std::min(w, max_w));
        width_choices = choices;

        fc1 = register_module("fc1", torch::nn::Linear(hidden_dim, max_width));
        fc2 = register_module("fc2", torch::nn::Linear(max_width, hidden_dim));
        ln_input = register_module("ln_input", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        ln_hidden = register_module("ln_hidden", torch::nn::LayerNorm(torch::nn::LayerNormOptions({max_width})));
    }

    torch::Tensor apply_activation(const torch::Tensor& h) {
        if (activation == "silu") return torch::silu(h);
        if (activation == "relu") return torch::relu(h);
        return torch::gelu(h);
    }

    // forward_single_width: matches Python SlicedFFN._forward_single_width
    torch::Tensor forward_single_width(const torch::Tensor& x_norm, int64_t width) {
        int64_t w = std::max(int64_t(1), std::min(width, max_width));
        auto fc1_w = fc1->weight.slice(0, 0, w);   // [w, H]
        auto fc1_b = fc1->bias.slice(0, 0, w);     // [w]
        auto fc2_w = fc2->weight.slice(1, 0, w);   // [H, w]
        auto fc2_b = fc2->bias;                     // [H]

        auto hidden = torch::linear(x_norm, fc1_w, fc1_b);
        hidden = apply_activation(hidden);
        // Skip ln_hidden when w < max_width (matches Python sliced_ffn.py:195-196)
        if (w == max_width) {
            hidden = ln_hidden->forward(hidden);
        }
        auto output = torch::linear(hidden, fc2_w, fc2_b);
        return output;
    }

    // forward_per_token_width: matches Python SlicedFFN._forward_per_token_width
    torch::Tensor forward_per_token_width(const torch::Tensor& x_norm,
                                           const torch::Tensor& width_idx) {
        // width_idx: [B, T] long tensor in [0, num_widths)
        auto opts = torch::TensorOptions().dtype(torch::kLong).device(x_norm.device());
        auto widths_tensor = torch::tensor(width_choices, opts);
        auto actual_widths = widths_tensor.index_select(0, width_idx.reshape(-1));

        auto output = torch::zeros_like(x_norm);
        auto x_flat = x_norm.reshape({-1, hidden_dim});

        // Get unique widths via std::set (torch::unique not in C++ API)
        auto widths_cpu = actual_widths.cpu().contiguous();
        auto widths_acc = widths_cpu.accessor<int64_t, 1>();
        std::set<int64_t> width_set;
        for (int64_t i = 0; i < widths_acc.size(0); ++i) {
            width_set.insert(widths_acc[i]);
        }

        for (int64_t w : width_set) {
            auto mask = (actual_widths == w);
            int64_t n_sel = mask.sum().item<int64_t>();
            if (n_sel == 0) continue;
            auto idx = mask.nonzero().squeeze(-1);
            auto x_sel = x_flat.index_select(0, idx);
            int64_t ww = std::max(int64_t(1), std::min(w, max_width));
            auto fc1_w = fc1->weight.slice(0, 0, ww);
            auto fc1_b = fc1->bias.slice(0, 0, ww);
            auto fc2_w = fc2->weight.slice(1, 0, ww);
            auto fc2_b = fc2->bias;
            auto hidden = torch::linear(x_sel, fc1_w, fc1_b);
            hidden = apply_activation(hidden);
            // Per-token path: NEVER apply ln_hidden (matches Python
            // _forward_single_width_flat which skips LN for all widths).
            // This is different from forward_single_width which DOES apply
            // ln_hidden when w == max_width.
            auto y_sel = torch::linear(hidden, fc2_w, fc2_b);
            output.view({-1, hidden_dim}).index_add_(0, idx, y_sel);
        }
        return output;
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& width_idx = {}) {
        auto xn = ln_input->forward(x);
        if (!width_idx.defined() || width_idx.numel() == 0) {
            return forward_single_width(xn, max_width);
        }
        return forward_per_token_width(xn, width_idx);
    }
};

// ─────────────────────────────────────────────────────────────────────
// Helper: read a .bin file as a torch tensor with given shape
// ─────────────────────────────────────────────────────────────────────
static torch::Tensor read_tensor(const std::string& path, const std::vector<int64_t>& shape,
                                  torch::ScalarType dtype = torch::kFloat32) {
    auto opts = torch::TensorOptions().dtype(dtype);
    auto tensor = torch::empty(shape, opts);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        std::exit(1);
    }
    f.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.nbytes());
    return tensor;
}

static std::map<std::string, std::string> parse_manifest(const std::string& dir) {
    std::map<std::string, std::string> kv;
    std::ifstream f(dir + "/manifest.txt");
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find('=');
        if (pos != std::string::npos) {
            kv[line.substr(0, pos)] = line.substr(pos + 1);
        }
    }
    return kv;
}

static std::vector<int64_t> parse_shape(const std::string& s) {
    std::vector<int64_t> shape;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) shape.push_back(std::stoll(item));
    }
    return shape;
}

// ─────────────────────────────────────────────────────────────────────
// Test one fixture
// ─────────────────────────────────────────────────────────────────────
struct TestResult {
    std::string name;
    bool pass;
    double max_abs_diff;
    std::string detail;
};

static TestResult test_fixture(const std::string& fixture_dir) {
    TestResult result;
    auto manifest = parse_manifest(fixture_dir);
    result.name = fs::path(fixture_dir).filename().string();

    int64_t hidden_dim = std::stoll(manifest["hidden_size"]);
    int64_t max_width = std::stoll(manifest["max_width"]);

    std::vector<int64_t> width_choices;
    {
        std::stringstream ss(manifest["width_choices"]);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) width_choices.push_back(std::stoll(item));
        }
    }

    auto x = read_tensor(fixture_dir + "/input_x.bin", parse_shape(manifest["shape_x"]));

    auto fc1_w = read_tensor(fixture_dir + "/param_fc1_weight.bin", parse_shape(manifest["shape_fc1_weight"]));
    auto fc1_b = read_tensor(fixture_dir + "/param_fc1_bias.bin", parse_shape(manifest["shape_fc1_bias"]));
    auto fc2_w = read_tensor(fixture_dir + "/param_fc2_weight.bin", parse_shape(manifest["shape_fc2_weight"]));
    auto fc2_b = read_tensor(fixture_dir + "/param_fc2_bias.bin", parse_shape(manifest["shape_fc2_bias"]));
    auto ln_iw = read_tensor(fixture_dir + "/param_ln_input_weight.bin", parse_shape(manifest["shape_ln_input_weight"]));
    auto ln_ib = read_tensor(fixture_dir + "/param_ln_input_bias.bin", parse_shape(manifest["shape_ln_input_bias"]));
    auto ln_hw = read_tensor(fixture_dir + "/param_ln_hidden_weight.bin", parse_shape(manifest["shape_ln_hidden_weight"]));
    auto ln_hb = read_tensor(fixture_dir + "/param_ln_hidden_bias.bin", parse_shape(manifest["shape_ln_hidden_bias"]));

    auto y_expected = read_tensor(fixture_dir + "/expected_y.bin", parse_shape(manifest["shape_y"]));

    SlicedFFN ffn(hidden_dim, max_width, "gelu", width_choices);
    {
        torch::NoGradGuard ng;
        ffn.fc1->weight.data().copy_(fc1_w);
        ffn.fc1->bias.data().copy_(fc1_b);
        ffn.fc2->weight.data().copy_(fc2_w);
        ffn.fc2->bias.data().copy_(fc2_b);
        ffn.ln_input->weight.data().copy_(ln_iw);
        ffn.ln_input->bias.data().copy_(ln_ib);
        ffn.ln_hidden->weight.data().copy_(ln_hw);
        ffn.ln_hidden->bias.data().copy_(ln_hb);
    }
    ffn.eval();

    torch::Tensor y_cpp;
    torch::NoGradGuard ng;

    if (result.name.find("_mixed") != std::string::npos) {
        int64_t B = parse_shape(manifest["shape_x"])[0];
        int64_t T = parse_shape(manifest["shape_x"])[1];
        auto width_idx = read_tensor(fixture_dir + "/input_width_idx.bin", {B, T}, torch::kInt64);
        y_cpp = ffn.forward(x, width_idx);
    } else {
        int64_t width = std::stoll(manifest["width_used"]);
        auto xn = ffn.ln_input->forward(x);
        y_cpp = ffn.forward_single_width(xn, width);
    }

    double max_abs = (y_cpp - y_expected).abs().max().item<double>();
    double tolerance = 1e-4;
    if (manifest.count("tolerance_max_abs")) {
        tolerance = std::stod(manifest["tolerance_max_abs"]);
    }

    result.max_abs_diff = max_abs;
    result.pass = (max_abs < tolerance);
    result.detail = "max_abs=" + std::to_string(max_abs) + " tol=" + std::to_string(tolerance);
    return result;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <fixtures_dir> <output_dir>\n";
        return 2;
    }
    std::string fixtures_dir = argv[1];
    std::string output_dir = argv[2];
    fs::create_directories(output_dir);

    torch::set_num_threads(1);

    std::vector<std::string> fixture_names = {
        "08_sliced_ffn_narrow",
        "08_sliced_ffn_intermediate",
        "08_sliced_ffn_max",
        "08_sliced_ffn_mixed",
    };

    std::vector<TestResult> results;
    int passed = 0, failed = 0;

    std::cout << "=== SlicedFFN Multi-Width Parity Test ===\n\n";

    for (const auto& name : fixture_names) {
        std::string dir = fixtures_dir + "/" + name;
        if (!fs::exists(dir)) {
            std::cout << "  [SKIP] " << name << " (fixture not found)\n";
            continue;
        }
        try {
            auto result = test_fixture(dir);
            results.push_back(result);
            if (result.pass) {
                std::cout << "  [PASS] " << result.name << " — " << result.detail << "\n";
                passed++;
            } else {
                std::cout << "  [FAIL] " << result.name << " — " << result.detail << "\n";
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] " << name << " — exception: " << e.what() << "\n";
            failed++;
        }
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "TOTAL: " << (passed + failed) << "  PASSED: " << passed << "  FAILED: " << failed << "\n";

    std::ofstream rep(output_dir + "/sliced_ffn_widths_report.txt");
    rep << "SlicedFFN Multi-Width Parity Report\n";
    rep << "====================================\n\n";
    for (const auto& r : results) {
        rep << (r.pass ? "PASS" : "FAIL") << " " << r.name << " — " << r.detail << "\n";
    }
    rep << "\nTOTAL: " << (passed + failed) << "  PASSED: " << passed << "  FAILED: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}

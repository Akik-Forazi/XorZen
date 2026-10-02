#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include "xorzen/model.h"
#include "xorzen/routing.h"

namespace fs = std::filesystem;

torch::Tensor load_tensor(const std::string& filename, torch::IntArrayRef shape, torch::ScalarType dtype) {
    std::string path = "tests/parity/" + filename;
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + path);
    auto t = torch::empty(shape, dtype);
    file.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
    return t;
}

int main() {
    try {
        xorzen::ModelConfig config;
        config.hidden_size = 768;
        config.cot_dim = 64;
        config.cot_components = 2;
        config.expert_count = 8;
        config.width_choices = std::vector<int64_t>{0, 1, 2}; // Use int64_t
        config.max_depth = 2;
        config.normalize(); 
        
        xorzen::AdaptiveRouter router(config);
        
        // 1. Load weights
        router->load_weights("C:/Users/akikf/programing/ai/xorzen_0.2.4/tests/parity/weights");
        router->eval();

        // 2. Load golden data
        torch::Tensor hidden = load_tensor("router_hidden.bin", {1, 16, 768}, torch::kFloat32);
        torch::Tensor cot = load_tensor("router_cot.bin", {1, 16, 128}, torch::kFloat32);
        torch::Tensor y_golden = load_tensor("router_y.bin", {1, 16, 2}, torch::kFloat32);
        
        // 3. Run C++ implementation
        auto decision = router->forward(hidden, cot, true, -1); // deterministic=true
        torch::Tensor y_cpp = decision.expert_weights;
        
        // 4. Compare
        bool success = torch::allclose(y_golden, y_cpp, 1e-4, 1e-4);
        
        if (success) {
            std::cout << "[PASS] AdaptiveRouter Numeric Parity" << std::endl;
            return 0;
        } else {
            std::cerr << "[FAIL] AdaptiveRouter Numeric Parity" << std::endl;
            std::cerr << "Max diff: " << (y_golden - y_cpp).abs().max().item<float>() << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

#include "xorzen/optimized/simd_ops.h" 

namespace fs = std::filesystem;

// Helper to load binary tensor
torch::Tensor load_tensor(const std::string& filename, torch::IntArrayRef shape) {
    std::vector<std::string> paths_to_try = {
        filename,
        "tests/parity/" + filename,
        "../../tests/parity/" + filename,
        "../tests/parity/" + filename
    };
    
    std::cerr << "Looking for: " << filename << " in current dir: " << fs::current_path() << std::endl;
    
    for (const auto& path : paths_to_try) {
        std::cerr << "  Trying: " << path << std::endl;
        if (fs::exists(path)) {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) continue;
            
            std::cerr << "  Found: " << path << std::endl;
            auto t = torch::empty(shape, torch::kFloat32);
            file.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
            return t;
        }
    }
    
    throw std::runtime_error("Could not find file: " + filename);
}

int main() {
    try {
        // 1. Load golden data
        torch::Tensor x = load_tensor("rmsnorm_x.bin", {2, 128, 768});
        torch::Tensor weight = load_tensor("rmsnorm_weight.bin", {768});
        torch::Tensor y_golden = load_tensor("rmsnorm_y.bin", {2, 128, 768});
        
        // 2. Run C++ implementation
        torch::Tensor y_cpp = xorzen::optimized::rmsnorm_simd(x, weight);
        
        // 3. Compare
        bool success = torch::allclose(y_golden, y_cpp, 1e-4, 1e-4);
        
        if (success) {
            std::cout << "[PASS] RMSNorm Numeric Parity" << std::endl;
            return 0;
        } else {
            std::cerr << "[FAIL] RMSNorm Numeric Parity" << std::endl;
            std::cerr << "Max diff: " << (y_golden - y_cpp).abs().max().item<float>() << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

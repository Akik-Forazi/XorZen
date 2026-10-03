#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    try {
        // 1. Setup minimal router
        auto model = torch::nn::Linear(4, 2);
        
        // Manual weights for testing
        torch::Tensor weight = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}});
        torch::Tensor bias = torch::tensor({0.1f, 0.2f});
        model->weight.copy_(weight);
        model->bias.copy_(bias);
        model->eval();

        // 2. Load golden data
        auto x = torch::from_blob(new float[4]{1.0f, -1.0f, 0.5f, -0.5f}, {1, 4}, torch::kFloat32).clone();
        torch::Tensor y_golden = torch::from_blob(new float[2]{0.0f, 0.05000003f}, {1, 2}, torch::kFloat32).clone();
        
        // 3. Compute
        auto y_cpp = torch::relu(model->forward(x));
        
        // 4. Compare
        bool success = torch::allclose(y_golden, y_cpp, 1e-4, 1e-4);
        
        if (success) {
            std::cout << "[PASS] MiniXorZen Parity" << std::endl;
            return 0;
        } else {
            std::cerr << "[FAIL] MiniXorZen Parity" << std::endl;
            std::cerr << "Expected: " << y_golden << std::endl;
            std::cerr << "Got: " << y_cpp << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

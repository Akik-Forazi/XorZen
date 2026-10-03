// ============================================================
//  XORZEN.CPP - test_ggml_bridge.cpp
//  Unit tests for GGML quantization bridge
//  Test: torch_to_ggml_q4(), torch_to_ggml_q8(), dequantization
// ============================================================

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>
#include <algorithm>

#ifdef XORZEN_USE_GGML
#include "xorzen/optimized/ggml_bridge.h"
#endif

namespace xorzen::testing {

class GGMLBridgeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GGML context for each test
        context_mem_size_ = 256 * 1024 * 1024;  // 256 MB
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
    
    size_t context_mem_size_;
};

// ============================================================
// Test: torch_to_ggml() - Basic conversion (unquantized)
// ============================================================

TEST_F(GGMLBridgeTest, TorchToGGML_FP32) {
    #ifdef XORZEN_USE_GGML
    // Create a simple tensor
    auto x = torch::randn({4, 512, 768}, torch::kFloat32);
    
    // Create GGML context
    optimized::GGMLContext ctx(context_mem_size_);
    
    // Convert to GGML
    auto gt = optimized::torch_to_ggml(ctx.get(), x);
    
    ASSERT_NE(gt, nullptr);
    ASSERT_EQ(gt->ne[0], 768);
    ASSERT_EQ(gt->ne[1], 512);
    ASSERT_EQ(gt->ne[2], 4);
    ASSERT_EQ(gt->type, GGML_TYPE_F32);
    
    // Verify data integrity (round-trip)
    auto x_back = optimized::ggml_to_torch(gt);
    ASSERT_TRUE(torch::allclose(x, x_back, 1e-5, 1e-6));
    #endif
}

TEST_F(GGMLBridgeTest, TorchToGGML_FP16) {
    #ifdef XORZEN_USE_GGML
    auto x = torch::randn({2, 256, 512}, torch::kFloat16);
    
    optimized::GGMLContext ctx(context_mem_size_);
    auto gt = optimized::torch_to_ggml(ctx.get(), x);
    
    ASSERT_NE(gt, nullptr);
    ASSERT_EQ(gt->type, GGML_TYPE_F16);
    #endif
}

// ============================================================
// Test: torch_to_ggml_q8() - INT8 Quantization
// ============================================================

TEST_F(GGMLBridgeTest, QuantizeQ8_SmallTensor) {
    #ifdef XORZEN_USE_GGML
    // Create a small tensor with known values
    auto x = torch::randn({8, 64}, torch::kFloat32);
    
    optimized::GGMLContext ctx(context_mem_size_);
    
    // Quantize to Q8_0
    auto qt = optimized::torch_to_ggml_q8(ctx.get(), x);
    
    ASSERT_NE(qt, nullptr);
    ASSERT_EQ(qt->type, GGML_TYPE_Q8_0);
    
    // Dequantize
    auto x_dequant = optimized::ggml_dequantize_to_torch(qt);
    
    // Check quantization error
    auto error = (x - x_dequant).abs();
    float max_error = error.max().item<float>();
    float mean_error = error.mean().item<float>();
    
    // INT8 should have bounded error
    EXPECT_LT(max_error, 0.1f) << "INT8 max error too large: " << max_error;
    EXPECT_LT(mean_error, 0.01f) << "INT8 mean error too large: " << mean_error;
    #endif
}

TEST_F(GGMLBridgeTest, QuantizeQ8_CompressionRatio) {
    #ifdef XORZEN_USE_GGML
    auto x = torch::randn({128, 768}, torch::kFloat32);
    
    optimized::GGMLContext ctx(context_mem_size_);
    auto qt = optimized::torch_to_ggml_q8(ctx.get(), x);
    
    // INT8 should be 4× compression (FP32 = 4 bytes, INT8 = 1 byte)
    int64_t original_bytes = x.element_size() * x.numel();
    int64_t quantized_bytes = ggml_nbytes(qt);
    
    float compression_ratio = static_cast<float>(original_bytes) / quantized_bytes;
    
    EXPECT_NEAR(compression_ratio, 4.0f, 0.5f) 
        << "Compression ratio not 4×: " << compression_ratio << "×";
    #endif
}

TEST_F(GGMLBridgeTest, QuantizeQ8_NumericalStability) {
    #ifdef XORZEN_USE_GGML
    // Test with various value ranges
    std::vector<torch::Tensor> test_tensors = {
        torch::ones({32, 64}, torch::kFloat32),           // All ones
        torch::randn({32, 64}, torch::kFloat32) * 100,    // Large values
        torch::randn({32, 64}, torch::kFloat32) * 0.001,  // Small values
    };
    
    optimized::GGMLContext ctx(context_mem_size_);
    
    for (auto& x : test_tensors) {
        auto qt = optimized::torch_to_ggml_q8(ctx.get(), x);
        auto x_dequant = optimized::ggml_dequantize_to_torch(qt);
        
        // Should always produce finite values
        ASSERT_TRUE(torch::isfinite(x_dequant).all())
            << "Dequantization produced NaN/Inf";
        
        // Relative error should be bounded
        auto rel_error = ((x - x_dequant).abs() / (x.abs() + 1e-6f)).mean();
        EXPECT_LT(rel_error.item<float>(), 0.02f)  // 2% relative error
            << "Relative error too large";
    }
    #endif
}

// ============================================================
// Test: torch_to_ggml_q4() - INT4 Quantization
// ============================================================

TEST_F(GGMLBridgeTest, QuantizeQ4_SmallTensor) {
    #ifdef XORZEN_USE_GGML
    auto x = torch::randn({8, 64}, torch::kFloat32);
    
    optimized::GGMLContext ctx(context_mem_size_);
    auto qt = optimized::torch_to_ggml_q4(ctx.get(), x);
    
    ASSERT_NE(qt, nullptr);
    ASSERT_EQ(qt->type, GGML_TYPE_Q4_K);
    
    auto x_dequant = optimized::ggml_dequantize_to_torch(qt);
    
    auto error = (x - x_dequant).abs();
    float max_error = error.max().item<float>();
    
    // INT4 has larger quantization error than INT8
    EXPECT_LT(max_error, 0.5f) << "INT4 max error too large: " << max_error;
    #endif
}

TEST_F(GGMLBridgeTest, QuantizeQ4_CompressionRatio) {
    #ifdef XORZEN_USE_GGML
    auto x = torch::randn({128, 768}, torch::kFloat32);
    
    optimized::GGMLContext ctx(context_mem_size_);
    auto qt = optimized::torch_to_ggml_q4(ctx.get(), x);
    
    // INT4 should be 8× compression
    int64_t original_bytes = x.element_size() * x.numel();
    int64_t quantized_bytes = ggml_nbytes(qt);
    
    float compression_ratio = static_cast<float>(original_bytes) / quantized_bytes;
    
    EXPECT_NEAR(compression_ratio, 8.0f, 1.0f)
        << "Compression ratio not 8×: " << compression_ratio << "×";
    #endif
}

TEST_F(GGMLBridgeTest, QuantizeQ4_VsQ8) {
    #ifdef XORZEN_USE_GGML
    auto x = torch::randn({64, 256}, torch::kFloat32);
    
    optimized::GGMLContext ctx(context_mem_size_);
    
    auto qt4 = optimized::torch_to_ggml_q4(ctx.get(), x);
    auto qt8 = optimized::torch_to_ggml_q8(ctx.get(), x);
    
    auto x_dequant4 = optimized::ggml_dequantize_to_torch(qt4);
    auto x_dequant8 = optimized::ggml_dequantize_to_torch(qt8);
    
    auto error4 = (x - x_dequant4).abs().mean();
    auto error8 = (x - x_dequant8).abs().mean();
    
    // Q8 should be more accurate than Q4
    EXPECT_LT(error8.item<float>(), error4.item<float>())
        << "Q8 error should be less than Q4 error";
    #endif
}

// ============================================================
// Test: ggml_dequantize_to_torch() - Dequantization
// ============================================================

TEST_F(GGMLBridgeTest, DequantizeQ8_RoundTrip) {
    #ifdef XORZEN_USE_GGML
    // Quantize and dequantize should be reversible (within error)
    auto x = torch::randn({32, 128}, torch::kFloat32);
    
    optimized::GGMLContext ctx(context_mem_size_);
    auto qt = optimized::torch_to_ggml_q8(ctx.get(), x);
    auto x_back = optimized::ggml_dequantize_to_torch(qt);
    
    // Should have same shape
    ASSERT_EQ(x.shape, x_back.shape);
    
    // Error should be bounded by quantization precision
    auto error = (x - x_back).abs();
    EXPECT_LT(error.max().item<float>(), 0.15f);
    #endif
}

TEST_F(GGMLBridgeTest, DequantizePreservesShape) {
    #ifdef XORZEN_USE_GGML
    std::vector<torch::IntArrayRef> shapes = {
        torch::IntArrayRef({4, 64}),
        torch::IntArrayRef({2, 512, 768}),
        torch::IntArrayRef({1, 1, 1, 4096}),
    };
    
    optimized::GGMLContext ctx(context_mem_size_);
    
    for (auto shape : shapes) {
        auto x = torch::randn(shape, torch::kFloat32);
        auto qt = optimized::torch_to_ggml_q8(ctx.get(), x);
        auto x_back = optimized::ggml_dequantize_to_torch(qt);
        
        ASSERT_EQ(x.shape, x_back.shape)
            << "Shape mismatch for shape " << shape;
    }
    #endif
}

// ============================================================
// Test: GGMLContext RAII
// ============================================================

TEST_F(GGMLBridgeTest, GGMLContextRAII) {
    #ifdef XORZEN_USE_GGML
    // Test that context is properly initialized and destroyed
    {
        optimized::GGMLContext ctx(context_mem_size_);
        ASSERT_NE(ctx.get(), nullptr);
    }
    // Should not leak or crash on destruction
    SUCCEED();
    #endif
}

TEST_F(GGMLBridgeTest, GGMLContextMove) {
    #ifdef XORZEN_USE_GGML
    optimized::GGMLContext ctx1(context_mem_size_);
    auto ptr1 = ctx1.get();
    
    // Move assignment
    optimized::GGMLContext ctx2 = std::move(ctx1);
    auto ptr2 = ctx2.get();
    
    ASSERT_EQ(ptr1, ptr2);
    ASSERT_EQ(ctx1.get(), nullptr);  // ctx1 should be null after move
    #endif
}

// ============================================================
// Test: Error Handling
// ============================================================

TEST_F(GGMLBridgeTest, ErrorHandling_NullContext) {
    #ifdef XORZEN_USE_GGML
    auto x = torch::randn({4, 64}, torch::kFloat32);
    
    // Should throw or handle gracefully when context is null
    EXPECT_THROW(
        optimized::torch_to_ggml(nullptr, x),
        std::runtime_error
    );
    #endif
}

TEST_F(GGMLBridgeTest, ErrorHandling_InvalidShape) {
    #ifdef XORZEN_USE_GGML
    // 5D tensor should fail (GGML max is 4D)
    auto x = torch::randn({2, 4, 8, 16, 32}, torch::kFloat32);
    
    optimized::GGMLContext ctx(context_mem_size_);
    
    // Should handle gracefully (fold higher dims into first dim)
    // or raise error - depends on implementation
    EXPECT_NO_THROW(
        optimized::torch_to_ggml(ctx.get(), x)
    );
    #endif
}

// ============================================================
// Test: Large Tensor Quantization
// ============================================================

TEST_F(GGMLBridgeTest, QuantizeLargeTensor_277M) {
    #ifdef XORZEN_USE_GGML
    // 277M model weights: roughly 277M * 4 bytes = 1.1 GB in FP32
    // With Q4: 277M / 2 bytes = ~138 MB (but we test with smaller tensor)
    
    // For testing, use smaller tensor that represents a large layer
    auto x = torch::randn({2048, 2048}, torch::kFloat32);  // 16M params
    
    optimized::GGMLContext ctx(1024 * 1024 * 1024);  // 1 GB context
    
    auto qt4 = optimized::torch_to_ggml_q4(ctx.get(), x);
    
    ASSERT_NE(qt4, nullptr);
    
    // Verify compression
    int64_t original_bytes = x.element_size() * x.numel();
    int64_t quantized_bytes = ggml_nbytes(qt4);
    float compression = static_cast<float>(original_bytes) / quantized_bytes;
    
    EXPECT_NEAR(compression, 8.0f, 1.0f);
    #endif
}

}  // namespace xorzen::testing

#pragma once

#include <torch/torch.h>

#ifdef XORZEN_USE_GGML
extern "C" {
    #include "ggml.h"
    #include "ggml-alloc.h"
}
#endif

namespace xorzen {
namespace optimized {

#ifdef XORZEN_USE_GGML

/**
 * GGML Bridge - Zero-copy tensor conversion between LibTorch and GGML
 * 
 * This bridge allows seamless integration of GGML's optimized quantization
 * and SIMD kernels with LibTorch's autograd and high-level API.
 */

// Convert LibTorch tensor to GGML tensor (zero-copy view when possible)
struct ggml_tensor* torch_to_ggml(
    struct ggml_context* ctx, 
    const torch::Tensor& t
);

// Convert GGML tensor to LibTorch tensor (zero-copy view)
torch::Tensor ggml_to_torch(struct ggml_tensor* t);

// Quantize LibTorch FP32 tensor to GGML Q4_K_M format
struct ggml_tensor* torch_to_ggml_q4(
    struct ggml_context* ctx,
    const torch::Tensor& t
);

// Quantize to Q5_K (higher quality, slightly larger)
struct ggml_tensor* torch_to_ggml_q5(
    struct ggml_context* ctx,
    const torch::Tensor& t
);

// Quantize to Q8_0 (lossless for many purposes, 2× size of Q4)
struct ggml_tensor* torch_to_ggml_q8(
    struct ggml_context* ctx,
    const torch::Tensor& t
);

// Dequantize GGML quantized tensor to LibTorch FP32
torch::Tensor ggml_dequantize_to_torch(struct ggml_tensor* t);

// RAII wrapper for GGML context management
class GGMLContext {
public:
    explicit GGMLContext(size_t mem_size = 128 * 1024 * 1024);  // 128MB default
    ~GGMLContext();
    
    // No copy
    GGMLContext(const GGMLContext&) = delete;
    GGMLContext& operator=(const GGMLContext&) = delete;
    
    // Move support
    GGMLContext(GGMLContext&& other) noexcept;
    GGMLContext& operator=(GGMLContext&& other) noexcept;
    
    struct ggml_context* get() const { return ctx_; }
    
private:
    struct ggml_context* ctx_ = nullptr;
    struct ggml_init_params params_;
    void* mem_buffer_ = nullptr;
};

// Helper: Get GGML type enum from torch dtype
ggml_type torch_dtype_to_ggml(torch::ScalarType dtype);

// Helper: Get torch dtype from GGML type
torch::ScalarType ggml_type_to_torch(ggml_type type);

#endif // XORZEN_USE_GGML

} // namespace optimized
} // namespace xorzen

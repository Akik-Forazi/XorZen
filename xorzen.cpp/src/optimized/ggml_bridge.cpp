// ============================================================
//  xorzen.cpp — src/optimized/ggml_bridge.cpp
//  GGML ↔ LibTorch tensor bridge + quantization utilities
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/optimized/ggml_bridge.h"

#ifdef XORZEN_USE_GGML

#include <cstring>
#include <stdexcept>
#include <vector>

namespace xorzen {
namespace optimized {

//==============================================================================
// Type conversion helpers
//==============================================================================

ggml_type torch_dtype_to_ggml(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return GGML_TYPE_F32;
        case torch::kFloat16: return GGML_TYPE_F16;
        case torch::kInt32:   return GGML_TYPE_I32;
        case torch::kInt16:   return GGML_TYPE_I16;
        case torch::kInt8:    return GGML_TYPE_I8;
        default:
            throw std::runtime_error("Unsupported torch dtype for GGML conversion");
    }
}

torch::ScalarType ggml_type_to_torch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return torch::kFloat32;
        case GGML_TYPE_F16: return torch::kFloat16;
        case GGML_TYPE_I32: return torch::kInt32;
        case GGML_TYPE_I16: return torch::kInt16;
        case GGML_TYPE_I8:  return torch::kInt8;
        default:
            throw std::runtime_error(
                "Cannot directly convert quantized GGML type to torch - dequantize first");
    }
}

//==============================================================================
// LibTorch → GGML (unquantized)
//==============================================================================

struct ggml_tensor* torch_to_ggml(
    struct ggml_context* ctx,
    const torch::Tensor& t)
{
    if (!ctx) throw std::runtime_error("GGML context is null");

    auto tc = t.contiguous();
    ggml_type type = torch_dtype_to_ggml(tc.scalar_type());

    // GGML tensors are max 4-D; store higher dims in leading axes
    int64_t ne[4] = {1, 1, 1, 1};
    int ndim = static_cast<int>(std::min(tc.dim(), int64_t(4)));
    for (int i = 0; i < ndim; ++i) ne[i] = tc.size(i);

    struct ggml_tensor* gt = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
    if (!gt) throw std::runtime_error("Failed to allocate GGML tensor");

    memcpy(gt->data, tc.data_ptr(), ggml_nbytes(gt));
    return gt;
}

//==============================================================================
// GGML → LibTorch (unquantized)
//==============================================================================

torch::Tensor ggml_to_torch(struct ggml_tensor* gt)
{
    if (!gt) throw std::runtime_error("GGML tensor is null");
    if (ggml_is_quantized(gt->type))
        throw std::runtime_error(
            "Cannot directly convert quantized GGML tensor. "
            "Use ggml_dequantize_to_torch() instead.");

    torch::ScalarType dtype = ggml_type_to_torch(gt->type);

    std::vector<int64_t> sizes;
    for (int i = 0; i < 4; ++i)
        if (gt->ne[i] > 1) sizes.push_back(gt->ne[i]);
    if (sizes.empty()) sizes.push_back(1);

    auto t = torch::empty(sizes, torch::TensorOptions().dtype(dtype));
    memcpy(t.data_ptr(), gt->data, ggml_nbytes(gt));
    return t;
}

//==============================================================================
// Quantize LibTorch FP32 → GGML Q4_K_M
//==============================================================================

struct ggml_tensor* torch_to_ggml_q4(
    struct ggml_context* ctx,
    const torch::Tensor& t)
{
    if (!ctx) throw std::runtime_error("GGML context is null");

    auto t_f32 = t.to(torch::kFloat32).contiguous();
    auto gt_f32 = torch_to_ggml(ctx, t_f32);

    auto gt_q4 = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_K,
        gt_f32->ne[0], gt_f32->ne[1], gt_f32->ne[2], gt_f32->ne[3]);
    if (!gt_q4) throw std::runtime_error("Failed to allocate GGML Q4_K tensor");

    int64_t hist[16] = {0};
    ggml_quantize_chunk(
        GGML_TYPE_Q4_K,
        static_cast<const float*>(gt_f32->data),
        gt_q4->data,
        /*start=*/0,
        /*nrows=*/1,
        ggml_nelements(gt_f32),
        /*imatrix=*/nullptr);

    return gt_q4;
}

//==============================================================================
// Quantize LibTorch FP32 → GGML Q5_K_M
//==============================================================================

struct ggml_tensor* torch_to_ggml_q5(
    struct ggml_context* ctx,
    const torch::Tensor& t)
{
    if (!ctx) throw std::runtime_error("GGML context is null");

    auto t_f32 = t.to(torch::kFloat32).contiguous();
    auto gt_f32 = torch_to_ggml(ctx, t_f32);

    auto gt_q5 = ggml_new_tensor_4d(ctx, GGML_TYPE_Q5_K,
        gt_f32->ne[0], gt_f32->ne[1], gt_f32->ne[2], gt_f32->ne[3]);
    if (!gt_q5) throw std::runtime_error("Failed to allocate GGML Q5_K tensor");

    ggml_quantize_chunk(
        GGML_TYPE_Q5_K,
        static_cast<const float*>(gt_f32->data),
        gt_q5->data,
        0,
        1,
        ggml_nelements(gt_f32),
        nullptr);

    return gt_q5;
}

//==============================================================================
// Quantize LibTorch FP32 → GGML Q8_0
//==============================================================================

struct ggml_tensor* torch_to_ggml_q8(
    struct ggml_context* ctx,
    const torch::Tensor& t)
{
    if (!ctx) throw std::runtime_error("GGML context is null");

    auto t_f32 = t.to(torch::kFloat32).contiguous();
    auto gt_f32 = torch_to_ggml(ctx, t_f32);

    auto gt_q8 = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0,
        gt_f32->ne[0], gt_f32->ne[1], gt_f32->ne[2], gt_f32->ne[3]);
    if (!gt_q8) throw std::runtime_error("Failed to allocate GGML Q8_0 tensor");

    ggml_quantize_chunk(
        GGML_TYPE_Q8_0,
        static_cast<const float*>(gt_f32->data),
        gt_q8->data,
        0,
        1,
        ggml_nelements(gt_f32),
        nullptr);

    return gt_q8;
}

//==============================================================================
// Dequantize GGML quantized tensor → LibTorch FP32
//==============================================================================

torch::Tensor ggml_dequantize_to_torch(struct ggml_tensor* gt)
{
    if (!gt) throw std::runtime_error("GGML tensor is null");
    if (!ggml_is_quantized(gt->type))
        return ggml_to_torch(gt);

    int64_t n = ggml_nelements(gt);

    std::vector<int64_t> sizes;
    for (int i = 0; i < GGML_MAX_DIMS; ++i)
        if (gt->ne[i] > 1) sizes.push_back(gt->ne[i]);
    if (sizes.empty()) sizes.push_back(1);

    auto t = torch::empty(sizes, torch::kFloat32);
    float* out = t.data_ptr<float>();

    const struct ggml_type_traits* tt = ggml_get_type_traits(gt->type);
    if (!tt || !tt->to_float)
        throw std::runtime_error("No dequantization function for this GGML type");
    tt->to_float(gt->data, out, n);

    return t;
}

//==============================================================================
// GGMLContext RAII
//==============================================================================

GGMLContext::GGMLContext(size_t mem_size)
{
    mem_buffer_ = std::malloc(mem_size);
    if (!mem_buffer_) throw std::bad_alloc();
    params_ = ggml_init_params{
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ mem_buffer_,
        /*.no_alloc   =*/ false
    };
    ctx_ = ggml_init(params_);
    if (!ctx_) {
        std::free(mem_buffer_);
        throw std::runtime_error("Failed to initialize GGML context");
    }
}

GGMLContext::~GGMLContext()
{
    if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }
    if (mem_buffer_) { std::free(mem_buffer_); mem_buffer_ = nullptr; }
}

GGMLContext::GGMLContext(GGMLContext&& o) noexcept
    : ctx_(o.ctx_), params_(o.params_), mem_buffer_(o.mem_buffer_)
{
    o.ctx_ = nullptr; o.mem_buffer_ = nullptr;
}

GGMLContext& GGMLContext::operator=(GGMLContext&& o) noexcept
{
    if (this != &o) {
        if (ctx_) ggml_free(ctx_);
        if (mem_buffer_) std::free(mem_buffer_);
        ctx_ = o.ctx_; params_ = o.params_; mem_buffer_ = o.mem_buffer_;
        o.ctx_ = nullptr; o.mem_buffer_ = nullptr;
    }
    return *this;
}

} // namespace optimized
} // namespace xorzen

#endif // XORZEN_USE_GGML

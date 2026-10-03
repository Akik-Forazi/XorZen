/*
 * xorzen_native_bridge.cpp — pybind11 bridge for xorzen C++ kernels
 * ==================================================================
 * This file wraps the extern "C" kernel functions in pybind11 bindings
 * so they can be loaded as a Python module via torch.utils.cpp_extension.
 *
 * The actual kernel implementations are in the other .cpp files:
 *   - math_utils.cpp    — GELU, softmax, residual add
 *   - fused_ops.cpp     — RMSNorm, SwiGLU, SSM scan, LayerNorm+GELU
 *   - attention_ops.cpp — Fused window attention
 *   - expert_dispatch.cpp — MoE capacity constraint + weighted sum
 *   - router_ops.cpp    — Router MLP fast path
 *   - ssm_scan.cpp      — Parallel prefix SSM scan
 */
#include <torch/extension.h>
#include "xorzen_kernels.h"

// ============================================================================
// Fused math operations
// ============================================================================

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor gamma, torch::Tensor out, int N, int D, float eps) {
    rms_norm_f32(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D, eps
    );
    return out;
}

void gelu_inplace(torch::Tensor x, int n) {
    gelu_f32_inplace(x.data_ptr<float>(), n);
}

torch::Tensor fused_swiglu(torch::Tensor gate, torch::Tensor up, torch::Tensor out, int N, int D) {
    fused_swiglu_f32(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D
    );
    return out;
}

void fused_layernorm_gelu(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int N, int D, float eps) {
    fused_layernorm_gelu_f32(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, D, eps
    );
}

void fused_residual_add(torch::Tensor dst, torch::Tensor src, int n) {
    fused_residual_add_f32(
        dst.data_ptr<float>(),
        src.data_ptr<float>(),
        n
    );
}

void softmax_inplace(torch::Tensor x, int rows, int cols) {
    softmax_f32_inplace(x.data_ptr<float>(), rows, cols);
}

// ============================================================================
// Expert dispatch
// ============================================================================

void expert_capacity_constraint(torch::Tensor indices, torch::Tensor weights,
                                 int N, int top_k, int n_experts, int capacity) {
    expert_capacity_constraint_f32(
        indices.data_ptr<int>(),
        weights.data_ptr<float>(),
        N, top_k, n_experts, capacity
    );
}

void expert_weighted_sum(torch::Tensor expert_out, torch::Tensor weights,
                          torch::Tensor token_ids, torch::Tensor dst,
                          int n_active, int N, int hidden) {
    expert_weighted_sum_f32(
        expert_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        token_ids.data_ptr<int>(),
        dst.data_ptr<float>(),
        n_active, N, hidden
    );
}

// ============================================================================
// SSM scan
// ============================================================================

void diagonal_ssm_scan(torch::Tensor b_seq, torch::Tensor a_diag,
                        torch::Tensor states, int batch, int T, int state) {
    diagonal_ssm_scan_f32(
        b_seq.data_ptr<float>(),
        a_diag.data_ptr<float>(),
        states.data_ptr<float>(),
        batch, T, state
    );
}

// ============================================================================
// Python module definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm, "RMSNorm (C++ AVX2)");
    m.def("gelu_f32_inplace", &gelu_inplace, "GELU in-place (C++ AVX2)");
    m.def("fused_swiglu_f32", &fused_swiglu, "Fused SwiGLU (C++ AVX2)");
    m.def("fused_layernorm_gelu_f32", &fused_layernorm_gelu, "Fused LayerNorm+GELU (C++)");
    m.def("fused_residual_add_f32", &fused_residual_add, "Fused residual add (C++ AVX2)");
    m.def("softmax_f32_inplace", &softmax_inplace, "Softmax in-place (C++)");
    m.def("expert_capacity_constraint_f32", &expert_capacity_constraint, "MoE capacity constraint (C++)");
    m.def("expert_weighted_sum_f32", &expert_weighted_sum, "MoE weighted sum (C++ AVX2)");
    m.def("diagonal_ssm_scan_f32", &diagonal_ssm_scan, "Diagonal SSM scan (C++)");
}

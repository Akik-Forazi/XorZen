#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/ssm.h
//  S4D (Diagonal SSM) Kernel + SSMBlock
//  Ported from xorzen/model/ssm.py v2.0
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================

#include <torch/torch.h>
#include "xorzen/types.h"

namespace xorzen {

// ============================================================
//  S4DKernel — Structured diagonal SSM with parallel scan
//  Continuous-time: h'(t) = A·h(t) + B·u(t), y(t) = C·h(t) + D·u(t)
//  Discretized via ZOH: A_bar = exp(dt·A), B_bar = (A_bar-1)/A · B
// ============================================================

struct S4DKernelImpl : torch::nn::Module {
    // Params
    int64_t hidden_dim;
    int64_t state_dim;
    int64_t dt_rank;

    // Learnable parameters
    torch::Tensor A_log;   // [H, N/2]   — real part of diagonal A
    torch::Tensor A_im;    // [H, N/2]   — imaginary part of diagonal A
    torch::Tensor B;       // [H, N/2, 2] — complex B matrix
    torch::Tensor C;       // [H, N/2, 2] — complex C matrix
    torch::Tensor D;       // [H]         — skip connection

    torch::nn::Linear dt_proj{nullptr};
    torch::nn::Linear x_proj_for_dt{nullptr};

    S4DKernelImpl(int64_t hidden_dim,
                  int64_t state_dim,
                  int64_t dt_rank     = -1,   // -1 = auto (hidden/16)
                  double  dt_min      = 0.001,
                  double  dt_max      = 0.1,
                  double  dt_init_floor = 1e-4);

    // Forward: [B, L, H] (real float) → [B, L, H] (real float)
    torch::Tensor forward(const torch::Tensor& u);

    // Parallel scan: h_k = A_k * h_{k-1} + u_k
    // u_in: [B, L, H, N/2] complex, A_in: [B, L, H, N/2] complex
    // returns: [B, L, H, N/2] complex
    static torch::Tensor parallel_scan(const torch::Tensor& u_in,
                                       const torch::Tensor& A_in);
};
TORCH_MODULE(S4DKernel);

// ============================================================
//  SSMBlock — Mamba-style gated SSM block
//  in_proj → conv1d → SiLU → S4DKernel → gate → out_proj
// ============================================================

struct SSMBlockImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t state_dim;

    torch::nn::Linear in_proj{nullptr};   // H → 2H (ssm + gate)
    torch::nn::Conv1d conv1d{nullptr};    // depthwise, kernel=3
    torch::nn::SiLU   activation;
    S4DKernel         ssm_kernel{nullptr};
    torch::nn::Linear out_proj{nullptr};
    torch::nn::LayerNorm norm{nullptr};

    explicit SSMBlockImpl(const ModelConfig& cfg);

    // [B, L, H] → [B, L, H]
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(SSMBlock);

} // namespace xorzen

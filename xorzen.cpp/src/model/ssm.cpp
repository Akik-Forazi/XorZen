// ============================================================
//  xorzen.cpp — src/model/ssm.cpp
//  S4D Kernel + SSMBlock full implementation
//  Ported from xorzen/model/ssm.py v2.0
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/ssm.h"
#include "xorzen/optimized/simd_ops.h"

#include <cmath>
#include <stdexcept>
#include <complex>

namespace xorzen {

// ============================================================
//  S4DKernel
// ============================================================

S4DKernelImpl::S4DKernelImpl(int64_t hidden,
                              int64_t state,
                              int64_t dt_rank_in,
                              double  dt_min,
                              double  dt_max,
                              double  dt_init_floor)
    : hidden_dim(hidden), state_dim(state) {
    dt_rank = (dt_rank_in < 0)
        ? static_cast<int64_t>(std::ceil(static_cast<double>(hidden) / 16.0))
        : dt_rank_in;

    int64_t N2 = state / 2;   // half-state (complex diagonal)

    // --- A matrix: complex diagonal parameterised as (A_log, A_im) ---
    // A = -exp(A_log) + j*A_im  →  stable (negative real part)
    auto A_re_init = torch::ones({hidden, N2});
    auto A_im_init = torch::arange(N2).to(torch::kFloat32)
                         .unsqueeze(0).expand({hidden, N2});
    A_log = register_parameter("A_log", torch::log(A_re_init));
    A_im  = register_parameter("A_im",  A_im_init.clone());

    // --- B and C: complex, stored as [..., 2] (real, imag) ---
    B = register_parameter("B", torch::randn({hidden, N2, 2}));
    C = register_parameter("C", torch::randn({hidden, N2, 2}));

    // --- D: skip connection ---
    D = register_parameter("D", torch::randn({hidden}));

    // --- dt projection ---
    dt_proj       = register_module("dt_proj",
        torch::nn::Linear(dt_rank, hidden));
    x_proj_for_dt = register_module("x_proj_for_dt",
        torch::nn::Linear(torch::nn::LinearOptions(hidden, dt_rank).bias(false)));

    // Initialise dt_proj bias so initial dt ~ Uniform(dt_init_floor, dt_max)
    {
        torch::NoGradGuard ng;
        auto dt_init = torch::exp(
            torch::rand({hidden}) * (std::log(dt_max) - std::log(dt_init_floor))
            + std::log(dt_init_floor)).clamp_min(dt_min);
        // inv_softplus(x) = x + log(1 - exp(-x))
        auto inv_dt = dt_init + torch::log(-torch::expm1(-dt_init));
        dt_proj->bias.copy_(inv_dt);
    }
}

torch::Tensor S4DKernelImpl::forward(const torch::Tensor& u_real) {
    // u_real: [B, L, H] float32
    const int64_t batch = u_real.size(0);
    const int64_t L = u_real.size(1);
    const int64_t H = u_real.size(2);
    const int64_t N2 = state_dim / 2;

    // Cast to complex float for SSM operations
    auto u = u_real.to(torch::kComplexFloat);  // [B, L, H] cfloat

    // Reconstruct complex A: A = -exp(A_log) + j*A_im  [H, N2]
    auto A = torch::complex(
        -torch::exp(A_log.to(torch::kFloat32)),
         A_im.to(torch::kFloat32));  // [H, N2] cfloat

    // Reconstruct complex B, C: [..., 0] + j*[..., 1]
    auto B_c = torch::complex(B.select(-1, 0), B.select(-1, 1)); // [H, N2]
    auto C_c = torch::complex(C.select(-1, 0), C.select(-1, 1)); // [H, N2]

    // --- Compute dt ---
    // dt_proj operates on real part (nn::Linear doesn't support cfloat)
    auto dt_input = x_proj_for_dt->forward(u_real);              // [B, L, dt_rank]
    auto dt = torch::nn::functional::softplus(
        dt_proj->forward(dt_input));                              // [B, L, H]

    // --- ZOH Discretization ---
    // dtA: [B, L, H, N2]
    auto dtA = dt.unsqueeze(-1).to(torch::kComplexFloat) *
               A.unsqueeze(0).unsqueeze(0);                       // broadcast to [B,L,H,N2]

    auto A_bar = torch::exp(dtA);                                 // [B, L, H, N2]

    // B_bar = (exp(dtA) - 1) / dtA * B  (Taylor expansion when |dtA| < 1e-4)
    auto z = dtA;
    auto z_abs = z.abs();
    auto B_bar_div = torch::where(
        z_abs < 1e-4f,
        torch::ones_like(z) + z / 2.0f,                          // Taylor: (e^z-1)/z ≈ 1 + z/2
        (torch::exp(z) - 1.0f) / z);
    auto B_bar = B_bar_div *
                 B_c.unsqueeze(0).unsqueeze(0);                   // [B, L, H, N2]

    // ub = u ⊗ B_bar   [B, L, H, N2]
    auto ub = u.unsqueeze(-1) * B_bar;

    // --- Parallel scan ---
    auto y = parallel_scan(ub, A_bar);                            // [B, L, H, N2]

    // --- Output projection: y_out = sum_n(C_n * y_n) ---
    auto y_out = (C_c.unsqueeze(0).unsqueeze(0) * y).sum(-1);   // [B, L, H]

    // Skip connection D (real)
    y_out = y_out + u * D.unsqueeze(0).unsqueeze(0).to(torch::kComplexFloat);

    // Return real part only
    return torch::real(y_out);  // [B, L, H] float32
}

torch::Tensor S4DKernelImpl::parallel_scan(const torch::Tensor& u_in,
                                           const torch::Tensor& A_in) {
    // u_in: [B, L, H, N]  A_in: [B, L, H, N]  (complex)
    const int64_t B  = u_in.size(0);
    const int64_t L  = u_in.size(1);
    const int64_t H  = u_in.size(2);
    const int64_t N  = u_in.size(3);
    
    // Serial scan on CPU is MUCH faster than log(L) rounds of LibTorch kernel launches
    // for typical sequence lengths (up to 4096-8192).
    auto y = torch::empty_like(u_in);
    
    auto u_ptr = u_in.to(torch::kCPU).contiguous();
    auto A_ptr = A_in.to(torch::kCPU).contiguous();
    auto y_ptr = y.to(torch::kCPU).contiguous();
    
    const std::complex<float>* p_u = reinterpret_cast<const std::complex<float>*>(u_ptr.data_ptr());
    const std::complex<float>* p_A = reinterpret_cast<const std::complex<float>*>(A_ptr.data_ptr());
    std::complex<float>* p_y = reinterpret_cast<std::complex<float>*>(y_ptr.data_ptr());
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t n = 0; n < N; ++n) {
                std::complex<float> state(0.0f, 0.0f);
                for (int64_t l = 0; l < L; ++l) {
                    int64_t idx = ((b * L + l) * H + h) * N + n;
                    state = p_A[idx] * state + p_u[idx];
                    p_y[idx] = state;
                }
            }
        }
    }
    
    return y_ptr.to(u_in.device());
}

// ============================================================
//  SSMBlock
// ============================================================

SSMBlockImpl::SSMBlockImpl(const ModelConfig& cfg)
    : hidden_dim(cfg.hidden_size),
      state_dim(cfg.ssm_state_dim) {

    in_proj  = register_module("in_proj",
        torch::nn::Linear(hidden_dim, hidden_dim * 2));

    conv1d   = register_module("conv1d",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(hidden_dim, hidden_dim, 3)
            .padding(1) // Fixed padding to maintain size
            .groups(hidden_dim)
            .bias(true)));

    ssm_kernel = register_module("ssm_kernel",
        S4DKernel(hidden_dim, state_dim));

    out_proj = register_module("out_proj",
        torch::nn::Linear(hidden_dim, hidden_dim));

    norm     = register_module("norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
}

torch::Tensor SSMBlockImpl::forward(const torch::Tensor& x) {
    // x: [B, L, H]
    const int64_t L = x.size(1);

    auto x_n    = norm->forward(x);

    // Split projection into SSM path and gate path
    auto x_proj = in_proj->forward(x_n);                            // [B, L, 2H]
    auto chunks  = x_proj.chunk(2, -1);
    auto x_ssm  = chunks[0];                                        // [B, L, H]
    auto x_gate = chunks[1];                                        // [B, L, H]

    // Depthwise conv1d (operates on channels dim)
    // conv1d expects [B, C, L]
    auto x_conv = conv1d->forward(x_ssm.transpose(1, 2))           // [B, H, L]
                        .transpose(1, 2);                            // [B, L, H]
    x_conv = optimized::silu_simd(x_conv);

    // S4D SSM
    auto y_ssm = ssm_kernel->forward(x_conv);                       // [B, L, H]

    // Gate: elementwise product
    auto output = y_ssm * optimized::silu_simd(x_gate);                       // [B, L, H]

    // Final projection
    return out_proj->forward(output);                               // [B, L, H]
}

} // namespace xorzen

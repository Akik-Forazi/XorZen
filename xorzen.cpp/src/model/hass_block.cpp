#include "xorzen/hass.h"
#include "xorzen/optimized/flash_attn.h"
#include "xorzen/optimized/simd_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xorzen {

LocalAttentionPathwayImpl::LocalAttentionPathwayImpl(int64_t hidden, int64_t heads, int64_t window,
                                                     double dropout, bool causal)
    : hidden_dim(hidden),
      num_heads(std::max<int64_t>(1, heads)),
      head_dim(hidden / std::max<int64_t>(1, heads)),
      window_size(window),
      causal(causal) {
    q_proj = register_module("q_proj", torch::nn::Linear(hidden_dim, hidden_dim));
    k_proj = register_module("k_proj", torch::nn::Linear(hidden_dim, hidden_dim));
    v_proj = register_module("v_proj", torch::nn::Linear(hidden_dim, hidden_dim));
    out_proj = register_module("out_proj", torch::nn::Linear(hidden_dim, hidden_dim));
    ln_q = register_module("ln_q", torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim})));
    ln_k = register_module("ln_k", torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim})));
    attn_dropout = register_module("attn_dropout", torch::nn::Dropout(dropout));
    resid_dropout = register_module("resid_dropout", torch::nn::Dropout(dropout));
    xavier_linear(q_proj, 1.0 / std::sqrt(2.0));
    xavier_linear(k_proj, 1.0 / std::sqrt(2.0));
    xavier_linear(v_proj, 1.0 / std::sqrt(2.0));
    xavier_linear(out_proj, 1.0 / std::sqrt(2.0));
}

torch::Tensor LocalAttentionPathwayImpl::forward(const torch::Tensor& x,
                                                 const torch::Tensor& attention_mask,
                                                 const torch::Tensor& position_bias) {
    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    
    auto q = q_proj->forward(x).view({B, S, num_heads, head_dim}).transpose(1, 2);
    auto k = k_proj->forward(x).view({B, S, num_heads, head_dim}).transpose(1, 2);
    auto v = v_proj->forward(x).view({B, S, num_heads, head_dim}).transpose(1, 2);
    
    q = ln_q->forward(q.transpose(1, 2)).transpose(1, 2);
    k = ln_k->forward(k.transpose(1, 2)).transpose(1, 2);

#ifdef XORZEN_ENABLE_FLASH_ATTN
    // Use optimized Flash Attention kernel
    auto out = optimized::flash_attention_cpu(q, k, v, attention_mask, causal);
    out = out.transpose(1, 2).contiguous().view({B, S, hidden_dim});
#else
    auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
    if (causal) {
        auto mask = make_local_causal_mask(S, window_size, x.device()).view({1, 1, S, S});
        scores = scores.masked_fill(mask.logical_not(), -std::numeric_limits<float>::infinity());
    }
    if (attention_mask.defined()) {
        auto am = attention_mask.dim() == 2 ? attention_mask.view({B, 1, 1, S}) : attention_mask;
        scores = scores.masked_fill(am.to(torch::kBool).logical_not(), -std::numeric_limits<float>::infinity());
    }
    if (position_bias.defined()) scores = scores + position_bias;
    auto probs = torch::softmax(scores, -1);
    probs = attn_dropout->forward(probs);
    auto out = torch::matmul(probs, v).transpose(1, 2).contiguous().view({B, S, hidden_dim});
#endif

    return resid_dropout->forward(out_proj->forward(out));
}

LowRankGlobalPathwayImpl::LowRankGlobalPathwayImpl(int64_t hidden, int64_t low_rank, int64_t heads, double dropout_p)
    : hidden_dim(hidden), low_rank_dim(low_rank), num_heads(heads) {
    const int64_t D = low_rank_dim * num_heads;
    to_low_rank = register_module("to_low_rank", torch::nn::Linear(hidden_dim, D));
    from_low_rank = register_module("from_low_rank", torch::nn::Linear(D, hidden_dim));
    ln_input = register_module("ln_input", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    // NOTE: ln_low_rank and context_weights REMOVED — Python LowRankGlobalPathway
    // does not have these. It uses causal self-attention with GELU, no learned
    // context query, no second LayerNorm. See hass_block.py:261-368.
    xavier_linear(to_low_rank, 1.0 / std::sqrt(2.0));
    xavier_linear(from_low_rank, 1.0 / std::sqrt(2.0));
}

torch::Tensor LowRankGlobalPathwayImpl::forward(const torch::Tensor& x) {
    // Mirror Python LowRankGlobalPathway.forward (hass_block.py:309-368) exactly.
    auto x_norm = ln_input->forward(x);
    auto low_rank = to_low_rank->forward(x_norm);
    low_rank = torch::gelu(low_rank);

    int64_t B = low_rank.size(0), S = low_rank.size(1);
    int64_t rk_dim = low_rank_dim * num_heads;

    // Causal lower-triangular mask
    auto causal_mask = torch::tril(torch::ones({S, S}, low_rank.options()));

    if (S <= 512) {
        // Full pairwise causal attention: O(T^2 * D)
        auto scores = torch::matmul(low_rank, low_rank.transpose(-1, -2)) /
                      std::sqrt(static_cast<double>(rk_dim));
        scores = scores.masked_fill(causal_mask.unsqueeze(0).eq(0),
                                     -std::numeric_limits<float>::infinity());
        auto attn_w = torch::softmax(scores, -1);
        auto global_context = torch::matmul(attn_w, low_rank);
        auto combined = low_rank + global_context;
        return dropout->forward(from_low_rank->forward(combined));
    } else {
        // Chunked causal attention for long sequences (matches Python fallback)
        auto global_context = torch::zeros_like(low_rank);
        int64_t chunk_size = 512;
        for (int64_t start = 0; start < S; start += chunk_size) {
            int64_t end = std::min(start + chunk_size, S);
            auto q_chunk = low_rank.slice(1, start, end);
            auto scores = torch::matmul(q_chunk, low_rank.transpose(-1, -2)) /
                          std::sqrt(static_cast<double>(rk_dim));
            auto causal_chunk = causal_mask.slice(0, start, end);
            scores = scores.masked_fill(causal_chunk.unsqueeze(0).eq(0),
                                         -std::numeric_limits<float>::infinity());
            auto attn_w = torch::softmax(scores, -1);
            global_context.slice(1, start, end) = torch::matmul(attn_w, low_rank);
        }
        auto combined = low_rank + global_context;
        return dropout->forward(from_low_rank->forward(combined));
    }
}

SSMPathwayImpl::SSMPathwayImpl(int64_t hidden, int64_t state, int64_t kernel,
                               double dropout_p, bool conv_enabled)
    : hidden_dim(hidden), state_dim(state), kernel_size(kernel), use_conv(conv_enabled) {
    A_log = register_parameter("A_log", torch::zeros({state_dim}));
    dt_proj = register_module("dt_proj", torch::nn::Linear(hidden_dim, state_dim));
    B_proj = register_module("B_proj", torch::nn::Linear(hidden_dim, state_dim));
    C_proj = register_module("C_proj", torch::nn::Linear(hidden_dim, state_dim));
    D_proj = register_module("D_proj", torch::nn::Linear(state_dim, hidden_dim));
    gate_proj = register_module("gate_proj", torch::nn::Linear(hidden_dim, hidden_dim * 2));
    if (use_conv) {
        // CAUSAL left-padding: padding = kernel_size - 1, then truncate to T.
        // Mirrors Python SSMPathway (hass_block.py:432-437). The previous
        // C++ used padding = kernel_size / 2 (center padding), which caused
        // future-token leakage — a correctness bug, not just a numerical diff.
        conv = register_module("conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(hidden_dim, hidden_dim, kernel_size)
            .padding(kernel_size - 1).groups(hidden_dim)));
    }
    ln_input = register_module("ln_input", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    ln_state = register_module("ln_state", torch::nn::LayerNorm(torch::nn::LayerNormOptions({state_dim})));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    xavier_linear(dt_proj, 1.0 / std::sqrt(2.0));
    xavier_linear(B_proj, 1.0 / std::sqrt(2.0));
    xavier_linear(C_proj, 1.0 / std::sqrt(2.0));
    xavier_linear(D_proj, 1.0 / std::sqrt(2.0));
    xavier_linear(gate_proj, 1.0 / std::sqrt(2.0));
}

#include <torch/autograd.h>

// SSMScanFunction: serial scan over the recurrence h_t = A_bar_t * h_{t-1} + B_bar_t.
//
// FIX vs previous version: this function now returns RAW STATES h_t (not C*h).
// C is no longer passed in — the caller applies C after the scan and after ln_state,
// matching Python SSMPathway.forward_parallel (hass_block.py:549-562, 612-615).
//
// Inputs:
//   Ab:     [B, T, N]  (already ZOH-discretized A_bar = exp(dt * a))
//   Bb:     [B, T, N]  (already ZOH-discretized B_bar = ((exp(z)-1)/z) * dt * Bv)
//
// Output:
//   states: [B, T, N]  (raw h_t values)
//
// Backward: dL/dAb, dL/dBb. (No dL/dC — C is applied in the caller's autograd graph.)
class SSMScanFunction : public torch::autograd::Function<SSMScanFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor Ab,
                                 torch::Tensor Bb) {
        auto Ab_cpu = Ab.to(torch::kCPU).contiguous();
        auto Bb_cpu = Bb.to(torch::kCPU).contiguous();

        const int64_t B = Ab.size(0);
        const int64_t T = Ab.size(1);
        const int64_t D = Ab.size(2);

        auto states_cpu = torch::empty({B, T, D}, Ab.options().device(torch::kCPU));

        const float* p_Ab = Ab_cpu.data_ptr<float>();
        const float* p_Bb = Bb_cpu.data_ptr<float>();
        float* p_states = states_cpu.data_ptr<float>();

        for (int64_t b = 0; b < B; ++b) {
            std::vector<float> h(D, 0.0f);
            for (int64_t t = 0; t < T; ++t) {
                int64_t offset = b * T * D + t * D;
                for (int64_t d = 0; d < D; ++d) {
                    float ab = p_Ab[offset + d];
                    float bb = p_Bb[offset + d];
                    h[d] = ab * h[d] + bb;
                    p_states[offset + d] = h[d];  // raw state, NOT c*h
                }
            }
        }

        auto states = states_cpu.to(Ab.device());
        ctx->save_for_backward({Ab_cpu, Bb_cpu});
        return states;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto Ab_cpu = saved[0];
        auto Bb_cpu = saved[1];
        auto grad_states_cpu = grad_outputs[0].to(torch::kCPU).contiguous();

        const int64_t B = Ab_cpu.size(0);
        const int64_t T = Ab_cpu.size(1);
        const int64_t D = Ab_cpu.size(2);

        auto grad_Ab_cpu = torch::zeros_like(Ab_cpu);
        auto grad_Bb_cpu = torch::zeros_like(Bb_cpu);

        const float* p_Ab = Ab_cpu.data_ptr<float>();
        const float* p_Bb = Bb_cpu.data_ptr<float>();
        const float* p_grad_states = grad_states_cpu.data_ptr<float>();

        float* p_grad_Ab = grad_Ab_cpu.data_ptr<float>();
        float* p_grad_Bb = grad_Bb_cpu.data_ptr<float>();

        for (int64_t b = 0; b < B; ++b) {
            // Recompute h values for numerical stability
            std::vector<float> h(T * D, 0.0f);
            std::vector<float> curr_h(D, 0.0f);
            for (int64_t t = 0; t < T; ++t) {
                int64_t offset = b * T * D + t * D;
                for (int64_t d = 0; d < D; ++d) {
                    float ab = p_Ab[offset + d];
                    float bb = p_Bb[offset + d];
                    curr_h[d] = ab * curr_h[d] + bb;
                    h[t * D + d] = curr_h[d];
                }
            }

            // Backward scan: dh accumulates gradient flowing back from later timesteps.
            std::vector<float> dh(D, 0.0f);
            for (int64_t t = T - 1; t >= 0; --t) {
                int64_t offset = b * T * D + t * D;
                for (int64_t d = 0; d < D; ++d) {
                    float ab = p_Ab[offset + d];
                    float h_prev = (t > 0) ? h[(t - 1) * D + d] : 0.0f;

                    float grad_h_t = p_grad_states[offset + d] + dh[d];
                    // h_t = ab * h_{t-1} + bb  →  dL/dbb = grad_h_t, dL/dab = grad_h_t * h_prev
                    p_grad_Bb[offset + d] = grad_h_t;
                    p_grad_Ab[offset + d] = grad_h_t * h_prev;
                    // dL/dh_{t-1} += grad_h_t * ab
                    dh[d] = grad_h_t * ab;
                }
            }
        }

        auto device = grad_outputs[0].device();
        return {grad_Ab_cpu.to(device), grad_Bb_cpu.to(device)};
    }
};

// Helper: ZOH discretization for diagonal A, mirroring Python discretize_zoh
// (ssm_scan.py:62-106). Returns (A_bar, B_bar).
//   a:  [N]            negative real diagonal of A
//   Bv: [B, T, N]      raw B projection
//   dt: [B, T, N]      input-dependent step size
//   A_bar = exp(dt * a)
//   B_bar = ((exp(z) - 1) / z) * dt * Bv    where z = dt * a
//   For |z| < 1e-4, use Taylor: 1 + z/2 + z²/6 (avoids 0/0).
static std::pair<torch::Tensor, torch::Tensor>
discretize_zoh(const torch::Tensor& a,     // [N]
               const torch::Tensor& Bv,    // [B, T, N]
               const torch::Tensor& dt,    // [B, T, N]
               double eps = 1e-4) {
    auto a_b = a.view({1, 1, -1});           // [1, 1, N]
    auto z = dt * a_b;                        // [B, T, N]
    auto A_bar = torch::exp(z);
    auto small = z.abs() < eps;
    auto safe_z = torch::where(small, torch::ones_like(z), z);
    auto exact = (A_bar - 1.0) / safe_z;     // (exp(z)-1)/z, stable for |z| >= eps
    auto taylor = 1.0 + z / 2.0 + (z * z) / 6.0;
    auto B_bar_div = torch::where(small, taylor, exact);
    auto B_bar = B_bar_div * dt * Bv;
    return {A_bar, B_bar};
}

torch::Tensor SSMPathwayImpl::forward(const torch::Tensor& x) {
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    auto xn = ln_input->forward(x);
    if (use_conv && conv) {
        // CAUSAL conv: padding = kernel_size - 1 produces output of length T + (kernel_size - 1).
        // Truncate to T to remove the right-side padding (mirror Python hass_block.py:525-530).
        auto conv_out = conv->forward(xn.transpose(1, 2));   // [B, H, T + k - 1]
        conv_out = conv_out.slice(/*dim=*/2, /*start=*/0, /*end=*/T);  // [B, H, T]
        xn = xn + conv_out.transpose(1, 2);
    }
    auto gate_pair = gate_proj->forward(xn).chunk(2, -1);
    auto gate = torch::sigmoid(gate_pair[0]);
    auto input_gate = torch::sigmoid(gate_pair[1]);
    auto Bv = B_proj->forward(xn * input_gate);
    auto C = C_proj->forward(xn);
    auto dt = torch::softplus(dt_proj->forward(xn));
    auto a = -torch::exp(A_log);

    // ZOH discretization of BOTH A and B (mirror Python discretize_zoh).
    auto [Ab, Bb] = discretize_zoh(a, Bv, dt);

    // Scan: returns raw states h_t (NOT C*h). C is applied after ln_state below.
    auto states = SSMScanFunction::apply(Ab, Bb);

    // Python order (hass_block.py:555-562, 612-615):
    //   states_ln = ln_state(states)        # LN(h)
    //   ssm_output = C * states_ln          # C * LN(h)
    //   output = D_proj(ssm_output) * gate
    auto states_ln = ln_state->forward(states);
    auto ssm_output = C * states_ln;
    auto out = D_proj->forward(ssm_output) * gate;
    return dropout->forward(out);
}

AdaptiveFFNImpl::AdaptiveFFNImpl(int64_t hidden, double multiplier, std::string act, double dropout_p)
    : hidden_dim(hidden), base_ffn_dim(static_cast<int64_t>(hidden * multiplier)), activation(std::move(act)) {
    fc1 = register_module("fc1", torch::nn::Linear(hidden_dim, base_ffn_dim));
    fc2 = register_module("fc2", torch::nn::Linear(base_ffn_dim, hidden_dim));
    ln_input = register_module("ln_input", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    ln_hidden = register_module("ln_hidden", torch::nn::LayerNorm(torch::nn::LayerNormOptions({base_ffn_dim})));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    ffn_dropout = register_module("ffn_dropout", torch::nn::Dropout(dropout_p));
    xavier_linear(fc1, 1.0 / std::sqrt(2.0));
    xavier_linear(fc2, 1.0 / std::sqrt(2.0));
}

torch::Tensor AdaptiveFFNImpl::forward(const torch::Tensor& x, const torch::Tensor& width_multiplier) {
    auto xn = ln_input->forward(x);
    auto h = fc1->forward(xn);
    
    if (activation == "silu") {
        h = optimized::silu_simd(h);
    } else if (activation == "relu") {
        h = torch::relu(h);
    } else {
        h = optimized::gelu_simd(h);
    }
    
    h = ffn_dropout->forward(ln_hidden->forward(h));
    auto out = fc2->forward(h);
    if (width_multiplier.defined()) out = out * width_multiplier;
    return dropout->forward(out);
}

HASSBlockImpl::HASSBlockImpl(ModelConfig cfg, int64_t idx)
    : config(std::move(cfg)), layer_idx(idx) {
    config.normalize();
    local = register_module("local", LocalAttentionPathway(config.hidden_size,
        std::max<int64_t>(1, config.num_attention_heads / 2), config.local_window_size, config.dropout, true));
    low_rank = register_module("low_rank", LowRankGlobalPathway(config.hidden_size, config.low_rank_dim, 4, config.dropout));
    ssm = register_module("ssm", SSMPathway(config.hidden_size, config.ssm_state_dim, config.ssm_kernel_size, config.dropout, true));
    // NOTE: pathway_gate REMOVED — Python removed it in v0.5.
    // When routing_decision is provided, path_probs are used directly.
    // When no routing_decision, uniform 1/3 weighting is used.
    ffn = register_module("ffn", AdaptiveFFN(config.hidden_size, 4.0, config.hidden_act, config.dropout));
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.hidden_size})));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.hidden_size})));
    dropout = register_module("dropout", torch::nn::Dropout(config.dropout));
}

torch::Tensor HASSBlockImpl::forward(const torch::Tensor& x,
                                     const RoutingDecision* routing_decision,
                                     const torch::Tensor& attention_mask,
                                     bool compute_all_pathways) {
    auto xa = ln1->forward(x);
    int64_t B = xa.size(0), T = xa.size(1), H = xa.size(2);

    torch::Tensor combined;
    if (routing_decision == nullptr || compute_all_pathways) {
        // No routing — compute all 3 pathways on full input, uniform 1/3 weighting
        auto local_out = local->forward(xa, attention_mask);
        auto low_rank_out = low_rank->forward(xa);
        auto ssm_out = ssm->forward(xa);
        auto w = torch::ones({B, T, 3}, xa.options()) / 3.0;
        combined = local_out * w.select(-1, 0).unsqueeze(-1) +
                   low_rank_out * w.select(-1, 1).unsqueeze(-1) +
                   ssm_out * w.select(-1, 2).unsqueeze(-1);
    } else {
        // SPARSE PATHWAY DISPATCH — mirror Python sparse_pathway_dispatch exactly.
        // CRITICAL: pathways are computed on TOKEN SUBSETS (only selected tokens),
        // not on the full input. This matters because SSM recurrence and local
        // attention causal mask depend on the full sequence context. Running on
        // a subset produces different outputs than running on all tokens then masking.
        auto path_probs = routing_decision->path_probs;  // [B, T, 3]
        int64_t num_paths = path_probs.size(-1);
        int64_t top_k = 2;  // Python config.pathway_top_k default

        // Build hard top-k mask and renormalized weights
        auto topk = path_probs.topk(top_k, /*dim=*/-1);
        auto topk_idx = std::get<1>(topk);
        auto hard_mask = torch::zeros_like(path_probs);
        hard_mask.scatter_(-1, topk_idx, 1.0);
        auto selected = path_probs * hard_mask;
        auto sel_sum = selected.sum(-1, /*keepdim=*/true);
        sel_sum = torch::where(sel_sum > 1e-8, sel_sum, torch::ones_like(sel_sum));
        auto norm_w = selected / sel_sum;  // [B, T, 3]

        if (top_k >= num_paths) {
            // All pathways selected — compute on full input
            auto local_out = local->forward(xa, attention_mask);
            auto low_rank_out = low_rank->forward(xa);
            auto ssm_out = ssm->forward(xa);
            combined = local_out * norm_w.select(-1, 0).unsqueeze(-1) +
                       low_rank_out * norm_w.select(-1, 1).unsqueeze(-1) +
                       ssm_out * norm_w.select(-1, 2).unsqueeze(-1);
        } else {
            // Sparse dispatch: for each pathway, find selected tokens, run forward
            // on ONLY those tokens, scatter back with index_add_.
            auto x_flat = xa.reshape({B * T, H});
            auto mask_flat = hard_mask.reshape({B * T, num_paths});
            auto w_flat = norm_w.reshape({B * T, num_paths});
            auto combined_flat = torch::zeros({B * T, H}, xa.options());

            // Pathway 0: local — pass empty attention_mask (Python passes None)
            {
                auto sel = mask_flat.select(1, 0) > 0.5;  // [B*T] bool
                if (sel.any().item<bool>()) {
                    auto idx = sel.nonzero().squeeze(-1);  // [n_sel]
                    auto x_slice = x_flat.index_select(0, idx);  // [n_sel, H]
                    auto x3d = x_slice.unsqueeze(0);
                    // Pass empty attention_mask — Python sparse dispatch passes None
                    auto y3d = local->forward(x3d, /*attention_mask=*/{});
                    auto y_slice = y3d.squeeze(0);
                    auto w_slice = w_flat.select(1, 0).index_select(0, idx).unsqueeze(-1);
                    combined_flat.index_add_(0, idx, y_slice * w_slice);
                }
            }
            // Pathway 1: low_rank
            {
                auto sel = mask_flat.select(1, 1) > 0.5;
                if (sel.any().item<bool>()) {
                    auto idx = sel.nonzero().squeeze(-1);
                    auto x_slice = x_flat.index_select(0, idx);
                    auto x3d = x_slice.unsqueeze(0);
                    auto y3d = low_rank->forward(x3d);
                    auto y_slice = y3d.squeeze(0);
                    auto w_slice = w_flat.select(1, 1).index_select(0, idx).unsqueeze(-1);
                    combined_flat.index_add_(0, idx, y_slice * w_slice);
                }
            }
            // Pathway 2: ssm
            {
                auto sel = mask_flat.select(1, 2) > 0.5;
                if (sel.any().item<bool>()) {
                    auto idx = sel.nonzero().squeeze(-1);
                    auto x_slice = x_flat.index_select(0, idx);
                    auto x3d = x_slice.unsqueeze(0);
                    auto y3d = ssm->forward(x3d);
                    auto y_slice = y3d.squeeze(0);
                    auto w_slice = w_flat.select(1, 2).index_select(0, idx).unsqueeze(-1);
                    combined_flat.index_add_(0, idx, y_slice * w_slice);
                }
            }
            combined = combined_flat.reshape({B, T, H});
        }
    }

    auto residual = x + dropout->forward(combined);
    auto xf = ln2->forward(residual);
    auto ffn_out = routing_decision ? ffn->forward(xf, routing_decision->width_multiplier) : ffn->forward(xf);
    return residual + dropout->forward(ffn_out);
}

// NOTE: The 2-gate GatedMergerImpl previously defined here has been REMOVED.
// The correct 3-gate GatedMergerImpl (matching Python xorzenMergerGate) lives
// in src/model/merger.cpp. The main model uses XorzenMergerGate from merger.h.

} // namespace xorzen

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
    ln_low_rank = register_module("ln_low_rank", torch::nn::LayerNorm(torch::nn::LayerNormOptions({D})));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    context_weights = register_parameter("context_weights", torch::randn({1, 1, D}) * 0.02);
    xavier_linear(to_low_rank, 1.0 / std::sqrt(2.0));
    xavier_linear(from_low_rank, 1.0 / std::sqrt(2.0));
}

torch::Tensor LowRankGlobalPathwayImpl::forward(const torch::Tensor& x) {
    auto xn = ln_input->forward(x);
    auto lr = optimized::fused_layernorm_gelu_simd(
        to_low_rank->forward(xn),
        ln_low_rank->weight,
        ln_low_rank->bias,
        ln_low_rank->options.eps()
    );
    auto attn = optimized::softmax_simd(torch::matmul(lr, context_weights.transpose(-1, -2)) /
                               std::sqrt(static_cast<double>(low_rank_dim)), 1);
    auto global_ctx = torch::matmul(attn.transpose(-1, -2), lr).expand_as(lr);
    return dropout->forward(from_low_rank->forward(lr + global_ctx));
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
        conv = register_module("conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(hidden_dim, hidden_dim, kernel_size)
            .padding(kernel_size / 2).groups(hidden_dim)));
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

class SSMScanFunction : public torch::autograd::Function<SSMScanFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor Ab,
                                 torch::Tensor Bv,
                                 torch::Tensor C) {
        auto Ab_cpu = Ab.to(torch::kCPU).contiguous();
        auto Bv_cpu = Bv.to(torch::kCPU).contiguous();
        auto C_cpu = C.to(torch::kCPU).contiguous();
        
        const int64_t B = Ab.size(0);
        const int64_t T = Ab.size(1);
        const int64_t D = Ab.size(2);
        
        auto states_cpu = torch::empty({B, T, D}, Ab.options().device(torch::kCPU));
        
        const float* p_Ab = Ab_cpu.data_ptr<float>();
        const float* p_Bv = Bv_cpu.data_ptr<float>();
        const float* p_C = C_cpu.data_ptr<float>();
        float* p_states = states_cpu.data_ptr<float>();
        
        for (int64_t b = 0; b < B; ++b) {
            std::vector<float> h(D, 0.0f);
            for (int64_t t = 0; t < T; ++t) {
                int64_t offset = b * T * D + t * D;
                for (int64_t d = 0; d < D; ++d) {
                    float ab = p_Ab[offset + d];
                    float bv = p_Bv[offset + d];
                    float c = p_C[offset + d];
                    float h_next = ab * h[d] + bv;
                    h[d] = h_next;
                    p_states[offset + d] = c * h_next;
                }
            }
        }
        
        auto states = states_cpu.to(Ab.device());
        ctx->save_for_backward({Ab_cpu, Bv_cpu, C_cpu});
        return states;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto Ab_cpu = saved[0];
        auto Bv_cpu = saved[1];
        auto C_cpu = saved[2];
        auto grad_states_cpu = grad_outputs[0].to(torch::kCPU).contiguous();
        
        const int64_t B = Ab_cpu.size(0);
        const int64_t T = Ab_cpu.size(1);
        const int64_t D = Ab_cpu.size(2);
        
        auto grad_Ab_cpu = torch::zeros_like(Ab_cpu);
        auto grad_Bv_cpu = torch::zeros_like(Bv_cpu);
        auto grad_C_cpu = torch::zeros_like(C_cpu);
        
        const float* p_Ab = Ab_cpu.data_ptr<float>();
        const float* p_Bv = Bv_cpu.data_ptr<float>();
        const float* p_C = C_cpu.data_ptr<float>();
        const float* p_grad_states = grad_states_cpu.data_ptr<float>();
        
        float* p_grad_Ab = grad_Ab_cpu.data_ptr<float>();
        float* p_grad_Bv = grad_Bv_cpu.data_ptr<float>();
        float* p_grad_C = grad_C_cpu.data_ptr<float>();
        
        for (int64_t b = 0; b < B; ++b) {
            // Recompute h values for numerical stability
            std::vector<float> h(T * D, 0.0f);
            std::vector<float> curr_h(D, 0.0f);
            for (int64_t t = 0; t < T; ++t) {
                int64_t offset = b * T * D + t * D;
                for (int64_t d = 0; d < D; ++d) {
                    float ab = p_Ab[offset + d];
                    float bv = p_Bv[offset + d];
                    float h_next = ab * curr_h[d] + bv;
                    curr_h[d] = h_next;
                    h[t * D + d] = h_next;
                }
            }
            
            std::vector<float> dh(D, 0.0f);
            for (int64_t t = T - 1; t >= 0; --t) {
                int64_t offset = b * T * D + t * D;
                for (int64_t d = 0; d < D; ++d) {
                    float ab = p_Ab[offset + d];
                    float c = p_C[offset + d];
                    float h_t = h[t * D + d];
                    float h_prev = (t > 0) ? h[(t - 1) * D + d] : 0.0f;
                    
                    float grad_out = p_grad_states[offset + d];
                    p_grad_C[offset + d] = grad_out * h_t;
                    
                    float grad_h_t = grad_out * c + dh[d];
                    p_grad_Bv[offset + d] = grad_h_t;
                    p_grad_Ab[offset + d] = grad_h_t * h_prev;
                    
                    dh[d] = grad_h_t * ab;
                }
            }
        }
        
        auto device = grad_outputs[0].device();
        return {grad_Ab_cpu.to(device), grad_Bv_cpu.to(device), grad_C_cpu.to(device)};
    }
};

torch::Tensor SSMPathwayImpl::forward(const torch::Tensor& x) {
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    auto xn = ln_input->forward(x);
    if (use_conv && conv) {
        xn = xn + conv->forward(xn.transpose(1, 2)).transpose(1, 2);
    }
    auto gate_pair = gate_proj->forward(xn).chunk(2, -1);
    auto gate = torch::sigmoid(gate_pair[0]);
    auto input_gate = torch::sigmoid(gate_pair[1]);
    auto Bv = B_proj->forward(xn * input_gate);
    auto C = C_proj->forward(xn);
    auto dt = torch::softplus(dt_proj->forward(xn));
    auto a = -torch::exp(A_log);
    auto Ab = torch::exp(dt * a.view({1, 1, state_dim}));
    
    // Call the hyper-fast custom autograd SSM scan function
    auto states = SSMScanFunction::apply(Ab, Bv, C);
    
    auto out = D_proj->forward(ln_state->forward(states)) * gate;
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
    pathway_gate = register_module("pathway_gate", torch::nn::Sequential(
        torch::nn::Linear(config.hidden_size, 128), torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})), torch::nn::GELU(),
        torch::nn::Linear(128, 3)));
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
    auto local_out = local->forward(xa, attention_mask);
    auto low_rank_out = low_rank->forward(xa);
    auto ssm_out = ssm->forward(xa);
    torch::Tensor w;
    if (routing_decision == nullptr || compute_all_pathways) {
        w = optimized::softmax_simd(pathway_gate->forward(xa), -1);
    } else if (is_training()) {
        auto gate_probs = optimized::softmax_simd(pathway_gate->forward(xa), -1);
        w = 0.9 * routing_decision->path_probs + 0.1 * gate_probs;
    } else {
        w = routing_decision->path_probs;
    }
    auto combined = local_out * w.slice(-1, 0, 1) +
                    low_rank_out * w.slice(-1, 1, 2) +
                    ssm_out * w.slice(-1, 2, 3);
    auto residual = x + dropout->forward(combined);
    auto xf = ln2->forward(residual);
    auto ffn_out = routing_decision ? ffn->forward(xf, routing_decision->width_multiplier) : ffn->forward(xf);
    return residual + dropout->forward(ffn_out);
}

// NOTE: The 2-gate GatedMergerImpl previously defined here has been REMOVED.
// The correct 3-gate GatedMergerImpl (matching Python xorzenMergerGate) lives
// in src/model/merger.cpp. The main model uses XorzenMergerGate from merger.h.

} // namespace xorzen

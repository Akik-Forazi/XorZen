"""
Production-grade HASS Block for xorzen-zero.
Hybrid Attention-Shard Switch with 3 pathways: Local Attention, Low-Rank Global, SSM.

FIXES applied:
  [1] SSM: diagonal A_log (no matrix_exp, no full-rank einsum) — O(state) scan
  [2] HASS training blend: 90% router / 10% gate (was 50/50, router had no authority)
  [3] Attention masks cached per seq_len (no reallocation every step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

from xorzen.config import ModelConfig
from xorzen.utils.logger import get_logger
from xorzen.utils.math_utils import TensorStability
from xorzen.model.components.routing import RoutingDecision

logger = get_logger()


# ==================== PATHWAY IMPLEMENTATIONS ====================

class LocalAttentionPathway(nn.Module):
    def __init__(self, hidden_dim, num_heads, window_size, dropout=0.0, causal=True):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        self.window_size = window_size
        self.causal      = causal

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln_q = nn.LayerNorm(self.head_dim)
        self.ln_k = nn.LayerNorm(self.head_dim)

        # Cached masks — rebuilt only when seq_len changes
        self._cached_seq_len: int = -1
        self.register_buffer("_mask_cache", None, persistent=False)

        self._init_weights()
        logger.debug("hass", f"LocalAttention: hidden={hidden_dim}, heads={num_heads}, window={window_size}")

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0 / math.sqrt(2))
            nn.init.zeros_(proj.bias)

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Combined causal + window mask, cached per seq_len."""
        if self._cached_seq_len != seq_len or self._mask_cache is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            if self.window_size > 0:
                pos  = torch.arange(seq_len)
                dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
                mask = mask & (dist <= self.window_size)
            self._mask_cache    = mask
            self._cached_seq_len = seq_len
        return self._mask_cache.to(device)

    def forward(self, x, attention_mask=None, position_bias=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.ln_q(q.transpose(1, 2)).transpose(1, 2)
        k = self.ln_k(k.transpose(1, 2)).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask   = self._get_mask(S, x.device)
        scores = scores.masked_fill(~mask[None, None], float("-inf"))
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        if position_bias is not None:
            scores = scores + position_bias

        probs  = F.softmax(scores, dim=-1)
        probs  = self.attn_dropout(probs)
        out    = (probs @ v).transpose(1, 2).contiguous().view(B, S, self.hidden_dim)
        return self.resid_dropout(self.out_proj(out))

    def get_compute_stats(self, seq_len, batch_size=1):
        qkv   = 3 * batch_size * seq_len * self.hidden_dim ** 2
        attn  = batch_size * self.num_heads * seq_len * seq_len * self.head_dim * 2
        out   = batch_size * seq_len * self.hidden_dim ** 2
        total = qkv + attn + out
        return {
            "flops_total": total, "flops_per_token": total / (batch_size * seq_len),
            "param_count": sum(p.numel() for p in self.parameters()),
            "param_memory_bytes": sum(p.numel() for p in self.parameters()) * 4,
            "activation_memory_bytes": batch_size * seq_len * self.hidden_dim * 4 * 10,
            "window_size": self.window_size,
        }


class LowRankGlobalPathway(nn.Module):
    def __init__(self, hidden_dim, low_rank_dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.low_rank_dim = low_rank_dim
        self.num_heads    = num_heads
        D = low_rank_dim * num_heads
        self.to_low_rank   = nn.Linear(hidden_dim, D)
        self.from_low_rank = nn.Linear(D, hidden_dim)
        self.context_weights = nn.Parameter(torch.randn(1, 1, D))
        self.ln_input    = nn.LayerNorm(hidden_dim)
        self.ln_low_rank = nn.LayerNorm(D)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()
        logger.debug("hass", f"LowRankGlobal: hidden={hidden_dim}, low_rank={low_rank_dim}")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.to_low_rank.weight,   gain=1.0/math.sqrt(2)); nn.init.zeros_(self.to_low_rank.bias)
        nn.init.xavier_uniform_(self.from_low_rank.weight, gain=1.0/math.sqrt(2)); nn.init.zeros_(self.from_low_rank.bias)
        nn.init.normal_(self.context_weights, std=0.02)

    def forward(self, x):
        xn         = self.ln_input(x)
        lr         = F.gelu(self.ln_low_rank(self.to_low_rank(xn)))
        attn_w     = F.softmax(torch.matmul(lr, self.context_weights.transpose(-1, -2)) / math.sqrt(self.low_rank_dim), dim=1)
        global_ctx = torch.matmul(attn_w.transpose(-1, -2), lr).expand(-1, x.size(1), -1)
        return self.dropout(self.from_low_rank(lr + global_ctx))

    def get_compute_stats(self, seq_len, batch_size=1):
        D     = self.low_rank_dim * self.num_heads
        total = batch_size * seq_len * (self.hidden_dim * D + D + D * self.hidden_dim)
        return {
            "flops_total": total, "flops_per_token": total / (batch_size * seq_len),
            "param_count": sum(p.numel() for p in self.parameters()),
            "param_memory_bytes": sum(p.numel() for p in self.parameters()) * 4,
            "activation_memory_bytes": batch_size * seq_len * D * 4 * 3,
            "low_rank_dim": self.low_rank_dim,
        }


class SSMPathway(nn.Module):
    """
    Mamba-style diagonal SSM.

    FIX [1]: A is diagonal (A_log: [state_dim]) — no matrix_exp, no full einsum.
             Discretisation: Ab_t = exp(dt_t * a)  where a = -exp(A_log) < 0.
             Scan recurrence: h_t = Ab_t * h_{t-1} + Bv_t   (elementwise multiply)
             This is O(T * state) not O(T * state²).
    """

    def __init__(self, hidden_dim, state_dim, kernel_size=3, dropout=0.0, use_conv=True):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.state_dim   = state_dim
        self.kernel_size = kernel_size
        self.use_conv    = use_conv

        # Diagonal A — stored as log(-a) so a = -exp(A_log) is always negative
        self.A_log   = nn.Parameter(torch.zeros(state_dim))
        # Input-dependent dt (one dt per state dim per token)
        self.dt_proj = nn.Linear(hidden_dim, state_dim, bias=True)
        self.B_proj  = nn.Linear(hidden_dim, state_dim)
        self.C_proj  = nn.Linear(hidden_dim, state_dim)
        self.D_proj  = nn.Linear(state_dim,  hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        if use_conv:
            self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                                  padding=kernel_size // 2, groups=hidden_dim)
        else:
            self.conv = None

        self.ln_input = nn.LayerNorm(hidden_dim)
        self.ln_state = nn.LayerNorm(state_dim)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()
        logger.debug("hass", f"SSMPathway(diag): hidden={hidden_dim}, state={state_dim}")

    def _init_weights(self):
        with torch.no_grad():
            nn.init.zeros_(self.A_log)          # a = -1 at init
            nn.init.zeros_(self.dt_proj.bias)   # dt starts near softplus(0)=ln2
        for m in [self.B_proj, self.C_proj, self.D_proj, self.gate_proj]:
            nn.init.xavier_uniform_(m.weight, gain=1.0 / math.sqrt(2))
            nn.init.zeros_(m.bias)
        if self.conv is not None:
            nn.init.xavier_uniform_(self.conv.weight, gain=1.0 / math.sqrt(2))
            if self.conv.bias is not None: nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        B, T, _ = x.shape
        xn = self.ln_input(x)

        if self.conv is not None:
            xn = xn + self.conv(xn.transpose(1, 2)).transpose(1, 2)

        gate, input_gate = self.gate_proj(xn).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        Bv = self.B_proj(xn * torch.sigmoid(input_gate))   # [B, T, state]
        C  = self.C_proj(xn)                                # [B, T, state]

        # Diagonal discretisation — purely elementwise
        dt  = F.softplus(self.dt_proj(xn))                  # [B, T, state]  > 0
        a   = -torch.exp(self.A_log)                        # [state]         < 0
        Ab  = torch.exp(dt * a)                             # [B, T, state] in (0,1)

        # Sequential scan — O(T * state), no matrix multiply
        h    = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = Ab[:, t] * h + Bv[:, t]                    # [B, state]
            outs.append(C[:, t] * h)                        # [B, state]
        states = torch.stack(outs, dim=1)                   # [B, T, state]

        out = self.D_proj(self.ln_state(states)) * gate
        return self.dropout(out)

    def forward_parallel(self, x):
        return self.forward(x)

    def get_compute_stats(self, seq_len, batch_size=1):
        proj  = 2 * batch_size * seq_len * self.hidden_dim * self.state_dim
        scan  = batch_size * seq_len * self.state_dim        # elementwise
        total = proj + scan
        if self.conv: total += batch_size * seq_len * self.hidden_dim * self.kernel_size * 2
        return {
            "flops_total": total, "flops_per_token": total / (batch_size * seq_len),
            "param_count": sum(p.numel() for p in self.parameters()),
            "param_memory_bytes": sum(p.numel() for p in self.parameters()) * 4,
            "activation_memory_bytes": batch_size * seq_len * self.state_dim * 4 * 5,
            "state_dim": self.state_dim, "complexity": "O(seq_len * state_dim)",
        }


# ==================== FEED-FORWARD NETWORK ====================

class AdaptiveFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_multiplier=4.0, activation="gelu",
                 dropout=0.0, width_choices=None):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.base_ffn_dim = int(hidden_dim * ffn_multiplier)
        self.activation   = activation
        self.width_choices = width_choices or [hidden_dim]

        self.fc1 = nn.Linear(hidden_dim, self.base_ffn_dim)
        self.fc2 = nn.Linear(self.base_ffn_dim, hidden_dim)

        self.width_adapters = nn.ModuleDict()
        for width in self.width_choices:
            if width != hidden_dim:
                self.width_adapters[str(width)] = nn.Sequential(
                    nn.Linear(hidden_dim, width), self._act(), nn.Linear(width, hidden_dim))

        self.ln_input  = nn.LayerNorm(hidden_dim)
        self.ln_hidden = nn.LayerNorm(self.base_ffn_dim)
        self.dropout     = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()
        logger.debug("hass", f"AdaptiveFFN: hidden={hidden_dim}, ffn={self.base_ffn_dim}")

    def _act(self):
        return {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}.get(self.activation, nn.GELU())

    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight, gain=1.0/math.sqrt(2)); nn.init.zeros_(m.bias)
        for adapter in self.width_adapters.values():
            for layer in adapter:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0/math.sqrt(2)); nn.init.zeros_(layer.bias)

    def forward(self, x, width_multiplier=None, width_idx=None):
        xn  = self.ln_input(x)
        h   = self.ffn_dropout(self.ln_hidden(self._act()(self.fc1(xn))))
        out = self.fc2(h)
        if width_multiplier is not None and width_idx is not None:
            ada = self._apply_width_adaptation(xn, width_idx, width_multiplier)
            out = out * width_multiplier + ada * (1 - width_multiplier)
        return self.dropout(out)

    def _apply_width_adaptation(self, x, width_idx, width_probs=None):
        if self.training and width_probs is not None and self.width_adapters:
            outs = [self.width_adapters[k](x) for k in sorted(self.width_adapters)]
            if outs:
                stacked = torch.stack(outs, dim=-1)
                w = width_probs[..., :len(outs)].unsqueeze(-2)
                return (stacked * w).sum(-1)
        output = torch.zeros_like(x)
        for wv in torch.unique(width_idx):
            wv = wv.item(); m = (width_idx == wv)
            if m.any():
                t = x[m]
                output[m] = self.width_adapters[str(wv)](t) if str(wv) in self.width_adapters else t
        return output

    def get_compute_stats(self, seq_len, batch_size=1, width_multiplier=1.0):
        base = batch_size * seq_len * (self.hidden_dim * self.base_ffn_dim * 2)
        return {
            "flops_total": base, "flops_per_token": base / (batch_size * seq_len),
            "param_count": sum(p.numel() for p in self.parameters()),
            "param_memory_bytes": sum(p.numel() for p in self.parameters()) * 4,
            "base_ffn_dim": self.base_ffn_dim,
        }


# ==================== HASS BLOCK ====================

class HASSBlock(nn.Module):
    """
    Hybrid Attention-Shard Switch Block.

    FIX [2]: Training blend is 90% router / 10% gate.
             Previously 50/50 — the router's learned path distribution was
             being averaged with a random gate, meaning the router signal was
             halved every step and never achieved dominance.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config     = config
        self.layer_idx  = layer_idx
        self.hidden_dim = config.hidden_size

        self.pathways = nn.ModuleDict({
            "local": LocalAttentionPathway(
                config.hidden_size, config.num_attention_heads // 2,
                config.local_window_size, config.dropout, causal=True),
            "low_rank": LowRankGlobalPathway(
                config.hidden_size, config.low_rank_dim, num_heads=4, dropout=config.dropout),
            "ssm": SSMPathway(
                config.hidden_size, config.ssm_state_dim,
                config.ssm_kernel_size, config.dropout, use_conv=True),
        })

        self.pathway_gate = nn.Sequential(
            nn.Linear(config.hidden_size, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 3))

        self.ffn = AdaptiveFFN(config.hidden_size, 4.0, config.hidden_act,
                               config.dropout, config.width_choices)

        self.ln1     = nn.LayerNorm(config.hidden_size)
        self.ln2     = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self._init_weights()

        logger.info("hass", f"HASSBlock {layer_idx}: hidden={config.hidden_size}, "
                    f"local_window={config.local_window_size}, "
                    f"low_rank={config.low_rank_dim}, ssm={config.ssm_state_dim}")

    def _init_weights(self):
        for layer in self.pathway_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0/math.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, x, routing_decision=None, attention_mask=None,
                position_bias=None, compute_all_pathways=False):
        B, T, _ = x.shape
        xa = self.ln1(x)

        if routing_decision is None or compute_all_pathways:
            local_out    = self.pathways["local"](xa, attention_mask, position_bias)
            low_rank_out = self.pathways["low_rank"](xa)
            ssm_out      = self.pathways["ssm"].forward_parallel(xa)
            if routing_decision is None:
                w = F.softmax(self.pathway_gate(xa), dim=-1)
            else:
                w = routing_decision.path_probs
            combined = local_out * w[..., 0:1] + low_rank_out * w[..., 1:2] + ssm_out * w[..., 2:3]

        else:
            path_probs = routing_decision.path_probs
            if self.training:
                local_out    = self.pathways["local"](xa, attention_mask, position_bias)
                low_rank_out = self.pathways["low_rank"](xa)
                ssm_out      = self.pathways["ssm"].forward_parallel(xa)
                # FIX [2]: 90% router authority — gate trains with small gradient share
                gate_probs = F.softmax(self.pathway_gate(xa), dim=-1)
                w = 0.9 * path_probs + 0.1 * gate_probs
                combined = local_out * w[..., 0:1] + low_rank_out * w[..., 1:2] + ssm_out * w[..., 2:3]
            else:
                # Inference: only compute pathways with non-trivial weight
                outs, weights = [], []
                if path_probs[..., 0].max() > 0.01:
                    outs.append(self.pathways["local"](xa, attention_mask, position_bias))
                    weights.append(path_probs[..., 0:1])
                if path_probs[..., 1].max() > 0.01:
                    outs.append(self.pathways["low_rank"](xa))
                    weights.append(path_probs[..., 1:2])
                if path_probs[..., 2].max() > 0.01:
                    outs.append(self.pathways["ssm"].forward_parallel(xa))
                    weights.append(path_probs[..., 2:3])
                if not outs:
                    outs   = [self.pathways["local"](xa), self.pathways["low_rank"](xa),
                              self.pathways["ssm"].forward_parallel(xa)]
                    weights = [torch.full_like(path_probs[..., 0:1], 1/3)] * 3
                ws = sum(weights)
                combined = sum(o * (w / (ws + 1e-12)) for o, w in zip(outs, weights))

        x = x + self.dropout(combined)
        xf = self.ln2(x)
        if routing_decision is not None:
            ffn_out = self.ffn(xf, routing_decision.width_multiplier, routing_decision.width_idx)
        else:
            ffn_out = self.ffn(xf)
        return x + self.dropout(ffn_out)

    def forward_with_depth(self, x, depth_mask, routing_decision=None, attention_mask=None):
        if depth_mask.sum() == 0:
            return x
        active_mask = depth_mask.bool()
        active_x    = x[active_mask].unsqueeze(0)

        if routing_decision is not None:
            mini = RoutingDecision(
                depth_logits=torch.zeros(1, active_x.shape[1], 1, device=x.device),
                depth_probs=torch.ones(1, active_x.shape[1], 1, device=x.device),
                depth_mask=torch.ones(1, active_x.shape[1], 1, device=x.device, dtype=torch.bool),
                width_logits=torch.zeros(1, active_x.shape[1], len(self.config.width_choices), device=x.device),
                width_probs=F.one_hot(routing_decision.width_idx[active_mask],
                                      num_classes=len(self.config.width_choices)).float().unsqueeze(0),
                width_idx=routing_decision.width_idx[active_mask].unsqueeze(0),
                width_multiplier=routing_decision.width_multiplier[active_mask].unsqueeze(0),
                path_logits=torch.zeros(1, active_x.shape[1], 3, device=x.device),
                path_probs=routing_decision.path_probs[active_mask].unsqueeze(0),
                expert_logits=torch.zeros(1, active_x.shape[1], self.config.expert_count, device=x.device),
                expert_probs=torch.zeros(1, active_x.shape[1], self.config.expert_count, device=x.device),
                expert_indices=torch.zeros(1, active_x.shape[1], self.config.top_k_experts, device=x.device, dtype=torch.long),
                expert_weights=torch.zeros(1, active_x.shape[1], self.config.top_k_experts, device=x.device),
                complexity=torch.ones(1, active_x.shape[1], 1, device=x.device),
                uncertainty=torch.zeros(1, active_x.shape[1], 1, device=x.device),
            )
            processed = self.forward(active_x, mini, attention_mask=None)
        else:
            processed = self.forward(active_x, None, None, None, compute_all_pathways=True)

        output = x.clone()
        output[active_mask] = processed.squeeze(0)
        return output

    def get_compute_stats(self, seq_len, batch_size=1, routing_decision=None, depth_active_ratio=1.0):
        active_tokens = int(batch_size * seq_len * depth_active_ratio)
        pathway_stats, total_flops, total_params = {}, 0, 0
        for name, pw in self.pathways.items():
            s = pw.get_compute_stats(seq_len, batch_size)
            pathway_stats[name] = s
            prob = routing_decision.path_probs[..., list(self.pathways.keys()).index(name)].mean().item() \
                   if routing_decision is not None else 1.0
            total_flops  += s["flops_total"] * prob * depth_active_ratio
            total_params += s["param_count"]
        wm = routing_decision.width_multiplier.mean().item() if routing_decision is not None else 1.0
        fs = self.ffn.get_compute_stats(seq_len, batch_size, wm)
        total_flops  += fs["flops_total"] * depth_active_ratio
        total_params += fs["param_count"]
        return {
            "total_flops": total_flops, "flops_per_active_token": total_flops / max(active_tokens, 1),
            "total_params": total_params, "depth_active_ratio": depth_active_ratio,
            "pathway_stats": pathway_stats, "ffn_stats": fs,
        }

    def analyze_pathway_usage(self, routing_decision):
        pp = routing_decision.path_probs
        dominant = torch.argmax(pp.mean(dim=(0, 1))).item()
        return {
            "local_prob": pp[..., 0].mean().item(),
            "low_rank_prob": pp[..., 1].mean().item(),
            "ssm_prob": pp[..., 2].mean().item(),
            "pathway_entropy": (-torch.sum(pp * torch.log(pp + 1e-12), dim=-1)).mean().item(),
            "dominant_pathway": ["local", "low_rank", "ssm"][dominant],
        }


# ==================== HASS BLOCK MANAGER ====================

class HASSBlockManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.blocks = nn.ModuleList([HASSBlock(config, i) for i in range(config.max_depth)])
        self.compute_history, self.pathway_history = [], []
        logger.info("hass", f"HASSBlockManager: {len(self.blocks)} blocks")

    def forward(self, x, routing_decision, attention_mask=None, position_bias=None, collect_stats=False):
        B, T, _ = x.shape
        current = x
        for i, block in enumerate(self.blocks):
            dm = routing_decision.depth_mask[..., i] if i < routing_decision.depth_mask.shape[-1] \
                 else torch.ones(B, T, device=x.device, dtype=torch.bool)
            if dm.sum() == 0: continue
            current = block.forward_with_depth(current, dm, routing_decision, attention_mask)
            if collect_stats:
                dar = dm.float().mean().item()
                s   = block.get_compute_stats(T, B, routing_decision, dar); s["layer"] = i
                self.compute_history.append(s)
                ps = block.analyze_pathway_usage(routing_decision); ps["layer"] = i
                self.pathway_history.append(ps)
        return current

    def get_compute_summary(self):
        if not self.compute_history: return {}
        return {
            "total_blocks": len(self.blocks),
            "total_flops": sum(s["total_flops"] for s in self.compute_history),
            "avg_depth_active_ratio": np.mean([s["depth_active_ratio"] for s in self.compute_history]),
        }

    def reset_statistics(self):
        self.compute_history.clear(); self.pathway_history.clear()


__all__ = ["LocalAttentionPathway", "LowRankGlobalPathway", "SSMPathway",
           "AdaptiveFFN", "HASSBlock", "HASSBlockManager"]

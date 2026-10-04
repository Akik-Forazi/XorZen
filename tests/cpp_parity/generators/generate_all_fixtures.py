"""
Generate deterministic, portable parity fixtures for every C++-portable component.

Outputs to ``tests/cpp_parity/fixtures/<component>/``:
  - ``meta.json``  — config + manifest of all tensors (name, shape, dtype, file)
  - ``input_*.bin``   — raw little-endian bytes for each input tensor
  - ``param_*.bin``   — raw little-endian bytes for each parameter tensor
  - ``expected_*.bin``— raw little-endian bytes for each expected output tensor

All tensors use float32 or int64. The format is intentionally NOT PyTorch-specific
so C++ can consume them with a simple ``std::ifstream::read``.

Determinism: every tensor is generated from a fresh ``torch.Generator().manual_seed(SEED)``
so the same commit + same PyTorch version produces byte-identical fixtures across runs.

Components covered:
   1. normalization      — LayerNorm(hidden) (Python: nn.LayerNorm; C++: torch::nn::LayerNorm)
   2. rope               — learned positional embedding (Python: nn.Embedding; C++: nn::Embedding)
   3. q_proj / k_proj / v_proj  — Linear(hidden, hidden)
   4. attention          — local causal SDPA, B=1,T=8,H=8,heads=2 (small but exercises full path)
   5. ssm_scan           — diagonal ZOH scan, B=1,T=8,N=4 (the recurrence only, NOT the full SSMPathway)
   6. sliced_ffn         — single-width forward at width=max (deterministic, no router dependency)
   7. router             — AdaptiveRouter with deterministic eval-mode routing
   8. moe_aggregation    — weighted top-k aggregation only (no expert compute, just the scatter/sum)
   9. merger             — GatedMerger forward (HASS+MoE+CoT)
  10. lm_head            — Linear(hidden, vocab, bias=False) + softmax
  11. embeddings         — token + position embeddings (combined)
  12. ssm_pathway_full   — full SSMPathway.forward (the C++ has SSMPathwayImpl; includes conv, gates, scan, ln)
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────
FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures"
SEED = 42

# ─────────────────────────────────────────────────────────────────
# Portable tensor I/O — raw little-endian bytes + JSON sidecar.
# ─────────────────────────────────────────────────────────────────
_DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int32:   "int32",
    torch.int64:   "int64",
    torch.bool:    "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _to_numpy(t: torch.Tensor):
    return t.detach().cpu().contiguous().numpy()


def write_tensor(t: torch.Tensor, path: Path) -> None:
    """Write a tensor as raw little-endian bytes (row-major / C-order)."""
    arr = _to_numpy(t)
    arr.tofile(path)


def tensor_meta(t: torch.Tensor, name: str, file_name: str) -> Dict[str, Any]:
    return {
        "name":   name,
        "shape":  list(t.shape),
        "dtype":  _DTYPE_TO_STR[t.dtype],
        "file":   file_name,
        "numel":  int(t.numel()),
        "nbytes": int(t.numel() * t.element_size()),
    }


def write_fixture(component: str,
                  config: Dict[str, Any],
                  inputs: Dict[str, torch.Tensor],
                  params: Dict[str, torch.Tensor],
                  expected: Dict[str, torch.Tensor],
                  tolerance: Dict[str, float]) -> Path:
    """Write a complete fixture directory for one component."""
    d = FIXTURE_ROOT / component
    if d.exists():
        import shutil
        shutil.rmtree(d)
    d.mkdir(parents=True)

    def _dump(prefix: str, tensors: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        out = []
        for name, t in tensors.items():
            fname = f"{prefix}_{name}.bin"
            write_tensor(t, d / fname)
            out.append(tensor_meta(t, name, fname))
        return out

    meta = {
        "component":  component,
        "seed":       SEED,
        "pytorch_version": torch.__version__,
        "config":     config,
        "tolerance":  tolerance,
        "tensors": {
            "inputs":   _dump("input",   inputs),
            "params":   _dump("param",   params),
            "expected": _dump("expected", expected),
        },
    }
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return d


# ─────────────────────────────────────────────────────────────────
# Helpers to pull live parameters from a freshly-seeded zero_tiny_23k
# model so the fixtures match the real production code paths.
# ─────────────────────────────────────────────────────────────────
def make_model_and_inputs():
    """Construct zero_tiny_23k in eval mode and deterministic inputs.

    Returns:
        model, input_ids, labels, hidden_states (token+pos embeddings summed)
    """
    torch.manual_seed(SEED)
    from xorzen.models.zero import zero_tiny_23k
    m = zero_tiny_23k(test_mode=False)
    m.eval()
    cfg = m.config
    B, T = 1, 8
    g_in = torch.Generator().manual_seed(SEED)
    g_lbl = torch.Generator().manual_seed(SEED + 1)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), generator=g_in)
    labels    = torch.randint(0, cfg.vocab_size, (B, T), generator=g_lbl)
    with torch.no_grad():
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        token_emb = m.token_embedding(input_ids)
        pos_emb   = m.position_embedding(pos_ids)
        hidden = token_emb + pos_emb
    return m, cfg, input_ids, labels, hidden


# ─────────────────────────────────────────────────────────────────
# Component fixture generators (1 per component)
# ─────────────────────────────────────────────────────────────────
def gen_normalization() -> Path:
    m, cfg, _, _, hidden = make_model_and_inputs()
    ln = m.blocks[0].ln1  # nn.LayerNorm(cfg.hidden_size)
    with torch.no_grad():
        y = ln(hidden)
    return write_fixture(
        "01_normalization",
        config={"eps": 1e-5, "normalized_shape": [cfg.hidden_size]},
        inputs={"x": hidden},
        params={"weight": ln.weight, "bias": ln.bias},
        expected={"y": y},
        tolerance={"max_abs_error": 1e-6, "max_rel_error": 1e-5},
    )


def gen_rope() -> Path:
    """Positional representation: we use LEARNED position embeddings,
    not sinusoidal RoPE. So this fixture exercises nn.Embedding lookup."""
    m, cfg, input_ids, _, _ = make_model_and_inputs()
    pos_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
    with torch.no_grad():
        pos_emb = m.position_embedding(pos_ids)
    return write_fixture(
        "02_positional_embedding",
        config={"num_embeddings": cfg.context_length,
                "embedding_dim":  cfg.hidden_size},
        inputs={"position_ids": pos_ids},
        params={"weight": m.position_embedding.weight},
        expected={"pos_emb": pos_emb},
        tolerance={"max_abs_error": 0.0, "max_rel_error": 0.0},  # exact lookup
    )


def gen_q_proj() -> Path:
    m, cfg, _, _, hidden = make_model_and_inputs()
    proj = m.blocks[0].pathways["local"].q_proj
    with torch.no_grad():
        y = proj(hidden)
    return write_fixture(
        "03_q_proj",
        config={"in_features": cfg.hidden_size, "out_features": cfg.hidden_size, "bias": True},
        inputs={"x": hidden},
        params={"weight": proj.weight, "bias": proj.bias},
        expected={"y": y},
        tolerance={"max_abs_error": 1e-6, "max_rel_error": 1e-5},
    )


def gen_k_proj() -> Path:
    m, cfg, _, _, hidden = make_model_and_inputs()
    proj = m.blocks[0].pathways["local"].k_proj
    with torch.no_grad():
        y = proj(hidden)
    return write_fixture(
        "04_k_proj",
        config={"in_features": cfg.hidden_size, "out_features": cfg.hidden_size, "bias": True},
        inputs={"x": hidden},
        params={"weight": proj.weight, "bias": proj.bias},
        expected={"y": y},
        tolerance={"max_abs_error": 1e-6, "max_rel_error": 1e-5},
    )


def gen_v_proj() -> Path:
    m, cfg, _, _, hidden = make_model_and_inputs()
    proj = m.blocks[0].pathways["local"].v_proj
    with torch.no_grad():
        y = proj(hidden)
    return write_fixture(
        "05_v_proj",
        config={"in_features": cfg.hidden_size, "out_features": cfg.hidden_size, "bias": True},
        inputs={"x": hidden},
        params={"weight": proj.weight, "bias": proj.bias},
        expected={"y": y},
        tolerance={"max_abs_error": 1e-6, "max_rel_error": 1e-5},
    )


def gen_attention() -> Path:
    """Local causal attention with window=128 (>= T so equivalent to full causal)."""
    m, cfg, _, _, hidden = make_model_and_inputs()
    pathway = m.blocks[0].pathways["local"]
    with torch.no_grad():
        y = pathway(hidden, attention_mask=None, position_bias=None)
    return write_fixture(
        "06_attention",
        config={"hidden_size": cfg.hidden_size,
                "num_heads":   pathway.num_heads,
                "head_dim":    pathway.head_dim,
                "window_size": pathway.window_size,
                "causal":      True},
        inputs={"x": hidden},
        params={
            "q_proj_weight": pathway.q_proj.weight, "q_proj_bias": pathway.q_proj.bias,
            "k_proj_weight": pathway.k_proj.weight, "k_proj_bias": pathway.k_proj.bias,
            "v_proj_weight": pathway.v_proj.weight, "v_proj_bias": pathway.v_proj.bias,
            "out_proj_weight": pathway.out_proj.weight, "out_proj_bias": pathway.out_proj.bias,
            "ln_q_weight": pathway.ln_q.weight, "ln_q_bias": pathway.ln_q.bias,
            "ln_k_weight": pathway.ln_k.weight, "ln_k_bias": pathway.ln_k.bias,
        },
        expected={"y": y},
        # SDPA vs matmul+softmax may differ slightly due to fused kernels
        tolerance={"max_abs_error": 1e-5, "max_rel_error": 1e-4},
    )


def gen_ssm_scan() -> Path:
    """The diagonal ZOH scan recurrence only (not the full SSMPathway).

    Inputs: A_bar [B,T,N], B_bar [B,T,N], C [B,T,N]
    Output: y [B,T,N] where y_t = C_t * h_t, h_t = A_bar_t * h_{t-1} + B_bar_t
    """
    torch.manual_seed(SEED)
    B, T, N = 1, 8, 4
    # Generate A_bar in (0, 1) and B_bar, C as small random values
    g = torch.Generator().manual_seed(SEED)
    A_bar = torch.rand(B, T, N, generator=g) * 0.5 + 0.25   # [0.25, 0.75]
    B_bar = torch.randn(B, T, N, generator=g) * 0.1
    C     = torch.randn(B, T, N, generator=g) * 0.1

    from xorzen.model.components.ssm_scan import sequential_scan
    states = sequential_scan(A_bar, B_bar, init_state=None)  # [B,T,N] h_t
    y = C * states  # [B,T,N]

    return write_fixture(
        "07_ssm_scan",
        config={"B": B, "T": T, "N": N,
                "note": "Diagonal ZOH scan. h_t = A_bar_t * h_{t-1} + B_bar_t. y_t = C_t * h_t."},
        inputs={"A_bar": A_bar, "B_bar": B_bar, "C": C},
        params={},
        expected={"y": y, "states": states},
        tolerance={"max_abs_error": 1e-6, "max_rel_error": 1e-5},
    )


def gen_sliced_ffn() -> Path:
    """SlicedFFN forward at single width = max_width (deterministic)."""
    m, cfg, _, _, hidden = make_model_and_inputs()
    ffn = m.blocks[0].ffn  # SlicedFFN
    # Use max width for determinism (no router dependency)
    max_width = ffn.max_width
    with torch.no_grad():
        y = ffn(hidden, width=max_width)
    return write_fixture(
        "08_sliced_ffn",
        config={"hidden_size": ffn.hidden_dim, "max_width": max_width,
                "activation": "gelu", "width_used": max_width},
        inputs={"x": hidden},
        params={
            "fc1_weight": ffn.fc1.weight, "fc1_bias": ffn.fc1.bias,
            "fc2_weight": ffn.fc2.weight, "fc2_bias": ffn.fc2.bias,
            "ln_input_weight": ffn.ln_input.weight, "ln_input_bias": ffn.ln_input.bias,
            "ln_hidden_weight": ffn.ln_hidden.weight, "ln_hidden_bias": ffn.ln_hidden.bias,
        },
        expected={"y": y},
        # GELU difference: PyTorch uses exact erf; C++ may use tanh approx
        tolerance={"max_abs_error": 1e-3, "max_rel_error": 1e-3},
    )


def gen_router() -> Path:
    """AdaptiveRouter in eval mode (deterministic).

    Captures: depth_logits, depth_probs, depth_mask,
              width_logits, width_probs, width_idx,
              path_logits, path_probs,
              expert_logits, expert_probs, expert_indices, expert_weights,
              complexity, uncertainty.
    """
    m, cfg, input_ids, _, hidden = make_model_and_inputs()
    router = m.router
    B, T = hidden.shape[:2]
    cot_features = torch.zeros(B, T, cfg.cot_dim * cfg.cot_components)
    with torch.no_grad():
        rd = router(x=hidden, cot_features=cot_features, training=False)
    return write_fixture(
        "09_router",
        config={"hidden_size": cfg.hidden_size, "max_depth": cfg.max_depth,
                "num_widths": len(cfg.width_choices), "num_paths": 3,
                "num_experts": cfg.expert_count, "top_k": cfg.top_k_experts,
                "temperature": cfg.router_temperature,
                "cost_aware_routing": bool(getattr(cfg, 'cost_aware_routing', True)),
                "compute_budget": float(getattr(cfg, 'compute_budget', 1.0))},
        inputs={"x": hidden, "cot_features": cot_features},
        params={
            # We capture ALL router parameters so C++ can reproduce exact matmuls.
            **{f"router.{n}": p for n, p in router.named_parameters()}
        },
        expected={
            "depth_logits": rd.depth_logits, "depth_probs": rd.depth_probs,
            "depth_mask":   rd.depth_mask,
            "width_logits": rd.width_logits, "width_probs": rd.width_probs,
            "width_idx":    rd.width_idx,
            "path_logits":  rd.path_logits,  "path_probs":  rd.path_probs,
            "expert_logits": rd.expert_logits, "expert_probs": rd.expert_probs,
            "expert_indices": rd.expert_indices, "expert_weights": rd.expert_weights,
            "complexity":   rd.complexity,    "uncertainty": rd.uncertainty,
        },
        tolerance={"max_abs_error": 1e-5, "max_rel_error": 1e-4},
    )


def gen_moe_aggregation() -> Path:
    """Weighted top-k expert aggregation only (no expert compute).

    Given expert_outputs [num_experts, H] and expert_indices [B, T, top_k] and
    expert_weights [B, T, top_k], compute the weighted sum per token.
    """
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    B, T, H = 1, 8, 8
    num_experts = 4
    top_k = 2
    expert_outputs = torch.randn(num_experts, H, generator=g) * 0.5
    expert_indices = torch.randint(0, num_experts, (B, T, top_k), generator=g)
    expert_weights = torch.rand(B, T, top_k, generator=g)
    expert_weights = expert_weights / expert_weights.sum(-1, keepdim=True)

    # Aggregation: y[b,t,h] = sum_k expert_weights[b,t,k] * expert_outputs[expert_indices[b,t,k], h]
    gathered = expert_outputs[expert_indices]  # [B, T, top_k, H]
    y = (gathered * expert_weights.unsqueeze(-1)).sum(-2)  # [B, T, H]

    return write_fixture(
        "10_moe_aggregation",
        config={"num_experts": num_experts, "top_k": top_k, "hidden_size": H},
        inputs={"expert_outputs": expert_outputs,
                "expert_indices": expert_indices,
                "expert_weights": expert_weights},
        params={},
        expected={"y": y},
        tolerance={"max_abs_error": 1e-6, "max_rel_error": 1e-5},
    )


def gen_merger() -> Path:
    m, cfg, _, _, hidden = make_model_and_inputs()
    merger = m.merger
    B, T = hidden.shape[:2]
    moe_out = torch.zeros_like(hidden)
    cot_vector = torch.zeros(B, T, cfg.cot_dim * cfg.cot_components)
    with torch.no_grad():
        y = merger(hass_output=hidden, moe_output=moe_out, cot_vector=cot_vector)
    return write_fixture(
        "11_merger",
        config={"hidden_size": cfg.hidden_size,
                "cot_dim":     cfg.cot_dim * cfg.cot_components,
                "merger_type": cfg.merger_type},
        inputs={"hass_output": hidden, "moe_output": moe_out, "cot_vector": cot_vector},
        params={f"merger.{n}": p for n, p in merger.named_parameters()},
        expected={"y": y},
        tolerance={"max_abs_error": 1e-5, "max_rel_error": 1e-4},
    )


def gen_lm_head() -> Path:
    m, cfg, _, _, hidden = make_model_and_inputs()
    head = m.lm_head  # tied with token_embedding
    with torch.no_grad():
        logits = head(hidden)
        probs = F.softmax(logits, dim=-1)
    return write_fixture(
        "12_lm_head",
        config={"hidden_size": cfg.hidden_size, "vocab_size": cfg.vocab_size, "bias": False},
        inputs={"x": hidden},
        params={"weight": head.weight},
        expected={"logits": logits, "probs": probs},
        tolerance={"max_abs_error": 1e-6, "max_rel_error": 1e-5},
    )


def gen_embeddings() -> Path:
    m, cfg, input_ids, _, _ = make_model_and_inputs()
    pos_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
    with torch.no_grad():
        token_emb = m.token_embedding(input_ids)
        pos_emb   = m.position_embedding(pos_ids)
        combined  = token_emb + pos_emb
    return write_fixture(
        "13_embeddings",
        config={"vocab_size": cfg.vocab_size, "context_length": cfg.context_length,
                "hidden_size": cfg.hidden_size, "pad_token_id": cfg.pad_token_id},
        inputs={"input_ids": input_ids, "position_ids": pos_ids},
        params={"token_embedding_weight": m.token_embedding.weight,
                "position_embedding_weight": m.position_embedding.weight},
        expected={"token_emb": token_emb, "pos_emb": pos_emb, "combined": combined},
        tolerance={"max_abs_error": 0.0, "max_rel_error": 0.0},
    )


def gen_ssm_pathway_full() -> Path:
    """Full SSMPathway.forward (conv + gates + scan + ln + D_proj)."""
    m, cfg, _, _, hidden = make_model_and_inputs()
    pathway = m.blocks[0].pathways["ssm"]
    with torch.no_grad():
        # Use forward_parallel — that's what the model uses in production
        y = pathway.forward_parallel(hidden)
    return write_fixture(
        "14_ssm_pathway_full",
        config={"hidden_size": pathway.hidden_dim, "state_dim": pathway.state_dim,
                "kernel_size": pathway.kernel_size, "use_conv": pathway.use_conv},
        inputs={"x": hidden},
        params={f"ssm_pathway.{n}": p for n, p in pathway.named_parameters()},
        expected={"y": y},
        tolerance={"max_abs_error": 1e-5, "max_rel_error": 1e-4},
    )


# ─────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────
GENERATORS = [
    ("01_normalization",         gen_normalization),
    ("02_positional_embedding",  gen_rope),
    ("03_q_proj",                gen_q_proj),
    ("04_k_proj",                gen_k_proj),
    ("05_v_proj",                gen_v_proj),
    ("06_attention",             gen_attention),
    ("07_ssm_scan",              gen_ssm_scan),
    ("08_sliced_ffn",            gen_sliced_ffn),
    ("09_router",                gen_router),
    ("10_moe_aggregation",       gen_moe_aggregation),
    ("11_merger",                gen_merger),
    ("12_lm_head",               gen_lm_head),
    ("13_embeddings",            gen_embeddings),
    ("14_ssm_pathway_full",      gen_ssm_pathway_full),
]


def main():
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(GENERATORS)} component fixtures into {FIXTURE_ROOT}")
    summary = []
    for name, fn in GENERATORS:
        try:
            d = fn()
            # Compute total fixture size
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            meta = json.loads((d / "meta.json").read_text())
            n_in = len(meta["tensors"]["inputs"])
            n_pa = len(meta["tensors"]["params"])
            n_ex = len(meta["tensors"]["expected"])
            print(f"  [OK] {name:32s} in={n_in:2d}  params={n_pa:2d}  expected={n_ex:2d}  size={total:>8,}B")
            summary.append({"component": name, "status": "ok",
                            "inputs": n_in, "params": n_pa, "expected": n_ex, "size_bytes": total})
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name:32s} {e}")
            traceback.print_exc()
            summary.append({"component": name, "status": "fail", "error": str(e)})

    # Write top-level manifest
    manifest = {
        "seed": SEED,
        "pytorch_version": torch.__version__,
        "components": summary,
        "total_components": len(summary),
        "ok": sum(1 for s in summary if s["status"] == "ok"),
        "fail": sum(1 for s in summary if s["status"] == "fail"),
    }
    (FIXTURE_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {FIXTURE_ROOT / 'manifest.json'}")
    print(f"OK: {manifest['ok']}/{manifest['total_components']}")


if __name__ == "__main__":
    main()

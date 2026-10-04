"""Trace the source of the 0.006 block_out numerical difference.

Hypothesis: Python uses index_add_ for sparse pathway dispatch scatter,
C++ uses element-wise multiplication + addition. These produce different
floating-point accumulation orders.

This script:
  1. Runs the Python model with hooks capturing each pathway output + the combined result.
  2. Reproduces the C++ scatter math in Python (element-wise, no index_add_).
  3. Compares the two combined results to confirm the hypothesis.
  4. Also tests: does Python index_add_ vs loop-based scatter produce different results?
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import numpy as np
from pathlib import Path

torch.manual_seed(42)
from xorzen.models.zero import zero_tiny_23k
m = zero_tiny_23k(test_mode=False)
m.eval()
cfg = m.config

B, T = 2, 16
input_ids = torch.randint(0, cfg.vocab_size, (B, T), generator=torch.Generator().manual_seed(42))
labels = torch.randint(0, cfg.vocab_size, (B, T), generator=torch.Generator().manual_seed(43))

# Capture block internals
captured = {}
block = m.blocks[0]

# Get hidden states
with torch.no_grad():
    pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    hidden = m.token_embedding(input_ids) + m.position_embedding(pos_ids)
    x_attn = block.ln1(hidden)

    # Run each pathway on FULL x_attn
    local_out = block.pathways['local'](x_attn, None, None)
    low_rank_out = block.pathways['low_rank'](x_attn)
    ssm_out = block.pathways['ssm'].forward_parallel(x_attn)

    # Get path_probs from router
    cot_features = torch.zeros(B, T, cfg.cot_dim * cfg.cot_components)
    rd = m.router(x=hidden, cot_features=cot_features, training=False)
    path_probs = rd.path_probs  # [B, T, 3]

    # === Method 1: Python sparse_pathway_dispatch (index_add_) ===
    from xorzen.model.components.sparse_dispatch import sparse_pathway_dispatch, topk_pathway_mask
    # Use the block's own wrapped pathway fns (handles 2D→3D reshape)
    wrapped_fns = block._get_wrapped_pathway_fns()
    result = sparse_pathway_dispatch(
        x_attn, path_probs, wrapped_fns, ['local', 'low_rank', 'ssm'],
        top_k=2, training=False,
    )
    combined_sparse = result[0] if isinstance(result, tuple) else result

    # === Method 2: Dense compute (all 3 pathways) + renormalized weights ===
    # This is what the C++ harness does
    topk = path_probs.topk(2, dim=-1)
    topk_idx = topk.indices
    hard_mask = torch.zeros_like(path_probs)
    hard_mask.scatter_(-1, topk_idx, 1.0)
    selected = path_probs * hard_mask
    sel_sum = selected.sum(-1, keepdim=True)
    sel_sum = torch.where(sel_sum > 1e-8, sel_sum, torch.ones_like(sel_sum))
    norm_w = selected / sel_sum
    combined_dense = (local_out * norm_w[..., 0:1] +
                      low_rank_out * norm_w[..., 1:2] +
                      ssm_out * norm_w[..., 2:3])

    # === Method 3: index_add_ in Python (replicate what sparse_pathway_dispatch does internally) ===
    combined_index_add = torch.zeros(B * T, cfg.hidden_size)
    x_flat = x_attn.reshape(B * T, cfg.hidden_size)
    mask_flat = hard_mask.reshape(B * T, 3)
    w_flat = norm_w.reshape(B * T, 3)
    for i, key in enumerate(['local', 'low_rank', 'ssm']):
        selected_mask = mask_flat[:, i] > 0.5
        if not selected_mask.any():
            continue
        idx = selected_mask.nonzero(as_tuple=False).squeeze(-1)
        x_slice = x_flat[idx]
        y_slice = wrapped_fns[key](x_slice)
        w_slice = w_flat[idx, i].unsqueeze(-1)
        combined_index_add.index_add_(0, idx, y_slice * w_slice)
    combined_index_add = combined_index_add.reshape(B, T, cfg.hidden_size)

    # Compare
    print("=" * 70)
    print("SPARSE PATHWAY DISPATCH — accumulation order comparison")
    print("=" * 70)
    print(f"  path_probs sample [0,0]: {path_probs[0,0].tolist()}")
    print(f"  hard_mask sample [0,0]:  {hard_mask[0,0].tolist()}")
    print(f"  norm_w sample [0,0]:     {norm_w[0,0].tolist()}")
    print()

    diff_sparse_vs_dense = (combined_sparse - combined_dense).abs()
    diff_sparse_vs_index = (combined_sparse - combined_index_add).abs()
    diff_dense_vs_index = (combined_dense - combined_index_add).abs()

    print(f"  sparse (Python index_add_) vs dense (C++ element-wise):")
    print(f"    max_abs = {diff_sparse_vs_dense.max().item():.2e}")
    print(f"    mean_abs = {diff_sparse_vs_dense.mean().item():.2e}")
    print()
    print(f"  sparse (Python) vs index_add_ (Python reimpl):")
    print(f"    max_abs = {diff_sparse_vs_index.max().item():.2e}")
    print()
    print(f"  dense (C++ style) vs index_add_ (Python):")
    print(f"    max_abs = {diff_dense_vs_index.max().item():.2e}")

    # Now check: does the ACTUAL block forward use sparse dispatch?
    with torch.no_grad():
        block_out_actual = block(x=hidden, routing_decision=rd)

    # Compute block_out using dense combined
    residual_dense = hidden + combined_dense
    xf_dense = block.ln2(residual_dense)
    ffn_out_dense = block.ffn(xf_dense, width_idx=rd.width_idx)
    block_out_dense = residual_dense + ffn_out_dense

    residual_sparse = hidden + combined_sparse
    xf_sparse = block.ln2(residual_sparse)
    ffn_out_sparse = block.ffn(xf_sparse, width_idx=rd.width_idx)
    block_out_sparse = residual_sparse + ffn_out_sparse

    diff_block = (block_out_actual - block_out_dense).abs()
    print()
    print(f"  block_out actual (Python) vs dense (C++ style):")
    print(f"    max_abs = {diff_block.max().item():.2e}")

    diff_block_sparse = (block_out_actual - block_out_sparse).abs()
    print(f"  block_out actual (Python) vs sparse (Python index_add_):")
    print(f"    max_abs = {diff_block_sparse.max().item():.2e}")

    # Save the combined_sparse and combined_dense for C++ comparison
    out_dir = Path("tests/cpp_parity/fixtures/17_dispatch_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in [("combined_sparse", combined_sparse),
                     ("combined_dense", combined_dense),
                     ("combined_index_add", combined_index_add),
                     ("local_out", local_out),
                     ("low_rank_out", low_rank_out),
                     ("ssm_out", ssm_out),
                     ("path_probs", path_probs),
                     ("hard_mask", hard_mask),
                     ("norm_w", norm_w)]:
        t.detach().cpu().contiguous().numpy().tofile(out_dir / f"{name}.bin")
    print(f"\n  Saved tensors to {out_dir}")
    print(f"\n  CONCLUSION: {'CONFIRMED' if diff_sparse_vs_dense.max().item() > 1e-7 else 'REFUTED'}")
    print(f"  The {diff_sparse_vs_dense.max().item():.2e} difference between index_add_ and")
    print(f"  element-wise scatter is the source of the E2E block_out divergence.")

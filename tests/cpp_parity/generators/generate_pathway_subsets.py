"""Generate per-pathway subset fixtures: extract the EXACT token subsets that
Python's sparse dispatch passes to each pathway, run Python's pathway on them,
and save everything for C++ comparison."""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, numpy as np
from pathlib import Path

torch.manual_seed(42)
from xorzen.models.zero import zero_tiny_23k
m = zero_tiny_23k(test_mode=False)
m.eval()
cfg = m.config

B, T = 2, 16
input_ids = torch.randint(0, cfg.vocab_size, (B, T), generator=torch.Generator().manual_seed(42))
labels = torch.randint(0, cfg.vocab_size, (B, T), generator=torch.Generator().manual_seed(43))

with torch.no_grad():
    pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    hidden = m.token_embedding(input_ids) + m.position_embedding(pos_ids)
    cot_features = torch.zeros(B, T, cfg.cot_dim * cfg.cot_components)
    rd = m.router(x=hidden, cot_features=cot_features, training=False)
    block = m.blocks[0]
    x_attn = block.ln1(hidden)

    # Build hard top-k mask (same as C++)
    topk = rd.path_probs.topk(2, dim=-1)
    hard_mask = torch.zeros_like(rd.path_probs)
    hard_mask.scatter_(-1, topk.indices, 1.0)
    selected = rd.path_probs * hard_mask
    sel_sum = selected.sum(-1, keepdim=True)
    sel_sum = torch.where(sel_sum > 1e-8, sel_sum, torch.ones_like(sel_sum))
    norm_w = selected / sel_sum

    # Flatten for dispatch
    x_flat = x_attn.reshape(B * T, cfg.hidden_size)
    mask_flat = hard_mask.reshape(B * T, 3)
    w_flat = norm_w.reshape(B * T, 3)

    pathway_names = ['local', 'low_rank', 'ssm']
    wrapped_fns = block._get_wrapped_pathway_fns()

    out_dir = Path("tests/cpp_parity/fixtures/19_pathway_subsets")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save state_dict (same as before)
    sd = m.state_dict()
    lines_sd = ["state_dict"]
    for name, t in sd.items():
        arr = t.detach().cpu().contiguous().numpy()
        fname = f"param_{name}.bin"
        arr.tofile(out_dir / fname)
        shape_csv = ",".join(str(s) for s in arr.shape)
        dtype = "float32" if arr.dtype == np.float32 else "int64" if arr.dtype == np.int64 else str(arr.dtype)
        lines_sd.append(f"tensor {name} {dtype} {fname} {shape_csv}")
    lines_sd.append("end")
    (out_dir / "state_dict_manifest.txt").write_text("\n".join(lines_sd) + "\n")

    # For each pathway, extract subset, run Python forward, save
    mlines = ["component 19_pathway_subsets"]
    mlines.append(f"config hidden_size {cfg.hidden_size}")
    mlines.append(f"config num_attention_heads {cfg.num_attention_heads}")
    mlines.append(f"config ssm_state_dim {cfg.ssm_state_dim}")
    mlines.append(f"config ssm_kernel_size {cfg.ssm_kernel_size}")
    mlines.append(f"config low_rank_dim {cfg.low_rank_dim}")
    mlines.append(f"config local_window_size {cfg.local_window_size}")
    mlines.append(f"config B {B}")
    mlines.append(f"config T {T}")
    mlines.append(f"config layer_norm_eps {cfg.layer_norm_eps}")
    mlines.append("tolerance max_abs 1e-5")
    mlines.append("tolerance max_rel 1e-4")

    # Save x_attn (full) for reference
    arr = x_attn.detach().cpu().contiguous().numpy()
    arr.tofile(out_dir / "expected_x_attn_full.bin")
    mlines.append(f"tensor expected x_attn_full float32 expected_x_attn_full.bin {','.join(str(s) for s in arr.shape)}")

    # Save path_probs, hard_mask, norm_w
    for name, t in [("path_probs", rd.path_probs), ("hard_mask", hard_mask), ("norm_w", norm_w)]:
        arr = t.detach().cpu().contiguous().numpy()
        arr.tofile(out_dir / f"expected_{name}.bin")
        mlines.append(f"tensor expected {name} float32 expected_{name}.bin {','.join(str(s) for s in arr.shape)}")

    for i, name in enumerate(pathway_names):
        sel = mask_flat[:, i] > 0.5
        n_sel = sel.sum().item()
        if n_sel == 0:
            print(f"  {name}: 0 tokens selected, skipping")
            continue

        idx = sel.nonzero(as_tuple=False).squeeze(-1)  # [n_sel]
        x_slice = x_flat[idx]  # [n_sel, H]
        x3d = x_slice.unsqueeze(0)  # [1, n_sel, H]

        # Run Python pathway forward on the subset
        if name == 'local':
            y3d = block.pathways['local'](x3d, None, None)
        elif name == 'low_rank':
            y3d = block.pathways['low_rank'](x3d)
        elif name == 'ssm':
            y3d = block.pathways['ssm'].forward_parallel(x3d)

        y_slice = y3d.squeeze(0)  # [n_sel, H]
        w_slice = w_flat[idx, i].unsqueeze(-1)  # [n_sel, 1]
        weighted = y_slice * w_slice  # [n_sel, H]

        # Save: x_slice (input), y_slice (output), idx (indices), w_slice (weights), weighted
        for tname, t in [
            (f"{name}_x_slice", x_slice),
            (f"{name}_y_slice", y3d.squeeze(0)),
            (f"{name}_idx", idx),
            (f"{name}_w_slice", w_slice.squeeze(-1)),
            (f"{name}_weighted", weighted),
        ]:
            arr = t.detach().cpu().contiguous().numpy()
            fname = f"expected_{tname}.bin"
            arr.tofile(out_dir / fname)
            shape_csv = ",".join(str(s) for s in arr.shape)
            dtype = "float32" if arr.dtype == np.float32 else "int64"
            mlines.append(f"tensor expected {tname} {dtype} {fname} {shape_csv}")

        print(f"  {name}: {n_sel} tokens selected, y_slice shape={tuple(y_slice.shape)}, "
              f"y max={y_slice.abs().max().item():.6e}")

    # Also save the full combined_sparse (Python's actual result)
    result = __import__('xorzen.model.components.sparse_dispatch', fromlist=['sparse_pathway_dispatch']).sparse_pathway_dispatch(
        x_attn, rd.path_probs, wrapped_fns, pathway_names, top_k=2, training=False)
    combined = result[0] if isinstance(result, tuple) else result
    arr = combined.detach().cpu().contiguous().numpy()
    arr.tofile(out_dir / "expected_combined_sparse.bin")
    mlines.append(f"tensor expected combined_sparse float32 expected_combined_sparse.bin {','.join(str(s) for s in arr.shape)}")

    # Save input_ids and labels
    for name, t in [("input_ids", input_ids), ("labels", labels)]:
        arr = t.detach().cpu().contiguous().numpy()
        arr.tofile(out_dir / f"expected_{name}.bin")
        mlines.append(f"tensor expected {name} int64 expected_{name}.bin {','.join(str(s) for s in arr.shape)}")

    mlines.append("end")
    (out_dir / "manifest.txt").write_text("\n".join(mlines) + "\n")
    print(f"\nSaved to {out_dir}")

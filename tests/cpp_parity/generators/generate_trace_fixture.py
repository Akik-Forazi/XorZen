"""Trace the EXACT first tensor where Python and C++ diverge.

Generates a step-by-step fixture with every intermediate tensor in the HASS block,
so the C++ harness can compare each one and find the first divergence.
"""
import sys, json
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

# Capture every intermediate tensor
captured = {}

# 1. Embeddings
with torch.no_grad():
    pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    token_emb = m.token_embedding(input_ids)
    pos_emb = m.position_embedding(pos_ids)
    hidden = token_emb + pos_emb
    captured["hidden"] = hidden

    # 2. Router
    cot_features = torch.zeros(B, T, cfg.cot_dim * cfg.cot_components)
    rd = m.router(x=hidden, cot_features=cot_features, training=False)
    captured["path_probs"] = rd.path_probs
    captured["depth_mask"] = rd.depth_mask
    captured["width_idx"] = rd.width_idx
    captured["width_multiplier"] = rd.width_multiplier

    # 3. HASS block internals
    block = m.blocks[0]
    x_attn = block.ln1(hidden)
    captured["x_attn"] = x_attn

    # 3a. Pathway outputs (on FULL x_attn)
    local_out = block.pathways['local'](x_attn, None, None)
    low_rank_out = block.pathways['low_rank'](x_attn)
    ssm_out = block.pathways['ssm'].forward_parallel(x_attn)
    captured["local_out"] = local_out
    captured["low_rank_out"] = low_rank_out
    captured["ssm_out"] = ssm_out

    # 3b. Sparse dispatch
    from xorzen.model.components.sparse_dispatch import sparse_pathway_dispatch
    wrapped_fns = block._get_wrapped_pathway_fns()
    result = sparse_pathway_dispatch(
        x_attn, rd.path_probs, wrapped_fns, ['local', 'low_rank', 'ssm'],
        top_k=2, training=False,
    )
    combined = result[0] if isinstance(result, tuple) else result
    captured["combined_sparse"] = combined

    # 3c. Dense element-wise (what C++ does)
    topk = rd.path_probs.topk(2, dim=-1)
    topk_idx = topk.indices
    hard_mask = torch.zeros_like(rd.path_probs)
    hard_mask.scatter_(-1, topk_idx, 1.0)
    selected = rd.path_probs * hard_mask
    sel_sum = selected.sum(-1, keepdim=True)
    sel_sum = torch.where(sel_sum > 1e-8, sel_sum, torch.ones_like(sel_sum))
    norm_w = selected / sel_sum
    combined_dense = (local_out * norm_w[..., 0:1] +
                      low_rank_out * norm_w[..., 1:2] +
                      ssm_out * norm_w[..., 2:3])
    captured["combined_dense"] = combined_dense
    captured["norm_w"] = norm_w
    captured["hard_mask"] = hard_mask

    # 3d. Block output
    block_out = block(x=hidden, routing_decision=rd)
    captured["block_out"] = block_out

    # 4. MoE
    hidden_flat = block_out.reshape(B * T, -1)
    expert_indices_flat = rd.expert_indices.reshape(B * T, -1)
    expert_weights_flat = rd.expert_weights.reshape(B * T, -1)
    moe_output_flat, moe_stats = m.moe(hidden_flat, expert_indices_flat, expert_weights_flat)
    moe_output = moe_output_flat.reshape(B, T, -1)
    captured["moe_output"] = moe_output

    # 5. Merger
    cot_vector = torch.zeros(B, T, cfg.cot_dim * cfg.cot_components)
    merged = m.merger(hass_output=block_out, moe_output=moe_output, cot_vector=cot_vector)
    captured["merged"] = merged

    # 6. Final norm + logits
    final_norm = m.final_norm(merged)
    captured["final_norm"] = final_norm
    logits = m.lm_head(final_norm)
    captured["logits"] = logits

    # 7. Loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    lm_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, cfg.vocab_size),
        shift_labels.view(-1),
        ignore_index=cfg.pad_token_id)
    captured["lm_loss"] = lm_loss

# Save everything
out_dir = Path("tests/cpp_parity/fixtures/18_trace_divergence")
out_dir.mkdir(parents=True, exist_ok=True)

# Save state_dict
sd = m.state_dict()
torch.save(sd, out_dir / "state_dict.pt")
# Per-tensor bins
lines = ["state_dict"]
for name, t in sd.items():
    arr = t.detach().cpu().contiguous().numpy()
    fname = f"param_{name}.bin"
    arr.tofile(out_dir / fname)
    shape_csv = ",".join(str(s) for s in arr.shape)
    dtype = "float32" if arr.dtype == np.float32 else "int64" if arr.dtype == np.int64 else str(arr.dtype)
    lines.append(f"tensor {name} {dtype} {fname} {shape_csv}")
lines.append("end")
(out_dir / "state_dict_manifest.txt").write_text("\n".join(lines) + "\n")

# Save captured tensors
mlines = ["component 18_trace_divergence"]
for k, v in [
    ("vocab_size", cfg.vocab_size), ("hidden_size", cfg.hidden_size),
    ("num_layers", cfg.num_layers), ("num_attention_heads", cfg.num_attention_heads),
    ("max_depth", cfg.max_depth), ("num_widths", len(cfg.width_choices)),
    ("num_paths", 3), ("num_experts", cfg.expert_count), ("top_k", cfg.top_k_experts),
    ("ssm_state_dim", cfg.ssm_state_dim), ("ssm_kernel_size", cfg.ssm_kernel_size),
    ("context_length", cfg.context_length), ("pad_token_id", cfg.pad_token_id),
    ("B", B), ("T", T),
    ("temperature", cfg.router_temperature), ("eval_routing_noise", cfg.eval_routing_noise),
    ("cot_dim", cfg.cot_dim), ("cot_components", cfg.cot_components),
    ("layer_norm_eps", cfg.layer_norm_eps),
    ("width_choices", ",".join(str(w) for w in cfg.width_choices)),
]:
    mlines.append(f"config {k} {v}")
mlines.append("tolerance max_abs 1e-5")
mlines.append("tolerance max_rel 1e-4")

for name, t in captured.items():
    arr = t.detach().cpu().contiguous().numpy()
    fname = f"expected_{name}.bin"
    arr.tofile(out_dir / fname)
    shape_csv = ",".join(str(s) for s in arr.shape)
    dtype = "float32" if arr.dtype == np.float32 else "int64" if arr.dtype == np.int64 else str(arr.dtype)
    mlines.append(f"tensor expected {name} {dtype} {fname} {shape_csv}")

# Also save input_ids and labels
for name, t in [("input_ids", input_ids), ("labels", labels)]:
    arr = t.detach().cpu().contiguous().numpy()
    fname = f"expected_{name}.bin"
    arr.tofile(out_dir / fname)
    shape_csv = ",".join(str(s) for s in arr.shape)
    dtype = "int64"
    mlines.append(f"tensor expected {name} {dtype} {fname} {shape_csv}")

mlines.append("end")
(out_dir / "manifest.txt").write_text("\n".join(mlines) + "\n")

print(f"Saved {len(captured)} + 2 = {len(captured)+2} tensors to {out_dir}")
print(f"State dict: {len(sd)} keys")
for name, t in captured.items():
    print(f"  {name}: {tuple(t.shape)} mean={t.float().mean().item():.6f}")
print(f"\n  combined_sparse vs combined_dense max_abs: {(captured['combined_sparse'] - captured['combined_dense']).abs().max().item():.6e}")

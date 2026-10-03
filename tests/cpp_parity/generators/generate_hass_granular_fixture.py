"""Generate a granular HASS-block fixture: captures each pathway output separately
so we can isolate which pathway diverges in C++."""
import sys
sys.path.insert(0, "/home/z/my-project/XorZen")
from pathlib import Path
import numpy as np
import torch

torch.manual_seed(42)
from xorzen.models.zero import zero_tiny_23k
m = zero_tiny_23k(test_mode=False)
m.eval()
cfg = m.config

B, T = 2, 16
input_ids = torch.randint(0, cfg.vocab_size, (B, T),
                          generator=torch.Generator().manual_seed(42))
labels = torch.randint(0, cfg.vocab_size, (B, T),
                       generator=torch.Generator().manual_seed(43))

# Get combined embeddings (block input)
with torch.no_grad():
    pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    hidden = m.token_embedding(input_ids) + m.position_embedding(pos_ids)

block = m.blocks[0]
with torch.no_grad():
    x_attn = block.ln1(hidden)
    # Run each pathway on the FULL x_attn (not sparse dispatch)
    local_out = block.pathways['local'](x_attn, None, None)
    low_rank_out = block.pathways['low_rank'](x_attn)
    ssm_out = block.pathways['ssm'].forward_parallel(x_attn)
    # Capture path_probs from router (eval mode, deterministic)
    cot_features = torch.zeros(B, T, cfg.cot_dim * cfg.cot_components)
    rd = m.router(x=hidden, cot_features=cot_features, training=False)
    path_probs = rd.path_probs
    # Aggregation
    combined = (local_out * path_probs[..., 0:1] +
                low_rank_out * path_probs[..., 1:2] +
                ssm_out * path_probs[..., 2:3])
    residual = hidden + combined
    xf = block.ln2(residual)
    ffn_out = block.ffn(xf, width=block.ffn.max_width)
    block_out = residual + ffn_out

OUT_DIR = ROOT = Path("/home/z/my-project/XorZen/tests/cpp_parity/fixtures/16_hass_block_granular")
if OUT_DIR.exists():
    import shutil
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

# Save state_dict for the block submodules + router
sd = {}
for prefix, mod in [("blocks.0", block), ("router", m.router)]:
    for n, p in mod.state_dict().items():
        sd[f"{prefix}.{n}"] = p
# Also need ln1, ln2 from block directly
# Actually block.state_dict() already includes ln1, ln2, pathways.*, ffn.*
# Let's also save token_embedding + position_embedding for completeness
sd["token_embedding.weight"] = m.token_embedding.weight
sd["position_embedding.weight"] = m.position_embedding.weight

# Save per-tensor bins + manifest
lines = ["state_dict"]
for name, t in sd.items():
    arr = t.detach().cpu().contiguous().numpy()
    fname = f"param_{name}.bin"
    arr.tofile(OUT_DIR / fname)
    shape_csv = ",".join(str(s) for s in arr.shape)
    dtype = "float32" if arr.dtype == np.float32 else "int64" if arr.dtype == np.int64 else str(arr.dtype
)
    lines.append(f"tensor {name} {dtype} {fname} {shape_csv}")
lines.append("end")
(OUT_DIR / "state_dict_manifest.txt").write_text("\n".join(lines) + "\n")

# Save expected tensors
captured = {
    "hidden": hidden,
    "x_attn": x_attn,
    "local_out": local_out,
    "low_rank_out": low_rank_out,
    "ssm_out": ssm_out,
    "path_probs": path_probs,
    "combined": combined,
    "residual": residual,
    "xf": xf,
    "ffn_out": ffn_out,
    "block_out": block_out,
}

# Manifest
mlines = ["component 16_hass_block_granular"]
mlines.append(f"config hidden_size {cfg.hidden_size}")
mlines.append(f"config num_attention_heads {cfg.num_attention_heads}")
mlines.append(f"config ssm_state_dim {cfg.ssm_state_dim}")
mlines.append(f"config ssm_kernel_size {cfg.ssm_kernel_size}")
mlines.append(f"config local_window_size {cfg.local_window_size}")
mlines.append(f"config low_rank_dim {cfg.low_rank_dim}")
mlines.append(f"config max_depth {cfg.max_depth}")
mlines.append(f"config num_widths {len(cfg.width_choices)}")
mlines.append(f"config num_paths 3")
mlines.append(f"config num_experts {cfg.expert_count}")
mlines.append(f"config top_k {cfg.top_k_experts}")
mlines.append(f"config temperature {cfg.router_temperature}")
mlines.append(f"config cost_aware_routing 1")
mlines.append(f"config compute_budget 1.0")
mlines.append(f"config eval_routing_noise {cfg.eval_routing_noise}")
mlines.append(f"config cot_dim {cfg.cot_dim}")
mlines.append(f"config cot_components {cfg.cot_components}")
mlines.append(f"config layer_norm_eps {cfg.layer_norm_eps}")
mlines.append(f"config B {B}")
mlines.append(f"config T {T}")
mlines.append(f"config width_choices {','.join(str(w) for w in cfg.width_choices)}")
mlines.append("tolerance max_abs 1e-5")
mlines.append("tolerance max_rel 1e-4")

for name, t in captured.items():
    arr = t.detach().cpu().contiguous().numpy()
    fname = f"expected_{name}.bin"
    arr.tofile(OUT_DIR / fname)
    shape_csv = ",".join(str(s) for s in arr.shape)
    dtype = "float32" if arr.dtype == np.float32 else "int64" if arr.dtype == np.int64 else str(arr.dtype
)
    mlines.append(f"tensor expected {name} {dtype} {fname} {shape_csv}")

mlines.append("end")
(OUT_DIR / "manifest.txt").write_text("\n".join(mlines) + "\n")

print(f"Saved granular HASS block fixture to {OUT_DIR}")
for name, t in captured.items():
    print(f"  {name}: shape={tuple(t.shape)}  mean={t.float().mean().item():.6f}")

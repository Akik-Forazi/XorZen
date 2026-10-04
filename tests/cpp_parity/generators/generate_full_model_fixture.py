"""Generate a full-model end-to-end fixture for the C++ harness.

Runs a deterministic zero_tiny_23k forward pass and captures:
  - input_ids, labels
  - token_emb, pos_emb, combined_embeddings
  - block_0_input, block_0_output
  - merger_output, final_norm_out, logits
  - lm_loss, total_loss

Also saves the full state_dict so the C++ harness can load real weights.

Output: tests/cpp_parity/fixtures/15_full_model/
  - state_dict.pt  (loadable via torch::load in C++)
  - manifest.txt   (config + expected tensors)
  - expected_*.bin (raw tensor bytes)
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_DIR = ROOT / "tests" / "cpp_parity" / "fixtures" / "15_full_model"

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

# Capture intermediate tensors via hooks
captured = {}
def make_hook(name):
    def h(mod, inp, out):
        if isinstance(out, torch.Tensor):
            captured[name] = out.detach().clone()
        elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            captured[name] = out[0].detach().clone()
    return h

hooks = []
hooks.append(m.token_embedding.register_forward_hook(make_hook("token_emb")))
hooks.append(m.position_embedding.register_forward_hook(make_hook("pos_emb")))
def block_input_hook(mod, args, kwargs):
    x = kwargs.get("x")
    if x is None and args:
        x = args[0]
    if isinstance(x, torch.Tensor):
        captured["block_0_input"] = x.detach().clone()

hooks.append(m.blocks[0].register_forward_pre_hook(block_input_hook, with_kwargs=True))
hooks.append(m.blocks[0].register_forward_hook(make_hook("block_0_output")))
hooks.append(m.merger.register_forward_hook(make_hook("merger_output")))
hooks.append(m.final_norm.register_forward_hook(make_hook("final_norm_out")))
hooks.append(m.lm_head.register_forward_hook(make_hook("logits")))

with torch.no_grad():
    out = m(input_ids=input_ids, labels=labels, return_dict=True)

for h in hooks:
    h.remove()

captured["input_ids"] = input_ids
captured["labels"] = labels
captured["combined_embeddings"] = captured["token_emb"] + captured["pos_emb"]
captured["lm_loss"] = out.lm_loss.detach().clone()
captured["total_loss"] = out.loss.detach().clone()
# logits already captured via hook

# Save state_dict — both as .pt (for reference) AND as per-tensor .bin files
# (C++ can't easily read torch.save(dict) format; it reads raw .bin via ifstream).
if OUT_DIR.exists():
    import shutil
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)
state_dict = m.state_dict()
torch.save(state_dict, OUT_DIR / "state_dict.pt")

# Dump each param as a .bin file with a sidecar manifest
sd_lines = ["state_dict"]
for name, t in state_dict.items():
    arr = t.detach().cpu().contiguous().numpy()
    fname = f"param_{name}.bin"
    arr.tofile(OUT_DIR / fname)
    shape_csv = ",".join(str(s) for s in arr.shape)
    dtype = "float32" if arr.dtype == np.float32 else ("int64" if arr.dtype == np.int64 else str(arr.dtype)
)
    sd_lines.append(f"tensor {name} {dtype} {fname} {shape_csv}")
sd_lines.append("end")
(OUT_DIR / "state_dict_manifest.txt").write_text("\n".join(sd_lines) + "\n")

# Write manifest + expected tensors
lines = ["component 15_full_model"]
# Config
config_lines = [
    ("vocab_size", cfg.vocab_size),
    ("hidden_size", cfg.hidden_size),
    ("num_layers", cfg.num_layers),
    ("num_attention_heads", cfg.num_attention_heads),
    ("max_depth", cfg.max_depth),
    ("num_widths", len(cfg.width_choices)),
    ("num_paths", 3),
    ("num_experts", cfg.expert_count),
    ("top_k", cfg.top_k_experts),
    ("temperature", cfg.router_temperature),
    ("cost_aware_routing", 1),
    ("compute_budget", 1.0),
    ("eval_routing_noise", cfg.eval_routing_noise),
    ("context_length", cfg.context_length),
    ("pad_token_id", cfg.pad_token_id),
    ("ssm_state_dim", cfg.ssm_state_dim),
    ("ssm_kernel_size", cfg.ssm_kernel_size),
    ("local_window_size", cfg.local_window_size),
    ("low_rank_dim", cfg.low_rank_dim),
    ("expert_hidden_multiplier", cfg.expert_hidden_multiplier),
    ("merger_hidden_multiplier", cfg.merger_hidden_multiplier),
    ("layer_norm_eps", cfg.layer_norm_eps),
    ("cot_dim", cfg.cot_dim),
    ("cot_components", cfg.cot_components),
    ("B", B),
    ("T", T),
    ("width_choices", ",".join(str(w) for w in cfg.width_choices)),
]
for k, v in config_lines:
    lines.append(f"config {k} {v}")
lines.append("tolerance max_abs 1e-5")
lines.append("tolerance max_rel 1e-4")

# Expected tensors
for name, t in captured.items():
    arr = t.detach().cpu().contiguous().numpy()
    fname = f"expected_{name}.bin"
    arr.tofile(OUT_DIR / fname)
    shape_csv = ",".join(str(s) for s in arr.shape)
    if arr.dtype == np.float32:
        dtype = "float32"
    elif arr.dtype == np.int64:
        dtype = "int64"
    else:
        dtype = str(arr.dtype)
    lines.append(f"tensor expected {name} {dtype} {fname} {shape_csv}")

lines.append("end")
(OUT_DIR / "manifest.txt").write_text("\n".join(lines) + "\n")

print(f"Saved full-model fixture to {OUT_DIR}")
print(f"  state_dict: {len(state_dict)} keys, {sum(v.numel() for v in state_dict.values()):,} params")
print(f"  captured tensors: {len(captured)}")
print(f"  logits shape: {tuple(captured['logits'].shape)}")
print(f"  lm_loss: {captured['lm_loss'].item():.6f}")
print(f"  total_loss: {captured['total_loss'].item():.6f}")

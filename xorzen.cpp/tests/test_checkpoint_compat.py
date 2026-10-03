"""
XorZen Checkpoint Compatibility Test
=====================================

Tests that a Python-trained checkpoint can be loaded into C++ (and vice
versa) with identical parameter names, shapes, and values.

This script:
1. Creates a Python xorzen model (zero_1M)
2. Saves its state_dict to a .pt file
3. Exports the state_dict keys + shapes + values to JSON
4. Runs the model forward and saves logits + loss
5. The C++ side loads the .pt file and compares

Usage:
    python test_checkpoint_compat.py --model zero_1M --output checkpoint_test/
"""
import sys
sys.path.insert(0, "/home/z/my-project/XorZen")

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero import zero_1M, zero_10M, zero_50M, zero_tiny_23k

MODEL_FACTORIES = {
    "zero_tiny_23k": (zero_tiny_23k, ModelSize.TINY_23K),
    "zero_1M": (zero_1M, ModelSize.NANO_1M),
    "zero_10M": (zero_10M, ModelSize.NANO_10M),
    "zero_50M": (zero_50M, ModelSize.MICRO_50M),
}


def tensor_to_list(t):
    if t is None:
        return None
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.float()
    return t.detach().cpu().numpy().tolist()


def test_checkpoint_compat(model_name: str, output_dir: str, seed: int = 42):
    """Generate checkpoint + reference data for C++ compatibility testing."""
    factory, size = MODEL_FACTORIES[model_name]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Checkpoint Compatibility Test: {model_name} ===")

    # 1. Create model with fixed seed
    torch.manual_seed(seed)
    model = factory(test_mode=False)
    model.eval()

    cfg = model.config
    print(f"  Model: {cfg.model_name}")
    print(f"  Params: {model.count_parameters():,}")
    print(f"  Vocab: {cfg.vocab_size}, Hidden: {cfg.hidden_size}, Layers: {cfg.num_layers}")

    # 2. Save state_dict as .pt (PyTorch native format)
    state_dict = model.state_dict()
    pt_path = out_dir / f"{model_name}_state_dict.pt"
    torch.save(state_dict, str(pt_path))
    print(f"  Saved state_dict: {pt_path} ({pt_path.stat().st_size / (1024*1024):.1f} MB)")

    # 3. Export state_dict metadata to JSON (for C++ to verify key names + shapes)
    sd_metadata = {}
    for name, tensor in state_dict.items():
        sd_metadata[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "numel": tensor.numel(),
        }
    sd_meta_path = out_dir / f"{model_name}_state_dict_metadata.json"
    with open(sd_meta_path, "w") as f:
        json.dump(sd_metadata, f, indent=2)
    print(f"  Saved metadata: {sd_meta_path} ({len(sd_metadata)} keys)")

    # 4. Generate fixed input + run forward
    torch.manual_seed(seed + 1)
    seq_len = min(cfg.context_length, 64)
    batch_size = 2
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels, return_dict=True)

    # 5. Save reference output
    ref_output = {
        "model_name": model_name,
        "config": {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "context_length": cfg.context_length,
            "expert_count": cfg.expert_count,
            "top_k_experts": cfg.top_k_experts,
            "width_choices": list(cfg.width_choices),
            "cot_dim": cfg.cot_dim,
            "cot_components": cfg.cot_components,
            "max_depth": cfg.max_depth,
            "min_depth": cfg.min_depth,
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "pad_token_id": cfg.pad_token_id,
        },
        "inputs": {
            "input_ids": tensor_to_list(input_ids),
            "labels": tensor_to_list(labels),
            "shape": [batch_size, seq_len],
        },
        "output": {
            "logits_shape": list(output.logits.shape),
            "logits": tensor_to_list(output.logits),
            "loss": output.loss.item() if output.loss is not None else None,
            "routing_loss": output.routing_loss.item() if output.routing_loss is not None else None,
            "load_balance_loss": output.load_balance_loss.item() if output.load_balance_loss is not None else None,
        },
        "state_dict_keys": sorted(state_dict.keys()),
        "seed": seed,
    }

    ref_path = out_dir / f"{model_name}_reference.json"
    with open(ref_path, "w") as f:
        json.dump(ref_output, f, indent=2)
    print(f"  Saved reference: {ref_path}")
    print(f"  Logits shape: {tuple(output.logits.shape)}")
    print(f"  Loss: {output.loss.item():.6f}")

    # 6. Verify state_dict key naming matches Python convention
    # Check for common parity issues
    issues = []
    expected_prefixes = ["token_embedding", "position_embedding", "embedding_dropout",
                         "cot.", "router.", "blocks.", "moe.", "merger.",
                         "final_norm", "lm_head"]
    found_prefixes = set()
    for key in state_dict.keys():
        for prefix in expected_prefixes:
            if key.startswith(prefix):
                found_prefixes.add(prefix)
                break

    missing_prefixes = set(expected_prefixes) - found_prefixes
    if missing_prefixes:
        issues.append(f"Missing expected state_dict prefixes: {missing_prefixes}")

    # Check for C++-only keys (like cot_loss_head)
    unexpected_keys = [k for k in state_dict.keys() if "cot_loss_head" in k]
    if unexpected_keys:
        issues.append(f"Unexpected keys (should not exist in Python): {unexpected_keys}")

    if issues:
        print(f"\n  ⚠️  PARITY ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ✅ State dict key naming looks correct")

    # 7. Summary
    print(f"\n=== Summary ===")
    print(f"  Files generated in {out_dir}/:")
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
    print(f"\n  C++ compatibility test steps:")
    print(f"    1. Load {pt_path.name} via torch::serialize::InputArchive")
    print(f"    2. Verify state_dict keys match {sd_meta_path.name}")
    print(f"    3. Run C++ forward with same input_ids from {ref_path.name}")
    print(f"    4. Compare C++ logits vs Python logits (tolerance: 1e-5 fp32, 1e-3 bf16)")
    print(f"    5. Compare C++ loss vs Python loss (tolerance: 1e-5)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate checkpoint compatibility test data.",
    )
    parser.add_argument("--model", default="zero_1M", choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--output", default="checkpoint_test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_checkpoint_compat(args.model, args.output, args.seed)


if __name__ == "__main__":
    main()

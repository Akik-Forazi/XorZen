"""
XorZen Python Golden Reference Generator
=========================================

Exports intermediate activations from the Python xorzen model at every
layer of the forward pass, so the C++ implementation can compare against
them for numerical parity.

Usage:
    python generate_golden_reference.py --model zero_1M --output golden_ref.json

The output JSON contains:
  - Model config (exact hyperparameters)
  - Input tokens (random, seeded)
  - State dict (all parameter names + shapes + values as lists)
  - Intermediate activations at each forward stage:
      embeddings, cot_vector, routing_decision.*, block_outputs[*],
      moe_output, merger_output, final_norm, logits, loss
  - Metadata (torch version, seed, dtype)

The C++ parity test loads this file and compares each tensor against
the C++ forward pass output.
"""
import sys
sys.path.insert(0, "/home/z/my-project/XorZen")

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero import (
    zero_tiny_23k, zero_1M, zero_10M, zero_50M, zero_277M
)

MODEL_FACTORIES = {
    "zero_tiny_23k": zero_tiny_23k,
    "zero_1M": zero_1M,
    "zero_10M": zero_10M,
    "zero_50M": zero_50M,
    "zero_277M": zero_277M,
}

MODEL_SIZES = {
    "zero_tiny_23k": ModelSize.TINY_23K,
    "zero_1M": ModelSize.NANO_1M,
    "zero_10M": ModelSize.NANO_10M,
    "zero_50M": ModelSize.MICRO_50M,
    "zero_277M": ModelSize.MINI_277M,
}


def tensor_to_list(t: torch.Tensor) -> List:
    """Convert tensor to nested list, handling bfloat16/fp16 by casting to fp32."""
    if t is None:
        return None
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.float()
    return t.detach().cpu().numpy().tolist()


def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    """Compute stats for parity comparison."""
    if t is None:
        return {"present": False}
    t = t.detach().float()
    return {
        "present": True,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std().item() if t.numel() > 1 else 0.0,
        "norm": t.norm().item(),
        "has_nan": bool(torch.isnan(t).any()),
        "has_inf": bool(torch.isinf(t).any()),
        "numel": t.numel(),
    }


def generate_golden_reference(model_name: str, output_path: str, seed: int = 42):
    """Generate golden reference for a specific model variant."""
    print(f"Generating golden reference for {model_name}...")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate model
    factory = MODEL_FACTORIES[model_name]
    model = factory(test_mode=False)
    model.eval()  # eval mode for deterministic output

    cfg = model.config
    seq_len = min(cfg.context_length, 64)  # small for fast comparison
    batch_size = 2
    vocab = cfg.vocab_size

    # Generate fixed input
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab, (batch_size, seq_len))
    labels = torch.randint(0, vocab, (batch_size, seq_len))

    # Collect intermediate activations via hooks
    intermediates: Dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                intermediates[name] = output.clone()
            elif hasattr(output, 'logits'):
                intermediates[name] = output.logits.clone()
        return hook

    # Register hooks on key modules
    hooks = []
    if hasattr(model, 'token_embedding'):
        hooks.append(model.token_embedding.register_forward_hook(make_hook("embeddings.token")))
    if hasattr(model, 'position_embedding'):
        hooks.append(model.position_embedding.register_forward_hook(make_hook("embeddings.position")))
    if hasattr(model, 'cot'):
        hooks.append(model.cot.register_forward_hook(make_hook("cot")))
    if hasattr(model, 'router'):
        hooks.append(model.router.register_forward_hook(make_hook("router")))
    if hasattr(model, 'moe'):
        hooks.append(model.moe.register_forward_hook(make_hook("moe")))
    if hasattr(model, 'merger'):
        hooks.append(model.merger.register_forward_hook(make_hook("merger")))
    if hasattr(model, 'final_norm'):
        hooks.append(model.final_norm.register_forward_hook(make_hook("final_norm")))
    if hasattr(model, 'lm_head'):
        hooks.append(model.lm_head.register_forward_hook(make_hook("lm_head")))

    # Register hooks on each HASS block
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_hook(f"blocks.{i}")))

    # Run forward pass
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels, return_dict=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Build state dict (parameter names + shapes only for the main report,
    # values in a separate section)
    state_dict = model.state_dict()
    state_dict_info = {}
    for name, tensor in state_dict.items():
        state_dict_info[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "numel": tensor.numel(),
        }

    # Build config dict
    config_dict = {}
    for attr in ["model_name", "vocab_size", "hidden_size", "num_layers",
                 "num_attention_heads", "context_length", "expert_count",
                 "top_k_experts", "width_choices", "cot_dim", "cot_components",
                 "max_depth", "min_depth", "ssm_state_dim", "local_window_size",
                 "low_rank_dim", "tie_word_embeddings", "gradient_checkpointing",
                 "pad_token_id", "router_hidden_dim", "target_active_ratio"]:
        val = getattr(cfg, attr, None)
        if val is not None:
            if isinstance(val, tuple):
                val = list(val)
            config_dict[attr] = val

    # Build the golden reference
    golden = {
        "metadata": {
            "model_name": model_name,
            "torch_version": torch.__version__,
            "seed": seed,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "vocab_size": vocab,
            "total_params": model.count_parameters(),
            "trainable_params": model.count_parameters(only_trainable=True),
            "generated_at": torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else None,
        },
        "config": config_dict,
        "inputs": {
            "input_ids": tensor_to_list(input_ids),
            "labels": tensor_to_list(labels),
        },
        "state_dict_info": state_dict_info,
        "state_dict_values": {
            name: tensor_to_list(tensor) for name, tensor in state_dict.items()
        },
        "intermediates_stats": {
            name: tensor_stats(t) for name, t in intermediates.items()
        },
        "intermediates_values": {
            name: tensor_to_list(t) for name, t in intermediates.items()
        },
        "output": {
            "logits": tensor_stats(output.logits),
            "logits_values": tensor_to_list(output.logits),
            "loss": output.loss.item() if output.loss is not None else None,
            "lm_loss": output.lm_loss.item() if hasattr(output, 'lm_loss') and output.lm_loss is not None else None,
            "routing_loss": output.routing_loss.item() if hasattr(output, 'routing_loss') and output.routing_loss is not None else None,
            "load_balance_loss": output.load_balance_loss.item() if hasattr(output, 'load_balance_loss') and output.load_balance_loss is not None else None,
        },
    }

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(golden, f, indent=2)

    print(f"  Total params: {model.count_parameters():,}")
    print(f"  State dict keys: {len(state_dict)}")
    print(f"  Intermediate activations: {len(intermediates)}")
    print(f"  Logits shape: {tuple(output.logits.shape)}")
    print(f"  Loss: {output.loss.item():.6f}")
    print(f"  Saved to: {out_path}")
    print(f"  File size: {out_path.stat().st_size / (1024*1024):.1f} MB")

    return golden


def main():
    parser = argparse.ArgumentParser(
        description="Generate Python golden reference for C++ parity testing.",
    )
    parser.add_argument(
        "--model", default="zero_1M",
        choices=list(MODEL_FACTORIES.keys()),
        help="Model variant to generate reference for.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Default: golden_ref_<model>.json",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    output_path = args.output or f"golden_ref_{args.model}.json"
    generate_golden_reference(args.model, output_path, args.seed)


if __name__ == "__main__":
    main()

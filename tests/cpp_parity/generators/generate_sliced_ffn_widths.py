#!/usr/bin/env python3
"""Generate SlicedFFN parity fixtures at multiple widths.

Creates 4 fixtures:
  - 08_sliced_ffn_narrow:   width = max_width // 4 (narrowest)
  - 08_sliced_ffn_intermediate: width = max_width // 2
  - 08_sliced_ffn_max:      width = max_width (existing, regenerated)
  - 08_sliced_ffn_mixed:    per-token mixed widths (different tokens use different widths)

Each fixture captures the Python SlicedFFN output so the C++ port can be
compared against it. The mixed fixture is the most important — it would
fail if the C++ implementation silently uses max width for all tokens.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import torch
import numpy as np
from pathlib import Path

torch.manual_seed(42)

from xorzen.models.zero import zero_tiny_23k
from xorzen.model.components.sliced_ffn import SlicedFFN

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "tests" / "cpp_parity" / "fixtures"


def write_fixture(name, config, inputs, params, expected, tolerance=None):
    """Write a parity fixture directory with .bin files and manifest.txt."""
    d = FIXTURES_DIR / name
    d.mkdir(parents=True, exist_ok=True)

    # Write config as manifest.txt (key=value lines)
    with open(d / "manifest.txt", "w") as f:
        for k, v in config.items():
            f.write(f"{k}={v}\n")
        f.write(f"tolerance_max_abs={tolerance.get('max_abs_error', 1e-5) if tolerance else 1e-5}\n")
        f.write(f"tolerance_max_rel={tolerance.get('max_rel_error', 1e-5) if tolerance else 1e-5}\n")

    # Write tensors as .bin (raw float32 bytes) + a listing in manifest
    for tensor_name, tensor in {**inputs, **params, **expected}.items():
        t = tensor.detach().contiguous().float().cpu()
        t.numpy().tofile(d / f"input_{tensor_name}.bin") if tensor_name in inputs else \
        t.numpy().tofile(d / f"param_{tensor_name}.bin") if tensor_name in params else \
        t.numpy().tofile(d / f"expected_{tensor_name}.bin")
        # Append shape to manifest
        shape_str = ",".join(str(s) for s in t.shape)
        with open(d / "manifest.txt", "a") as f:
            f.write(f"shape_{tensor_name}={shape_str}\n")

    print(f"  Wrote {name}/ ({d.stat().st_size} bytes manifest)")
    return d


def main():
    print("Generating SlicedFFN multi-width parity fixtures...")

    torch.manual_seed(42)
    m = zero_tiny_23k(test_mode=False)
    m.eval()
    ffn = m.blocks[0].ffn  # SlicedFFN instance
    assert isinstance(ffn, SlicedFFN), f"Expected SlicedFFN, got {type(ffn)}"

    hidden_dim = ffn.hidden_dim       # 8 for tiny_23k
    max_width = ffn.max_width         # 32 for tiny_23k
    width_choices = ffn.width_choices  # e.g. [8, 32] for tiny_23k

    print(f"  hidden_dim={hidden_dim}, max_width={max_width}")
    print(f"  width_choices={width_choices}")

    # Create deterministic input
    B, T = 2, 16
    torch.manual_seed(42)
    x = torch.randn(B, T, hidden_dim)

    # Common params
    params = {
        "fc1_weight": ffn.fc1.weight,
        "fc1_bias": ffn.fc1.bias,
        "fc2_weight": ffn.fc2.weight,
        "fc2_bias": ffn.fc2.bias,
        "ln_input_weight": ffn.ln_input.weight,
        "ln_input_bias": ffn.ln_input.bias,
        "ln_hidden_weight": ffn.ln_hidden.weight,
        "ln_hidden_bias": ffn.ln_hidden.bias,
    }

    # Fixture 1: narrow width (width_choices[0] — the smallest)
    narrow_w = width_choices[0]
    with torch.no_grad():
        y_narrow = ffn(x, width=narrow_w)
    write_fixture(
        "08_sliced_ffn_narrow",
        config={"hidden_size": hidden_dim, "max_width": max_width,
                "activation": "gelu", "width_used": narrow_w,
                "width_choices": ",".join(str(w) for w in width_choices)},
        inputs={"x": x},
        params=params,
        expected={"y": y_narrow},
        tolerance={"max_abs_error": 1e-4, "max_rel_error": 1e-4},
    )

    # Fixture 2: intermediate width (middle choice or max_width//2)
    if len(width_choices) >= 3:
        inter_w = width_choices[len(width_choices) // 2]
    else:
        inter_w = max_width // 2
    # Ensure inter_w is different from narrow and max
    if inter_w == narrow_w or inter_w == max_width:
        inter_w = max(1, (narrow_w + max_width) // 2)
    with torch.no_grad():
        y_inter = ffn(x, width=inter_w)
    write_fixture(
        "08_sliced_ffn_intermediate",
        config={"hidden_size": hidden_dim, "max_width": max_width,
                "activation": "gelu", "width_used": inter_w,
                "width_choices": ",".join(str(w) for w in width_choices)},
        inputs={"x": x},
        params=params,
        expected={"y": y_inter},
        tolerance={"max_abs_error": 1e-4, "max_rel_error": 1e-4},
    )

    # Fixture 3: max width (regenerate existing)
    with torch.no_grad():
        y_max = ffn(x, width=max_width)
    write_fixture(
        "08_sliced_ffn_max",
        config={"hidden_size": hidden_dim, "max_width": max_width,
                "activation": "gelu", "width_used": max_width,
                "width_choices": ",".join(str(w) for w in width_choices)},
        inputs={"x": x},
        params=params,
        expected={"y": y_max},
        tolerance={"max_abs_error": 1e-4, "max_rel_error": 1e-4},
    )

    # Fixture 4: MIXED per-token widths (the critical test)
    # Assign different widths to different tokens. This would FAIL if the
    # C++ implementation silently uses max width for all tokens.
    # Use width_idx: [B, T] long tensor indexing into width_choices.
    # Pattern: alternate between narrow and max width across tokens.
    width_idx = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            if t % 3 == 0:
                width_idx[b, t] = 0  # narrow (width_choices[0])
            elif t % 3 == 1:
                width_idx[b, t] = len(width_choices) - 1  # max (last choice)
            else:
                # Use intermediate if available, else narrow
                width_idx[b, t] = min(1, len(width_choices) - 1)
    with torch.no_grad():
        y_mixed = ffn(x, width_idx=width_idx)
    write_fixture(
        "08_sliced_ffn_mixed",
        config={"hidden_size": hidden_dim, "max_width": max_width,
                "activation": "gelu",
                "width_choices": ",".join(str(w) for w in width_choices),
                "width_idx_pattern": "per_token_mixed"},
        inputs={"x": x, "width_idx": width_idx},
        params=params,
        expected={"y": y_mixed},
        tolerance={"max_abs_error": 1e-4, "max_rel_error": 1e-4},
    )

    # Also write the width_idx as a separate input file for the mixed fixture
    mixed_dir = FIXTURES_DIR / "08_sliced_ffn_mixed"
    width_idx.numpy().tofile(mixed_dir / "input_width_idx.bin")

    print()
    print("All 4 SlicedFFN fixtures generated:")
    for name in ["08_sliced_ffn_narrow", "08_sliced_ffn_intermediate",
                 "08_sliced_ffn_max", "08_sliced_ffn_mixed"]:
        d = FIXTURES_DIR / name
        files = list(d.glob("*.bin"))
        print(f"  {name}: {len(files)} .bin files")


if __name__ == "__main__":
    main()

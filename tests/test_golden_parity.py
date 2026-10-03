"""
Phase 5: Deterministic Python golden parity test (.pt format).

Captures a comprehensive set of intermediate tensors from a fixed-seed
zero_tiny_23k forward pass and saves them in PyTorch's native .pt format
(via ``torch.save``). A companion manifest JSON records tensor metadata
(shapes, dtypes, min/max/mean/std), the full model config, seed, PyTorch
version, the XorZen git commit, and a SHA256 hash of the .pt file.

The test verifies determinism by re-running the forward pass with a fresh
model instance (same seed) and comparing EVERY captured tensor for exact
equality. It also re-loads the .pt file and verifies the round-trip.

Configuration:
  - Model: zero_tiny_23k  (37K params, 1 layer, 1 expert, top_k=1, 1 width)
  - Batch: B=2, T=16
  - Seed:  42
  - Mode:  eval (deterministic routing, no dropout)

Captured tensors (stored in the .pt file as ``dict[str, torch.Tensor]``):

  Inputs:
    - input_ids, labels

  Embeddings:
    - token_emb_out, pos_emb_out, combined_embeddings

  Router (re-run separately — RoutingDecision is a dataclass, not a tensor,
          so module hooks cannot capture its fields directly):
    - router_feature_encoder_out       (hook on router.feature_encoder)
    - depth_logits, depth_probs, depth_mask
    - width_logits, width_probs, width_idx
    - path_logits, path_probs
    - expert_logits, expert_probs, expert_indices, expert_weights
    - complexity, uncertainty

  HASS block 0 (the only block — tiny_23k has num_layers=1):
    - block_0_input, block_0_output
    - block_0_local_out, block_0_low_rank_out, block_0_ssm_out
        (re-run on full ln1(block_input) — the model's sparse-dispatch path
         only invokes pathways on token subsets, so hooks would capture
         partial slices, not the full pathway output)
    - block_0_ffn_out                   (hook on block.ffn)

  MoE:
    - moe_input, moe_output, moe_load_balance_loss

  Merger:
    - merger_hass_input, merger_moe_input, merger_cot_input, merger_output

  Final:
    - final_norm_out, logits

  Losses (from ModelOutput):
    - lm_loss, routing_loss, load_balance_loss,
    - cot_consistency_loss, total_loss
"""
import sys
sys.path.insert(0, "/home/z/my-project/XorZen")

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# ─── Test result tracking ─────────────────────────────────────────
results: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


print("=" * 70)
print("PHASE 5: DETERMINISTIC GOLDEN PARITY TEST (.pt format)")
print("=" * 70)

# ─── Setup ────────────────────────────────────────────────────────
SEED = 42
B, T = 2, 16

torch.manual_seed(SEED)

from xorzen.models.zero import zero_tiny_23k

model = zero_tiny_23k(test_mode=False)
model.eval()  # eval mode: no dropout, deterministic routing

cfg = model.config

# Deterministic inputs (seeded generators, independent of global RNG state)
input_ids = torch.randint(
    0, cfg.vocab_size, (B, T),
    generator=torch.Generator().manual_seed(SEED),
)
labels = torch.randint(
    0, cfg.vocab_size, (B, T),
    generator=torch.Generator().manual_seed(SEED + 1),
)

print(f"\n  Model:           {cfg.model_name}")
print(f"  Params:          {model.count_parameters():,}")
print(f"  Vocab:           {cfg.vocab_size}")
print(f"  Hidden:          {cfg.hidden_size}")
print(f"  Layers:          {cfg.num_layers}")
print(f"  Experts:         {cfg.expert_count} (top_k={cfg.top_k_experts})")
print(f"  Width choices:   {tuple(cfg.width_choices)}")
print(f"  Pathway top-k:   {getattr(cfg, 'pathway_top_k', 2)}")
print(f"  Input:           batch={B}, seq={T}")
print(f"  Seed:            {SEED}")
print(f"  Mode:            eval")

# ─── Expected tensor list (used for verification & manifest ordering) ─
EXPECTED_TENSORS: List[str] = [
    # Inputs
    "input_ids", "labels",
    # Embeddings
    "token_emb_out", "pos_emb_out", "combined_embeddings",
    # Router
    "router_feature_encoder_out",
    "depth_logits", "depth_probs", "depth_mask",
    "width_logits", "width_probs", "width_idx",
    "path_logits", "path_probs",
    "expert_logits", "expert_probs", "expert_indices", "expert_weights",
    "complexity", "uncertainty",
    # HASS block 0
    "block_0_input", "block_0_output",
    "block_0_local_out", "block_0_low_rank_out", "block_0_ssm_out",
    "block_0_ffn_out",
    # MoE
    "moe_input", "moe_output", "moe_load_balance_loss",
    # Merger
    "merger_hass_input", "merger_moe_input", "merger_cot_input", "merger_output",
    # Final
    "final_norm_out", "logits",
    # Losses
    "lm_loss", "routing_loss", "load_balance_loss",
    "cot_consistency_loss", "total_loss",
]

# RoutingDecision fields captured by re-running the router
RD_FIELDS: List[str] = [
    "depth_logits", "depth_probs", "depth_mask",
    "width_logits", "width_probs", "width_idx",
    "path_logits", "path_probs",
    "expert_logits", "expert_probs", "expert_indices", "expert_weights",
    "complexity", "uncertainty",
]


# ─── Hook factories ───────────────────────────────────────────────
def make_tensor_hook(captured: Dict[str, torch.Tensor], name: str) -> Callable:
    """Forward hook that captures a tensor output (or first tensor of a tuple)."""
    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            captured[name] = out.detach().clone()
        elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            captured[name] = out[0].detach().clone()
    return hook


def make_pre_hook_args(captured: Dict[str, torch.Tensor], name: str, idx: int = 0) -> Callable:
    """Forward pre-hook that captures args[idx]."""
    def hook(module, args):
        if args and isinstance(args[idx], torch.Tensor):
            captured[name] = args[idx].detach().clone()
    return hook


def make_pre_hook_kwargs(captured: Dict[str, torch.Tensor], name: str, key: str) -> Callable:
    """Forward pre-hook (with_kwargs=True) that captures kwargs[key]."""
    def hook(module, args, kwargs):
        v = (kwargs or {}).get(key)
        if isinstance(v, torch.Tensor):
            captured[name] = v.detach().clone()
    return hook


def make_moe_post_hook(captured: Dict[str, torch.Tensor]) -> Callable:
    """Forward hook for the MoE — captures output tensor + load_balance_loss."""
    def hook(module, inp, out):
        # ShardedExpertFabric.forward returns (output, stats_dict)
        if isinstance(out, tuple) and len(out) == 2:
            output, stats = out
            if isinstance(output, torch.Tensor):
                captured["moe_output"] = output.detach().clone()
            if isinstance(stats, dict) and "load_balance_loss" in stats:
                lb = stats["load_balance_loss"]
                if isinstance(lb, torch.Tensor):
                    captured["moe_load_balance_loss"] = lb.detach().clone()
                else:
                    captured["moe_load_balance_loss"] = torch.tensor(float(lb))
    return hook


# ─── Full capture pipeline (used for both runs) ───────────────────
def capture_all(model: torch.nn.Module,
                input_ids: torch.Tensor,
                labels: torch.Tensor,
                cfg: Any,
                B: int, T: int) -> Dict[str, torch.Tensor]:
    """
    Run a full forward pass with hooks and return all captured tensors.

    This is factored into a function so the determinism test can re-run the
    EXACT same capture pipeline on a fresh model instance.
    """
    captured: Dict[str, torch.Tensor] = {}
    hooks: List[Any] = []

    # ---- Register hooks ----
    # Embeddings
    hooks.append(model.token_embedding.register_forward_hook(
        make_tensor_hook(captured, "token_emb_out")))
    hooks.append(model.position_embedding.register_forward_hook(
        make_tensor_hook(captured, "pos_emb_out")))

    # Router sub-modules (tensor outputs only)
    hooks.append(model.router.feature_encoder.register_forward_hook(
        make_tensor_hook(captured, "router_feature_encoder_out")))

    # HASS block 0 (the only block)
    hooks.append(model.blocks[0].register_forward_pre_hook(
        make_pre_hook_kwargs(captured, "block_0_input", "x"), with_kwargs=True))
    hooks.append(model.blocks[0].register_forward_hook(
        make_tensor_hook(captured, "block_0_output")))
    hooks.append(model.blocks[0].ffn.register_forward_hook(
        make_tensor_hook(captured, "block_0_ffn_out")))

    # MoE
    hooks.append(model.moe.register_forward_pre_hook(
        make_pre_hook_args(captured, "moe_input", 0)))
    hooks.append(model.moe.register_forward_hook(make_moe_post_hook(captured)))

    # Merger (called with all kwargs)
    hooks.append(model.merger.register_forward_pre_hook(
        make_pre_hook_kwargs(captured, "merger_hass_input", "hass_output"),
        with_kwargs=True))
    hooks.append(model.merger.register_forward_pre_hook(
        make_pre_hook_kwargs(captured, "merger_moe_input", "moe_output"),
        with_kwargs=True))
    hooks.append(model.merger.register_forward_pre_hook(
        make_pre_hook_kwargs(captured, "merger_cot_input", "cot_vector"),
        with_kwargs=True))
    hooks.append(model.merger.register_forward_hook(
        make_tensor_hook(captured, "merger_output")))

    # Final norm + LM head
    hooks.append(model.final_norm.register_forward_hook(
        make_tensor_hook(captured, "final_norm_out")))
    hooks.append(model.lm_head.register_forward_hook(
        make_tensor_hook(captured, "logits")))

    # ---- Run forward pass ----
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels, return_dict=True)

    # Remove hooks BEFORE re-running sub-modules (otherwise the re-runs
    # would trigger the hooks again and overwrite the captured values).
    for h in hooks:
        h.remove()

    # ---- Capture inputs & combined embeddings ----
    captured["input_ids"] = input_ids.detach().clone()
    captured["labels"] = labels.detach().clone()
    # combined_embeddings = token_emb + pos_emb
    # (embedding_dropout is identity in eval mode, so this equals the
    #  hidden_states that the router and block 0 receive)
    captured["combined_embeddings"] = (
        captured["token_emb_out"] + captured["pos_emb_out"]
    ).detach().clone()

    # ---- Capture losses from ModelOutput ----
    captured["lm_loss"] = output.lm_loss.detach().clone()
    captured["routing_loss"] = output.routing_loss.detach().clone()
    captured["load_balance_loss"] = output.load_balance_loss.detach().clone()
    captured["cot_consistency_loss"] = output.cot_consistency_loss.detach().clone()
    captured["total_loss"] = output.loss.detach().clone()

    # ---- Re-run router to capture RoutingDecision fields ----
    # RoutingDecision is a dataclass, not a tensor — module hooks cannot
    # capture its fields directly. We re-run the router with the EXACT same
    # inputs the model used internally (eval mode => deterministic=True
    # inside the router, so the re-run produces identical routing decisions).
    with torch.no_grad():
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        hidden_for_router = (
            model.token_embedding(input_ids)
            + model.position_embedding(pos_ids)
        )
        cot_features = torch.zeros(B, T, cfg.cot_dim * cfg.cot_components)
        rd = model.router(
            x=hidden_for_router,
            cot_features=cot_features,
            training=False,
        )
    for field_name in RD_FIELDS:
        t = getattr(rd, field_name)
        captured[field_name] = t.detach().clone()

    # ---- Re-run HASS block 0 pathways on full input ----
    # The model uses sparse pathway dispatch (top-k of 3 pathways per token),
    # so hooks on individual pathways would only capture PARTIAL slices
    # (the token subset that selected each pathway). Instead, we re-run
    # each pathway on the FULL ln1(block_0_input) to get the complete
    # "raw" pathway output — what the C++ port needs to reproduce.
    #
    # NOTE: SSMPathway.forward_parallel is called directly (not via
    # __call__), so it cannot be hooked with standard forward hooks anyway.
    with torch.no_grad():
        block = model.blocks[0]
        x_attn = block.ln1(captured["block_0_input"])
        captured["block_0_local_out"] = block.pathways["local"](
            x_attn, None, None).detach().clone()
        captured["block_0_low_rank_out"] = block.pathways["low_rank"](
            x_attn).detach().clone()
        captured["block_0_ssm_out"] = block.pathways["ssm"].forward_parallel(
            x_attn).detach().clone()

    return captured


# ─── Run #1 (primary capture) ─────────────────────────────────────
print("\n--- Run #1: primary capture ---")
captured1 = capture_all(model, input_ids, labels, cfg, B, T)

# Verify all expected tensors captured
missing = [n for n in EXPECTED_TENSORS if n not in captured1]
extra = [n for n in captured1 if n not in EXPECTED_TENSORS]
check("All expected tensors captured (run 1)", not missing,
      f"missing={missing}" if missing else f"count={len(EXPECTED_TENSORS)}")
if extra:
    print(f"  [INFO] Extra tensors (not in expected list): {extra}")

# Print tensor stats for run 1
print(f"\n--- Captured {len(captured1)} tensors (run 1) ---")
for name in EXPECTED_TENSORS:
    if name not in captured1:
        continue
    t = captured1[name]
    if t.is_floating_point():
        std_val = t.float().std().item() if t.numel() > 1 else 0.0
        print(f"  {name:32s} shape={str(tuple(t.shape)):20s} dtype={str(t.dtype):10s} "
              f"min={t.min().item():+.6f} max={t.max().item():+.6f} "
              f"mean={t.float().mean().item():+.6f} std={std_val:+.6f}")
    else:
        print(f"  {name:32s} shape={str(tuple(t.shape)):20s} dtype={str(t.dtype):10s} "
              f"min={int(t.min().item())} max={int(t.max().item())}")

# ─── Determinism test: re-run with fresh model (same seed) ────────
print(f"\n--- Determinism test (fresh model, same seed={SEED}) ---")
torch.manual_seed(SEED)
model2 = zero_tiny_23k(test_mode=False)
model2.eval()
captured2 = capture_all(model2, input_ids, labels, cfg, B, T)

# Compare EVERY captured tensor for exact equality
print(f"--- Comparing {len(EXPECTED_TENSORS)} tensors between run 1 and run 2 ---")
all_match = True
mismatches: List[str] = []
for name in EXPECTED_TENSORS:
    if name not in captured1 or name not in captured2:
        mismatches.append(f"{name}: missing in one run "
                          f"(run1={'Y' if name in captured1 else 'N'}, "
                          f"run2={'Y' if name in captured2 else 'N'})")
        all_match = False
        continue
    t1, t2 = captured1[name], captured2[name]
    if t1.shape != t2.shape:
        mismatches.append(f"{name}: shape mismatch {tuple(t1.shape)} vs {tuple(t2.shape)}")
        all_match = False
        continue
    if t1.dtype != t2.dtype:
        mismatches.append(f"{name}: dtype mismatch {t1.dtype} vs {t2.dtype}")
        all_match = False
        continue
    if not torch.equal(t1, t2):
        if t1.is_floating_point():
            diff = (t1.float() - t2.float()).abs().max().item()
            mismatches.append(f"{name}: value mismatch (max_abs_diff={diff:.2e})")
        else:
            n_diff = int((t1 != t2).sum().item())
            mismatches.append(f"{name}: value mismatch ({n_diff}/{t1.numel()} elements differ)")
        all_match = False

check("All tensors exactly equal across runs", all_match,
      "all match" if all_match else f"{len(mismatches)} mismatches")
for m in mismatches[:15]:
    print(f"    {m}")

# ─── Save tensors to .pt ──────────────────────────────────────────
print(f"\n--- Saving golden reference to .pt ---")
golden_dir = Path(__file__).parent
pt_path = golden_dir / "golden_reference.pt"
manifest_path = golden_dir / "golden_reference_manifest.json"

tensors_to_save: Dict[str, torch.Tensor] = {
    name: captured1[name] for name in EXPECTED_TENSORS if name in captured1
}
torch.save(tensors_to_save, pt_path)
print(f"  Saved {len(tensors_to_save)} tensors to: {pt_path}")
print(f"  File size: {pt_path.stat().st_size:,} bytes")

# ─── Compute SHA256 hash of .pt ───────────────────────────────────
sha256_hash = hashlib.sha256(pt_path.read_bytes()).hexdigest()
print(f"  SHA256: {sha256_hash}")

# ─── Verify .pt round-trip ────────────────────────────────────────
print(f"\n--- Verifying .pt round-trip ---")
try:
    reloaded = torch.load(pt_path, weights_only=True)
except TypeError:
    # Older PyTorch without weights_only kwarg
    reloaded = torch.load(pt_path)
except Exception:
    # Fall back to weights_only=False (we trust the file we just wrote)
    reloaded = torch.load(pt_path, weights_only=False)

roundtrip_ok = True
roundtrip_mismatches: List[str] = []
for name in EXPECTED_TENSORS:
    if name not in reloaded:
        roundtrip_mismatches.append(f"{name}: missing in reloaded .pt")
        roundtrip_ok = False
        continue
    if name not in captured1:
        continue
    if not torch.equal(reloaded[name], captured1[name]):
        roundtrip_mismatches.append(f"{name}: value mismatch after reload")
        roundtrip_ok = False
check(".pt round-trip preserves all tensors", roundtrip_ok,
      "all match" if roundtrip_ok else f"{len(roundtrip_mismatches)} mismatches")
for m in roundtrip_mismatches[:15]:
    print(f"    {m}")

# ─── Build manifest JSON ──────────────────────────────────────────

def tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    """Compute stats for a tensor, handling non-float dtypes appropriately."""
    stats: Dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": int(t.numel()),
    }
    if t.numel() == 0:
        stats["note"] = "empty tensor"
        return stats
    if t.is_floating_point():
        tf = t.float()
        stats["min"] = t.min().item()
        stats["max"] = t.max().item()
        stats["mean"] = tf.mean().item()
        stats["std"] = tf.std().item() if t.numel() > 1 else 0.0
    else:
        # Integer / bool tensors: min/max are integers.
        # mean/std are not meaningful for discrete indices, but we include
        # them as float diagnostics (clearly labelled).
        stats["min"] = int(t.min().item())
        stats["max"] = int(t.max().item())
        if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                       torch.uint8):
            tf = t.float()
            stats["mean_float"] = tf.mean().item()
            stats["std_float"] = tf.std().item() if t.numel() > 1 else 0.0
    return stats


def config_to_dict(c) -> Dict[str, Any]:
    """Convert a ModelConfig (BaseConfig) to a JSON-serializable dict.

    Uses BaseConfig.to_dict() which handles Enum/tuple/nested-BaseConfig
    conversions. Falls back to ``str()`` for any non-serializable remainder.
    """
    if hasattr(c, "to_dict"):
        return c.to_dict()
    # Fallback: manual dataclass conversion
    from dataclasses import fields as dc_fields
    out: Dict[str, Any] = {}
    for f in dc_fields(c):
        val = getattr(c, f.name)
        try:
            json.dumps(val)
            out[f.name] = val
        except (TypeError, ValueError):
            out[f.name] = str(val)
    return out


# Get XorZen git commit (run `git rev-parse HEAD` in the repo root)
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(golden_dir.parent),
        stderr=subprocess.DEVNULL,
    ).decode().strip()
    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(golden_dir.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_branch = "unknown"
    try:
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(golden_dir.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip())
    except Exception:
        git_dirty = None
except Exception as e:
    git_commit = f"unknown ({e})"
    git_branch = "unknown"
    git_dirty = None

manifest: Dict[str, Any] = {
    "metadata": {
        "model": cfg.model_name,
        "seed": SEED,
        "batch_size": B,
        "seq_len": T,
        "mode": "eval",
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "pt_file": pt_path.name,
        "pt_sha256": sha256_hash,
        "pt_size_bytes": pt_path.stat().st_size,
        "num_tensors": len(tensors_to_save),
        "created_by": "tests/test_golden_parity.py",
        "description": (
            "Golden reference intermediate tensors from a deterministic "
            "zero_tiny_23k forward pass. Used for regression detection and "
            "as the expected-output baseline for the C++ port."
        ),
    },
    "config": config_to_dict(cfg),
    "tensors": {
        name: tensor_stats(tensors_to_save[name])
        for name in EXPECTED_TENSORS if name in tensors_to_save
    },
    "output_summary": {
        "logits_shape": list(captured1["logits"].shape),
        "lm_loss": captured1["lm_loss"].item(),
        "routing_loss": captured1["routing_loss"].item(),
        "load_balance_loss": captured1["load_balance_loss"].item(),
        "cot_consistency_loss": captured1["cot_consistency_loss"].item(),
        "total_loss": captured1["total_loss"].item(),
        "moe_load_balance_loss": captured1["moe_load_balance_loss"].item(),
    },
    "determinism": {
        "verified": all_match,
        "compared_tensors": len(EXPECTED_TENSORS),
        "mismatches": mismatches if not all_match else [],
    },
}

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"  Manifest saved to:    {manifest_path}")

# ─── Print output summary ─────────────────────────────────────────
print(f"\n--- Output summary ---")
print(f"  Logits shape:          {tuple(captured1['logits'].shape)}")
print(f"  LM loss:               {captured1['lm_loss'].item():.6f}")
print(f"  Routing loss:          {captured1['routing_loss'].item():.6f}")
print(f"  Load-balance loss:     {captured1['load_balance_loss'].item():.6f}")
print(f"  CoT consistency loss:  {captured1['cot_consistency_loss'].item():.6f}")
print(f"  Total loss:            {captured1['total_loss'].item():.6f}")
print(f"  MoE load-balance loss: {captured1['moe_load_balance_loss'].item():.6f}")

# ─── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"TOTAL: {len(results)}  |  PASSED: {passed}  |  FAILED: {failed}")
if failed:
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL: {name}: {detail}")
else:
    print("ALL TESTS PASSED — golden reference is deterministic")
print(f"{'='*70}")

sys.exit(1 if failed else 0)

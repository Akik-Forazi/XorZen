#!/usr/bin/env python3
"""
XORZEN Routing Consistency Tests

Verifies that routing decisions are consistent between full-sequence and
incremental (KV-cached) execution. The key concern is that
AdaptiveRouter._eval_gumbel_noise was previously shape-dependent — a token
at absolute position N received different noise when processed as part of a
T-token sequence versus as a single incremental token.

After the Sprint 2 fix, noise is indexed by absolute position, so:
  - path_probs, depth_mask, width_idx, expert_indices for position N
    must be identical whether computed in a full-sequence forward or
    in an incremental forward with past_key_values.

Run: python tests/test_routing_consistency.py
"""
import sys
import warnings
warnings.simplefilter("ignore")

import torch
torch.manual_seed(42)

from xorzen.models.zero import zero_tiny_23k


def _pass(msg):
    print(f"  [PASS] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")
    return False


def _make_model():
    torch.manual_seed(42)
    m = zero_tiny_23k(test_mode=False)
    m.eval()
    m.gradient_checkpointing = False
    return m


# ─────────────────────────────────────────────────────────────────────
# Test 1: path_probs consistency
# ─────────────────────────────────────────────────────────────────────
def test_path_probs_consistency():
    """path_probs for position N must match between full and incremental."""
    print("\n=== Test 1: path_probs consistency (full vs incremental) ===")
    m = _make_model()
    x = torch.randint(0, m.config.vocab_size, (1, 8))
    new_tok = torch.tensor([[3]])
    x_full = torch.cat([x, new_tok], dim=1)

    with torch.no_grad():
        # Full-sequence forward (9 tokens) with routing info
        out_full = m(input_ids=x_full, return_dict=True, use_cache=True,
                      output_routing_info=True)
        pp_full = out_full.routing_info.path_probs  # [1, 9, 3]

        # Prefill (8 tokens) + incremental (1 token)
        out_prefill = m(input_ids=x, return_dict=True, use_cache=True,
                         output_routing_info=True)
        out_incr = m(input_ids=new_tok, return_dict=True, use_cache=True,
                      past_key_values=out_prefill.past_key_values,
                      output_routing_info=True)
        pp_incr = out_incr.routing_info.path_probs  # [1, 1, 3]

    # Compare position 8 (the new token) — should be identical
    diff = (pp_incr[0, 0] - pp_full[0, 8]).abs().max().item()

    # Also compare positions 0-7 (prefill should match full)
    diff_prefill = (out_prefill.routing_info.path_probs[0] - pp_full[0, :8]).abs().max().item()

    if diff < 1e-6:
        _pass(f"path_probs position 8 match (diff={diff:.2e})")
    else:
        return _fail(f"path_probs position 8 differ (diff={diff:.2e})")

    if diff_prefill < 1e-6:
        _pass(f"Prefill path_probs positions 0-7 match (diff={diff_prefill:.2e})")
        return True
    else:
        return _fail(f"Prefill path_probs differ (diff={diff_prefill:.2e})")


# ─────────────────────────────────────────────────────────────────────
# Test 2: depth_mask consistency
# ─────────────────────────────────────────────────────────────────────
def test_depth_mask_consistency():
    """depth_mask for position N must match between full and incremental."""
    print("\n=== Test 2: depth_mask consistency ===")
    m = _make_model()
    x = torch.randint(0, m.config.vocab_size, (1, 8))
    new_tok = torch.tensor([[3]])
    x_full = torch.cat([x, new_tok], dim=1)

    with torch.no_grad():
        out_full = m(input_ids=x_full, return_dict=True, use_cache=True,
                      output_routing_info=True)
        dm_full = out_full.routing_info.depth_mask  # [1, 9, n_layers]

        out_prefill = m(input_ids=x, return_dict=True, use_cache=True,
                         output_routing_info=True)
        out_incr = m(input_ids=new_tok, return_dict=True, use_cache=True,
                      past_key_values=out_prefill.past_key_values,
                      output_routing_info=True)
        dm_incr = out_incr.routing_info.depth_mask  # [1, 1, n_layers]

    diff = (dm_incr[0, 0] - dm_full[0, 8]).abs().max().item()

    if diff < 1e-6:
        _pass(f"depth_mask position 8 match (diff={diff:.2e})")
        return True
    else:
        return _fail(f"depth_mask position 8 differ (diff={diff:.2e})")


# ─────────────────────────────────────────────────────────────────────
# Test 3: width_idx consistency
# ─────────────────────────────────────────────────────────────────────
def test_width_idx_consistency():
    """width_idx for position N must match between full and incremental."""
    print("\n=== Test 3: width_idx consistency ===")
    m = _make_model()
    x = torch.randint(0, m.config.vocab_size, (1, 8))
    new_tok = torch.tensor([[3]])
    x_full = torch.cat([x, new_tok], dim=1)

    with torch.no_grad():
        out_full = m(input_ids=x_full, return_dict=True, use_cache=True,
                      output_routing_info=True)
        wi_full = out_full.routing_info.width_idx  # [1, 9]

        out_prefill = m(input_ids=x, return_dict=True, use_cache=True,
                         output_routing_info=True)
        out_incr = m(input_ids=new_tok, return_dict=True, use_cache=True,
                      past_key_values=out_prefill.past_key_values,
                      output_routing_info=True)
        wi_incr = out_incr.routing_info.width_idx  # [1, 1]

    match = (wi_incr[0, 0] == wi_full[0, 8]).item()

    if match:
        _pass(f"width_idx position 8 match (incr={wi_incr[0,0].item()}, full={wi_full[0,8].item()})")
        return True
    else:
        return _fail(f"width_idx position 8 differ (incr={wi_incr[0,0].item()}, full={wi_full[0,8].item()})")


# ─────────────────────────────────────────────────────────────────────
# Test 4: expert_indices consistency
# ─────────────────────────────────────────────────────────────────────
def test_expert_indices_consistency():
    """expert_indices for position N must match between full and incremental."""
    print("\n=== Test 4: expert_indices consistency ===")
    m = _make_model()
    x = torch.randint(0, m.config.vocab_size, (1, 8))
    new_tok = torch.tensor([[3]])
    x_full = torch.cat([x, new_tok], dim=1)

    with torch.no_grad():
        out_full = m(input_ids=x_full, return_dict=True, use_cache=True,
                      output_routing_info=True)
        ei_full = out_full.routing_info.expert_indices  # [1, 9, top_k]

        out_prefill = m(input_ids=x, return_dict=True, use_cache=True,
                         output_routing_info=True)
        out_incr = m(input_ids=new_tok, return_dict=True, use_cache=True,
                      past_key_values=out_prefill.past_key_values,
                      output_routing_info=True)
        ei_incr = out_incr.routing_info.expert_indices  # [1, 1, top_k]

    match = torch.equal(ei_incr[0, 0], ei_full[0, 8])

    if match:
        _pass(f"expert_indices position 8 match (incr={ei_incr[0,0].tolist()}, full={ei_full[0,8].tolist()})")
        return True
    else:
        return _fail(f"expert_indices differ (incr={ei_incr[0,0].tolist()}, full={ei_full[0,8].tolist()})")


# ─────────────────────────────────────────────────────────────────────
# Test 5: Multiple positions consistency
# ─────────────────────────────────────────────────────────────────────
def test_multiple_positions():
    """Check consistency across multiple incremental steps."""
    print("\n=== Test 5: Multiple positions consistency ===")
    m = _make_model()
    x = torch.randint(0, m.config.vocab_size, (1, 4))
    new_tokens = [torch.tensor([[t]]) for t in [5, 2, 7, 1]]
    x_full = torch.cat([x] + [t for t in new_tokens], dim=1)  # [1, 8]

    with torch.no_grad():
        # Full forward
        out_full = m(input_ids=x_full, return_dict=True, use_cache=True,
                      output_routing_info=True)
        pp_full = out_full.routing_info.path_probs  # [1, 8, 3]

        # Prefill + 4 incremental steps
        out_prefill = m(input_ids=x, return_dict=True, use_cache=True,
                         output_routing_info=True)
        past_kv = out_prefill.past_key_values

        all_match = True
        for i, new_tok in enumerate(new_tokens):
            pos = 4 + i  # absolute position
            out_incr = m(input_ids=new_tok, return_dict=True, use_cache=True,
                         past_key_values=past_kv, output_routing_info=True)
            past_kv = out_incr.past_key_values
            pp_incr = out_incr.routing_info.path_probs[0, 0]  # [3]
            pp_at_pos = pp_full[0, pos]  # [3]
            diff = (pp_incr - pp_at_pos).abs().max().item()
            if diff >= 1e-6:
                _fail(f"Position {pos}: path_probs diff={diff:.2e}")
                all_match = False

    if all_match:
        _pass(f"All {len(new_tokens)} incremental positions match full-sequence routing")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Test 6: Routing noise disabled (eval_routing_noise=0)
# ─────────────────────────────────────────────────────────────────────
def test_noise_disabled():
    """When eval_routing_noise=0, routing should be fully deterministic."""
    print("\n=== Test 6: Routing noise disabled (eval_routing_noise=0) ===")
    m = _make_model()
    m.router.eval_routing_noise = 0.0

    x = torch.randint(0, m.config.vocab_size, (1, 8))
    new_tok = torch.tensor([[3]])
    x_full = torch.cat([x, new_tok], dim=1)

    with torch.no_grad():
        out_full = m(input_ids=x_full, return_dict=True, use_cache=True,
                      output_routing_info=True)
        out_prefill = m(input_ids=x, return_dict=True, use_cache=True,
                         output_routing_info=True)
        out_incr = m(input_ids=new_tok, return_dict=True, use_cache=True,
                      past_key_values=out_prefill.past_key_values,
                      output_routing_info=True)

        pp_full = out_full.routing_info.path_probs[0, 8]
        pp_incr = out_incr.routing_info.path_probs[0, 0]
        diff = (pp_full - pp_incr).abs().max().item()

    if diff < 1e-6:
        _pass(f"With noise=0, path_probs match exactly (diff={diff:.2e})")
        return True
    else:
        return _fail(f"With noise=0, path_probs still differ (diff={diff:.2e})")


# ─────────────────────────────────────────────────────────────────────
# Test 7: Batch independence
# ─────────────────────────────────────────────────────────────────────
def test_batch_independence():
    """Routing noise must not leak across batch items."""
    print("\n=== Test 7: Batch independence ===")
    m = _make_model()
    x1 = torch.randint(0, m.config.vocab_size, (1, 8))
    x2 = torch.randint(0, m.config.vocab_size, (1, 8))
    x_batch = torch.cat([x1, x2], dim=0)  # [2, 8]

    with torch.no_grad():
        # Batch forward
        out_batch = m(input_ids=x_batch, return_dict=True, use_cache=True,
                       output_routing_info=True)
        pp_batch = out_batch.routing_info.path_probs  # [2, 8, 3]

        # Individual forwards
        out1 = m(input_ids=x1, return_dict=True, use_cache=True,
                  output_routing_info=True)
        out2 = m(input_ids=x2, return_dict=True, use_cache=True,
                  output_routing_info=True)

        diff1 = (pp_batch[0] - out1.routing_info.path_probs[0]).abs().max().item()
        diff2 = (pp_batch[1] - out2.routing_info.path_probs[0]).abs().max().item()

    if diff1 < 1e-6 and diff2 < 1e-6:
        _pass(f"Batch routing matches individual routing (diff1={diff1:.2e}, diff2={diff2:.2e})")
        return True
    else:
        return _fail(f"Batch routing differs (diff1={diff1:.2e}, diff2={diff2:.2e})")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("XORZEN Routing Consistency Tests")
    print("=" * 70)

    tests = [
        test_path_probs_consistency,
        test_depth_mask_consistency,
        test_width_idx_consistency,
        test_expert_indices_consistency,
        test_multiple_positions,
        test_noise_disabled,
        test_batch_independence,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            if t():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__} raised: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    print(f"TOTAL: {passed + failed}  |  PASSED: {passed}  |  FAILED: {failed}")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

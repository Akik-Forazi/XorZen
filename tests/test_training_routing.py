#!/usr/bin/env python3
"""
XORZEN Training Routing Audit Tests

Verifies that routing parameters receive gradients from the language-model
loss, and that the SlicedFFN STE fix provides differentiable gradient flow
to the width router.

Key test: the width_router parameters must receive nonzero, finite gradients
from the LM loss alone (auxiliary routing losses disabled). This would FAIL
under the original disconnected implementation that returned y_hard without
any STE connection to width_probs.

Run: python tests/test_training_routing.py
"""
import sys
import warnings
warnings.simplefilter("ignore")

import torch
torch.manual_seed(42)


def _pass(msg):
    print(f"  [PASS] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")
    return False


# ─────────────────────────────────────────────────────────────────────
# Test 1: SlicedFFN STE — width_probs receives gradient
# ─────────────────────────────────────────────────────────────────────
def test_sliced_ffn_ste_gradient():
    """width_probs must receive nonzero gradient from a loss on the FFN output."""
    print("\n=== Test 1: SlicedFFN STE gradient to width_probs ===")
    from xorzen.model.components.sliced_ffn import SlicedFFN

    torch.manual_seed(42)
    ffn = SlicedFFN(hidden_dim=64, max_width=256, activation="gelu", width_choices=[64, 128])
    ffn.train()

    x = torch.randn(2, 8, 64, requires_grad=False)
    width_probs = torch.randn(2, 8, 2, requires_grad=True)
    width_probs.data = width_probs.data.softmax(dim=-1)

    y = ffn(x, width_probs=width_probs)
    # Simulate LM loss: sum of all outputs
    loss = y.sum()
    loss.backward()

    if width_probs.grad is None:
        return _fail("width_probs.grad is None — no gradient flow")

    grad_norm = width_probs.grad.norm().item()
    is_finite = torch.isfinite(width_probs.grad).all().item()

    if grad_norm > 0 and is_finite:
        _pass(f"width_probs receives gradient (norm={grad_norm:.6e}, finite={is_finite})")
        return True
    else:
        return _fail(f"width_probs gradient is zero or non-finite (norm={grad_norm:.6e})")


# ─────────────────────────────────────────────────────────────────────
# Test 2: SlicedFFN STE — forward uses hard width (not soft mix)
# ─────────────────────────────────────────────────────────────────────
def test_sliced_ffn_ste_forward_is_hard():
    """The forward output must match the hard-width output, not the soft mix."""
    print("\n=== Test 2: SlicedFFN STE forward uses hard width ===")
    from xorzen.model.components.sliced_ffn import SlicedFFN

    torch.manual_seed(42)
    ffn = SlicedFFN(hidden_dim=64, max_width=256, activation="gelu", width_choices=[64, 128])
    ffn.train()

    x = torch.randn(2, 8, 64)
    width_probs = torch.tensor([[[1.0, 0.0]]] * 2 * 8).reshape(2, 8, 2)  # all select width 0

    with torch.no_grad():
        y_ste = ffn(x, width_probs=width_probs)
        # Compute hard output directly (width=64 for all tokens)
        x_norm = ffn.ln_input(x)
        y_hard = ffn._forward_single_width(x_norm, 64)

    diff = (y_ste - y_hard).abs().max().item()
    if diff < 1e-5:
        _pass(f"STE forward matches hard output (diff={diff:.2e})")
        return True
    else:
        return _fail(f"STE forward differs from hard output (diff={diff:.2e})")


# ─────────────────────────────────────────────────────────────────────
# Test 3: Full model — width_router receives LM loss gradient
# ─────────────────────────────────────────────────────────────────────
def test_model_width_router_gradient():
    """The width_router parameters must receive nonzero gradients from LM loss."""
    print("\n=== Test 3: Model width_router receives LM loss gradient ===")
    from xorzen.models.zero import zero_10M

    torch.manual_seed(42)
    m = zero_10M(test_mode=False)
    m.train()
    m.gradient_checkpointing = False

    width_router_params = list(m.router.width_router.parameters())
    for p in width_router_params:
        p.grad = None

    x = torch.randint(0, m.config.vocab_size, (2, 16))
    labels = torch.randint(0, m.config.vocab_size, (2, 16))

    out = m(input_ids=x, labels=labels, return_dict=True)
    out.loss.backward()

    has_grad = True
    for i, p in enumerate(width_router_params):
        if p.grad is None:
            _fail(f"width_router[{i}] grad is None")
            has_grad = False
            continue
        grad_norm = p.grad.norm().item()
        is_finite = torch.isfinite(p.grad).all().item()
        if grad_norm == 0 or not is_finite:
            _fail(f"width_router[{i}] grad norm={grad_norm:.6e} finite={is_finite}")
            has_grad = False

    if has_grad:
        _pass("All width_router params receive nonzero finite gradients from LM loss")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Test 4: Depth router receives gradient
# ─────────────────────────────────────────────────────────────────────
def test_depth_router_gradient():
    """The depth_router parameters must receive nonzero gradients from LM loss."""
    print("\n=== Test 4: Depth router receives LM loss gradient ===")
    from xorzen.models.zero import zero_10M

    torch.manual_seed(42)
    m = zero_10M(test_mode=False)
    m.train()
    m.gradient_checkpointing = False

    depth_router_params = list(m.router.depth_router.parameters())
    for p in depth_router_params:
        p.grad = None

    x = torch.randint(0, m.config.vocab_size, (2, 16))
    labels = torch.randint(0, m.config.vocab_size, (2, 16))

    out = m(input_ids=x, labels=labels, return_dict=True)
    out.loss.backward()

    has_grad = True
    for i, p in enumerate(depth_router_params):
        if p.grad is None:
            _fail(f"depth_router[{i}] grad is None")
            has_grad = False
            continue
        grad_norm = p.grad.norm().item()
        is_finite = torch.isfinite(p.grad).all().item()
        if grad_norm == 0 or not is_finite:
            _fail(f"depth_router[{i}] grad norm={grad_norm:.6e} finite={is_finite}")
            has_grad = False

    if has_grad:
        _pass("All depth_router params receive nonzero finite gradients from LM loss")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Test 5: Expert router receives gradient
# ─────────────────────────────────────────────────────────────────────
def test_expert_router_gradient():
    """The expert_router parameters must receive nonzero gradients from LM loss."""
    print("\n=== Test 5: Expert router receives LM loss gradient ===")
    from xorzen.models.zero import zero_10M

    torch.manual_seed(42)
    m = zero_10M(test_mode=False)
    m.train()
    m.gradient_checkpointing = False

    expert_router_params = list(m.router.expert_router.parameters())
    for p in expert_router_params:
        p.grad = None

    x = torch.randint(0, m.config.vocab_size, (2, 16))
    labels = torch.randint(0, m.config.vocab_size, (2, 16))

    out = m(input_ids=x, labels=labels, return_dict=True)
    out.loss.backward()

    has_grad = True
    for i, p in enumerate(expert_router_params):
        if p.grad is None:
            _fail(f"expert_router[{i}] grad is None")
            has_grad = False
            continue
        grad_norm = p.grad.norm().item()
        is_finite = torch.isfinite(p.grad).all().item()
        if grad_norm == 0 or not is_finite:
            _fail(f"expert_router[{i}] grad norm={grad_norm:.6e} finite={is_finite}")
            has_grad = False

    if has_grad:
        _pass("All expert_router params receive nonzero finite gradients from LM loss")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Test 6: Path router receives gradient
# ─────────────────────────────────────────────────────────────────────
def test_path_router_gradient():
    """The path_router parameters must receive nonzero gradients from LM loss."""
    print("\n=== Test 6: Path router receives LM loss gradient ===")
    from xorzen.models.zero import zero_10M

    torch.manual_seed(42)
    m = zero_10M(test_mode=False)
    m.train()
    m.gradient_checkpointing = False

    path_router_params = list(m.router.path_router.parameters())
    for p in path_router_params:
        p.grad = None

    x = torch.randint(0, m.config.vocab_size, (2, 16))
    labels = torch.randint(0, m.config.vocab_size, (2, 16))

    out = m(input_ids=x, labels=labels, return_dict=True)
    out.loss.backward()

    has_grad = True
    for i, p in enumerate(path_router_params):
        if p.grad is None:
            _fail(f"path_router[{i}] grad is None")
            has_grad = False
            continue
        grad_norm = p.grad.norm().item()
        is_finite = torch.isfinite(p.grad).all().item()
        if grad_norm == 0 or not is_finite:
            _fail(f"path_router[{i}] grad norm={grad_norm:.6e} finite={is_finite}")
            has_grad = False

    if has_grad:
        _pass("All path_router params receive nonzero finite gradients from LM loss")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Test 7: CoT frozen state verification
# ─────────────────────────────────────────────────────────────────────
def test_cot_frozen_state():
    """CoT parameters must be frozen by default during pretraining."""
    print("\n=== Test 7: CoT frozen state ===")
    from xorzen.models.zero import zero_10M

    torch.manual_seed(42)
    m = zero_10M(test_mode=False)

    # Check that CoT parameters are frozen
    cot_params = list(m.cot.parameters())
    if not cot_params:
        _pass("CoT has no parameters (trivially frozen)")
        return True

    frozen_count = 0
    unfrozen_count = 0
    for p in cot_params:
        if not p.requires_grad:
            frozen_count += 1
        else:
            unfrozen_count += 1

    if frozen_count > 0 and unfrozen_count == 0:
        _pass(f"All {frozen_count} CoT parameter tensors are frozen (requires_grad=False)")
        return True
    else:
        return _fail(f"CoT: {frozen_count} frozen, {unfrozen_count} unfrozen — expected all frozen")


# ─────────────────────────────────────────────────────────────────────
# Test 8: CoT unfreeze works
# ─────────────────────────────────────────────────────────────────────
def test_cot_unfreeze():
    """enable_cot() should unfreeze CoT parameters."""
    print("\n=== Test 8: CoT unfreeze ===")
    from xorzen.models.zero import zero_10M

    torch.manual_seed(42)
    m = zero_10M(test_mode=False)

    # Initially frozen
    initial_frozen = all(not p.requires_grad for p in m.cot.parameters())

    # Unfreeze
    m.enable_cot()

    after_unfreeze = all(p.requires_grad for p in m.cot.parameters())

    if initial_frozen and after_unfreeze:
        _pass("CoT: frozen initially, unfrozen after enable_cot()")
        return True
    else:
        return _fail(f"CoT: initial_frozen={initial_frozen}, after_unfreeze={after_unfreeze}")


# ─────────────────────────────────────────────────────────────────────
# Test 9: Training step produces finite loss and gradients
# ─────────────────────────────────────────────────────────────────────
def test_training_step_finite():
    """One optimizer step must produce finite loss and parameter updates."""
    print("\n=== Test 9: Training step finite ===")
    from xorzen.models.zero import zero_10M

    torch.manual_seed(42)
    m = zero_10M(test_mode=False)
    m.train()
    m.gradient_checkpointing = False

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)

    x = torch.randint(0, m.config.vocab_size, (2, 16))
    labels = torch.randint(0, m.config.vocab_size, (2, 16))

    optimizer.zero_grad()
    out = m(input_ids=x, labels=labels, return_dict=True)
    loss_before = out.loss.item()
    out.loss.backward()

    # Check all gradients are finite
    all_finite = True
    for name, p in m.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            _fail(f"Non-finite gradient on {name}")
            all_finite = False
            break

    if not all_finite:
        return False

    optimizer.step()

    # Second forward to check loss is still finite
    optimizer.zero_grad()
    out2 = m(input_ids=x, labels=labels, return_dict=True)
    loss_after = out2.loss.item()

    if torch.isfinite(torch.tensor(loss_after)):
        _pass(f"Training step: loss_before={loss_before:.4f}, loss_after={loss_after:.4f} (both finite)")
        return True
    else:
        return _fail(f"Loss after step is non-finite: {loss_after}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("XORZEN Training Routing Audit Tests")
    print("=" * 70)

    tests = [
        test_sliced_ffn_ste_gradient,
        test_sliced_ffn_ste_forward_is_hard,
        test_model_width_router_gradient,
        test_depth_router_gradient,
        test_expert_router_gradient,
        test_path_router_gradient,
        test_cot_frozen_state,
        test_cot_unfreeze,
        test_training_step_finite,
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

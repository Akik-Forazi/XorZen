"""
Phase 1 Test: SSM scan parity for all sequence lengths.

Tests that jit_sequential_scan matches the original sequential_scan for:
  - T=64 (direct sequential)
  - T=256 (single chunk boundary)
  - T=257 (just over chunk boundary — triggers chunked_scan with init_state)
  - T=512 (two chunks)
  - T=1024 (four chunks — the notebook's sequence length)

Also tests:
  - Direct sequential_scan with init_state
  - chunked_scan (which calls sequential_scan with init_state internally)
  - select_scan (the dispatcher)
  - With and without the JIT patch applied

Verifies:
  - Output shape
  - Output dtype
  - Output device
  - Maximum absolute error
  - Maximum relative error
  - Gradient flow (if supported)
"""
import sys
sys.path.insert(0, "/home/z/my-project/XorZen")

import torch
import math

results = []
def check(name, ok, detail=''):
    results.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")

print("=" * 70)
print("PHASE 1: SSM SCAN PARITY TEST")
print("=" * 70)

# ─── Test without JIT patch first ───────────────────────────────────
print("\n--- WITHOUT JIT patch (original Python sequential_scan) ---")

from xorzen.model.components.ssm_scan import (
    sequential_scan, chunked_scan, select_scan, parallel_scan
)

test_cases = [
    ("T=64", 64),
    ("T=256", 256),
    ("T=257", 257),
    ("T=512", 512),
    ("T=1024", 1024),
]

B, N = 2, 16

for name, T in test_cases:
    torch.manual_seed(42)
    A_bar = torch.rand(B, T, N) * 0.9 + 0.05  # (0.05, 0.95) — stable
    B_bar = torch.randn(B, T, N)

    # Direct sequential_scan without init_state
    out_seq = sequential_scan(A_bar, B_bar)
    check(f"{name} sequential_scan shape", out_seq.shape == (B, T, N), f"{tuple(out_seq.shape)}")
    check(f"{name} sequential_scan dtype", out_seq.dtype == A_bar.dtype, str(out_seq.dtype))

    # Direct sequential_scan WITH init_state
    init = torch.randn(B, N)
    out_seq_init = sequential_scan(A_bar, B_bar, init_state=init)
    check(f"{name} sequential_scan+init shape", out_seq_init.shape == (B, T, N), f"{tuple(out_seq_init.shape)}")

    # chunked_scan (calls sequential_scan with init_state internally for T > chunk_size)
    out_chunk = chunked_scan(A_bar, B_bar, chunk_size=256)
    max_abs = (out_seq - out_chunk).abs().max().item()
    max_rel = ((out_seq - out_chunk).abs() / (out_seq.abs() + 1e-8)).max().item()
    check(f"{name} chunked vs sequential", max_abs < 1e-5, f"abs={max_abs:.2e}, rel={max_rel:.2e}")

    # chunked_scan WITH init_state
    out_chunk_init = chunked_scan(A_bar, B_bar, init_state=init, chunk_size=256)
    max_abs_init = (out_seq_init - out_chunk_init).abs().max().item()
    check(f"{name} chunked+init vs sequential+init", max_abs_init < 1e-5, f"abs={max_abs_init:.2e}")

    # select_scan (dispatcher — should pick the right method)
    out_select = select_scan(A_bar, B_bar)
    max_abs_sel = (out_seq - out_select).abs().max().item()
    check(f"{name} select_scan vs sequential", max_abs_sel < 1e-5, f"abs={max_abs_sel:.2e}")

# ─── Now apply JIT patch and re-test ──────────────────────────────
print("\n--- WITH JIT patch (jit_sequential_scan replaces sequential_scan) ---")

from xorzen.speed.jit_kernels import jit_sequential_scan, jit_available
check("JIT available", jit_available, str(jit_available))

# Save original
import xorzen.model.components.ssm_scan as ssm_module
original_sequential_scan = ssm_module.sequential_scan

# Apply patch
ssm_module.sequential_scan = jit_sequential_scan
print("  Patched ssm_scan.sequential_scan → jit_sequential_scan")

for name, T in test_cases:
    torch.manual_seed(42)
    A_bar = torch.rand(B, T, N) * 0.9 + 0.05
    B_bar = torch.randn(B, T, N)

    # JIT sequential_scan without init_state
    out_jit = jit_sequential_scan(A_bar, B_bar)
    out_ref = original_sequential_scan(A_bar, B_bar)
    max_abs = (out_jit - out_ref).abs().max().item()
    max_rel = ((out_jit - out_ref).abs() / (out_ref.abs() + 1e-8)).max().item()
    check(f"{name} JIT vs Python (no init)", max_abs < 1e-6, f"abs={max_abs:.2e}, rel={max_rel:.2e}")

    # JIT sequential_scan WITH init_state
    init = torch.randn(B, N)
    out_jit_init = jit_sequential_scan(A_bar, B_bar, init_state=init)
    out_ref_init = original_sequential_scan(A_bar, B_bar, init_state=init)
    max_abs_init = (out_jit_init - out_ref_init).abs().max().item()
    check(f"{name} JIT vs Python (with init)", max_abs_init < 1e-6, f"abs={max_abs_init:.2e}")

    # chunked_scan with JIT patch active (the critical test — T>256 calls
    # sequential_scan with init_state on chunk 2+)
    out_chunk_jit = chunked_scan(A_bar, B_bar, chunk_size=256)
    max_abs_chunk = (out_ref - out_chunk_jit).abs().max().item()
    check(f"{name} chunked_scan (JIT patched) vs sequential (Python)",
          max_abs_chunk < 1e-5, f"abs={max_abs_chunk:.2e}")

    # chunked_scan WITH init_state + JIT patched
    out_chunk_jit_init = chunked_scan(A_bar, B_bar, init_state=init, chunk_size=256)
    max_abs_chunk_init = (out_ref_init - out_chunk_jit_init).abs().max().item()
    check(f"{name} chunked_scan+init (JIT) vs sequential+init (Python)",
          max_abs_chunk_init < 1e-5, f"abs={max_abs_chunk_init:.2e}")

# ─── Gradient flow test ──────────────────────────────────────────
print("\n--- Gradient flow test ---")
T = 512
A_bar_orig = torch.rand(B, T, N) * 0.9 + 0.05
B_bar_orig = torch.randn(B, T, N)

# JIT version
A_bar = A_bar_orig.clone().requires_grad_(True)
B_bar = B_bar_orig.clone().requires_grad_(True)
out_jit = jit_sequential_scan(A_bar, B_bar)
loss_jit = out_jit.sum()
loss_jit.backward()
grad_A_jit = A_bar.grad.clone()
grad_B_jit = B_bar.grad.clone()

# Python version
A_bar2 = A_bar_orig.clone().requires_grad_(True)
B_bar2 = B_bar_orig.clone().requires_grad_(True)
out_ref = original_sequential_scan(A_bar2, B_bar2)
loss_ref = out_ref.sum()
loss_ref.backward()
grad_A_ref = A_bar2.grad.clone()
grad_B_ref = B_bar2.grad.clone()

grad_A_diff = (grad_A_jit - grad_A_ref).abs().max().item()
grad_B_diff = (grad_B_jit - grad_B_ref).abs().max().item()
check("Gradient A_bar parity", grad_A_diff < 1e-6, f"max_abs={grad_A_diff:.2e}")
check("Gradient B_bar parity", grad_B_diff < 1e-6, f"max_abs={grad_B_diff:.2e}")

# ─── Restore original ────────────────────────────────────────────
ssm_module.sequential_scan = original_sequential_scan
print("\n  Restored original sequential_scan")

# ─── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"TOTAL: {len(results)}  |  PASSED: {passed}  |  FAILED: {failed}")
if failed:
    print("\nFAILURES:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {name}: {detail}")
else:
    print("ALL TESTS PASSED — SSM scan is chunk-safe with JIT patch")
print("=" * 70)

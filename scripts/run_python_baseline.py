"""Run the full Python baseline test suite + CPU performance baseline.

This script:
  1. Runs every tests/test_*.py file individually.
  2. Captures PASS/FAIL/UNVERIFIED per test.
  3. Runs a CPU performance benchmark (forward + backward pass on zero_1M).
  4. Writes a JSON report to docs/parity/PYTHON_BASELINE.json.

Exit code 0 if all tests pass; nonzero otherwise.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/z/my-project/XorZen")
TESTS = ROOT / "tests"
REPORT = ROOT / "docs" / "parity" / "PYTHON_BASELINE.json"

# The tests we want to run, in priority order
TEST_FILES = [
    "test_phase1_correctness.py",
    "test_phase2_regression.py",
    "test_phase4_v04.py",
    "test_v05_fixes.py",
    "test_v101_regression.py",
    "test_ssm_scan_parity.py",
    "test_golden_parity.py",
    "test_beam_search.py",
    "test_fixes.py",
    "test_blocker_fixes.py",
    "test_causal_leakage_fix.py",
    "test_cot_consistency_loss.py",
    "test_fix_p5_load_balance.py",
    "test_fix_sppq_schedule.py",
    "test_fix_tokenizer_roundtrip.py",
    "test_micro_overfit.py",
    "test_phase2_sparsity.py",
]

results = []
n_pass = 0
n_fail = 0
n_skip = 0

print("=" * 70)
print("PYTHON BASELINE TEST SUITE")
print("=" * 70)

for tf in TEST_FILES:
    p = TESTS / tf
    if not p.exists():
        results.append({"file": tf, "status": "MISSING"})
        n_skip += 1
        print(f"  [SKIP] {tf} (file not found)")
        continue
    print(f"  Running {tf}...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(p)],
        capture_output=True, text=True, timeout=180
    )
    dt = time.time() - t0
    # Look for PASS/FAIL markers in the output
    out = proc.stdout + proc.stderr
    if proc.returncode == 0:
        # Try to count individual test PASS/FAIL
        n_p = out.count("[PASS]") + out.count("PASS ")
        n_f = out.count("[FAIL]") + out.count("FAIL ")
        results.append({
            "file": tf, "status": "PASS",
            "returncode": proc.returncode,
            "duration_s": round(dt, 2),
            "individual_pass": n_p,
            "individual_fail": n_f,
        })
        n_pass += 1
        print(f"    PASS ({dt:.1f}s, {n_p} individual pass, {n_f} fail)")
    else:
        # Even on failure, see if there were some passes
        n_p = out.count("[PASS]") + out.count("PASS ")
        n_f = out.count("[FAIL]") + out.count("FAIL ")
        # Extract last error line
        last_err = ""
        for line in out.splitlines()[::-1]:
            if "Error" in line or "FAIL" in line or "assert" in line.lower():
                last_err = line.strip()[:200]
                break
        results.append({
            "file": tf, "status": "FAIL",
            "returncode": proc.returncode,
            "duration_s": round(dt, 2),
            "individual_pass": n_p,
            "individual_fail": n_f,
            "last_error": last_err,
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
        })
        n_fail += 1
        print(f"    FAIL ({dt:.1f}s, {n_p} pass, {n_f} fail): {last_err[:100]}")

print()
print("=" * 70)
print("CPU PERFORMANCE BASELINE (zero_1M)")
print("=" * 70)

# Run a simple CPU benchmark
perf = {"status": "UNVERIFIED"}
try:
    sys.path.insert(0, str(ROOT))
    import torch
    torch.manual_seed(42)
    from xorzen.models.zero import zero_1M
    m = zero_1M()
    m.eval()
    cfg = m.config
    B, T = 1, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = torch.randint(0, cfg.vocab_size, (B, T))
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = m(input_ids=input_ids, labels=labels)
    # Measure forward
    n_iter = 5
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = m(input_ids=input_ids, labels=labels)
    fwd_dt = (time.time() - t0) / n_iter
    print(f"  forward:  {fwd_dt*1000:.1f} ms/step (avg of {n_iter})")
    print(f"  params:   {m.count_parameters():,}")
    print(f"  config:   {cfg.model_name}")

    # Measure backward (training step)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    n_iter = 3
    t0 = time.time()
    for _ in range(n_iter):
        opt.zero_grad()
        out = m(input_ids=input_ids, labels=labels)
        out.loss.backward()
        opt.step()
    train_dt = (time.time() - t0) / n_iter
    print(f"  train:    {train_dt*1000:.1f} ms/step (avg of {n_iter}, includes backward + AdamW)")

    # Measure SSM scan at various T
    from xorzen.model.components.ssm_scan import sequential_scan, parallel_scan, chunked_scan, select_scan
    import torch
    ssm_results = {}
    for T_test in [64, 256, 257, 512, 1024]:
        B_test, N_test = 1, 16
        g = torch.Generator().manual_seed(42)
        A_bar = torch.rand(B_test, T_test, N_test, generator=g) * 0.5 + 0.25
        B_bar = torch.randn(B_test, T_test, N_test, generator=g) * 0.1
        # warmup
        for _ in range(2):
            _ = select_scan(A_bar, B_bar)
        # measure
        t0 = time.time()
        for _ in range(5):
            _ = select_scan(A_bar, B_bar)
        dt_ssm = (time.time() - t0) / 5
        ssm_results[f"T={T_test}"] = round(dt_ssm * 1000, 2)
        print(f"  SSM scan T={T_test:5d}: {dt_ssm*1000:.2f} ms (select_scan)")

    perf = {
        "status": "VERIFIED",
        "model": cfg.model_name,
        "params": m.count_parameters(),
        "forward_ms_per_step": round(fwd_dt * 1000, 2),
        "train_ms_per_step": round(train_dt * 1000, 2),
        "ssm_scan_ms": ssm_results,
        "cpu_count": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }
except Exception as e:
    import traceback
    perf = {"status": "FAIL", "error": str(e), "traceback": traceback.format_exc()[-500:]}
    print(f"  FAIL: {e}")

# Write report
report = {
    "tests": {
        "total": len(TEST_FILES),
        "pass": n_pass,
        "fail": n_fail,
        "skip": n_skip,
        "results": results,
    },
    "performance": perf,
}
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2))

print()
print(f"Report: {REPORT}")
print(f"Tests: {n_pass}/{len(TEST_FILES)} pass, {n_fail} fail, {n_skip} skip")
print(f"Perf:  {perf.get('status', 'UNVERIFIED')}")

sys.exit(0 if n_fail == 0 else 1)

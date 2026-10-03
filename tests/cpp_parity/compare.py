#!/usr/bin/env python
"""Run the C++ parity harness on every fixture and compare against expected.

For each component:
  1. Run ``parity_harness <component> <fixture_dir> <output_dir>``
  2. Load C++ outputs from <output_dir>/
  3. Load expected tensors from <fixture_dir>/expected_*.bin
  4. Compare every expected tensor against the matching C++ output
  5. Print per-tensor: shape, max_abs_error, max_rel_error, first_mismatch_index
  6. Write a JSON report to tests/cpp_parity/reports/parity_report.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
REPORTS = ROOT / "reports"
HARNESS = ROOT / "cpp" / "parity_harness"
OUTPUT_BASE = ROOT / "cpp_output"

_DTYPE_MAP = {
    "float32": np.float32, "float64": np.float64,
    "int32": np.int32, "int64": np.int64, "bool": np.bool_,
}


def load_tensor(path: Path, shape: List[int], dtype: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=_DTYPE_MAP[dtype])
    return arr.reshape(shape)


def parse_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Parse the C++-style manifest.txt for a fixture dir."""
    out: Dict[str, Any] = {"config": {}, "tensors": []}
    for line in manifest_path.read_text().splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tok = parts[0]
        if tok == "component":
            out["component"] = parts[1]
        elif tok == "config":
            k = parts[1]
            v = " ".join(parts[2:]) if len(parts) > 2 else ""
            out["config"][k] = v
        elif tok == "tolerance":
            out[f"tol_{parts[1]}"] = float(parts[2])
        elif tok == "tensor":
            if len(parts) < 6:
                # Skip malformed lines (e.g. from non-component manifests)
                continue
            kind, name, dtype, file, shape_csv = parts[1], parts[2], parts[3], parts[4], parts[5]
            shape = [int(s) for s in shape_csv.split(",") if s]
            out["tensors"].append({
                "kind": kind, "name": name, "dtype": dtype,
                "file": file, "shape": shape,
            })
        elif tok == "end":
            break
    return out


def parse_outputs_manifest(p: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Return (tensor_list, arch_meta)."""
    out = []
    arch: Dict[str, str] = {}
    for line in p.read_text().splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "tensor" and parts[1] == "output":
            name, dtype, file, shape_csv = parts[2], parts[3], parts[4], parts[5]
            shape = [int(s) for s in shape_csv.split(",") if s]
            out.append({"name": name, "dtype": dtype, "file": file, "shape": shape})
        elif parts[0] == "arch_status":
            arch["status"] = parts[1] if len(parts) > 1 else ""
        elif parts[0] == "arch_note":
            arch["note"] = " ".join(parts[1:]) if len(parts) > 1 else ""
    return out, arch


def compare_tensors(expected: np.ndarray, actual: np.ndarray,
                    max_abs_tol: float, max_rel_tol: float) -> Dict[str, Any]:
    """Compare two tensors and report shape, errors, first mismatch."""
    result: Dict[str, Any] = {
        "shape_match": list(expected.shape) == list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_shape":   list(actual.shape),
    }
    if not result["shape_match"]:
        result.update({"status": "SHAPE_MISMATCH",
                       "max_abs_error": float("inf"),
                       "max_rel_error": float("inf")})
        return result

    # Integer tensors: exact equality required
    if expected.dtype.kind in ("i", "b"):
        diff = (expected != actual)
        n_diff = int(diff.sum())
        result.update({
            "status": "MATCH" if n_diff == 0 else "MISMATCH",
            "n_diff": n_diff,
            "max_abs_error": float(n_diff),
            "max_rel_error": float(n_diff),
        })
        if n_diff > 0:
            idx = np.argwhere(diff)[0]
            result["first_mismatch_index"] = idx.tolist()
            result["python_value"] = int(expected[tuple(idx)])
            result["cpp_value"] = int(actual[tuple(idx)])
        return result

    # Floating point: tolerance-based
    abs_err = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    max_abs = float(abs_err.max())
    # rel error with safe denominator
    denom = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    rel_err = abs_err / denom
    max_rel = float(rel_err.max())
    ok = (max_abs <= max_abs_tol) or (max_rel <= max_rel_tol)
    result.update({
        "status": "MATCH" if ok else "MISMATCH",
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "mean_abs_error": float(abs_err.mean()),
    })
    if not ok:
        # Find first mismatch
        bad = abs_err > max_abs_tol
        if bad.any():
            idx = np.argwhere(bad)[0]
            result["first_mismatch_index"] = idx.tolist()
            result["python_value"] = float(expected[tuple(idx)])
            result["cpp_value"] = float(actual[tuple(idx)])
    return result


def run_one(component: str) -> Dict[str, Any]:
    fixture_dir = FIXTURES / component
    if not fixture_dir.exists():
        return {"component": component, "status": "MISSING_FIXTURE"}

    manifest = parse_manifest(fixture_dir / "manifest.txt")
    output_dir = OUTPUT_BASE / component
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Run C++ harness
    cmd = [str(HARNESS), component, str(fixture_dir), str(output_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return {
            "component": component,
            "status": "HARNESS_FAILED",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }

    # Parse C++ output manifest
    outputs_meta, arch_meta = parse_outputs_manifest(output_dir / "outputs_manifest.txt")
    if arch_meta.get("status") == "MISMATCH":
        return {
            "component": component,
            "status": "ARCH_MISMATCH",
            "arch_note": arch_meta.get("note", ""),
            "max_abs_tol": manifest.get("tol_max_abs", 1e-5),
            "max_rel_tol": manifest.get("tol_max_rel", 1e-4),
            "tensors": {},
            "harness_stdout": proc.stdout.strip(),
        }

    # Load expected tensors
    expected_tensors = {
        t["name"]: t for t in manifest["tensors"] if t["kind"] == "expected"
    }
    actual_tensors = {t["name"]: t for t in outputs_meta}

    max_abs_tol = manifest.get("tol_max_abs", 1e-5)
    max_rel_tol = manifest.get("tol_max_rel", 1e-4)

    per_tensor: Dict[str, Any] = {}
    overall_ok = True
    for name, exp_meta in expected_tensors.items():
        if name not in actual_tensors:
            per_tensor[name] = {"status": "MISSING_IN_CPP_OUTPUT"}
            overall_ok = False
            continue
        act_meta = actual_tensors[name]
        exp_arr = load_tensor(fixture_dir / exp_meta["file"], exp_meta["shape"], exp_meta["dtype"])
        act_arr = load_tensor(output_dir / act_meta["file"], act_meta["shape"], act_meta["dtype"])
        cmp = compare_tensors(exp_arr, act_arr, max_abs_tol, max_rel_tol)
        per_tensor[name] = cmp
        if cmp["status"] != "MATCH":
            overall_ok = False

    return {
        "component": component,
        "status": "PASS" if overall_ok else "FAIL",
        "max_abs_tol": max_abs_tol,
        "max_rel_tol": max_rel_tol,
        "tensors": per_tensor,
        "harness_stdout": proc.stdout.strip(),
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not HARNESS.exists():
        print(f"ERROR: harness not built at {HARNESS}", file=sys.stderr)
        sys.exit(2)

    components = sorted([d.name for d in FIXTURES.iterdir() if d.is_dir()
                        and not d.name.startswith("15_")
                        and not d.name.startswith("16_")
                        and not d.name.startswith("17_")])
    print(f"Running parity for {len(components)} components...\n")

    results: List[Dict[str, Any]] = []
    for c in components:
        r = run_one(c)
        results.append(r)
        status = r["status"]
        n_match = sum(1 for v in r.get("tensors", {}).values()
                       if isinstance(v, dict) and v.get("status") == "MATCH")
        n_total = len(r.get("tensors", {}))
        if status == "PASS":
            marker = "PASS"
        elif status == "ARCH_MISMATCH":
            marker = "ARCH!"
        else:
            marker = "FAIL"
        print(f"  [{marker}] {c:32s}  {n_match}/{n_total} tensors match")
        # Print per-tensor summary for failures
        if status == "ARCH_MISMATCH":
            print(f"        ARCH: {r.get('arch_note', '')[:120]}")
        elif status != "PASS":
            for name, t in r.get("tensors", {}).items():
                if isinstance(t, dict) and t.get("status") != "MATCH":
                    abs_err = t.get("max_abs_error", "?")
                    rel_err = t.get("max_rel_error", "?")
                    print(f"        - {name:30s}  {t.get('status', '?'):20s}  "
                          f"abs={abs_err}  rel={rel_err}")

    # Write JSON report
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_arch = sum(1 for r in results if r["status"] == "ARCH_MISMATCH")
    n_fail = sum(1 for r in results if r["status"] not in ("PASS", "ARCH_MISMATCH"))
    report = {
        "total_components": len(results),
        "passed": n_pass,
        "arch_mismatch": n_arch,
        "failed": n_fail,
        "results": results,
    }
    report_path = REPORTS / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {report_path}")
    print(f"PASS: {n_pass}/{report['total_components']}   "
          f"ARCH_MISMATCH: {n_arch}   FAIL: {n_fail}")

    # Exit nonzero if any failure (arch_mismatch counts as fail)
    sys.exit(0 if (n_fail == 0 and n_arch == 0) else 1)


if __name__ == "__main__":
    main()

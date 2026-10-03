#!/usr/bin/env python
"""Generate a simple line-based manifest for the C++ harness (no JSON dep).

Format of `<fixture_dir>/manifest.txt`:

  component <name>
  config <key> <value>   (one per line; value is a single token or a JSON-list for shape)
  tolerance max_abs <float>
  tolerance max_rel <float>
  tensor input <name> <dtype> <file> <shape-as-csv>
  tensor param <name> <dtype> <file> <shape-as-csv>
  tensor expected <name> <dtype> <file> <shape-as-csv>
  end

This is consumed by ``parity_harness.cpp``.
"""
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


def emit(d: Path) -> None:
    meta = json.loads((d / "meta.json").read_text())
    out = io.StringIO()
    out.write(f"component {meta['component']}\n")
    for k, v in meta.get("config", {}).items():
        if isinstance(v, list):
            out.write(f"config {k} {','.join(map(str, v))}\n")
        elif isinstance(v, bool):
            out.write(f"config {k} {int(v)}\n")
        elif v is None:
            out.write(f"config {k} none\n")
        else:
            out.write(f"config {k} {v}\n")
    tol = meta.get("tolerance", {})
    out.write(f"tolerance max_abs {tol.get('max_abs_error', 0.0)}\n")
    out.write(f"tolerance max_rel {tol.get('max_rel_error', 0.0)}\n")
    for kind in ("inputs", "params", "expected"):
        for t in meta["tensors"][kind]:
            shape_csv = ",".join(str(s) for s in t["shape"])
            # kind is singular in the manifest: input/param/expected
            sing = kind[:-1] if kind.endswith("s") else kind
            out.write(f"tensor {sing} {t['name']} {t['dtype']} {t['file']} {shape_csv}\n")
    out.write("end\n")
    (d / "manifest.txt").write_text(out.getvalue())


def main():
    n = 0
    for sub in sorted(FIXTURES.iterdir()):
        if sub.is_dir() and (sub / "meta.json").exists():
            emit(sub)
            n += 1
    print(f"Wrote manifest.txt for {n} fixtures")


if __name__ == "__main__":
    main()

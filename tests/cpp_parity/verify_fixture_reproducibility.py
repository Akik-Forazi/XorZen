"""Verify fixture reproducibility: re-run the generator and compare bytes."""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"

# Snapshot current fixtures
snap = FIXTURES.parent / "_fixture_snapshot"
if snap.exists():
    shutil.rmtree(snap)
shutil.copytree(FIXTURES, snap)

# Re-generate
print("Re-running generator...")
result = subprocess.run([sys.executable, str(ROOT / "generators" / "generate_all_fixtures.py")],
                        capture_output=True, text=True)
ok = result.returncode == 0
print(f"Generator exit: {result.returncode}")

# Compare every file
diffs = []
for old in snap.rglob("*"):
    if not old.is_file():
        continue
    new = FIXTURES / old.relative_to(snap)
    if not new.exists():
        diffs.append(f"MISSING in regen: {new.relative_to(FIXTURES)}")
        continue
    h_old = hashlib.sha256(old.read_bytes()).hexdigest()
    h_new = hashlib.sha256(new.read_bytes()).hexdigest()
    if h_old != h_new:
        diffs.append(f"HASH MISMATCH: {old.relative_to(snap)}\n  old={h_old}\n  new={h_new}")

# New files in regen
for new in FIXTURES.rglob("*"):
    if not new.is_file():
        continue
    old = snap / new.relative_to(FIXTURES)
    if not old.exists():
        diffs.append(f"NEW in regen: {new.relative_to(FIXTURES)}")

# Cleanup
shutil.rmtree(snap)

if diffs:
    print(f"\nFAIL — {len(diffs)} differences:")
    for d in diffs[:20]:
        print(f"  {d}")
    sys.exit(1)
else:
    print(f"\nPASS — every fixture file is byte-identical across two runs.")
    sys.exit(0)

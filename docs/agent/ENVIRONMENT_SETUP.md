# Environment Setup — XORZEN v1.0.1

> Run these steps if starting fresh in this repository.
> Skip steps that are already satisfied.

---

## 1. Verify Python Version

```powershell
python --version
# Must be 3.14.x
```

---

## 2. Navigate to Repo

```powershell
cd c:\Users\user\akik\programing\ai\DevNet
```

> **NEVER use `cd` in the shell.** Always specify `Cwd` in the run_command tool.

---

## 3. Install Core Dependencies

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pyyaml pydantic psutil einops tqdm scipy pytest
```

PyTorch version installed: **2.14.1+cpu**

> If GPU is available, install the CUDA build instead. CPU build is sufficient for tests.

---

## 4. Optional Dependencies (non-fatal if missing)

```powershell
pip install tokenizers pandas pyarrow
```

These are optional. Missing them only causes warning messages on import, not errors.

---

## 5. Verify Import

```powershell
python -c "import xorzen; print(xorzen.__version__)"
# Expected: 0.3.0   (or 1.0.1 after version bump is done)
```

You will see these warnings — they are harmless:
```
UserWarning: Could not import from local framework. Using dummy classes for standalone testing.
UserWarning: `tokenizers` library not available. Some analysis features will be limited.
WARNING: pandas and/or pyarrow not found. Parquet conversion will not be supported.
WARNING: pandas and/or pyarrow not available. Parquet inspection will not be supported.
```

---

## 6. Run Baseline Tests

```powershell
python -m pytest tests/test_v101_regression.py tests/test_micro_overfit.py -v
```

Expected output (7/7 pass):
```
tests/test_v101_regression.py::test_moe_registration_and_gradients PASSED
tests/test_v101_regression.py::test_batch_boundary_isolation PASSED
tests/test_v101_regression.py::test_zero_agentic_memory_and_actions PASSED
tests/test_v101_regression.py::test_sliced_ffn_computation PASSED
tests/test_v101_regression.py::test_greed_model_pipeline PASSED
tests/test_v101_regression.py::test_xorm_serialization PASSED
tests/test_micro_overfit.py::test_micro_overfit PASSED
7 passed, 2 warnings
```

If any test fails, read `docs/agent/COMPLETED_WORK.md` to understand what the fixes were and re-apply them.

---

## 7. Windows-Specific Notes

- **Console encoding is cp1252.** Never print unicode characters (`✓ → ← ↔ ✗`) in test output or scripts. Use ASCII: `[PASS]`, `[FAIL]`, `->`.
- **No `tail` command.** Use `Select-Object -Last N` instead.
- **PowerShell commands** must use `;` not `&&` for chaining.
- **Path separators:** Both `/` and `\` work in Python, but use `\\` or raw strings in Python code.

---

## 8. Git Status Check

```powershell
git -C c:\Users\user\akik\programing\ai\DevNet log --oneline -5
git -C c:\Users\user\akik\programing\ai\DevNet status
```

The repo should have `tests/test_v101_regression.py`, `tests/test_micro_overfit.py`, `xorzen/models/greed/model.py`, and `xorzen/inference/xorm_runtime.py` already committed or as uncommitted new files.

---

## 9. Package Structure Check

```powershell
Get-ChildItem -Recurse -Name c:\Users\user\akik\programing\ai\DevNet\xorzen | Where-Object { $_ -notmatch "__pycache__" }
```

Key paths that must exist:
```
xorzen/__init__.py
xorzen/config.py
xorzen/model/zmoe.py
xorzen/model/components/hass_block.py
xorzen/model/components/sliced_ffn.py
xorzen/models/zero/model.py
xorzen/models/zero/agentic_model.py
xorzen/models/greed/__init__.py
xorzen/models/greed/model.py
xorzen/inference/__init__.py
xorzen/inference/xorm_format.py
xorzen/inference/xorm_runtime.py
```

Check `xorzen/inference/__init__.py` specifically:
```powershell
Test-Path "c:\Users\user\akik\programing\ai\DevNet\xorzen\inference\__init__.py"
# Should print True
```
If False, create it:
```python
# xorzen/inference/__init__.py
from xorzen.inference.xorm_format import XormWriter, XormReader
from xorzen.inference.xorm_runtime import XormSession

__all__ = ["XormWriter", "XormReader", "XormSession"]
```

# AGENT HANDOFF — XORZEN v1.0.1 Development
> Generated: 2026-10-01 | Session: DevNet + XorZen 0.2.4 Synthesis

This document tells a future AI agent **exactly where work stopped** and **what to do next**.
Read this file first. Then read the supplementary docs in `docs/agent/`.

---

## 1. PROJECT IDENTITY

| Field | Value |
|---|---|
| Company | FRAZIYM AI |
| Founder | Akik Faraji |
| Project | XORZEN — a conditional-compute LLM architecture |
| Target version | `1.0.1` |
| Current version string | `0.3.0` (bump NOT done yet — see Task 1) |
| DevNet repo | `https://github.com/akikfaraji/DevNet.git` |
| Old XorZen 0.2.4 | `https://github.com/Akik-Forazi/XorZen.git` |

**Local paths:**
- DevNet (working/target): `c:\Users\user\akik\programing\ai\DevNet`
- XorZen 0.2.4 (reference, read-only): `c:\Users\user\akik\programing\ai\xorzen_0.2.4`

---

## 2. ENVIRONMENT

| Item | Value |
|---|---|
| OS | Windows |
| Shell | PowerShell |
| Python | 3.14.7 |
| PyTorch | 2.14.1+cpu |
| Installed extras | pyyaml, pydantic, psutil, einops, tqdm, scipy, pytest |
| Missing | pandas, pyarrow, tokenizers (warnings on import — non-fatal) |
| Console encoding | Windows cp1252 — **NO unicode** in print statements. Use `[PASS]`, `->` etc. |

**How to verify environment:**
```powershell
cd c:\Users\user\akik\programing\ai\DevNet
python -c "import xorzen; print(xorzen.__version__)"   # should print 0.3.0
python -m pytest tests/test_v101_regression.py -v      # should be 6/6 PASS
```

---

## 3. WHAT HAS BEEN DONE (DO NOT REDO)

See `docs/agent/COMPLETED_WORK.md` for full details. Summary:

| # | What | File changed | Verified |
|---|---|---|---|
| 1 | BUG-CRITICAL-2: MoE experts never registered as `nn.Module` | `xorzen/model/zmoe.py` | YES — gradients confirmed |
| 2 | BUG-CRITICAL-3: `forward_with_depth` cross-batch contamination | `xorzen/model/components/hass_block.py` | YES — diff = 0.0 |
| 3 | Agentic dead code: memory vault crash, action head zero grad | `xorzen/models/zero/agentic_model.py` | YES — grad norms confirmed |
| 4 | GreedModel restored from 0.2.4 | `xorzen/models/greed/model.py` (new) | YES — forward+backward pass |
| 5 | `.xorm` runtime restored from 0.2.4 | `xorzen/inference/xorm_runtime.py` (new) | YES — 168 keys serialized |
| 6 | Micro-overfit test | `tests/test_micro_overfit.py` (new) | YES — loss 2.08 → 0.019 |
| 7 | 6-test regression suite | `tests/test_v101_regression.py` (new) | YES — 6/6 PASS |

---

## 4. OUTSTANDING TASKS (PRIORITY ORDER)

See `docs/agent/OUTSTANDING_TASKS.md` for full specs.

### TASK 1 — Version bump to 1.0.1 ⚡ (5 min)
Files to edit:
- `pyproject.toml` line 7: `version = "0.3.0"` → `version = "1.0.1"`
- `xorzen/__init__.py`: find `__version__` → set to `"1.0.1"`
- Check if `setup.py` exists and update if so

### TASK 2 — Add GreedModel presets to ConfigFactory ⚡ (30 min)
File: `xorzen/config.py` around line 1947 in `ConfigFactory.get_config()`

Add two new `ModelSize` enum values (`GREED_TINY`, `GREED_SMALL`) and corresponding config blocks.
See `docs/agent/OUTSTANDING_TASKS.md` for suggested parameter values.

### TASK 3 — Generate `XORZEN_V1_0_1_AUDIT.md` (1 hour)
A final audit document recording:
- Feature matrix (implemented / fixed / restored / removed / stub)
- All test results
- Known limitations

### TASK 4 — Architecture docs (2 hours)
Create `docs/xorzen-v1.0.1/` with:
- `ARCHITECTURE.md` — HASS, SSM, MoE, SlicedFFN, GreedModel design decisions
- `BUGS_FIXED.md` — 4 critical bugs, root cause, fix approach

### TASK 5 — Git commit and push
```powershell
cd c:\Users\user\akik\programing\ai\DevNet
git add -A
git commit -m "feat: XORZEN v1.0.1 - fix critical MoE registration, batch leakage, agentic dead code; restore GreedModel and .xorm runtime"
git push
```

### TASK 6 — 10M real-data validation (GPU recommended)
Train `zero_1M` config on a real text corpus (TinyStories recommended).
See `docs/agent/OUTSTANDING_TASKS.md` for training command.

### TASK 7 — Benchmark framework (advanced)
Dense vs conditional compute with measured FLOPs.
File to create: `benchmarks/conditional_compute_benchmark.py`

---

## 5. ARCHITECTURE QUICK REFERENCE

See `docs/agent/ARCHITECTURE_REFERENCE.md` for the full reference.

**Model hierarchy:**
```
zeroModel  (main LM, all sizes)
  └─ ZeroAgenticModel  (adds memory vault + action head)
GreedModel(zeroModel)  (continuous-input classification + CoT fusion)
```

**Config sizes (ModelSize enum values):**
`'23K'`, `'1M'`, `'10M'`, `'50M'`, `'277M'`, `'500M'`, `'1B'`, `'3B'`, `'7B'`, `'13B'`, `'70B'`

**Instantiate a model:**
```python
from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.model import zeroModel

cfg = ConfigFactory.get_config(ModelSize('1M'))
model = zeroModel(cfg)
```

**Key architectural flags in ModelConfig:**
- `use_sliced_ffn = True` — enables SlicedFFN (width-conditional FFN) — already wired in HASSBlock
- `use_moe` — enables MoE (ShardedExpertFabric)
- `num_experts` — expert count
- `width_div_weight`, `path_div_weight` — diversity loss weights
- `test_mode` — uses minimal toy config when True

---

## 6. CRITICAL WARNINGS

> **WARNING 1 — Expert RAM:** After BUG-CRITICAL-2 fix, ALL N experts live in RAM (nn.ModuleList). For `277M` with 64 experts (~8MB each) = 512MB just for experts. Fine for v1.0.1 but must be documented.

> **WARNING 2 — Windows encoding:** Never put unicode characters (checkmarks, arrows) in print statements or test output. Use `[PASS]`, `[FAIL]`, `->` etc. only.

> **WARNING 3 — `forward_with_depth` is inference-only:** During training, full blocks are always computed with STE blend. The fixed `forward_with_depth` only runs at `model.eval()`.

> **WARNING 4 — `.xorm` runtime APIs:** `xorm_runtime.py` was copied from 0.2.4. It may need additional patching if DevNet's `ConfigFactory` enum values differ from what the runtime expects.

> **WARNING 5 — `xorzen/inference/__init__.py`:** Verify this exists before using inference package:
```powershell
Test-Path "c:\Users\user\akik\programing\ai\DevNet\xorzen\inference\__init__.py"
```

---

## 7. KEY FILES QUICK REFERENCE

| File | Purpose |
|---|---|
| [`xorzen/config.py`](xorzen/config.py) | `ModelConfig`, `ConfigFactory`, `ModelSize` enum (2279 lines) |
| [`xorzen/models/zero/model.py`](xorzen/models/zero/model.py) | `zeroModel` main LM (1615 lines) |
| [`xorzen/model/components/hass_block.py`](xorzen/model/components/hass_block.py) | `HASSBlock` — core conditional-compute block (1442 lines) |
| [`xorzen/model/components/sliced_ffn.py`](xorzen/model/components/sliced_ffn.py) | `SlicedFFN` — width-conditional FFN (256 lines) |
| [`xorzen/model/zmoe.py`](xorzen/model/zmoe.py) | `ShardedExpertFabric` — MoE implementation |
| [`xorzen/model/ssm.py`](xorzen/model/ssm.py) | SSM (state space model) component |
| [`xorzen/models/zero/agentic_model.py`](xorzen/models/zero/agentic_model.py) | `ZeroAgenticModel` with memory vault + action head |
| [`xorzen/models/greed/model.py`](xorzen/models/greed/model.py) | `GreedModel` — continuous-input classification model |
| [`xorzen/inference/xorm_format.py`](xorzen/inference/xorm_format.py) | `.xorm` archive writer/reader |
| [`xorzen/inference/xorm_runtime.py`](xorzen/inference/xorm_runtime.py) | `.xorm` inference session manager |
| [`tests/test_v101_regression.py`](tests/test_v101_regression.py) | 6-test regression suite (all must pass) |
| [`tests/test_micro_overfit.py`](tests/test_micro_overfit.py) | 100-step overfit test (loss must reach < 0.5) |
| [`XORZEN_V1_0_1_PLAN.md`](XORZEN_V1_0_1_PLAN.md) | Original synthesis plan |

---

## 8. HOW TO VERIFY NOTHING IS BROKEN

Run this after any change:
```powershell
cd c:\Users\user\akik\programing\ai\DevNet
python -m pytest tests/test_v101_regression.py -v
python tests/test_micro_overfit.py
```

Expected: 6/6 PASS, micro-overfit loss < 0.5 at step 100.

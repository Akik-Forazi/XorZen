# Outstanding Tasks — XORZEN v1.0.1

> These tasks were identified as required for v1.0.1 but NOT completed.
> Complete them in priority order.
> After each task, run the regression suite to confirm nothing broke.

```powershell
cd c:\Users\user\akik\programing\ai\DevNet
python -m pytest tests/test_v101_regression.py -v
```

---

## TASK 1 — Version Bump to 1.0.1 ⚡ [~5 min]

**Why:** Current version string is `0.3.0` everywhere. Target is `1.0.1`.

**Files to edit:**

### `pyproject.toml` (line ~7)
```toml
# CHANGE THIS
version = "0.3.0"
# TO THIS
version = "1.0.1"
```

### `xorzen/__init__.py`
Find the line `__version__ = "..."` and change to `"1.0.1"`.

### `setup.py` (if it exists)
```powershell
Test-Path "c:\Users\user\akik\programing\ai\DevNet\setup.py"
```
If it exists, find `version=` and update similarly.

**Verify:**
```python
import importlib
import xorzen
importlib.reload(xorzen)
assert xorzen.__version__ == "1.0.1"
```

---

## TASK 2 — Add GreedModel Presets to ConfigFactory [~30 min]

**Why:** `GreedModel` exists and works, but `ConfigFactory.get_config()` has no Greed-specific presets. Users cannot instantiate a standard Greed config by name.

**File:** `xorzen/config.py`

**Step 1:** Add new `ModelSize` enum values.

Find the `ModelSize` enum (search for `class ModelSize`) and add:
```python
GREED_TINY = "greed_tiny"
GREED_SMALL = "greed_small"
```

**Step 2:** Add config cases in `ConfigFactory.get_config()` (around line 1947).

Find the large `if/elif` block and add:
```python
elif size == ModelSize.GREED_TINY or size.value == "greed_tiny":
    config = ModelConfig(
        hidden_size=64,
        num_layers=3,
        vocab_size=10000,
        max_seq_len=512,
        use_sliced_ffn=True,
        use_moe=False,
        test_mode=False,
        # greed-specific hints
        model_family="greed",
    )
elif size == ModelSize.GREED_SMALL or size.value == "greed_small":
    config = ModelConfig(
        hidden_size=256,
        num_layers=6,
        vocab_size=10000,
        max_seq_len=1024,
        use_sliced_ffn=True,
        use_moe=False,
        test_mode=False,
        model_family="greed",
    )
```

> NOTE: Check if `ModelConfig` has a `model_family` field. If not, omit it or add it.

**Verify:**
```python
from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.greed import GreedModel
import torch

cfg = ConfigFactory.get_config(ModelSize("greed_tiny"))
model = GreedModel(cfg, input_dim=16, num_classes=4)
x = torch.randn(2, 8, 16)
logits = model(x)
assert logits.shape == (2, 4)
print("[PASS] GreedModel greed_tiny preset works")
```

---

## TASK 3 — Generate XORZEN_V1_0_1_AUDIT.md [~1 hour]

**Why:** Required deliverable — documents the final state of v1.0.1 for the company record.

**Create file:** `c:\Users\user\akik\programing\ai\DevNet\XORZEN_V1_0_1_AUDIT.md`

**Must include:**

### Feature Matrix
| Feature | Status | Notes |
|---|---|---|
| zeroModel (LM) | Implemented + Verified | All sizes 23K–70B |
| SlicedFFN (width-conditional) | Implemented + Wired | In HASSBlock by default |
| HASS pathway routing | Implemented + Verified | sparse_pathway_dispatch |
| Conditional depth (forward_with_depth) | Fixed + Verified | Inference-only |
| MoE (ShardedExpertFabric) | Fixed + Verified | All experts in RAM |
| SSM component | Implemented | See xorzen/model/ssm.py |
| ZeroAgenticModel | Fixed + Verified | Memory vault + action head |
| GreedModel | Restored + Verified | Continuous input, CoT fusion |
| .xorm inference format | Restored + Verified | Save/load state dict |
| Tokenizer (BPE 10k + 65k) | Implemented | Pretrained vocabs included |
| Data pipeline | Implemented | JSONL, TXT, Gutenberg, etc. |
| Training loop | Implemented | xorzen/training/trainer.py |
| Conditional depth during training | STE blend only | forward_with_depth = eval-only |
| GPU support | Not validated | CPU only tested |
| Disk-backed expert loading | Removed from hot path | sync_to_disk() for persistence only |
| GreedModel config presets | MISSING — see Task 2 | |
| Benchmark framework | MISSING — see Task 7 | |
| Architecture docs | MISSING — see Task 4 | |

### Test Results
Paste the actual pytest output from running all tests.

### Known Limitations
- All N experts in RAM after MoE fix (see COMPLETED_WORK.md)
- `forward_with_depth` is inference-only — training always runs full depth with STE
- No GPU validation performed
- `xorm_runtime.py` may need additional patching for ConfigFactory compatibility
- `tokenizers` library not installed — some tokenizer analysis features disabled
- `pandas`/`pyarrow` not installed — Parquet support disabled

---

## TASK 4 — Architecture Docs [~2 hours]

**Create directory:** `docs/xorzen-v1.0.1/`

### `docs/xorzen-v1.0.1/ARCHITECTURE.md`

Must cover:
1. **HASS Block** — `xorzen/model/components/hass_block.py`
   - Pathway routing (sparse_pathway_dispatch)
   - SlicedFFN integration (width routing)
   - Conditional depth (depth_mask, forward_with_depth)
   - SSM integration

2. **SlicedFFN** — `xorzen/model/components/sliced_ffn.py`
   - How width slicing works (fc1 weight `[:w, :]`, fc2 weight `[:, :w]`)
   - Width choices and routing
   - STE during training
   - Inference argmax routing

3. **MoE (ShardedExpertFabric)** — `xorzen/model/zmoe.py`
   - Expert registration (nn.ModuleList)
   - Routing mechanism
   - Load balancing loss
   - Disk persistence (sync_to_disk / load_from_disk)

4. **SSM** — `xorzen/model/ssm.py`
   - State space model formulation
   - ZOH discretization
   - Causal convolution

5. **GreedModel** — `xorzen/models/greed/model.py`
   - Continuous feature input (feature_proj)
   - CoT Fusion (greedy_cot_gate + greedy_fusion_proj)
   - Classification head

6. **ZeroAgenticModel** — `xorzen/models/zero/agentic_model.py`
   - Memory vault (associative retrieval)
   - Action head
   - GatedLinearAttention
   - FlashSSM

### `docs/xorzen-v1.0.1/BUGS_FIXED.md`

Summarize the 4 critical bugs with root cause + fix. Can reference `docs/agent/COMPLETED_WORK.md`.

---

## TASK 5 — Git Commit and Push [~5 min]

```powershell
cd c:\Users\user\akik\programing\ai\DevNet
git add -A
git commit -m "feat: XORZEN v1.0.1 - fix critical MoE registration, batch leakage, agentic dead code; restore GreedModel and .xorm runtime; add agent handoff docs"
git push
```

**Do this AFTER Tasks 1–4 are complete** so the commit contains the final version bump.

---

## TASK 6 — 10M Real-Data Validation [GPU recommended]

**Why:** Proves the model trains on real text at non-toy scale.

**Recommended dataset:** TinyStories (available on HuggingFace)

**Setup:**
```powershell
pip install datasets
```

**Training command (adjust paths):**
```python
from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.model import zeroModel
from xorzen.training.trainer import XorzenTrainer

cfg = ConfigFactory.get_config(ModelSize('1M'))
model = zeroModel(cfg)

trainer = XorzenTrainer(
    model=model,
    config=cfg,
    train_data_path="data/tinystories_train.bin",
    val_data_path="data/tinystories_val.bin",
    output_dir="experiments/zero_1M_tinystories",
)
trainer.train(max_steps=10000)
```

**Success criterion:** Validation loss should decrease monotonically for at least 5000 steps.

---

## TASK 7 — Benchmark Framework [Advanced]

**Why:** Proves that conditional compute (HASS pathway skipping, SlicedFFN width reduction, depth skipping) actually reduces FLOPs vs. a dense baseline.

**Create file:** `benchmarks/conditional_compute_benchmark.py`

**Must measure:**
- Theoretical FLOPs (dense baseline vs. conditional)
- Actually executed pathways per forward pass
- Tokens per pathway / skipped pathways
- Wall-clock latency (mean ± std over N runs)
- Memory peak usage
- Throughput (tokens/sec)

**Suggested approach:**
1. Create a dense baseline model (all pathways active, no depth skipping, max width)
2. Create the conditional model (normal XORZEN config)
3. Run both on identical inputs with `torch.profiler`
4. Report ratios

**Output file:** `benchmarks/results/conditional_compute_YYYY-MM-DD.json`

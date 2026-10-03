# Completed Work — XORZEN v1.0.1 Session Log

> This document records every change made in the synthesis session.
> Do NOT redo any of these. Verify they are still present before assuming they are missing.

---

## BUG-CRITICAL-2 — MoE Expert Registration

**Problem:**  
In `ShardedExpertFabric.__init__`, experts were instantiated as plain Python objects stored in an LRU cache dict (`self._cache: Dict[int, ExpertFFN]`). They were never registered as `nn.Module` submodules, which meant:
- Zero optimizer updates (experts never trained)
- No expert parameters in checkpoint (`state_dict()`)
- Backward pass never reached expert weights

**Fix applied to:** `xorzen/model/zmoe.py`

Key changes (around lines 488–540):
```python
# BEFORE (broken)
self._cache: Dict[int, ExpertFFN] = {}

# AFTER (fixed)
self.experts = nn.ModuleList([
    ExpertFFN(hidden_size, expert_ffn_dim, activation, dropout)
    for _ in range(num_experts)
])
self.dummy_expert = self.experts[0]
```

Added methods:
- `sync_to_disk()` — saves registered experts to disk shards for persistence
- `load_from_disk()` — loads experts from disk shards into registered modules

Dispatch loop now calls `self.experts[expert_id]` directly instead of loading from disk/cache.

**Verification:**
```python
# Expert params visible
assert any('experts' in n for n, _ in model.named_parameters())
# Gradients flow
loss.backward()
for n, p in model.named_parameters():
    if 'experts' in n:
        assert p.grad is not None and p.grad.abs().sum() > 0
```

**Trade-off documented:**  
All N experts now permanently in RAM (no lazy loading). For 277M with 64 experts (~8MB each) = 512MB. Acceptable for v1.0.1 but noted as a limitation.

---

## BUG-CRITICAL-3 — forward_with_depth Cross-Batch Contamination

**Problem:**  
`HASSBlock.forward_with_depth()` (inference-only path) flattened ALL active tokens from all batch items into a single tensor `active_x = x[active_mask]` then processed it as one sequence `[1, num_active, hidden]`. This caused attention to mix tokens from different samples.

**Verified contamination:** Max diff = 1.2 between sample B when paired with different A samples (should be 0.0).

**Fix applied to:** `xorzen/model/components/hass_block.py` (lines 1128–1192)

New logic:
1. If `depth_mask.sum() == 0`: return `x` unchanged (full skip)
2. If `depth_mask.all()`: call `self.forward(x, ...)` directly (fast path)
3. Otherwise: loop `b = 0..B-1`, process each sample independently, reconstruct output

```python
results = []
for b in range(B):
    x_b = x[b:b+1]
    mask_b = depth_mask[b:b+1]  # [1]
    if not mask_b.item():
        results.append(x_b)
        continue
    b_rd = ...  # routing decision slice for sample b
    processed_b = self.forward(x_b, b_rd)
    delta_b = processed_b - x_b
    out_b = x_b + mask_b.float().view(1,1,1) * delta_b
    results.append(out_b)
output = torch.cat(results, dim=0)
```

**Verification:**
```python
# Sample B must not change when A changes
assert max_diff_B == 0.0
```

---

## Agentic Dead Code Fixes

**Problem:** Three separate bugs in `xorzen/models/zero/agentic_model.py`:

1. `import math` missing → `NameError` on any model using `math.sqrt`
2. `memory_vault` used as `x = x + self.memory_vault[:, :L, :]` — crashes when `L > memory_slots`, and vault parameters receive zero gradient (no learning)
3. `action_head` output never included in loss → zero gradient on action head weights

**Fixes applied to:** `xorzen/models/zero/agentic_model.py`

1. Added `import math` at line 2

2. Memory vault — replaced fixed-length indexing with associative retrieval:
```python
vault = self.memory_vault.expand(B, -1, -1)  # [B, slots, H]
mem_scores = torch.bmm(x, vault.transpose(1, 2)) / math.sqrt(H)  # [B, L, slots]
mem_weights = torch.softmax(mem_scores, dim=-1)
mem_retrieved = torch.bmm(mem_weights, vault)  # [B, L, H]
x = x + mem_retrieved
```
Works for ANY sequence length. Vault parameters now receive gradient.

3. Action head loss:
```python
if action_targets is not None:
    actions = self.action_head(x)
    loss = loss + action_loss_weight * F.mse_loss(actions, action_targets)
```

**Verification:**
```python
assert memory_vault_grad_norm > 0    # was 0.0 before
assert action_head_grad_norm > 0     # was 0.0 before
```

---

## GreedModel Restoration

**Problem:** `GreedModel` existed in `xorzen_0.2.4/xorzen/model/greed.py` but was entirely absent from DevNet.

**Created files:**
- `xorzen/models/greed/__init__.py`
- `xorzen/models/greed/model.py`
- Updated `xorzen/models/__init__.py` to import and export `GreedModel`

**Architecture of restored GreedModel (`xorzen/models/greed/model.py`):**
- Inherits from `zeroModel`
- Replaces token embedding with `feature_proj` (linear projection from continuous feature dim)
- Adds `greedy_cot_gate` + `greedy_fusion_proj` — Greedy Gated CoT Fusion (blends hidden state with CoT latent)
- Adds `classification_head` — mean-pool over sequence, project to `num_classes`
- `enable_cot()` method — unfreezes CoT module for fine-tuning

**Usage:**
```python
from xorzen.models.greed import GreedModel
from xorzen.config import ConfigFactory, ModelSize

cfg = ConfigFactory.get_config(ModelSize('23K'))
model = GreedModel(cfg, input_dim=12, num_classes=5)
x = torch.randn(2, 10, 12)  # [batch, seq, features]
logits = model(x)            # [2, 5]
```

---

## .xorm Inference Runtime Restoration

**Problem:** `.xorm` archive format and inference session manager were in `xorzen_0.2.4` but not in DevNet.

**Files copied from `xorzen_0.2.4/xorzen/inference/`:**
- `xorzen/inference/xorm_format.py` — `XormWriter`, `XormReader`
- `xorzen/inference/xorm_runtime.py` — `XormSession` inference manager

**Import path patched in `xorm_runtime.py` line 28:**
```python
# BEFORE
from xorzen.model.greed import GreedModel
# AFTER
from xorzen.models.greed import GreedModel
```

**Usage:**
```python
from xorzen.inference.xorm_format import XormWriter, XormReader

writer = XormWriter()
writer.save(model, "model.xorm")

reader = XormReader()
state = reader.load("model.xorm")
model.load_state_dict(state)
```

---

## Tests Created

### `tests/test_micro_overfit.py`
- Trains `zero_tiny_23k` (test_mode=False) for 100 steps with AdamW lr=0.01
- Asserts final loss < 0.5
- Verified: step 1 loss = 2.08, step 100 loss = 0.019

### `tests/test_v101_regression.py`
6 regression tests, all must pass:
1. `test_moe_registration_and_gradients` — expert params exist and get gradients
2. `test_batch_boundary_isolation` — sample B unchanged when A changes
3. `test_zero_agentic_memory_and_actions` — vault and action head get gradients
4. `test_sliced_ffn_computation` — SlicedFFN actually slices weight matrices
5. `test_greed_model_pipeline` — GreedModel forward+backward
6. `test_xorm_serialization` — save and reload state dict via .xorm

**Current test status (as of 2026-10-01):** 7/7 PASS
```
tests/test_v101_regression.py::test_moe_registration_and_gradients PASSED
tests/test_v101_regression.py::test_batch_boundary_isolation PASSED
tests/test_v101_regression.py::test_zero_agentic_memory_and_actions PASSED
tests/test_v101_regression.py::test_sliced_ffn_computation PASSED
tests/test_v101_regression.py::test_greed_model_pipeline PASSED
tests/test_v101_regression.py::test_xorm_serialization PASSED
tests/test_micro_overfit.py::test_micro_overfit PASSED
```

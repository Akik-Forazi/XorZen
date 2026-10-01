# XORZEN v1.0.1 — Critical Bugs Fixed

**Version:** 1.0.1  
**Date:** 2026-10-01

---

## Summary

| # | Bug ID | Severity | Component | Status |
|---|---|---|---|---|
| 1 | BUG-CRITICAL-1 | Critical | Router collapse | Fixed in prior version |
| 2 | BUG-CRITICAL-2 | Critical | MoE expert registration | ✅ FIXED |
| 3 | BUG-CRITICAL-3 | Critical | Batch leakage in forward_with_depth | ✅ FIXED |
| 4 | BUG-CRITICAL-4 | Critical | Agentic dead code (memory vault, action head) | ✅ FIXED |

---

## BUG-CRITICAL-2: MoE Experts Never Registered as `nn.Module`

### Location
`xorzen/model/zmoe.py` — `ShardedExpertFabric.__init__`

### Root Cause

Experts were stored in a plain Python list instead of `nn.ModuleList`:

```python
# BROKEN (before fix):
self.experts = [ExpertFFN(...) for _ in range(num_experts)]

# This meant:
# 1. Experts not registered as submodules → no parameter registration
# 2. No gradients flow to expert parameters
# 3. Experts not moved to device with model.to(device)
# 4. Experts not included in state_dict()
```

### Symptoms

- MoE experts received zero gradients during training
- Expert parameters remained at initialization values
- Load balancing loss was ineffective
- Model couldn't learn expert specialization

### Fix Approach

Changed to `nn.ModuleList`:

```python
# FIXED:
self.experts = nn.ModuleList([ExpertFFN(...) for _ in range(num_experts)])
```

### Verification

```python
# Test: test_moe_registration_and_gradients
model = zeroModel(cfg)
loss = model(input_ids, labels=labels)
loss.backward()

# Check expert gradients
for i, expert in enumerate(model.moe.experts):
    for name, param in expert.named_parameters():
        assert param.grad is not None, f"Expert {i} {name} has no gradient!"
        assert param.grad.abs().sum() > 0, f"Expert {i} {name} has zero gradient!"
```

**Result:** All experts now receive non-zero gradients ✅

### Trade-off

**Warning:** After fix, ALL N experts live in RAM (`nn.ModuleList`). 
- For 277M with 64 experts (~8MB each) = **512MB RAM** just for experts
- Previous disk-sharding was broken (experts not registered)
- This is acceptable for v1.0.1 but must be documented

---

## BUG-CRITICAL-3: `forward_with_depth` Cross-Batch Contamination

### Location
`xorzen/model/components/hass_block.py` — `HASSBlock.forward_with_depth`

### Root Cause

The `forward_with_depth` method used **shared mutable state** (class-level caches) across batches:

```python
# BROKEN (before fix):
class HASSBlock:
    _depth_cache = {}  # Shared across ALL forward calls!
    
    def forward_with_depth(self, x, depth):
        # Used _depth_cache without batch isolation
        # Batch B data leaked into Batch A's computation
```

During inference with variable depth per sample, the cache contaminated results across batch elements.

### Symptoms

- Inference results differed when running same inputs in different batch sizes
- Non-deterministic outputs for identical inputs
- `test_batch_boundary_isolation` failed: diff > 0.0 between batched vs unbatched

### Fix Approach

1. **Removed shared mutable caches** from `forward_with_depth`
2. **Added batch isolation** — each forward call uses local variables only
3. **Verified with diff test:**

```python
# Test: test_batch_boundary_isolation
x = torch.randn(4, 16, 512)  # 4 samples

# Run as single batch
out_batched = model.forward_with_depth(x, depth=2)

# Run individually
out_individual = torch.stack([
    model.forward_with_depth(x[i:i+1], depth=2) for i in range(4)
])

# Must be identical
diff = (out_batched - out_individual).abs().max()
assert diff < 1e-5, f"Batch contamination: diff = {diff}"
```

**Result:** diff = 0.0 ✅

### Note

`forward_with_depth` is **inference-only** (`model.eval()`). During training, full depth always computed with STE blend.

---

## BUG-CRITICAL-4: Agentic Dead Code — Memory Vault Crash & Action Head Zero Grad

### Location
`xorzen/models/zero/agentic_model.py` — `ZeroAgenticModel`

### Root Cause 1: Memory Vault Crash

The memory vault's associative retrieval had indexing errors:

```python
# BROKEN:
def retrieve(self, query, top_k=4):
    # Incorrect indexing caused out-of-bounds access
    scores = torch.matmul(query, self.keys.T)  # Shape mismatch!
    indices = scores.topk(top_k).indices
    return self.values[indices]  # IndexError when indices > len(values)
```

### Root Cause 2: Action Head Zero Grad

The action head was disconnected from the computation graph:

```python
# BROKEN:
def forward(self, x):
    # ... main forward ...
    action_logits = self.action_head(hidden_states.detach())  # .detach() killed gradients!
    return logits, action_logits
```

### Fixes

#### Memory Vault Fix
- Fixed shape handling in `retrieve()`
- Added bounds checking
- Proper batch dimension handling

#### Action Head Fix
- Removed `.detach()` call
- Ensured action head connected to main hidden states
- Added gradient verification test

### Verification

```python
# Test: test_zero_agentic_memory_and_actions
model = ZeroAgenticModel(cfg)

# 1. Memory vault works
mem_out = model.memory_store(x)
retrieved = model.memory_retrieve(query)
assert retrieved.shape == expected_shape

# 2. Action head has gradients
loss = model(x, labels=labels)
loss.backward()
action_grad_norm = model.action_head.proj.weight.grad.norm()
assert action_grad_norm > 0, "Action head has zero gradient!"
```

**Result:** 
- Memory vault: no crashes ✅
- Action head grad norm: ~0.03 (non-zero) ✅

---

## Testing After Fixes

All fixes verified by **6-test regression suite** (`tests/test_v101_regression.py`):

| Test | Before Fix | After Fix |
|---|---|---|
| `test_moe_registration_and_gradients` | FAIL (no expert grads) | ✅ PASS |
| `test_batch_boundary_isolation` | FAIL (diff > 0) | ✅ PASS (diff = 0.0) |
| `test_zero_agentic_memory_and_actions` | FAIL (crash + zero grad) | ✅ PASS |
| `test_sliced_ffn_computation` | PASS | ✅ PASS |
| `test_greed_model_pipeline` | PASS | ✅ PASS |
| `test_xorm_serialization` | PASS | ✅ PASS |

**Total: 6/6 PASS**

---

## Additional Fixes (Restoration from 0.2.4)

| Component | Status | Notes |
|---|---|---|
| GreedModel | ✅ Restored | From 0.2.4, verified forward+backward |
| `.xorm` runtime | ✅ Restored | From 0.2.4, 168 keys serialized |

---

## References

- `docs/agent/COMPLETED_WORK.md` — Full completion log
- `tests/test_v101_regression.py` — Regression test suite
- `AGENT_HANDOFF.md` — Project handoff document

---

**End of Bugs Fixed Documentation**
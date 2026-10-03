# XORZEN v1.0.1 — Forensic Bug Report

> This document records all critical bugs discovered and fixed during the v1.0.1 synthesis session.
> For implementation details, see `docs/agent/COMPLETED_WORK.md`.

---

## BUG-CRITICAL-1 (Pre-existing, fixed before this session)
**CoT consistency loss was hardcoded to zero**
- Commit: `49b3ac3`
- File: CoT loss computation
- Effect: Model never learned CoT consistency, but claimed it did

---

## BUG-CRITICAL-2 — MoE Expert Registration
**Severity:** Critical — model could not be trained at all with MoE

**Root cause:**  
`ShardedExpertFabric` stored experts as plain Python objects in an LRU dict:
```python
self._cache: Dict[int, ExpertFFN] = {}
```
PyTorch has no knowledge of these objects. Result:
- Zero parameters registered in `model.state_dict()`
- Zero optimizer updates for expert weights
- Zero gradients through expert parameters
- Checkpoint saved/loaded zero expert state

**Fix:**  
Register all experts permanently in `nn.ModuleList`:
```python
self.experts = nn.ModuleList([
    ExpertFFN(hidden_size, expert_ffn_dim, activation, dropout)
    for _ in range(num_experts)
])
```

**File:** `xorzen/model/zmoe.py`  
**Verified:** Gradients confirmed non-zero on `experts.0.gate_proj.weight`

---

## BUG-CRITICAL-3 — Cross-Batch Contamination in forward_with_depth
**Severity:** Critical — inference outputs contaminated across batch samples

**Root cause:**  
`HASSBlock.forward_with_depth()` aggregated ALL active tokens from ALL batch items:
```python
active_mask = depth_mask.bool()          # [B]
active_x = x[active_mask]               # [num_active, L, H]
active_x = active_x.unsqueeze(0)        # [1, num_active*L, H]  ← WRONG
# attention is now across all samples!
```

**Effect:**  
Sample B's output depends on sample A's content. Max difference = 1.2 when A changes while B is held constant. Should be 0.0.

**Fix:**  
Process each batch sample independently in a loop:
```python
for b in range(B):
    x_b = x[b:b+1]
    if not depth_mask[b]:
        results.append(x_b)
        continue
    processed_b = self.forward(x_b, routing_decision_for_b)
    ...
```

**File:** `xorzen/model/components/hass_block.py` (lines 1128–1192)  
**Verified:** Max diff = 0.0 when A changes and B held constant

---

## BUG-CRITICAL-4 — Agentic Model Dead Code (Three Sub-Bugs)
**Severity:** High — ZeroAgenticModel untrainable in 3 ways

### Sub-bug 4a: Missing `import math`
```python
# NameError: name 'math' is not defined
dim = math.sqrt(self.head_dim)
```
**Fix:** Add `import math` at top of file.

### Sub-bug 4b: Memory Vault Hard Crash + Zero Gradient
```python
# BEFORE — crashes when L > memory_slots, and vault gets no gradient
x = x + self.memory_vault[:, :L, :]

# AFTER — associative retrieval, any L, vault gets gradient
mem_scores = torch.bmm(x, vault.T) / math.sqrt(H)   # [B, L, slots]
mem_weights = torch.softmax(mem_scores, dim=-1)
x = x + torch.bmm(mem_weights, vault)               # [B, L, H]
```

### Sub-bug 4c: Action Head Zero Gradient
```python
# BEFORE — action_head output never used in loss
actions = self.action_head(x)
# actions discarded

# AFTER — action_head included in loss
if action_targets is not None:
    actions = self.action_head(x)
    loss = loss + action_loss_weight * F.mse_loss(actions, action_targets)
```

**File:** `xorzen/models/zero/agentic_model.py`  
**Verified:** memory_vault grad norm = 0.014, action_head grad norm = 0.010 (both were 0.0 before)

---

## KNOWN REMAINING ISSUES (Not Yet Fixed)

### ISSUE-1 — Expert RAM Usage at Scale
**Severity:** Medium (limitation, not a bug)

After BUG-CRITICAL-2 fix, all N experts permanently in RAM. For large configs:
- `277M` with 64 experts: ~512MB RAM just for experts
- `70B` with 128 experts: impractical on consumer hardware

**Status:** Documented. `sync_to_disk()` / `load_from_disk()` added for persistence, but active experts always in RAM during training/inference.

**Future fix:** Implement parameter-swapping expert execution (experts paged to/from VRAM on demand).

### ISSUE-2 — xorm_runtime.py ConfigFactory Compatibility
**Severity:** Low

`xorm_runtime.py` was copied from `xorzen_0.2.4` and imports `ConfigFactory`, `ModelSize` etc. If DevNet's enum values differ from 0.2.4, session loading may fail.

**Status:** Import path patched (`xorzen.model.greed` → `xorzen.models.greed`). Full end-to-end session test not performed.

### ISSUE-3 — No GPU Validation
**Severity:** Medium

All tests run on CPU (`PyTorch 2.14.1+cpu`). GPU behavior untested:
- CUDA kernel compatibility unknown
- Memory behavior at scale unknown
- Mixed-precision (fp16/bf16) untested

### ISSUE-4 — forward_with_depth Training Gap
**Severity:** Low (by design)

During training, depth gating uses STE blend (all blocks always executed). The sparse `forward_with_depth` only runs at `model.eval()`. This means:
- Training FLOPs are always dense
- Inference FLOPs can be sparse
- The "conditional compute" benefit is inference-only

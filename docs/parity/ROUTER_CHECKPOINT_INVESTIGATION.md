# Router & Checkpoint Investigation — Python vs C++

---

## Section 1 — Router

### 1.1 Feature encoder (`build_network`)

| Layer | Python (`routing.py:347-359`) | C++ (`routing.cpp:53-58`) | Match? |
|---|---|---|---|
| Layer 0 | `Linear(input_dim, enc1)` | `Linear(input_dim, enc1)` | ✓ |
| Layer 1 | `LayerNorm(enc1)` | `LayerNorm(enc1)` | ✓ |
| Layer 2 | `GELU()` | `GELU()` | ✓ |
| Layer 3 | `Dropout(router_dropout)` | `Dropout(router_dropout)` | ✓ |
| Layer 4 | `Linear(enc1, enc2)` | `Linear(enc1, enc2)` | ✓ |
| Layer 5 | `LayerNorm(enc2)` | `LayerNorm(enc2)` | ✓ |
| Layer 6 | `GELU()` | `GELU()` | ✓ |
| Layer 7 | `Dropout(router_dropout)` | `Dropout(router_dropout)` | ✓ |
| Layer 8 | `Linear(enc2, enc3)` | `Linear(enc2, enc3)` | ✓ |
| Layer 9 | `LayerNorm(enc3)` | `LayerNorm(enc3)` | ✓ |
| Layer 10 | `GELU()` | `GELU()` | ✓ |

Dim formulas identical: `enc1 = max(128, h*4)`, `enc2 = max(64, h*2)`, `enc3 = max(32, h)`, `head = max(32, h//2)`. **Result: exact match**.

### 1.2 Depth routing

**Python `_route_depth` (`routing.py:662-725`)**

```python
depth_bias   = linspace(0, 1, max_depth)[1,1,D]
adjusted     = logits + complexity * depth_bias * 2.0
scaled       = adjusted / max(temp, 1e-8)
# training (gumbel-sigmoid):
g            = -log(-log(rand_like(scaled) + 1e-10) + 1e-10)
noisy        = (scaled + g) / temp           # NOTE: temp applied twice!
probs        = sigmoid(noisy)
mask_hard    = (probs > 0.5).float()
mask         = mask_hard - probs.detach() + probs   # STE
# min_depth override:
forced[..., :min_depth] = 1
mask = where(forced.bool(), ones, mask)
```

**C++ `route_depth` (`routing.cpp:176-197`)** — math matches exactly (both apply temp twice, both use STE).

**Behavioral divergences**:
- Python v0.5 (`routing.py:685-692`) adds **eval-time Gumbel noise** via `_eval_gumbel_noise(..., axis_id=0)` in the deterministic branch before sigmoid — C++ has NO eval noise.
- **Cost-aware modulation is MISSING in C++**: Python (lines `546-588`) modifies `depth_logits` **before** calling `_route_depth`:
  - `depth_layer_bias = linspace(0, -sparsity_pressure*3, max_depth)` (`routing.py:580-583`)
  - `depth_shift = -sparsity_pressure * 4.0 * (1 - complexity.squeeze(-1))` per-token (`routing.py:560`)
  - `depth_logits = depth_logits + depth_layer_bias + depth_shift.unsqueeze(-1)` (`routing.py:584`)
  - C++ has none of this. With `cost_aware_routing=True` (Python default at `routing.py:546`), depth decisions will diverge sharply under any `compute_budget < 1.0`.

**Parity harness fixture 09_router**: depth_probs MISMATCH (`max_abs=0.047`), depth_mask MISMATCH (1.0). Confirmed.

### 1.3 Width routing

Python `_route_width` (`routing.py:727-777`) and C++ `route_width` (`routing.cpp:199-209`) — **math: exact match** for the core formula. Both normalize by `hidden_size`, both use the registered buffer `width_values`.

**Behavioral divergences**:
- Python v0.5 adds eval Gumbel noise (`routing.py:751-755`) in deterministic branch — C++ does not.
- **Cost-aware width_bias_axis (`routing.py:564-568`) MISSING in C++**: `width_bias_axis = linspace(sparsity_pressure*2, -sparsity_pressure*2, num_widths)` is added to `width_logits` before `_route_width`. Under `compute_budget<1`, Python biases toward smaller widths; C++ has no such bias.

### 1.4 Pathway routing

Python `_route_path` (`routing.py:779-830`) and C++ `route_path` (`routing.cpp:211-219`) — **math: exact match**. Same `prior_weight = max(0.05, 0.5 * 0.995^step)` formula. Same `explore_tau = max(temp*2, 1.0)`. Same probability-space blending.

**Behavioral divergences**:
- Python v0.5 eval noise on path (`routing.py:805-809`) — C++ uses bare softmax in eval.
- **Cost-aware path_bias_axis (`routing.py:572-576`) MISSING in C++**: `path_bias_axis = linspace(sparsity_pressure*1.5, -sparsity_pressure*0.5, num_paths)` is added to `path_logits` before `_route_path`. Python biases toward SSM (index 2) under low budget; C++ does not.

**Parity harness fixture 09_router**: path_probs MISMATCH (`max_abs=0.121`). Confirmed.

### 1.5 Expert routing

Python `_route_experts` (`routing.py:832-902`) and C++ `route_experts` (`routing.cpp:221-244`) — **math: exact match.** Capacity default `1.25 * N / num_experts` identical. Top-k via `torch.topk` in both.

**Behavioral divergence**:
- Python v0.5 adds eval Gumbel noise to `logits_flat` in deterministic branch (`routing.py:858-864`) — C++ does not.

### 1.6 Capacity constraint algorithm

Python `_apply_capacity_constraint` (`routing.py:904-954`) vs C++ `apply_capacity_constraint` (`routing.cpp:246-282`) — **same algorithm**:
1. Flatten weights to `[N*top_k]`, sort descending by weight via `argsort(descending=True)`.
2. For each expert `e`, mask `(e_idx == e)`, take the first `capacity` positions in sorted order, mark as keep.
3. Apply keep-mask to weights, renormalize per-token by `weights / (sum + 1e-12)`.

Both run inside `torch.no_grad()` for the mask construction. **Match**.

### 1.7 Complexity & uncertainty estimators

**Python** (`routing.py:393-409`): both are `Linear(enc3, head) → LayerNorm(head) → GELU → Linear(head, 1) → Sigmoid`.
**C++** (`routing.cpp:76-81`): identical 5-element Sequential. **Match.**

### 1.8 Compute-budget modulation (Python ONLY)

The entire cost-aware block (`routing.py:546-588`) is **absent in C++**. Summary:

| Axis | Python code | Effect |
|---|---|---|
| `depth_shift` | `-sparsity_pressure * 4.0 * (1 - complexity.squeeze(-1))` per-token (`routing.py:560`) | Easier tokens (low complexity) get bigger negative shift → fewer layers |
| `width_bias_axis` | `linspace(+2*SP, -2*SP, num_widths)` (`routing.py:564-567`) added to `width_logits` | Biases toward small widths under low budget |
| `path_bias_axis` | `linspace(+1.5*SP, -0.5*SP, num_paths)` (`routing.py:572-575`) added to `path_logits` | Biases toward SSM (last index) under low budget |
| `depth_layer_bias` | `linspace(0, -3*SP, max_depth)` (`routing.py:580-583`) added to `depth_logits` | Prunes deeper layers more aggressively under low budget |
| Expert | (comment only at `routing.py:585-588`) | No explicit bias; Switch-formula load balance handles it |

Where `SP = sparsity_pressure = 1 - clip(compute_budget, 0.05, 1.0)`.

**C++ has ZERO of these.** A model trained with `compute_budget<1.0` in Python cannot be reproduced by the C++ inference path.

### 1.9 Temperature annealing

| | Code |
|---|---|
| Python (`routing.py:529`) | `current_temp = max(0.1, self.temperature * (0.99 ** (self.training_step / 1000)))` |
| C++ (`routing.cpp:141`) | `temp = max(0.1, temperature * pow(0.99, (double)training_step / 1000.0))` |

**Match.** Both gate the annealing on `is_training() && temperature_annealing`.

### 1.10 Straight-through behavior

| | Code |
|---|---|
| Python (`routing.py:705`) | `mask = mask_hard - probs.detach() + probs` (training, deterministic=False branch) |
| C++ (`routing.cpp:190`) | `mask = (!deterministic && is_training()) ? hard - probs.detach() + probs : hard` |

**Match** in math. C++ explicitly handles the deterministic branch (returns hard mask); Python falls through to the `else` clause which gives the same hard mask but also returns soft probs.

### 1.11 Top-k selection

Python: `torch.topk(weights, self.top_k, dim=-1)` (`routing.py:881, 894`).
C++: `torch::topk(probs_flat, top_k, -1)` (`routing.cpp:235`). **Match.**

### 1.12 Auxiliary losses inside the router

| Loss key | Python (`routing.py:644-658`) | C++ (`routing.cpp:167-171`) | Match? |
|---|---|---|---|
| `load_balance_loss` | `0.0001 * load_balance_loss(expert_probs, expert_indices, num_experts)` → **Switch formula** `N·Σ fₑ·Pₑ` via `load_balance_loss_switch` (`load_balance.py:151-177`) | `0.0001 * load_balance_loss(expert_probs, expert_indices, num_experts)` → **L2 against uniform** `(P̄ₑ - 1/N)².Σ` (`routing.cpp:11-17`) | ❌ **DIFFERENT FORMULA** |
| `router_z_loss` | `0.0001 * (logsumexp(expert_logits,-1)²).mean()` (`routing.py:38-44`) | `0.0001 * logsumexp(expert_logits,-1).pow(2).mean()` (`routing.cpp:19-21`) | ✓ |
| `path_div_loss` | `0.2 * path_diversity_loss(path_probs)` → `0.2 * (-mean_entropy)` (`routing.py:47-58`) | `0.02 * path_diversity_loss(path_probs)` → `0.02 * (avg - 1/N)².Σ` L2 against uniform (`routing.cpp:23-27`) | ❌ **DIFFERENT FORMULA & DIFFERENT WEIGHT (0.2 vs 0.02)** |
| `width_div_loss` | `0.1 * width_diversity_loss(width_probs)` → `0.1 * (-mean_entropy)` (`routing.py:61-76, 656-658`) | **(NOT PRESENT)** | ❌ **MISSING in C++** |

The C++ `load_balance_loss` function signature declares the second arg as unnamed (`const torch::Tensor&`), so `expert_indices` is **completely ignored** — only `expert_probs.mean(0).mean(0)` is used. The Switch formula's `f_e` (token dispatch fractions) is never computed in C++.

### 1.13 `RoutingRegularizer` (top-level)

**Python `RoutingRegularizer.forward` (`routing.py:1691-1694`)**:
```python
uncertainty_loss = decision.uncertainty.mean() * self.uncertainty_weight  # 0.01
return uncertainty_loss
```
Just returns the uncertainty penalty. The path_div_loss / load_balance / z_loss / width_div are summed **separately** in `zeroModel.forward` (`model.py:621-626`) under a strict allow-list `_AUX_LOSS_KEYS`.

**C++ `RoutingRegularizerImpl.forward` (`routing.cpp:284-293`)**:
```cpp
auto loss = decision.uncertainty.mean() * 0.0001;          // 0.0001, not 0.01
loss = loss + path_diversity_loss(decision.path_probs) * 0.02;   // re-adds path_div_loss
for (auto& kv : decision.auxiliary) {
    if (kv.first.find("loss") != string::npos && kv.second.defined())
        loss = loss + kv.second;                           // re-adds ALL aux losses
}
return loss;
```

**Divergences**:
1. Uncertainty weight: Python `0.01` (default from `routing_loss_weight`), C++ hard-coded `0.0001`.
2. C++ re-adds `path_div_loss` AND iterates over every `auxiliary` key whose name contains `"loss"` and adds them — **double-counting**: `load_balance_loss`, `router_z_loss`, `path_div_loss` are all already inserted into `auxiliary` at `routing.cpp:168-170`, then re-summed by the regularizer.
3. C++ adds an extra `path_div_loss` term that Python doesn't.

### 1.14 Expert usage tracking

**Python** (`routing.py:324-325, 956-986`): keeps `self.expert_usage` and `self.expert_load` as **plain attributes** (not registered buffers). Updated each training step via `_update_expert_usage` using `scatter_add_` on CPU buffers. Stats queryable via `get_expert_statistics`.

**C++**: has NO equivalent usage tracking. The only MoE-side stats come from `ShardedExpertFabric` (cache hits, etc.), NOT a per-expert running tally. **C++ side is missing this feature.**

### 1.15 Init weights

| Module | Python | C++ |
|---|---|---|
| feature_encoder Linear | `xavier_uniform_(gain=0.5)` (`routing.py:424`) | `xavier_uniform_(gain=0.5)` (`routing.cpp:93`) | ✓ |
| depth/width/path/expert_router Linear | `xavier_uniform_(gain=0.1)` (`routing.py:432`) | `xavier_uniform_(gain=0.5)` (`routing.cpp:93` — uses same 0.5 loop over ALL modules) | ❌ **DIFFERENT** — Python uses 0.1 for routers, C++ uses 0.5 for everything |
| complexity/uncertainty Linear | `xavier_uniform_(gain=0.5)` (`routing.py:439`) | `xavier_uniform_(gain=0.5)` (`routing.cpp:93`) | ✓ |

The C++ `init_weights` (`routing.cpp:90-97`) iterates `modules(false)` and applies `xavier_uniform_(gain=0.5)` uniformly to all Linear layers, missing Python's per-group `0.1` for the four routers.

### 1.16 Router module registration

| Module | Python (`routing.py`) | C++ (`routing.cpp`) | Match? |
|---|---|---|---|
| `feature_encoder` | ✓ | ✓ | ✓ |
| `depth_router` | ✓ | ✓ | ✓ |
| `width_router` | ✓ | ✓ | ✓ |
| `path_router` | ✓ | ✓ | ✓ |
| `expert_router` | ✓ | ✓ | ✓ |
| `complexity_estimator` | ✓ | ✓ | ✓ |
| `uncertainty_estimator` | ✓ | ✓ | ✓ |
| `character_router` | ✗ (NOT IN PYTHON) | ✓ (line 72) | ❌ **EXTRA IN C++** |
| `width_values` buffer | ✓ | ✓ | ✓ |

**C++ has an extra `character_router` Sequential** (`routing.cpp:72-75`: `Linear(enc3, head) → LN → GELU → Linear(head, max_characters) → Sigmoid`) that does **not exist** in Python. This registers `router.character_router.0.weight`, `router.character_router.0.bias`, `router.character_router.1.{weight,bias}`, `router.character_router.3.{weight,bias}` — all extra parameters that will appear in C++ checkpoints and be flagged as unexpected by Python.

---

## Section 2 — Checkpoint

### 2.A Python `model.state_dict()` top-level keys

Derived from `model.py:108-188` (instantiations) + submodule register calls:

```
token_embedding.weight
position_embedding.weight
embedding_dropout                       (no params; stateless)
router.*                                (see §1.16)
routing_regularizer.*                   (no params — just stores config)
blocks.{0..N-1}.*                       (HASSBlock; see below)
moe.*                                   (ShardedExpertFabric)
merger.merger_impl.*                    (GatedMerger or LinearMerger)
final_norm.{weight,bias}
lm_head.weight                          (bias=False; tied to token_embedding if config.tie_word_embeddings)
cot.*                                   (InternalLatentCoT, frozen pre-training)
```

Per HASSBlock (`hass_block.py` — `blocks.{i}.`):
```
local.{q_proj,k_proj,v_proj,out_proj,ln_q,ln_k}.{weight,bias}
low_rank.{to_low_rank,from_low_rank}.{weight,bias}
low_rank.ln_input.{weight,bias}
ssm.A_log                               (Parameter, not buffer)
ssm.{dt_proj,B_proj,C_proj,D_proj,gate_proj}.{weight,bias}
ssm.conv.{weight,bias}                  (only if use_conv)
ssm.ln_input.{weight,bias}
ssm.ln_state.{weight,bias}
ffn.*                                   (SlicedFFN or AdaptiveFFN)
ln1.{weight,bias}  ln2.{weight,bias}
dropout                                 (stateless)
```

Per expert (`zmoe.py:97-101` — `moe.experts.{i}.`):
```
gate_proj.{weight,bias?}   up_proj.{weight,bias?}   down_proj.{weight,bias?}   dropout
```

Plus `moe.dummy_expert.{gate_proj,up_proj,down_proj,...}` (alias to `experts[0]`).

Merger (`merger.py:163-174` — for the default GatedMerger):
```
merger.merger_impl.gate_controller.0.{weight,bias}     # Linear
merger.merger_impl.gate_controller.2.{weight,bias}     # Linear (index 1 is SiLU, no params)
merger.merger_impl.cot_proj.{weight,bias}
merger.merger_impl.output_norm.{weight,bias}
```

CoT (`cot_vector.py:80-174`):
```
cot.gru_cell.{weight_ih,weight_hh,bias_ih,bias_hh}                       # if updater == gru
cot.transformer_encoder.*                                                # if updater == transformer
cot.component_projections.{intention,decomposition,confidence,contradiction,direction,summary}.*
cot.updater.*
cot.component_norm.{weight,bias}
cot.cot_norm.{weight,bias}
cot.output_proj.{weight,bias}
cot.injection_gate.{weight,bias}
cot.update_gate.0.{weight,bias}    # Sequential; index 1 = SiLU
```

**NOTE**: `zeroModel` does NOT instantiate `CoTAuxiliaryLoss` (only `InternalLatentCoT` is imported at `model.py:54` and instantiated at `model.py:129`). So Python `state_dict()` has **no** `cot_loss_head.*` keys.

### 2.B C++ top-level keys

From `xorzen_model.cpp:12-31`:

```
token_embedding.weight
position_embedding.weight
embedding_dropout
router.*                                (see §1.16 — includes character_router!)
routing_regularizer.*                   (no params)
blocks.{0..N-1}.*                       (HASSBlock — see below)
moe.*                                   (ShardedExpertFabric)
merger.merger_impl.*                    (GatedMerger)
final_norm.{weight,bias}
lm_head.weight
cot.*                                   (InternalLatentCoT)
cot_loss_head.*                         # << EXTRA in C++
```

C++ HASSBlock (`hass_block.cpp:283-293` — `blocks.{i}.`):
```
local.{q_proj,k_proj,v_proj,out_proj,ln_q,ln_k}.{weight,bias}
low_rank.{to_low_rank,from_low_rank}.{weight,bias}
low_rank.ln_input.{weight,bias}
low_rank.ln_low_rank.{weight,bias}      # << EXTRA in C++
low_rank.context_weights                # << EXTRA in C++ (Parameter)
ssm.A_log                               # ✓ present
ssm.{dt_proj,B_proj,C_proj,D_proj,gate_proj}.{weight,bias}
ssm.conv.{weight,bias}
ssm.ln_input.{weight,bias}
ssm.ln_state.{weight,bias}
pathway_gate.0.{weight,bias}            # << EXTRA in C++ (Python removed in v0.5)
ffn.{fc1,fc2}.{weight,bias}             # AdaptiveFFN, not SlicedFFN
ffn.ln_input.{weight,bias}
ffn.ln_hidden.{weight,bias}
ln1.{weight,bias}  ln2.{weight,bias}
```

### 2.C Missing parameters in C++ (relative to Python)

| Python key | Why missing in C++ |
|---|---|
| `blocks.{i}.ffn.*` (SlicedFFN fields) | C++ uses `AdaptiveFFN` (`hass_block.cpp:252-257`) — `fc1`, `fc2`, `ln_input`, `ln_hidden`. SlicedFFN's nested-slice structure has no C++ equivalent. **Names match** (fc1, fc2, ln_input, ln_hidden) but the **compute path** is different. |
| `cot.update_gate.0` (Python) | C++ may register `update_gate` differently — needs verification but C++ CoT code (`cot_vector.cpp:97-122`) shows similar structure. |
| `router.width_div_loss` (auxiliary scalar) | N/A — auxiliary losses aren't parameters; not relevant for state_dict. |

### 2.D Extra parameters in C++ (relative to Python)

| C++ key | Source | Python status |
|---|---|---|
| `router.character_router.0.{weight,bias}` | `routing.cpp:72-75` | **Not in Python at all** |
| `router.character_router.1.{weight,bias}` | same | Same |
| `router.character_router.3.{weight,bias}` | same | Same |
| `blocks.{i}.pathway_gate.0.{weight,bias}` | `hass_block.cpp:287-289` | **Python removed in v0.5** (`hass_block.py:897-904`) — was dead code |
| `blocks.{i}.low_rank.context_weights` | `hass_block.cpp:76` (register_parameter) | **Not in Python** — Python low_rank uses pure self-attention (no learned context vector), see `hass_block.py:309-368` |
| `blocks.{i}.low_rank.ln_low_rank.{weight,bias}` | `hass_block.cpp:74` | **Not in Python** — Python low_rank has only `ln_input` |
| `cot_loss_head.token_complexity_head.{0,2}.{weight,bias}` | `cot_vector.cpp:331-335` | **Not in Python `zeroModel`** — `CoTAuxiliaryLoss` class exists in `cot_vector.py:327` but is never instantiated by `zeroModel` |
| `cot_loss_head.next_token_head.{0,1,2}.{weight,bias}` | `cot_vector.cpp:336-340` | Same |

### 2.E Transpositions / concatenations

All Linear weights in both implementations follow PyTorch's standard `[out_features, in_features]` convention. No manual `.t()` is applied during save/load.

Conv1d weights: both `[out_channels, in_channels/groups, kernel_size]` (depthwise, `groups=hidden_dim`). Match.

Embedding weights: both `[num_embeddings, embedding_dim]`. Match.

LayerNorm: both `{weight, bias}` of shape `[normalized_shape]`. Match.

A_log (SSM): `[state_dim]` 1-D Parameter in both. Match.

**No transposition/concat issues detected.**

### 2.F Serialization format

| Aspect | Python (`checkpoint.py:255-288`) | C++ (`checkpoint.cpp:99-123`) |
|---|---|---|
| Container | `dict` pickled via `torch.save(checkpoint, path)` | `torch::serialize::OutputArchive` written via `archive.save_to(path)` |
| Model state location | Nested under `checkpoint['model_state_dict']` (`checkpoint.py:260`) | **Flat at archive root** via `model.save(archive)` (`checkpoint.cpp:117`) — no `model_state_dict` wrapper |
| Extra scalar keys | `epoch`, `global_step`, `metrics`, `checkpoint_version`, `xorzen_version`, `created_at`, optional `optimizer_state_dict`, `scheduler_state_dict`, `model_config` (`checkpoint.py:257-285`) | `epoch`, `global_step`, `train_loss`, `eval_loss` only (`checkpoint.cpp:119-122`). Optimizer state saved **separately** as `.opt.pt` (`checkpoint.cpp:126-132`). |
| File extension | `.pt` | `.pt` |
| Filename pattern | `checkpoint_epoch_{epoch}_step_{step}_{YYYYMMDD_HHMMSS}.pt` (`checkpoint.py:248-249`) | `checkpoint_epoch_{epoch}_step_{step}.pt` (no timestamp, `checkpoint.cpp:109-112`) |
| Sidecar metadata | `.meta.json` (via `CheckpointMetadata.save`, `checkpoint.py:291-307`) | `.meta.json` (via `write_meta`, `checkpoint.cpp:321-348`) — different schema |
| Best model | `best_model.pt` (`checkpoint.py:312-318`) | `best_model.pt` (`checkpoint.cpp:213-220`, `copy_as_best` at 301-315) — also copies `best_model.opt.pt` |
| Version constants | `CHECKPOINT_VERSION="1.0"`, `XORZENX_VERSION="0.2.2"` (`checkpoint.py:163-164`) | `CHECKPOINT_VERSION="1.0"`, `XORZEN_VERSION="0.2.5"` (`checkpoint.h:49-50`) — version strings differ |

**Per-parameter .bin loading** (`routing.cpp:99-118`): the C++ `AdaptiveRouter::load_weights(weights_dir)` method is a **separate** load path that reads raw bytes from `<name_with_dots_replaced_by_underscores>.bin` files (e.g. `router_feature_encoder_0_weight.bin`). This is unrelated to the `.pt` checkpoint path and is incompatible with `torch.save` output — it expects a directory of flat binary files, one per parameter. **No Python equivalent exists.**

### 2.G Practical compatibility

**Can `model.py` state_dict be loaded into `xorzen_model.cpp` (via `model->load(archive)`)?** — **NO**, not directly. Reasons:

1. **Container mismatch**: Python wraps the state_dict under the `'model_state_dict'` key in the pickle. C++ `CheckpointManager::load` calls `archive.load_from(path); model->load(archive);` (`checkpoint.cpp:169-174`) expecting parameters at the archive root. The C++ loader would not find any parameters (they're nested under `model_state_dict`).

2. **Unexpected keys (C++ side will refuse or warn)**:
   - Python has `blocks.{i}.ffn.<sliced_ffn_keys>` — C++ expects `blocks.{i}.ffn.{fc1,fc2,ln_input,ln_hidden}` — **shape & name mismatch**.
   - Python does NOT have `router.character_router.*`, `blocks.{i}.pathway_gate.*`, `blocks.{i}.low_rank.context_weights`, `blocks.{i}.low_rank.ln_low_rank.*`, `cot_loss_head.*` — C++ will report these as **missing keys** when loading Python's state_dict.

3. **Missing keys (Python's state_dict lacks C++ extras)**:
   - All the extras listed in §2.D would appear as "expected but not found" when C++ loads Python's state.
   - Python's SlicedFFN params have no C++ counterpart.

4. **Tied weights**: When `config.tie_word_embeddings=True`, Python aliases `lm_head.weight = token_embedding.weight` (`model.py:196`). C++ does the same aliasing (`xorzen_model.cpp:34-36`). However, when serializing, PyTorch deduplicates tied tensors; libtorch's `model.save(archive)` may or may not deduplicate depending on version — possible shape/duplication mismatch when round-tripping.

**Can a C++ checkpoint be loaded back into Python?** — **NO**, for the inverse reasons plus:
- C++ saves parameters flat (no `model_state_dict` wrapper), so Python's `model.load_state_dict(checkpoint['model_state_dict'], strict=strict)` would fail because Python expects the dict to be under that key.
- Workaround would be: load C++ `.pt` via `torch.load`, manually construct `{'model_state_dict': <flat_dict>}`, then call `model.load_state_dict(..., strict=False)` to allow the extra C++ keys (`character_router.*`, `pathway_gate.*`, `context_weights`, `ln_low_rank.*`, `cot_loss_head.*`) to be silently dropped.
- Even with `strict=False`, the `SlicedFFN` ↔ `AdaptiveFFN` mismatch remains — the FFN sub-module keys won't line up, so all FFN weights would be silently dropped, degrading the loaded model.

**Actual delta (strict load, Python → C++)**:

| Category | Specific keys |
|---|---|
| Missing in Python's state_dict (C++ will complain) | `router.character_router.{0,1,3}.{weight,bias}`, `blocks.{i}.pathway_gate.0.{weight,bias}` ∀i, `blocks.{i}.low_rank.context_weights` ∀i, `blocks.{i}.low_rank.ln_low_rank.{weight,bias}` ∀i, `cot_loss_head.token_complexity_head.{0,2}.{weight,bias}`, `cot_loss_head.next_token_head.{0,1,2}.{weight,bias}` |
| Unexpected in Python's state_dict (C++ will complain) | `blocks.{i}.ffn.*` (SlicedFFN-specific; **names match** but compute path differs — see SlicedFFN section) |
| Shape mismatches | None expected for shared keys (both follow standard libtorch/PyTorch layouts) |
| Top-level structural | Python nests under `model_state_dict`; C++ expects flat — breaks load before key-level comparison even happens |

---

## Summary of Critical Divergences

**Router (top-priority bugs)**:
1. `load_balance_loss` formula: Python uses Switch `N·ΣfₑPₑ`, C++ uses L2 against uniform with indices IGNORED (`routing.cpp:11-17`).
2. `path_diversity_loss` formula: Python uses `-mean_entropy`, C++ uses L2 against uniform (`routing.cpp:23-27`). Weight also differs: 0.2 vs 0.02.
3. `width_div_loss` MISSING in C++ (`routing.cpp:167-171` vs `routing.py:644-658`).
4. `RoutingRegularizer.forward` double-counts aux losses in C++ (`routing.cpp:284-293`) — Python avoids this.
5. Cost-aware modulation (`routing.py:546-588`) entirely absent in C++.
6. v0.5 eval Gumbel noise (`routing.py:442-475`) absent in C++.
7. C++ has extra `character_router` module not present in Python (`routing.cpp:72-75`).
8. Init weights: C++ uses uniform `gain=0.5` for all Linears; Python uses `gain=0.1` for the four routers (`routing.py:432`).

**Checkpoint (top-priority bugs)**:
1. Container format incompatible: Python nests under `model_state_dict`, C++ writes flat at root.
2. C++ registers 5+ parameter groups Python doesn't have (`character_router`, `pathway_gate`, `low_rank.context_weights`, `low_rank.ln_low_rank`, `cot_loss_head`).
3. Python uses `SlicedFFN` in HASS blocks; C++ uses `AdaptiveFFN` — different parameter names/shapes.
4. C++ has a separate per-parameter `.bin` load path (`routing.cpp:99-118`) that has no Python equivalent and isn't interoperable with `torch.save`.
5. Version strings differ: Python `XORZENX_VERSION="0.2.2"`, C++ `XORZEN_VERSION="0.2.5"` (`checkpoint.py:164` vs `checkpoint.h:50`).

**Net assessment**: As written, Python-trained checkpoints **cannot** be loaded into the C++ build (and vice versa) without a converter script that (a) re-wraps the state_dict container, (b) drops the C++-only keys, (c) remaps SlicedFFN → AdaptiveFFN (lossy), and (d) handles tied-weight deduplication. Even then, the router's auxiliary-loss math and cost-aware modulation diverge enough that **numerically the C++ model will produce different routing decisions** from the Python model on the same input.

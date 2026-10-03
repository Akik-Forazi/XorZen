# CPP_PORT_DECISION — Three-Path Evidence-Only Comparison

**Purpose**: Compare three C++ port strategies against the evidence gathered in
`CPP_PARITY_MATRIX.md`, `SSM_INVESTIGATION.md`, `HASS_SlicedFFN_Loss_INVESTIGATION.md`,
and `ROUTER_CHECKPOINT_INVESTIGATION.md`. **No path is recommended here — only evidence.**

---

## Evidence baseline (recap)

| | Count | Source |
|---|---|---|
| Components total | 31 | `CPP_PARITY_MATRIX.md` |
| MATCHED (runtime-verified PASS) | 9 | parity harness fixtures |
| PARTIAL (runs but diverges) | 9 | static analysis + harness |
| MISMATCH (architecturally different) | 7 | static analysis + harness |
| MISSING in C++ | 5 | static analysis |
| EXTRA in C++ (Python doesn't have) | 1 | static analysis |

Runtime verification status:
- 14 components have deterministic Python fixtures in `tests/cpp_parity/fixtures/`
- 11 PASS, 2 informative FAIL (router, ssm_pathway_full), 1 ARCH_MISMATCH (merger)
- 17 components are documented via static analysis only

Python specification is **frozen**: golden reference (40 tensors) is byte-reproducible across runs (SHA256 `6906aee00addcd56a72c441c5eeb63d9466bec4dccb5911e294ed421733f13f8`).

---

## PATH A — Incremental port (adapt the existing C++ component by component)

### Strategy
Keep the existing `xorzen.cpp/src/model/*.cpp` files. Fix each component in place
to match the Python behavior. Do not restructure.

### Reusable code (no changes needed)
- `xorzen_model.cpp:12-16` — token + position embeddings
- `hass_block.cpp:11-66` — LocalAttentionPathway (math matches Python SDPA path)
- `hass_block.cpp:18-21` — separate q/k/v projections
- `hass_block.cpp:250-278` — AdaptiveFFN (works at max_width; equivalent to SlicedFFN)
- `hass_block.cpp:322-343` — `GatedMerger` 2-gate version — **NO, REPLACE** with `merger.cpp:54-133` 3-gate version
- `routing.cpp:120-174` — AdaptiveRouter forward skeleton
- `routing.cpp:221-244` — `route_experts` (matches Python)
- `routing.cpp:246-282` — `apply_capacity_constraint` (matches Python)
- `xorzen_model.cpp:127-135` — CE loss (matches Python)
- `xorzen_model.cpp:34-36` — tied embeddings

### Code requiring replacement/fix (in place)
| File:line | Component | Fix |
|---|---|---|
| `hass_block.cpp:105-106` | Conv1d padding | Change `padding(kernel_size/2)` → `padding(kernel_size-1)` + truncate to T |
| `hass_block.cpp:151` | SSM C application | Move C multiplication out of scan loop, store raw `h_t` |
| `hass_block.cpp:239-244` | SSM B_bar | Add `discretize_zoh` step before passing to scan |
| `hass_block.cpp:246` | SSM LN order | Reorder: `D_proj(C * ln_state(states)) * gate` |
| `hass_block.cpp:81-93` | LowRankGlobalPathway | Rewrite to use causal tril mask + chunked fallback + remove `context_weights` + remove `ln_low_rank` |
| `hass_block.cpp:287-289` | `pathway_gate` Sequential | **Delete** (Python removed in v0.5) |
| `hass_block.cpp:296-320` | Sparse pathway dispatch | **Add** top-k sparse dispatch (Python `sparse_pathway_dispatch`) |
| `routing.cpp:11-17` | `load_balance_loss` | Replace L2 formula with Switch formula `N·Σ f_e·p_e` |
| `routing.cpp:23-27` | `path_diversity_loss` | Replace L2 with `-mean_entropy` (Python `routing.py:47-58`); weight 0.2 not 0.02 |
| `routing.cpp:72-75` | `character_router` | **Delete** (not in Python) |
| `routing.cpp:90-97` | `init_weights` | Use `gain=0.1` for the four routers, `gain=0.5` for the rest |
| `routing.cpp:167-171` | Auxiliary loss dict | Add `width_div_loss` |
| `routing.cpp:176-197` | `route_depth` | Add cost-aware modulation (`depth_shift`, `depth_layer_bias`); add eval Gumbel noise |
| `routing.cpp:199-209` | `route_width` | Add cost-aware `width_bias_axis`; add eval Gumbel noise |
| `routing.cpp:211-219` | `route_path` | Add cost-aware `path_bias_axis`; add eval Gumbel noise |
| `routing.cpp:284-293` | `RoutingRegularizerImpl::forward` | Remove double-counted `path_div_loss`; use config-driven weights; use allowlist filter |
| `xorzen_model.cpp:24` | Merger instantiation | Switch from `hass.h` `GatedMerger` to `merger.h` `GatedMerger` (3-gate, already in `merger.cpp`) |
| `xorzen_model.cpp:125-148` | Loss computation | Add `cot_consistency_loss`, add `unify_load_balance` toggle, fix `width_div_loss`, fix double-counting |
| `xorzen_model.cpp:102-113` | Depth masking | Add per-token residual gating inside block during training (not just inference skip) |
| `xorzen_model.cpp` (new) | Generation strategies | Add temperature, top-k, top-p, repetition penalty, beam search (port from `model.py:871-1283`) |
| `checkpoint.cpp:99-123` | Checkpoint format | Wrap state_dict under `model_state_dict` key to match Python |
| `xorzen_model.cpp` (new) | .xorm v2 encryption | Port `xorm_crypto.py` (AES-256-GCM + HMAC) |

### Estimated scope
- **Lines changed**: ~1,500 (modifications) + ~800 (new code for sparse dispatch, cost-aware, generation, encryption)
- **Components touched**: 18 of 31
- **Files modified**: `hass_block.cpp`, `routing.cpp`, `xorzen_model.cpp`, `checkpoint.cpp` (4 files)
- **Files added**: probably 2 new (generation.cpp, xorm_crypto.cpp)

### Checkpoint implications
- Existing C++ checkpoints (if any) will be incompatible after the merger swap (`gate.*` → `gate_controller.*`) and the `character_router`/`pathway_gate` removal.
- After Path A, Python ↔ C++ checkpoint compatibility becomes POSSIBLE but still requires a converter script (container format `model_state_dict` wrapping).
- A converter script can be written in ~150 lines of Python.

### Testing burden
- 14 component-level parity fixtures already exist — re-run them after each fix.
- Add ~10 new fixtures for components not yet covered (loss, generation, checkpoint round-trip).
- Full-model parity test (golden reference, 40 tensors) needs a C++ implementation that can load Python checkpoints.

### Performance opportunities
- Once parity is achieved, can apply: SIMD for serial SSM scan, OpenMP parallelism for batch dim, fused QKV (only if not breaking checkpoint compat).
- These are POST-parity optimizations — explicitly out of scope per the user's directive.

### Technical risks
1. **Cascading bugs**: a fix in one component (e.g. SSM B_bar ZOH) changes the input distribution to downstream components (LN, D_proj), which may expose previously-hidden numerical issues.
2. **Hidden coupling**: the `pathway_gate` Sequential in C++ is wired into `HASSBlockImpl::forward` at line 305-312. Removing it requires touching the dispatch logic carefully.
3. **Merger swap risk**: switching from `hass.h` `GatedMerger` to `merger.h` `GatedMerger` may break the include graph if other code depends on the `hass.h` declaration.
4. **Loss rewrite risk**: the C++ loss has 7 distinct divergences — fixing all of them in one pass risks introducing new bugs. Recommend fixing one at a time with a fixture per fix.
5. **Custom autograd**: `SSMScanFunction` (`hass_block.cpp:120-224`) has hand-derived backward. The 4 SSM fixes require updating both forward AND backward.

### Dependencies
- LibTorch (already linked)
- No new external dependencies

### Likely failure modes
1. The C++ build system (CMake + gcc + LibTorch) is fragile; small changes can break the build.
2. The `pathway_gate` removal could break other C++ code that references `pathway_gate` (search needed).
3. The merger swap may require updating `xorzen_model.cpp` include directives.
4. The loss rewrite may produce NaN gradients if the Switch LB formula is implemented incorrectly.

---

## PATH B — Partial rewrite (keep infrastructure, replace model components)

### Strategy
Keep the C++ infrastructure (tokenizer, serialization, build system, threading, SIMD utilities,
kernel infrastructure, ggml bridge). Replace the model components that are architecturally
incompatible with Python-faithful reimplementations.

### Reusable code (infrastructure)
- `xorzen.cpp/CMakeLists.txt` — build system
- `xorzen.cpp/include/xorzen/optimized/*.h` — SIMD utilities, thread pool, flash_attn, quantize
- `xorzen.cpp/extern/ggml/*` — GGML bridge for quantization
- `xorzen.cpp/include/xorzen/data_pipeline.h` — data loading
- `xorzen.cpp/include/xorzen/tokenizer.h` — tokenizer infrastructure (BebpeTokenizer needs review but skeleton is reusable)
- `xorzen.cpp/include/xorzen/checkpoint.h` — checkpoint infrastructure (serialization format needs fix)
- `xorzen.cpp/include/xorzen/curriculum.h`, `trainer.h`, `lora.h` — training infrastructure
- `xorzen.cpp/src/optimized/*.cpp` — kernel infrastructure
- `xorzen.cpp/src/training/*.cpp` — training loops

### Code requiring replacement
| Component | Reason | Replacement source |
|---|---|---|
| `hass_block.cpp` (entire file) | 7+ divergences (sparse dispatch, low-rank, SSM 4 fixes, pathway_gate removal) | Reimplement from `hass_block.py` |
| `routing.cpp` (entire file) | 8+ divergences (cost-aware, eval noise, LB formula, init gain, character_router, width_div) | Reimplement from `routing.py` |
| `xorzen_model.cpp` (model construction + forward + loss) | Loss 7 divergences, merger swap, depth masking, generation strategies | Reimplement from `model.py` |
| `merger.cpp` (already correct, just unlink `hass_block.cpp` version) | Use existing 3-gate version | Already implemented |
| `cot_vector.cpp` (CoT aux head) | Extra `cot_loss_head.*` not in Python | Reimplement from `cot_vector.py` |
| `checkpoint.cpp` (format) | Container mismatch | Reimplement to match `checkpoint.py` |
| `variants.cpp` (config) | Missing `cost_aware_routing`, `compute_budget` fields | Add fields |

### Estimated scope
- **Lines rewritten**: ~3,500 (model + routing + HASS + loss + checkpoint + CoT)
- **Lines kept**: ~6,000 (infrastructure: optimized/, training/, data/, tokenizer/, extern/ggml/)
- **Files rewritten**: 6 files
- **Files added**: 2 new (generation.cpp, xorm_crypto.cpp)
- **Total C++ codebase after**: ~10,000 lines (down from 12,474)

### Checkpoint implications
- Path B produces a C++ build that can load Python checkpoints natively (with a converter for the `model_state_dict` wrapping).
- Existing C++ checkpoints become incompatible — but there are no production C++ checkpoints to preserve.

### Testing burden
- 14 existing fixtures can be reused verbatim (they test component behavior, not implementation).
- Add ~10 new fixtures for components not yet covered.
- Full-model parity test becomes feasible.

### Performance opportunities
- Same as Path A (post-parity).
- Plus: the rewrite can use the existing SIMD/OpenMP infrastructure from day one.

### Technical risks
1. **API mismatch**: the rewritten components must match the existing infrastructure APIs (e.g. `Module`'s forward signature, `Tensor` usage). If the infrastructure APIs are too restrictive, the rewrite may be constrained.
2. **Include graph fragility**: `xorzen_model.cpp` includes `hass.h` which declares the 2-gate `GatedMerger`. Replacing the merger requires updating either `hass.h` (remove the declaration) or `xorzen_model.cpp` (include `merger.h` instead). Either change can cascade.
3. **Behavioral drift during rewrite**: rewriting 3,500 lines of model code in one pass risks introducing subtle bugs that the existing fixtures won't catch (they only cover 14 components).

### Dependencies
- LibTorch (already linked)
- No new external dependencies

### Likely failure modes
1. The rewritten `HASSBlockImpl` may not match the Python sparse dispatch semantics exactly (the Python `sparse_pathway_dispatch` is non-trivial — it uses straight-through estimators and per-token masking).
2. The rewritten `AdaptiveRouterImpl` may not match the Python cost-aware modulation exactly (the `sparsity_pressure` formula has 4 axes with different scaling constants).
3. The rewritten loss may not match the Python loss exactly (the `_AUX_LOSS_KEYS` allowlist + `unify_load_balance` toggle + Switch LB formula interact subtly).
4. The rewritten checkpoint format may not be byte-compatible with Python's `torch.save` output (PyTorch's pickle format is complex).

---

## PATH C — Full model rewrite (reimplement C++ model directly from Python specification)

### Strategy
Throw away `xorzen.cpp/src/model/*.cpp` entirely. Reimplement the model from scratch,
following the Python specification as the source of truth. Keep only the infrastructure
directories (`optimized/`, `training/`, `data/`, `tokenizer/`, `extern/ggml/`).

### Reusable code (infrastructure only)
- Same as Path B's infrastructure list.
- Plus: the existing `xorzen.cpp/src/model/*.cpp` files become reference material (read-only) for understanding the prior C++ approach.

### Code requiring replacement
All model code:
- `hass_block.cpp` (345 lines)
- `routing.cpp` (295 lines)
- `xorzen_model.cpp` (241 lines)
- `merger.cpp` (164 lines, already correct — keep)
- `ssm.cpp` (216 lines — dead code, delete)
- `cot_vector.cpp` (419 lines)
- `expert.cpp` (329 lines)
- `zmoe.cpp` (477 lines)
- `variants.cpp` (311 lines — keep, add missing fields)
- `igris.cpp` (34 lines — appears unused)
- `coherence_field.cpp` (192 lines)

### Estimated scope
- **Lines rewritten**: ~2,500 (model code only, excluding merger.cpp which is kept)
- **Lines kept**: ~6,500 (infrastructure)
- **Files rewritten**: 9 files
- **Files added**: 2-3 new (generation.cpp, xorm_crypto.cpp, possibly a Python-spec annotation file)
- **Total C++ codebase after**: ~9,500 lines

### Checkpoint implications
- Same as Path B — Python checkpoints loadable natively (with converter for container format).

### Testing burden
- 14 existing fixtures can be reused.
- Full-model parity test becomes feasible immediately.
- The rewrite can be tested component-by-component as each module is completed.

### Performance opportunities
- Same as Path A and B (post-parity).
- Plus: the rewrite can be designed from the start to use the SIMD/OpenMP infrastructure idiomatically.

### Technical risks
1. **Highest upfront cost**: rewriting 2,500 lines of model code is a significant time investment.
2. **Risk of inconsistency**: if the rewrite doesn't follow the Python spec exactly, the same bugs as Path A could re-emerge.
3. **Loss of incremental validation**: Path A and B allow incremental fixes validated against fixtures; Path C requires the full rewrite to be done before any validation.

### Dependencies
- LibTorch (already linked)
- No new external dependencies

### Likely failure modes
1. The rewrite may take longer than expected (underestimation of model complexity).
2. The rewrite may introduce bugs in components that currently work (e.g. LocalAttentionPathway is currently MATCHED — a rewrite risks breaking it).
3. The rewrite may diverge from Python in subtle ways that the fixtures don't catch.

---

## Cross-path comparison

| Dimension | Path A (incremental) | Path B (partial rewrite) | Path C (full rewrite) |
|---|---|---|---|
| Lines changed | ~2,300 | ~3,500 | ~2,500 |
| Files touched | 4 modified + 2 new | 6 rewritten + 2 new | 9 rewritten + 2-3 new |
| Components requiring work | 18 of 31 | 22 of 31 | 31 of 31 (all reimplemented) |
| Existing MATCHED components preserved | 9 of 9 | 9 of 9 | 0 of 9 (all rewritten) |
| Risk of breaking working components | MEDIUM (cascading fixes) | LOW-MEDIUM (rewrite isolates changes) | HIGH (everything rewritten) |
| Time to first parity milestone | LOW (fix SSM → run fixture 14) | MEDIUM (rewrite HASS → run all fixtures) | HIGH (rewrite model → run all fixtures) |
| Checkpoint compat achieved | After all 18 components fixed | After 6 files rewritten | After 9 files rewritten |
| Final code clarity | LOW (patched code, mixed styles) | MEDIUM (rewritten modules, kept infra) | HIGH (consistent style throughout) |
| Performance ceiling (post-parity) | Same | Same | Same |
| Reuses existing SIMD/kernel infra | YES | YES | YES |
| Reuses existing tokenizer/data/training infra | YES | YES | YES |
| Reuses existing model code | YES (modified) | PARTIAL (merger.cpp kept) | NO (everything rewritten) |

---

## What the evidence says about each path

### Evidence favoring Path A (incremental)
- 9 components are already MATCHED — no need to rewrite them.
- The 4 SSM fixes are local and well-understood (~30 lines).
- The merger fix is a one-line include change (use `merger.h` instead of `hass.h`'s 2-gate version).
- The LocalAttentionPathway is already correct — rewriting it (Path C) would risk breaking it.
- Smallest absolute code change.

### Evidence favoring Path B (partial rewrite)
- The C++ model code has 7+ architectural mismatches (low-rank, loss, merger, sparse dispatch, cost-aware, depth masking, generation) that are not local fixes.
- The `RoutingRegularizer` has 3 interlocking bugs (double-counting, wrong weight, substring filter) — easier to rewrite than fix piecemeal.
- The loss has 7 distinct divergences — fixing one at a time risks introducing new bugs.
- `merger.cpp` is ALREADY correct — just needs to be wired in.
- The infrastructure (SIMD, tokenizer, data, training) is solid and worth keeping.
- Medium absolute code change.

### Evidence favoring Path C (full rewrite)
- The C++ model code has divergent naming conventions, inconsistent style, and dead code (`S4DKernel`, `character_router`, 2-gate merger).
- A rewrite produces a consistent, Python-faithful codebase.
- The 14 component fixtures provide a safety net — the rewrite can be validated component-by-component.
- The infrastructure (kept) is well-isolated from the model code.
- Highest absolute code change, but produces the cleanest result.

### Evidence against Path A
- The 7+ architectural mismatches are not local fixes — patching them in place produces a codebase that is harder to maintain than a rewrite.
- The `pathway_gate` removal + sparse dispatch addition + low-rank rewrite are all invasive changes to `hass_block.cpp` — they amount to a partial rewrite of that file anyway.
- The loss has 7 interlocking divergences — piecemeal fixes risk introducing new bugs.

### Evidence against Path B
- Rewriting 6 files is more upfront work than fixing 4 files.
- The rewrite must match the existing infrastructure APIs, which may constrain the implementation.

### Evidence against Path C
- The LocalAttentionPathway is already MATCHED — rewriting it risks breaking it.
- The AdaptiveFFN at max_width is already MATCHED — rewriting it risks breaking it.
- The MoE aggregation is already MATCHED — rewriting it risks breaking it.
- Highest upfront cost with no incremental validation.

---

## Final notes

1. **The 14 parity fixtures are path-independent** — they validate component behavior, not implementation. Any of the three paths can use them.

2. **The Python specification is frozen** — the golden reference (40 tensors) is byte-reproducible. Any path can be validated against it.

3. **The C++ build system works** — the parity harness compiles directly via `g++ + LibTorch` (no cmake required for the harness). The full `xorzen.cpp` build via cmake has not been verified in this environment.

4. **No path has been chosen in this document.** The evidence is presented for the user to decide.

5. **Regardless of path chosen, the next step is the same**: write a converter script that re-wraps Python `model_state_dict` into the C++ flat format (or vice versa). This is ~150 lines of Python and unblocks checkpoint-based parity testing.

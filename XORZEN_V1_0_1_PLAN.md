# XORZEN v1.0.1 Architectural Synthesis & Implementation Plan

**Author:** Antigravity Engineering Team  
**Date:** October 2026  
**Target Version:** XORZEN v1.0.1  
**Base Systems:** DevNet (main branch `c5f7af7`) + XORZEN v0.2.4 (`aaafbfe`)

---

## 1. Executive Summary & Design Philosophy

XORZEN v1.0.1 is a clean, technically defensible synthesis of the strongest verified architectural innovations from XORZEN v0.2.4 and the subsequent DevNet codebase. 

Rather than superficial branding or unverified claims, XORZEN v1.0.1 establishes:
1. **Mathematical Correctness:** Correct ZOH discretization for SSMs, proper Switch-Transformer load balancing, and mathematically rigorous straight-through estimators.
2. **Elimination of Critical Forensic Bugs:** 
   - Fix MoE expert parameter registration so all active/cached experts receive genuine optimizer updates and are saved in checkpoints.
   - Eliminate cross-sample attention leakage and causal mask destruction in conditional depth processing (`forward_with_depth`).
   - Fix agentic model dead code (`action_head` and `memory_vault`).
   - Ensure `SlicedFFN` is fully integrated with verified gradient flow and compute reduction.
3. **Selective Restoration:**
   - **Restore `GreedModel`**: Port the hybrid architecture to v1.0.1 standards with modern SSM scans and Switch load balancing.
   - **Modernize `.xorm` Runtime**: Provide a lightweight, zero-dependency serialization and inspection container format.
   - **Retain Clean Deprecations**: Keep heavyweight C++/Cython dead-ends and out-of-core Web/Studio components decoupled from the core ML library.
4. **Empirical Verifiability:** All FLOP reductions and speedups are measured on real hardware, and every architectural change is verified through micro-overfit tests and regression suites.

---

## 2. Component Disposition Matrix

| Component | XORZEN 0.2.4 | DevNet | v1.0.1 Decision | Technical Rationale |
|---|---|---|---|---|
| **HASS Pathway Dispatch** | Soft blend (all 3 computed) | Sparse dispatch (`sparse_dispatch.py`) | **RETAIN DevNet** | DevNet's sparse dispatch provides genuine computation skipping (40.6% measured reduction). |
| **FFN Sparsity** | `AdaptiveFFN` (compute-then-blend) | `SlicedFFN` (`sliced_ffn.py`) | **RETAIN & WIRE SlicedFFN** | True nested matrix slicing reduces FLOPs proportionally to width choice without running parallel sub-networks. |
| **SSM Discretization** | ZOH on $A$ only, raw $B$ | ZOH on both $A$ and $B$ | **RETAIN DevNet** | DevNet formulation $B_{bar} = \frac{\exp(\Delta A)-1}{A} B$ is physically consistent with continuous-time LTI theory. |
| **MoE Load Balancing** | Normalized by $N$ (loss $= K$ at equilibrium) | Switch Transformer ($N \times K$ normalization) | **RETAIN DevNet** | Switch Transformer formula is standard, bounded, and invariant to top-$k$ choice. |
| **MoE Expert Registration** | Unregistered in disk mode (0 optimizer updates) | Unregistered in disk mode (0 optimizer updates) | **CRITICAL FIX** | Wrap all experts (in-memory or LRU-cached) in `nn.ModuleDict` to ensure registration in `model.parameters()` and optimizer state. |
| **Conditional Depth** | Broken cross-batch flattening | Broken cross-batch flattening | **CRITICAL FIX** | Prevent cross-sample token concatenation in `forward_with_depth`. Preserve causal isolation per sequence. |
| **`GreedModel`** | Implemented (215 lines) | Deleted | **RESTORE & MODERNIZE** | Restore as a standalone lightweight hybrid model using v1.0.1 SSM and routing primitives. |
| **`.xorm` Archive Runtime** | Custom binary serializer | Deleted | **REIMPLEMENT (Modern)** | Implement modern, self-contained container for model weights + metadata. |
| **`ZeroAgenticModel`** | In `models/jima/` | In `models/zero/agentic_model.py` | **FIX & HARMONIZE** | Fix dead `action_head` (connect to action loss) and dead `memory_vault` (integrate into recurrent state). |
| **C++ Native Engine** | `xorzen.cpp/` | Deleted | **KEEP REMOVED** | LibTorch C++ bindings add build fragility without performance advantage over PyTorch C++ extensions / TorchScript. |
| **AVX2 SIMD Kernels** | Raw C/Cython | Deleted | **KEEP REMOVED** | PyTorch native CPU ATen kernels outperform unrolled handwritten loops and maintain cross-platform portability. |

---

## 3. Forensic Bug Remediation Plan

### 3.1 BUG-CRITICAL-2: MoE Expert Parameter Registration & Checkpoint Serialization
- **File:** `xorzen/model/zmoe.py` (`ShardedExpertFabric`)
- **Root Cause:** When `test_mode=False`, experts are loaded dynamically from `.pt` files and stored in a standard Python dictionary `self.cache.cache`. Because PyTorch's `nn.Module` does not track contents of arbitrary Python dicts, `self.parameters()` returns empty for all disk-loaded experts. Optimizer steps update 0 expert weights. Furthermore, `model.state_dict()` omits them.
- **Solution:** 
  1. Add `self.registered_experts = nn.ModuleDict()` to `ShardedExpertFabric`.
  2. For in-memory mode (`shard_experts=False` or when pre-loading), populate `self.registered_experts` with all $N$ experts.
  3. When an expert is fetched from disk/cache, ensure it is registered in `self.registered_experts[str(expert_id)]`.
  4. Implement `sync_experts_to_disk()` and an integrated state dict hook so `state_dict()` and `load_state_dict()` serialize the full ensemble of experts.

### 3.2 BUG-CRITICAL-3: `forward_with_depth` Cross-Batch Boundary Leakage
- **File:** `xorzen/model/components/hass_block.py`
- **Root Cause:** Lines 1136–1142 flatten active tokens across all batch items using `active_x = x[active_mask]`, creating a synthetic sequence of length `[1, num_active, hidden]`. This contaminates attention masks, destroys causal isolation, and leaks tokens from sample $A$ into sample $B$.
- **Solution:** 
  1. Process sequences retaining batch dimensionality `[B, T, H]`.
  2. When depth skipping is active, apply residual gating:
     $$x_{\text{out}} = x + \text{depth\_mask} \odot \Delta x$$
     where $\Delta x = \text{HASSBlock}(x)$.
  3. For genuine compute-skipping at inference on batched sequences, process each sample independently or apply block-diagonal attention masking so cross-sample interaction is strictly mathematically impossible.

### 3.3 AGENTIC DEAD CODE: `action_head` and `memory_vault`
- **File:** `xorzen/models/zero/agentic_model.py`
- **Root Cause:**
  - `action_head` computed logits in `forward()`, but returned output was never included in loss calculation, leading to 0 gradient updates.
  - `memory_vault` was instantiated as an `nn.Parameter`, but was never referenced in `forward()`.
- **Solution:**
  1. Wire `action_head` outputs into the model return dictionary and provide an optional `action_targets` loss term in `compute_loss()`.
  2. Integrate `memory_vault` as learned key-value memory prefix in the attention/SSM recurrence or as an initial working memory state.

---

## 4. Architectural Synthesis Details

### 4.1 HASS & SlicedFFN Integration
- Ensure `HASSBlock` uses `SlicedFFN` by default.
- Verify straight-through estimator (STE) or auxiliary diversity loss so that width routing logits receive informative gradients during training.
- Add FLOP counting instrumentation to verify actual matrix multiplication dimension reduction ($H \times W_i + W_i \times H$).

### 4.2 GreedModel Restoration
- Re-introduce `xorzen/models/greed/model.py` and `xorzen/models/greed/config.py`.
- Incorporate DevNet's corrected SSM scan (`xorzen.model.components.ssm_scan`) and Switch load balancing.
- Add `greed_tiny` and `greed_small` presets to `ConfigFactory`.

### 4.3 .xorm Container Specification
- Create `xorzen/serialization/xorm.py` supporting:
  - Header metadata (model architecture, version, hyper-parameters, commit hash)
  - Tensor payloads (FP32, FP16, BF16)
  - Checksum validation (SHA-256)
  - Memory-mapped zero-copy loading

---

## 5. Verification & Test Strategy

1. **Unit & Regression Suite:**
   - `tests/test_v101_moe_registration.py`: Verify that expert parameters appear in `model.parameters()` and receive non-zero gradients during `.backward()`.
   - `tests/test_v101_batch_isolation.py`: Verify that `forward_with_depth` produces identical outputs for sample $A$ regardless of what sample $B$ contains in the batch.
   - `tests/test_v101_sliced_ffn.py`: Verify that smaller width choices reduce intermediate tensor dimensions and FLOPs.
   - `tests/test_v101_agentic.py`: Verify that `action_head` and `memory_vault` receive gradients.
2. **Micro-Overfit Test:**
   - Train `zero_tiny_23k` / `zero_1M` on 4 synthetic sequences for 100 iterations.
   - Verify that training loss monotonically decreases from initial cross-entropy (~5.6) to < 0.5.
3. **Release Documentation & Packaging:**
   - Bump version to `1.0.1` across `setup.py`, `pyproject.toml`, and `xorzen/__init__.py`.
   - Generate `XORZEN_V1_0_1_AUDIT.md`.

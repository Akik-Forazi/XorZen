# XorZen Speed Architecture Audit

**Scope:** `xorzen/speed/*` — what actually runs, what is experimental scaffolding, and what is unreachable dead code.
**Method:** static call-graph analysis (grep + manual read of every speed module + every caller in `xorzen/`, `notebooks/`, `scripts/`).
**Date:** audit pass on the current tree.

---

## 1. The ACTUAL activation path — what runs when you call `xorzen_compile(model)`

### 1.1 Who calls it

`xorzen_compile` (re-exported as `xorzen.speed.compile`) is the **only** speed entry point reached in any production / training notebook:

| Caller | Location | Condition |
|---|---|---|
| `XORZEN_Kaggle_T4.ipynb` | cell `USE_TORCH_COMPILE` block, line ~1794 | `if USE_TORCH_COMPILE: model = xorzen_compile(model, mode='reduce-overhead')` |
| `XORZEN_Colab_GPU.ipynb` | same cell, line ~1794 | identical |
| `tests/test_ssm_scan_parity.py` | line 92 | imports `jit_sequential_scan` directly (not `xorzen_compile`), to verify parity |

No script under `scripts/`, no trainer under `xorzen/training/`, and no model code under `xorzen/model/` imports `xorzen_compile` or `boost_model`. The third notebook (`XORZEN_zero_277M_Colab.ipynb`) does not use the speed package at all.

### 1.2 What `xorzen_compile.compile()` actually does

Read `xorzen/speed/xorzen_compile.py:45-107`. Despite the name and the `mode` / `fullgraph` / `dynamic` / `backend` kwargs, **it does NOT call `torch.compile`**. The docstring (lines 51-73) is explicit about this:

> "This function does NOT use torch.compile (which is incompatible with xorzen's dynamic control flow — MoE dispatch loops, SlicedFFN width grouping, and router conditionals cause recompilation limit hits)."

The real control flow is:

```
xorzen_compile(model, mode=...)          # mode/fullgraph/dynamic/backend all IGNORED
  │
  ├── print banner (lines 75-81)
  │
  ├── try:
  │     from .jit_kernels import auto_patch_model as jit_patch, get_jit_status
  │     if get_jit_status()["available"]:        # always True on modern PyTorch
  │         jit_patch(model)                     # ← THE ONLY REAL PATCH
  │
  └── try:
        from .native_kernels import native_available, auto_patch_model as native_patch
        if native_available:                     # True only if a C++ compiler is present at import
            native_patch(model)                  # ← NO-OP (see §4)
  │
  └── return model   # unchanged object; only the module-level ssm_scan was monkey-patched
```

### 1.3 The one real patch: `jit_kernels.auto_patch_model`

`xorzen/speed/jit_kernels.py:209-247`. The body that actually runs:

```python
import xorzen.model.components.ssm_scan as ssm_module
ssm_module.sequential_scan = jit_sequential_scan   # monkey-patch the module attribute
```

That's it. Consequences:

* Every call site that resolves `sequential_scan` through the module namespace (`ssm_scan.sequential_scan(...)`) at runtime will hit the JIT-compiled version. This includes `select_scan` (T ≤ 64 path) **and** `chunked_scan` (which calls `sequential_scan` per chunk — `ssm_scan.py:253, 267`). So the patch covers all production paths.
* The MoE dispatch patch is **commented out** (lines 226-233): "SKIP for now, the MoEOutput unpacking is too fragile across different call sites."
* The SlicedFFN width-grouping patch is mentioned in the docstring (line 217) but is **not implemented** — only the SSM patch is applied.
* `jit_sequential_scan` is verified bit-equivalent to the Python `sequential_scan` (abs < 1e-6) by `tests/test_ssm_scan_parity.py:113`. **No numerical-semantic risk.**

### 1.4 Net effect on a model passed through `xorzen_compile`

1. `ssm_scan.sequential_scan` is replaced by a TorchScript-compiled version with identical numerics. The Python for-loop overhead on the SSM scan disappears; outputs are unchanged.
2. Nothing else is modified. The model object, its parameters, its MoE dispatch loop, its SlicedFFN width grouping, and its attention are all untouched.
3. If a C++ compiler is available, `native_kernels` is silently JIT-compiled and cached (10-30 s on first run, instant after) but **no kernel is auto-wired into the model** (see §4).
4. `torch.compile` is NOT applied — by design.

---

## 2. What `SpeedBooster` does — and who calls it

### 2.1 What it does

`xorzen/speed/booster.py:61-118`. `SpeedBooster.apply(model)` walks `model.named_modules()` and **replaces** submodules in place:

| `cls_name` matched | Replacement | Condition |
|---|---|---|
| `LocalAttentionPathway` | `FlashLocalAttention.from_slow(module)` | always |
| `SSMPathway` | `FastSSMPathway.from_slow(module)` | always |
| `AdaptiveRouter` | `CachedRouter.from_slow(module)` | `profile != CONSERVATIVE` |
| `ShardedExpertFabric` | `PreloadedExpertFabric.from_slow(module)` | `profile != CONSERVATIVE` |
| `RMSNorm` | `_FastRMSNorm(module)` | `_cpp_ok and profile == LIGHTNING` |

After the walk, in `LIGHTNING` profile it additionally calls `torch.compile(model, mode="reduce-overhead")` (skipped on CPU-Windows).

`_FastRMSNorm` (booster.py:46-58) calls `_ext.cpp_rms_norm(...)` directly — there is no Python fallback inside `forward`.

### 2.2 Who calls it

**Nobody.** Search results:

```
boost_model(           → never invoked anywhere
SpeedBooster(          → only invoked inside boost_model()
FastTrainer(           → never instantiated
```

The only external reference is `xorzen/gui/bootloader.py:361`:
```python
def _check_speed_booster(self):
    try:
        from xorzen.speed import SpeedBooster  # noqa
        return "ARMED", "green", "LIGHTNING profile ready"
    except ImportError as e:
        return "FAILED", "red", str(e)
```
This is a POST-style import probe that reports "ARMED" on the bootloader dashboard. It never instantiates `SpeedBooster`, never calls `apply()`. So `SpeedBooster`, `boost_model`, `SpeedProfile`, `_FastRMSNorm`, `FlashLocalAttention`, `FastSSMPathway`, `CachedRouter`, `PreloadedExpertFabric`, and `FastTrainer` are **all defined, exported, and never invoked** by any production code path.

### 2.3 Why this matters

Because nothing calls `SpeedBooster.apply`, the `fast_*.py` drop-in replacements are never wired into a live model. Their `forward()` methods — including the dead `xorzen_ext` code paths inside them — are unreachable from any training or inference entry point. They exist as an alternative optimization surface that was scaffolded but never activated, presumably because `xorzen_compile` (which only patches SSM) turned out to be sufficient and safer.

---

## 3. What `jit_kernels` patches — and who activates it

### 3.1 Patches applied

`auto_patch_model` (jit_kernels.py:209-247) does **one** thing:

```python
ssm_module.sequential_scan = jit_sequential_scan
```

The MoE dispatch and SlicedFFN patches described in the docstring are **not implemented**:
* MoE dispatch (lines 226-233): commented out, "MoEOutput unpacking is too fragile across different call sites."
* SlicedFFN width grouping: listed in the docstring (line 217) but no corresponding code in `auto_patch_model`.

The `@torch.jit.script`-decorated helpers that ARE defined but never wired in:
* `_jit_moe_group_indices` (line 51) — defined, used by nobody (the MoE patch is commented out)
* `jit_width_group_indices` (line 184) — defined, exported in `__init__.py`, used by nobody

### 3.2 Who activates it

* `xorzen_compile.compile()` at `xorzen_compile.py:85-88` — gated on `get_jit_status()["available"]` (always True on PyTorch ≥ 1.0 with `torch.jit.script`).
* `tests/test_ssm_scan_parity.py:100` — directly patches `ssm_module.sequential_scan = jit_sequential_scan` for parity verification.

### 3.3 Numerical-semantic risk

**None.** `jit_sequential_scan` (jit_kernels.py:131-163) is a literal TorchScript translation of `sequential_scan` (ssm_scan.py:132-157). The parity test asserts `max_abs < 1e-6` across multiple T values. The JIT version uses the same `torch.zeros` / `torch.stack` ops, just without Python interpreter overhead on the inner `for t in range(T_size)` loop.

---

## 4. What `native_kernels` patches — and who activates it

### 4.1 What it loads

`native_kernels.py:51-116`. On import, `_try_compile()` calls `torch.utils.cpp_extension.load()` on the seven `.cpp` files in `xorzen/speed/csrc/`:

```
math_utils.cpp, fused_ops.cpp, attention_ops.cpp,
expert_dispatch.cpp, router_ops.cpp, ssm_scan.cpp,
xorzen_native_bridge.cpp   # pybind11 bridge — must be last
```

If a C++ compiler is present (g++/clang/MSVC), this produces a cached `xorzen_native_kernels` extension module and sets `native_available = True`. Compilation flags include `-O3 -ffast-math -march=native -mfma -mavx2 -fopenmp` on Linux/Mac.

### 4.2 What `auto_patch_model` does

`native_kernels.py:331-355`. Read the full body:

```python
def auto_patch_model(model: nn.Module) -> bool:
    if not native_available:
        return False
    patched = False
    # NOTE: We do NOT patch SlicedFFN._activation with fused_gelu because the
    # C++ kernel uses tanh-approximate GELU while PyTorch uses exact GELU (erf).
    # This would silently change model semantics. The native GELU kernel remains
    # available as fused_gelu() for explicit opt-in use, but is never auto-patched.
    #
    # To re-enable auto-patching, the C++ kernel must be updated to use the exact
    # GELU formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    return patched
```

**It patches nothing.** Even when `native_available` is True, `auto_patch_model` returns False without modifying the model. The wrappers `fused_rmsnorm`, `fused_gelu`, `fused_swiglu`, `fused_layernorm_gelu`, `expert_dispatch`, `diagonal_ssm_scan` are exported in `__init__.py` and reachable as opt-in APIs, but **no model code calls them** (verified by grep across `xorzen/` and `scripts/`).

### 4.3 Who activates it

* `xorzen_compile.compile()` at `xorzen_compile.py:96-104` calls `native_patch(model)` if `native_available`. Because `auto_patch_model` is a no-op, this just prints "Native C++ kernels also loaded" and changes nothing.
* `bootloader.py` `_check_kernels()` (line 320) counts the `.cpp` files in `csrc/` — a file-existence probe, not an activation.
* Nobody else.

### 4.4 Numerical-semantic risks (per kernel, if it WERE activated)

| Wrapper | Risk | Notes |
|---|---|---|
| `fused_rmsnorm` (line 123) | low | float32 internal cast; `-ffast-math` may reorder rsqrt. Diff from `F.rms_norm` < 1e-6 at fp32, larger at fp16/bf16. |
| `fused_gelu` (line 148) | **HIGH** | Uses **tanh approximation** `x * 0.5 * (1 + tanh(√(2/π)·(x+0.044715x³)))`. PyTorch default is **exact** `x * 0.5 * (1 + erf(x/√2))`. Diff up to ~1e-3 in the tail. **This is why `auto_patch_model` is a no-op.** |
| `fused_swiglu` (line 161) | low | `silu(gate) * up`; uses `-ffast-math` silu. Diff < 1e-6. |
| `fused_layernorm_gelu` (line 184) | **HIGH** | Inherits the tanh-approx GELU. |
| `expert_dispatch` (line 202) | **HIGH** | Moves `expert_indices` and `expert_weights` to CPU (`.cpu()` calls at lines 245-246, 273-277), runs the kernel, moves back. Breaks autograd graph for any expert not on CPU. Only supports `torch.float32` (line 271 guard). The Python fallback (lines 226-240) is what actually runs in all current configs. |
| `diagonal_ssm_scan` (line 285) | **HIGH** | Forces `.cpu().float()` (lines 309-310), runs scalar scan in C++, returns to original device/dtype. Breaks autograd (no `torch.autograd.Function` wrapper). Signature is also mismatched: takes `b_seq, a_diag` while `jit_sequential_scan` takes `A_bar, B_bar, init_state` — these are NOT interchangeable. |

---

## 5. Which `fast_*.py` files have dead `xorzen_ext` code paths

### 5.1 Why `xorzen_ext` is always missing

`setup_speed.py:99-113` declares a Cython extension named `xorzen_ext` whose first source file is `xorzen_ext.pyx`:

```python
ext = Extension(
    name="xorzen_ext",
    sources=[
        str(HERE / "xorzen_ext.pyx"),   # ← THIS FILE DOES NOT EXIST
        *CPP_SOURCES,
    ],
    ...
)
```

A glob for `**/xorzen_ext*` and `**/*.pyx` across the entire repo returns **zero matches**. There is no `xorzen_ext.pyx`, no `xorzen_ext.py`, no compiled `xorzen_ext*.pyd` / `xorzen_ext*.so`. Therefore:

* `python setup_speed.py build_ext --inplace` would fail at the `cythonize` step (missing source).
* `from xorzen.speed import xorzen_ext` raises `ImportError` in every environment, every time.
* Every `try: from xorzen.speed import xorzen_ext as _ext; _CPP_AVAILABLE = True; except ImportError: _CPP_AVAILABLE = False` block resolves to `_CPP_AVAILABLE = False`.

The `csrc/*.cpp` files DO exist and are reachable through `native_kernels._try_compile()` via `torch.utils.cpp_extension.load()` — a completely separate mechanism that does not depend on `xorzen_ext`. So `native_available` can be True on a machine with a compiler, while `xorzen_ext` is permanently unimportable.

### 5.2 Dead `xorzen_ext` reference inventory

For each reference: file, line, what the dead branch does, what the fallback does, and a recommendation.

#### (a) `xorzen/speed/fast_attention.py:17`

```python
try:
    from xorzen.speed import xorzen_ext as _ext
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
```

* **`_CPP_AVAILABLE` always False** — confirmed (see §5.1).
* **Dead branch:** the C++ window-attention path at lines 95-104. Gated by `if _CPP_AVAILABLE and not x.is_cuda and position_bias is None and attention_mask is None:`. The body calls `_ext.cpp_window_attention(q_c, k_c, v_c, self.window_size)`. Never executes.
* **Fallback used:** SDPA path at lines 128-132 (`F.scaled_dot_product_attention`), or the manual softmax path at lines 134-145 if SDPA is unavailable. The fallback is the only path that ever runs.
* **Stats report:** `get_compute_stats` (line 162) reports `"cpp_kernel_active": _CPP_AVAILABLE` — always `False`.
* **Recommendation:** **Remove** the `try/except` import and the C++ branch. The SDPA path is what runs in every real configuration (notebooks use GPU + attention masks, so the C++ branch was doubly unreachable: `_CPP_AVAILABLE=False` AND `attention_mask is not None` during training). Keep the `_SDPA_AVAILABLE` and manual fallback paths.

#### (b) `xorzen/speed/fast_ssm.py:28`

```python
try:
    from xorzen.speed import xorzen_ext as _ext
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
```

* **`_CPP_AVAILABLE` always False** — confirmed.
* **Dead branch:** none — `_CPP_AVAILABLE` is referenced only at line 226 (`"cpp_kernel_active": _CPP_AVAILABLE` in `get_compute_stats`). There is no C++ call path in `FastSSMPathway.forward` or `forward_parallel`; both delegate to `xorzen.model.components.ssm_scan.{discretize_zoh, select_scan, parallel_scan, sequential_scan}`.
* **Fallback used:** the entire class is the fallback. The C++ flag is cosmetic.
* **Recommendation:** **Remove** the `try/except` import and the `cpp_kernel_active` stats field. `_ext` is imported but never used.

#### (c) `xorzen/speed/fast_router.py:17`

```python
try:
    from xorzen.speed import xorzen_ext as _ext
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
```

* **`_CPP_AVAILABLE` always False** — confirmed.
* **Dead branch:** the C++ MLP fast path at lines 100-108. Gated by `if self._cpp_mlp_enabled and not router_input.is_cuda:` (where `self._cpp_mlp_enabled = _CPP_AVAILABLE` from line 74). The body calls `_ext.cpp_router_mlp(x_flat, w1, b1, w2, b2, w3, b3)`. Never executes.
* **Fallback used:** `return self._router.feature_encoder(router_input)` (line 111) — the original PyTorch `nn.Sequential` forward. Always runs.
* **Recommendation:** **Remove** the `try/except` import, the `_cpp_mlp_enabled` attribute, the `_extract_encoder_weights` method (lines 82-95, only used by the dead C++ branch), and the C++ branch in `_encode_features`. The class becomes a pure inference-cache wrapper around the original router.

#### (d) `xorzen/speed/fast_moe.py:14`

```python
try:
    from xorzen.speed import xorzen_ext as _ext
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
```

* **`_CPP_AVAILABLE` always False** — confirmed.
* **Dead branch 1:** the C++ capacity constraint at lines 85-90. Gated by `if _CPP_AVAILABLE and not hidden_states.is_cuda:`. The body calls `_ext.cpp_expert_capacity(idx_2d, w_2d, fab.num_experts, cap)`. Never executes.
* **Dead branch 2 (the whole method):** `_fast_forward` (lines 107-197). Calls `_ext.cpp_expert_weighted_sum(...)` at line 181. The method is **never called from `forward()`** — the `forward` body at line 93 says: `# Delegate to original fabric (fast path removed — caused regression)`. So `_fast_forward` is unreachable even if `_CPP_AVAILABLE` were True. Additionally, `_fast_forward` uses `self._prefetch_executor` (never set in `__init__`), so it would `AttributeError` if called.
* **Dead branch 3:** `_schedule_prefetch` (lines 52-69) uses `self._prefetch_lock`, `self._last_expert_ids`, `self._prefetch_executor` — none of which are initialized in `__init__` (lines 28-30 only set `self._fabric`). The only call site (line 102) is guarded by `hasattr(self, '_prefetch_executor')` which is always False, so the call never fires. Even if it did, `with self._prefetch_lock:` would `AttributeError`.
* **Fallback used:** `forward` delegates entirely to `fab.forward(...)` (the original `ShardedExpertFabric`), so the wrapper is a no-op pass-through (with a never-fired prefetch side-channel).
* **Recommendation:** **Remove** the `try/except` import, the C++ capacity block, the entire `_fast_forward` method, and the entire `_schedule_prefetch` / `_prefetch_experts` machinery (the prefetch executor is never created). What's left is a pass-through wrapper with no behavior — at which point `PreloadedExpertFabric` itself can be deleted (see §6).

#### (e) `xorzen/speed/booster.py:54`

```python
class _FastRMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from xorzen.speed import xorzen_ext as _ext
        ...
        out = _ext.cpp_rms_norm(x2d, self.weight.float(), self.eps)
```

* **Import always raises ImportError.** Unlike (a)-(d), this import is inside `forward()`, not at module top. So calling `_FastRMSNorm.forward` would raise `ImportError` at runtime.
* **Dead branch:** the entire `_FastRMSNorm` class (lines 46-58). It is only instantiated at line 98 inside `SpeedBooster.apply`, gated by `elif cls_name == "RMSNorm" and _cpp_ok and self.profile == SpeedProfile.LIGHTNING:`. But `_cpp_ok` is set at lines 75-79 by another `try: from xorzen.speed import xorzen_ext as _ext; _cpp_ok = True; except ImportError: _cpp_ok = False` — and `_cpp_ok` is **always False**. So the `elif` branch never fires, and `_FastRMSNorm` is never instantiated.
* **Fallback used:** RMSNorm modules are left untouched in the model. The original `RMSNorm` (presumably from `xorzen.model.components`) runs as-is.
* **Recommendation:** **Remove** the `_FastRMSNorm` class entirely, the `_cpp_ok` probe (lines 75-79), the `elif cls_name == "RMSNorm"` branch (lines 97-99), and the `rmsnorm_patched` field from `BoosterStats`. If `SpeedBooster` itself is removed (see §6), this is automatic.

#### (f) `xorzen/speed/booster.py:76`

```python
def apply(self, model: nn.Module) -> BoosterStats:
    ...
    try:
        from xorzen.speed import xorzen_ext as _ext
        _cpp_ok = True
    except ImportError:
        _cpp_ok = False
```

* **Import always raises ImportError** → `_cpp_ok = False` always.
* **Dead branch:** the `elif cls_name == "RMSNorm" and _cpp_ok and ...` branch at line 97. Never fires.
* **Fallback used:** the entire `apply()` body runs without ever entering the RMSNorm branch. The other four replacements (LocalAttention, SSM, Router, ExpertFabric) proceed unconditionally on `cls_name` — but they are themselves never invoked because `apply()` is never called (see §2.2).
* **Recommendation:** **Remove** the `_cpp_ok` probe and the RMSNorm branch. Or, given §2.2, remove `SpeedBooster` entirely.

### 5.3 `xorzen_ext` references that are NOT dead

Two references in `xorzen/gui/bootloader.py` are intentional import-availability probes, not dead code paths:

| File | Line | Function | Verdict |
|---|---|---|---|
| `xorzen/gui/bootloader.py` | 287 | `_check_avx2` (Windows branch) | **Reserved.** Probes whether the extension built. Reports "AVX2 (MSVC /arch:AVX2)" if importable, falls through to scalar-fallback message otherwise. The probe itself is the point. Keep, but consider relabeling the success message since the extension can never build (see §5.1). |
| `xorzen/gui/bootloader.py` | 329 | `_check_cython_ext` | **Reserved.** Same pattern. Reports "LOADED" or "NOT COMPILED". Will always report "NOT COMPILED" until `xorzen_ext.pyx` is created. Keep — it correctly diagnoses the missing build artifact. |

The reference in `setup_speed.py:11` (docstring), `:100` (extension name), `:102` (missing `.pyx` source), `:139` (post-build `.pyd` glob), and `:146` (success print) constitute a **build script that cannot succeed**. Recommendation: either restore `xorzen_ext.pyx` or delete `setup_speed.py` and the `xorzen_ext`-as-extension concept entirely, since `native_kernels.py` provides a working alternative path via `torch.utils.cpp_extension.load()`.

---

## 6. What's experimental vs active vs dead

### 6.1 ACTIVE — runs in production training

| Symbol | File | What it does |
|---|---|---|
| `xorzen_compile` | `xorzen_compile.py:45` | Entry point; prints banner, calls jit_patch, calls native_patch. |
| `jit_kernels.auto_patch_model` | `jit_kernels.py:209` | Patches `ssm_scan.sequential_scan → jit_sequential_scan`. **The only model-modifying patch in the entire package.** |
| `jit_kernels.jit_sequential_scan` | `jit_kernels.py:131` | TorchScript-compiled scan; numerically identical to Python version (verified by parity test). |
| `jit_kernels.get_jit_status` | `jit_kernels.py:250` | Returns `{available, error, torch_version}`. Used by `xorzen_compile` to gate the patch. |
| `native_kernels._try_compile` | `native_kernels.py:51` | Runs at import time. Compiles+caches `csrc/*.cpp` via `torch.utils.cpp_extension.load()`. Sets `native_available`. |
| `native_kernels.native_available` | `native_kernels.py:46` | Flag checked by `xorzen_compile`. May be True or False depending on compiler availability. |
| `native_kernels.auto_patch_model` | `native_kernels.py:331` | **Called by `xorzen_compile` but is a no-op** (returns False without patching). Reserved for future use once the GELU semantics issue is resolved. |
| `warmup_compiled` | `xorzen_compile.py:110` | Imported by notebooks alongside `xorzen_compile`. Not actually invoked in the notebook cell (only `xorzen_compile` is called). Listed here under "active" because it's imported, but functionally it is dormant. |

### 6.2 EXPERIMENTAL — defined, exported, never invoked

These are reachable via `from xorzen.speed import ...` but no code in `xorzen/`, `scripts/`, or `notebooks/` calls them.

| Symbol | File | Why dormant |
|---|---|---|
| `SpeedBooster`, `boost_model`, `SpeedProfile` | `booster.py:12, 61, 121` | `boost_model` has zero callers; `SpeedBooster()` is only instantiated inside `boost_model`; the `bootloader.py:361` reference is an import-only POST probe. |
| `FlashLocalAttention` | `fast_attention.py:42` | Only referenced by `SpeedBooster.apply` (which is never called). |
| `FastSSMPathway` | `fast_ssm.py:34` | Same. |
| `CachedRouter` | `fast_router.py:59` | Same. |
| `PreloadedExpertFabric` | `fast_moe.py:20` | Same. |
| `FastTrainer` | `fast_trainer.py:24` | Never instantiated. |
| `_FastRMSNorm` | `booster.py:46` | Only instantiated inside `SpeedBooster.apply` under `_cpp_ok and LIGHTNING` — both conditions unreachable. |
| `native_kernels.fused_rmsnorm` | `native_kernels.py:123` | Exported opt-in; no caller. |
| `native_kernels.fused_gelu` | `native_kernels.py:148` | Same. |
| `native_kernels.fused_swiglu` | `native_kernels.py:161` | Same. |
| `native_kernels.fused_layernorm_gelu` | `native_kernels.py:184` | Same. |
| `native_kernels.expert_dispatch` | `native_kernels.py:202` | Same. |
| `native_kernels.diagonal_ssm_scan` | `native_kernels.py:285` | Same. |
| `native_kernels.get_native_status` | `native_kernels.py:317` | Same. |
| `jit_kernels.jit_moe_dispatch` | `jit_kernels.py:80` | Exported opt-in; the `auto_patch_model` MoE patch is commented out, so this is never wired into a model. |
| `jit_kernels._jit_moe_group_indices` | `jit_kernels.py:51` | Defined; used by nobody (MoE patch commented out). |
| `jit_kernels.jit_width_group_indices` | `jit_kernels.py:184` | Exported; SlicedFFN patch never implemented. |
| `jit_kernels.jit_diagonal_ssm_scan` | `jit_kernels.py:167` | Alias for `jit_sequential_scan`. Kept for backward compat. |

### 6.3 DEAD — unreachable code

| Location | What's dead | Why |
|---|---|---|
| `fast_attention.py:16-20` | `try: from xorzen.speed import xorzen_ext` block | `xorzen_ext.pyx` does not exist; `_CPP_AVAILABLE` always False. |
| `fast_attention.py:95-104` | C++ window-attention forward branch | Gated on `_CPP_AVAILABLE`. |
| `fast_attention.py:162` | `"cpp_kernel_active": _CPP_AVAILABLE` stats field | Always False. |
| `fast_ssm.py:27-31` | `try: from xorzen.speed import xorzen_ext` block | Same. |
| `fast_ssm.py:226` | `"cpp_kernel_active": _CPP_AVAILABLE` stats field | Always False; `_ext` imported but never used elsewhere in file. |
| `fast_router.py:16-20` | `try: from xorzen.speed import xorzen_ext` block | Same. |
| `fast_router.py:74` | `self._cpp_mlp_enabled = _CPP_AVAILABLE` | Always False. |
| `fast_router.py:82-95` | `_extract_encoder_weights` method | Only called from the dead C++ branch at line 101. |
| `fast_router.py:100-108` | C++ MLP fast path in `_encode_features` | Gated on `self._cpp_mlp_enabled`. |
| `fast_moe.py:13-17` | `try: from xorzen.speed import xorzen_ext` block | Same. |
| `fast_moe.py:85-90` | C++ capacity-constraint block in `forward` | Gated on `_CPP_AVAILABLE`. |
| `fast_moe.py:107-197` | `_fast_forward` method | Never called from `forward` (comment: "fast path removed — caused regression"). Also references unimplemented `self._prefetch_executor`. |
| `fast_moe.py:52-69` | `_schedule_prefetch` method | Only call site (line 102) is guarded by `hasattr(self, '_prefetch_executor')` which is never True. Also references `self._prefetch_lock` and `self._last_expert_ids` which are never initialized. |
| `fast_moe.py:38-49` | `_prefetch_experts` method | Only called from `_schedule_prefetch` (dead). |
| `fast_moe.py:206-211` | `__del__` method | Only shuts down `self._prefetch_executor` which is never created. Harmless but dead. |
| `booster.py:46-58` | `_FastRMSNorm` class | Only instantiated under `_cpp_ok and LIGHTNING`; `_cpp_ok` always False. |
| `booster.py:54` | `from xorzen.speed import xorzen_ext as _ext` inside `_FastRMSNorm.forward` | Would raise ImportError if reached; never reached. |
| `booster.py:75-79` | `_cpp_ok` probe in `SpeedBooster.apply` | Always sets `_cpp_ok = False`. |
| `booster.py:97-99` | `elif cls_name == "RMSNorm" and _cpp_ok and ...` branch | Never fires. |
| `booster.py:25, 35` | `rmsnorm_patched` field in `BoosterStats` | Never incremented. |
| `native_kernels.py:345-355` | `auto_patch_model` body (between `patched = False` and `return patched`) | Empty — comment-only. Functionally a no-op. |
| `setup_speed.py:99-113` | `Extension(name="xorzen_ext", sources=[str(HERE / "xorzen_ext.pyx"), ...])` | `xorzen_ext.pyx` does not exist; `cythonize` would fail. |
| `setup_speed.py:139-143` | Post-build `.pyd` copy loop | No `.pyd` is ever produced. |
| `jit_kernels.py:226-233` | Commented-out MoE dispatch patch | Dead by being commented out; mentioned in docstring as if active. |
| `jit_kernels.py:217` | SlicedFFN patch mention in docstring | No corresponding code in `auto_patch_model`. |

---

## 7. Summary of recommendations

### Tier 1 — Safe to delete now (zero behavior change)

1. **All six `from xorzen.speed import xorzen_ext` blocks** listed in §5.2 (a)–(f). `_CPP_AVAILABLE` is always False; the dead branches are unreachable; the fallbacks are what actually run.
2. **`fast_moe._fast_forward`** (lines 107-197). Never called; would crash if called (missing `self._prefetch_executor`).
3. **`fast_moe._schedule_prefetch`, `_prefetch_experts`, `__del__`** (lines 38-49, 52-69, 206-211). The prefetch executor is never created.
4. **`fast_router._extract_encoder_weights`** (lines 82-95). Only called from the dead C++ branch.
5. **`booster._FastRMSNorm`** and the `_cpp_ok` probe + RMSNorm branch (lines 46-58, 75-79, 97-99, 25, 35). `_cpp_ok` is always False.
6. **`setup_speed.py`**. References a nonexistent `xorzen_ext.pyx`. Either restore the `.pyx` or delete the script and the entire `xorzen_ext` concept (since `native_kernels.py` covers the same ground via `torch.utils.cpp_extension.load()`).

### Tier 2 — Delete if SpeedBooster is retired

If `SpeedBooster` / `boost_model` are confirmed unused (they are — see §2.2), the entire `booster.py` and the four `fast_*.py` drop-in modules (`fast_attention`, `fast_ssm`, `fast_router`, `fast_moe`) become dead, because they are only referenced by `SpeedBooster.apply` and `__init__.py` re-exports. This would remove ~1,200 LOC of unreachable code.

`FastTrainer` (fast_trainer.py, 103 LOC) is in the same boat — exported but never instantiated.

### Tier 3 — Label as "reserved" if kept

1. **`native_kernels.auto_patch_model`** — currently a no-op by design (GELU semantics). Add a docstring banner `# RESERVED — see native_kernels.py:347 for re-enablement criteria` so future readers don't expect it to patch.
2. **`native_kernels.{fused_rmsnorm, fused_gelu, fused_swiglu, fused_layernorm_gelu, expert_dispatch, diagonal_ssm_scan}`** — opt-in APIs. Either wire at least one into a model code path (with a parity test) or mark as `# RESERVED — explicit opt-in only, not auto-patched` to make the contract explicit.
3. **`jit_kernels.{jit_moe_dispatch, _jit_moe_group_indices, jit_width_group_indices}`** — opt-in APIs for patches that were planned but commented out. Either implement the patches (with the MoEOutput contract resolved) or mark `# RESERVED — see jit_kernels.py:226`.
4. **`bootloader.py:287, 329`** — these probes are intentional. Keep, but relabel the "AVX2 (MSVC /arch:AVX2)" success message since it can never fire (the extension cannot build without `xorzen_ext.pyx`).

### Tier 4 — Active, leave alone

1. **`xorzen_compile.compile`** — entry point; called by notebooks.
2. **`jit_kernels.auto_patch_model`** (the SSM-patching part) and **`jit_sequential_scan`** — the only patch that actually runs.
3. **`native_kernels._try_compile`** and the `csrc/*.cpp` sources — produces a working (if unused) `xorzen_native_kernels` extension. The compile-and-cache mechanism is sound; only the auto-patch layer above it is a no-op.

---

## Appendix A — Dead `xorzen_ext` references (quick index)

| # | File:Line | Context | `_CPP_AVAILABLE` | Fallback used? | Recommendation |
|---|---|---|---|---|---|
| 1 | `xorzen/speed/fast_attention.py:17` | module-top `try/except` import | always False | SDPA path (lines 128-132) | remove import + C++ branch (lines 95-104) |
| 2 | `xorzen/speed/fast_ssm.py:28` | module-top `try/except` import | always False | entire class is fallback; `_ext` unused | remove import + `cpp_kernel_active` field |
| 3 | `xorzen/speed/fast_router.py:17` | module-top `try/except` import | always False | `feature_encoder` PyTorch path (line 111) | remove import + `_cpp_mlp_enabled` + `_extract_encoder_weights` + C++ branch |
| 4 | `xorzen/speed/fast_moe.py:14` | module-top `try/except` import | always False | delegates to `fab.forward` (line 93) | remove import + C++ capacity block + `_fast_forward` + prefetch machinery |
| 5 | `xorzen/speed/booster.py:54` | inside `_FastRMSNorm.forward` | n/a (import inside method; class never instantiated) | RMSNorm left untouched | remove `_FastRMSNorm` class entirely |
| 6 | `xorzen/speed/booster.py:76` | inside `SpeedBooster.apply` | always sets `_cpp_ok = False` | RMSNorm branch skipped | remove `_cpp_ok` probe + RMSNorm branch |

**Related (not dead, but broken build):**

| # | File:Line | Context | Status | Recommendation |
|---|---|---|---|---|
| 7 | `xorzen/speed/setup_speed.py:102` | references `xorzen_ext.pyx` | file does not exist; build fails | restore `.pyx` or delete `setup_speed.py` |
| 8 | `xorzen/gui/bootloader.py:287` | POST probe in `_check_avx2` | always reports fallback | keep probe; relabel success message |
| 9 | `xorzen/gui/bootloader.py:329` | POST probe in `_check_cython_ext` | always reports "NOT COMPILED" | keep probe (correctly diagnoses missing build) |

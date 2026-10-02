# xorzen.cpp ↔ Python Parity Audit Report

**Date:** 2026-10-02
**Auditor:** Automated systematic audit
**Scope:** All C++ source files vs Python reference

## Executive Summary

The C++ port has **14 major parity gaps** spanning configuration, forward
pass, loss computation, initialization, and generation. The most critical
finding is that **every model variant had completely wrong hyperparameters**
— the C++ ConfigFactory values were invented, not ported from Python.

## Parity Matrix

### CRITICAL (breaks checkpoint compatibility or produces wrong results)

| # | Component | Issue | Status |
|---|-----------|-------|--------|
| 1 | Variant configs | All model sizes had wrong vocab/hidden/layers/experts/ctx | **FIXED** (commit `2c11b38`) |
| 2 | Loss computation | C++ adds L2 load-balance loss that Python zeros by default (`unify_load_balance=True`) | **NOT FIXED** |
| 3 | Loss computation | C++ adds `total_auxiliary` CoT loss term Python doesn't have | **NOT FIXED** |
| 4 | CoT injection | C++ injects `cot_influence * sigmoid(gate)` into hidden before routing; Python does not | **NOT FIXED** |
| 5 | state_dict keys | C++ registers `cot_loss_head` module absent in Python → key mismatch | **NOT FIXED** |
| 6 | Init weights | Padding row not re-zeroed after init when weights are tied | **NOT FIXED** |
| 7 | Init weights | Missing LayerNorm/RMSNorm weight=1/bias=0 init | **NOT FIXED** |
| 8 | Init weights | Missing `1/sqrt(hidden_size)` scaling for untied lm_head | **NOT FIXED** |

### HIGH (breaks feature parity)

| # | Component | Issue | Status |
|---|-----------|-------|--------|
| 9 | Gradient checkpointing | `config.gradient_checkpointing` is never consulted in forward() | **NOT FIXED** |
| 10 | Inference sparse compute | No `forward_with_depth` equivalent (per-token gather/scatter) | **NOT FIXED** |
| 11 | validate_config | 6 of 13 validation checks missing | **NOT FIXED** |
| 12 | Router aux losses | `z_loss`, `path_div_loss`, `width_div_loss` not aggregated into routing_loss | **NOT FIXED** |
| 13 | generate() | Repetition penalty wired in config but never applied | **NOT FIXED** |
| 14 | generate() | No beam search; no per-row unfinished tracking; `.all()` EOS check too strict | **NOT FIXED** |

### MEDIUM (functionality gaps)

| # | Component | Issue | Status |
|---|-----------|-------|--------|
| 15 | ModelOutput | Missing `routing_info` field; extra `agentic_actions` field | **NOT FIXED** |
| 16 | total_loss() | C++ returns `loss.clone()`; Python sums loss + aux (double-counts) | **NOT FIXED** |
| 17 | ModelSize enum | `XL_3B` renamed to `LARGE_3B`; added `XXL_13B`/`XXXL_70B`/`GREED_*` | **FIXED** (commit `2c11b38`) |
| 18 | Helper methods | Missing `get_memory_footprint`, `print_model_summary`, `get_routing_statistics`, `from_pretrained` | **NOT FIXED** |
| 19 | Dead code | `action_head`, `critique_module`, `recursive_router` declared but never constructed | **NOT FIXED** |

### ALREADY FIXED (from previous sessions)

| # | Component | Issue | Fix Commit |
|---|-----------|-------|------------|
| 20 | routing.cpp | Debug `print_tensor_stats()` calls in forward hot path | `8551537` |
| 21 | zmoe.cpp | Per-element `.item<int64_t>()` loop for expert grouping | `8551537` |
| 22 | zmoe.cpp | `mask.any().item<bool>()` unnecessary CPU sync | `8551537` |
| 23 | zmoe.cpp | `.item<double>()` stats update during training | `8551537` |
| 24 | zmoe.cpp | Weight normalization parity bug (C++ normalized, Python doesn't) | `8551537` |

## Test Infrastructure Created

| File | Purpose |
|------|---------|
| `xorzen.cpp/tests/generate_golden_reference.py` | Exports Python model intermediate activations for C++ parity comparison |
| `xorzen.cpp/tests/test_checkpoint_compat.py` | Generates checkpoint + reference data for cross-format compatibility testing |

## Remaining Work

The following items require C++ compilation to verify (not available in
this environment):

1. Fix loss computation in `xorzen_model.cpp` (remove L2 load-balance when `unify_load_balance=true`, remove `total_auxiliary`)
2. Remove `cot_loss_head` from C++ model constructor
3. Fix CoT injection (remove hidden-state modification before routing)
4. Add gradient checkpointing support in forward()
5. Fix init_weights (padding row re-zero, LayerNorm init, lm_head scaling)
6. Add missing validate_config checks
7. Port router auxiliary loss aggregation
8. Fix generate() (repetition penalty, beam search, EOS tracking)
9. Implement `forward_with_depth` for inference sparse compute
10. Remove dead code (action_head, critique_module, recursive_router declarations)
11. Add missing helper methods
12. Build and run parity tests with the golden reference generator

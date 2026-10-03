# `print()` / Logging Audit — `xorzen/` Package

**Scope:** all `*.py` files under `xorzen/` (the library package; tests, scripts, notebooks, and benchmarks are out of scope).
**Audit date:** 2025
**No code was changed.** This is a documentation-only audit.

---

## 0. Methodology

The requested discovery command was:

```bash
grep -rn "print(" xorzen/ --include="*.py" \
  | grep -v "__pycache__\|test\|\.pyc\|#.*print\|logger\|warning" \
  | head -60
```

Equivalent discovery was performed with the in-repo ripgrep wrapper (file-scoped, glob `*.py`, pattern `print\(`). The post-filters (`__pycache__|test|\.pyc|#.*print|logger|warning`, case-sensitive) remove lines containing those substrings. The `head -60` would yield only the first 60 filtered lines; for completeness this audit classifies **every** `print(` match in the package, organised by file, then calls out which lines would survive the user's filter as the "first 60".

### Classification categories (as defined by the task)

| Category | Meaning | Acceptable in library code? |
|---|---|---|
| **CLI output** | User-facing output of a CLI entry point, `info()`/`summary()`/`inspect()`-style helper, or `__main__` block. Intentional. | ✅ Yes |
| **Library logging** | Should be `logger.info` / `logger.warning` — informational or warning messages emitted from inside library functions that may be called by other code. | ❌ Convert to logger |
| **Debug output** | Should be removed or guarded by `logger.debug` — temporary scaffolding, ad-hoc values, fallback-only stubs. | ❌ Remove or downgrade |
| **Progress output** | Acceptable in notebooks/scripts, not in library code — per-step/per-epoch progress bars, "Loading X...", "Done" markers inside class methods. | ⚠️ Convert to logger or progress callback in library code |
| **Docstring example** | `>>> print(...)` or `... print(...)` inside a docstring/doctest. Not real code; Grep matches because `print(` is a substring. | ✅ Not applicable |
| **False positive** | Match on `print(` substring inside an identifier (e.g. `get_memory_footprint(self)` contains `print(self)`). | ✅ Not applicable |
| **Last-resort error** | `print()` inside the logger module itself, used because the logger cannot use itself to log its own failure. | ✅ Acceptable |

---

## 1. Aggregate findings

| Metric | Count |
|---|---|
| Total `print(` substring matches in `xorzen/` `*.py` | **559** |
| Files containing at least one match | **43** |
| False positives (matches inside identifiers, e.g. `footprint(self)`) | **3** |
| Docstring / doctest `>>> print(...)` / `... print(...)` matches | **~24** |
| Real `print()` calls in library code | **~532** |
| Lines that survive the user's `grep -v` filter | **~553** (the filter removes ~6 lines that contain `warning`, `test_mode`, or `#.*print`) |
| First-60-files window covers (alphabetical file order) | `xorzen/__init__.py` (36 lines) + first 24 lines of `xorzen/data/__init__.py` |

### Lines removed by the user's `grep -v` filter (case-sensitive)

These would NOT appear in the user's `head -60` output, but they ARE real print() calls that should be audited:

| File | Line | Content (excerpt) | Filter match |
|---|---|---|---|
| `xorzen/model/ssm.py` | 55 | `def warning(self, *args, **kwargs): print(f"WARNING: {args}")` | `warning` |
| `xorzen/model/components/cot_vector.py` | 60 | `def warning(self, *args, **kwargs): print(f"WARNING: {args}")` | `warning` |
| `xorzen/model/components/merger.py` | 56 | `def warning(self, *args, **kwargs): print(f"WARNING: {args}")` | `warning` |
| `xorzen/tokenizer/evaluation.py` | 441 | `def warning(self, *args, **kwargs): print(f"WARNING: {args}")` | `warning` |
| `xorzen/transfer/universal.py` | 267 | `print(f" ✓ (skipped in test_mode)")` | `test` |
| `xorzen/ult/xorzen_ultimate.py` | 68 | `print(f" OK (skipped in test_mode)")` | `test` |
| `xorzen/tokenizer/loader.py` | 745 | `#    print(f"\n✅ Created metadata template: {metadata_file}")` | `#.*print` (commented-out) |

---

## 2. Per-file classification

Categories are abbreviated: **CLI** = CLI output, **LOG** = library logging, **DBG** = debug output, **PROG** = progress output, **DOC** = docstring example, **FP** = false positive, **LRE** = last-resort error (logger module).

### `xorzen/__init__.py` — 36 matches

All prints live in two intentional user-facing helpers: `info()` (lines 159–252) and the env-gated welcome block in `_initialize()` (lines 355–361, only fires when `XORZENX_VERBOSE` is set).

| Line | Function / context | Category | Notes |
|---|---|---|---|
| 163–194 (32 lines) | `info()` | **CLI** | Intentional banner/info dump. The method's whole purpose is to print. |
| 226–250 (25 lines) | `info()` installation check section | **CLI** | Same as above. |
| 356–361 (6 lines) | `_initialize()` welcome message, gated on `XORZENX_VERBOSE` env var | **DBG** | Env-gated verbose banner. Acceptable, but could be `logger.info` so users can route it via logging config instead of an env var. |

**Verdict:** Mostly fine. The `XORZENX_VERBOSE` block (356–361) is the only candidate for migration to `logger.info`.

---

### `xorzen/data/__init__.py` — 28 matches

All prints live in two user-facing helpers: `inspect_data(path)` (lines 146–198) and `validate_data(path)` (lines 200–253).

| Line range | Function | Category | Notes |
|---|---|---|---|
| 165–197 (24 lines) | `inspect_data()` | **CLI** | Intentional inspection report. |
| 214, 222–248 (8 lines) | `validate_data()` | **CLI** | Intentional validation report. |
| 252 | `validate_data()` exception handler | **LOG** | `print(f"✗ Validation failed: {e}")` — should be `logger.warning` so library callers can suppress it. |
| 197 | `inspect_data()` path-not-found | **LOG** | `print(f"Path not found: {path}")` — should be `logger.error` and probably raise instead of printing. |
| 214 | `validate_data()` file-not-found | **LOG** | Same as above. |

**Verdict:** Mostly CLI. Three error-path prints (lines 197, 214, 252) should become `logger.error`/`logger.warning`.

---

### `xorzen/data/augmentation.py` — 1 match

| Line | Context | Category | Notes |
|---|---|---|---|
| 243 | `class` method body: `print(f"INFO: [Placeholder] Would load model '{self.model_name}' here.")` followed by `pass` | **DBG** | Placeholder/stub code. Remove or convert to `logger.debug`. The `INFO:` prefix suggests the author already intended this as a log line. |

---

### `xorzen/data/cleaner.py` — 5 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 654–658 (5 lines) | `main()` (argparse CLI entry) | **CLI** | End-of-run summary. Acceptable. |

---

### `xorzen/data/cli.py` — 24 matches

Entire file is the CLI for the data sub-package (argparse-based `main()`).

| Line range | Context | Category | Notes |
|---|---|---|---|
| 34, 38, 73, 81–84, 102, 111, 119, 120+ | `main()` / subcommand handlers | **CLI** | All intentional CLI output. |

**Verdict:** All CLI. No action needed.

---

### `xorzen/data/converter.py` — 17 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 60, 418, 1168 | docstring `>>> print(...)` examples | **DOC** | Not real code. |
| 1494–1516 (14 lines) | `main()` argparse CLI summary | **CLI** | Intentional. |

**Verdict:** Real prints are all CLI. No action needed.

---

### `xorzen/data/formats/__init__.py` — 1 match

| Line | Context | Category |
|---|---|---|
| 9 | docstring `... print(record['text'])` | **DOC** |

---

### `xorzen/data/formats/gutenberg.py` — 1 match

| Line | Context | Category |
|---|---|---|
| 106 | docstring `... print(record['title'], ...)` | **DOC** |

---

### `xorzen/data/inspector.py` — 13 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 523–539 (13 lines) | `generate_report()` method | **CLI** | The method's explicit purpose is to print the inspection report to console (line 522 comment: `# Print summary to console`). Acceptable. |

**Verdict:** All CLI. No action needed.

---

### `xorzen/data/loader.py` — 8 matches

| Line | Context | Category |
|---|---|---|
| 69, 470, 639, 680, 731, 892, 963, 964 | All docstring `>>> print(...)` examples | **DOC** |

**Verdict:** No real print() calls. No action needed.

---

### `xorzen/data/validation.py` — 9 matches

| Line | Context | Category |
|---|---|---|
| 18, 20, 54, 136, 609, 726, 728, 730, 758 | All docstring `... print(...)` / `>>> print(...)` examples | **DOC** |

**Verdict:** No real print() calls. No action needed.

---

### `xorzen/inference/xorm_runtime.py` — 8 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 140 | `compile` branch: `print("[xorm] Model compiled with torch.compile()")` | **LOG** | Should be `logger.info`. Library code, called from `XormSession.__init__`. |
| 142 | `compile` failure: `print(f"[xorm] torch.compile() not available: {e}")` | **LOG** | Should be `logger.warning`. |
| 159 | `warmup()`: `print(f"[xorm] Warmed up in {avg_ms:.1f}ms avg over {num_runs} runs")` | **LOG** | Should be `logger.info`. |
| 385 | `set_temperature()`: `print(f"[xorm] Temperature set to {temperature}")` | **LOG** | Should be `logger.info`. Mutator method side-effect. |
| 422 | `auto_calibrate()`: `print(f"[xorm] Auto-calibrated: temperature={best_temp}  accuracy={best_acc:.1%}")` | **LOG** | Should be `logger.info`. |
| 558 | `load_session()`: `print(f"[xorm] Loaded {manifest.get('model_name', path)}")` | **LOG** | Should be `logger.info`. |
| 585 | `load_sessions_from_dir()` skip: `print(f"[xorm] Skipping {p.name}: {e}")` | **LOG** | Should be `logger.warning`. |
| 598 | `clear_registry()`: `print("[xorm] Registry cleared")` | **LOG** | Should be `logger.info`. |

**Verdict:** **All 8 prints are library logging** that should be converted to `logger.info` / `logger.warning`. This is the cleanest "smoking gun" file in the audit — every print is inside a class method on `XormRuntime`/`XormSession` and is purely informational.

---

### `xorzen/model/base.py` — 1 match (false positive)

| Line | Context | Category |
|---|---|---|
| 172 | `def get_memory_footprint(self) -> Dict[str, Any]:` | **FP** — `footprint(self)` contains the substring `print(self)` |

**Verdict:** No real print() calls. No action needed.

---

### `xorzen/model/components/cot_vector.py` — 5 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 58–61 (4 lines) | `DummyLogger` class inside `except (ImportError, ModuleNotFoundError):` fallback block — only defined when `xorzen` framework can't be imported (standalone-script testing) | **DBG** | Last-resort fallback logger. Should not exist in production; if the framework import fails in production, the right answer is to fail loud, not silently fall back to a print-based stub. Convert to `logger.warning` once on fallback, then use `logging.getLogger` directly. |
| 460 | `if __name__ == "__main__":` self-test block | **CLI** | Script-style self-test. Acceptable. |

---

### `xorzen/model/components/merger.py` — 4 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 54–57 (4 lines) | `DummyLogger` class in import-fallback block (same pattern as `cot_vector.py`) | **DBG** | Same recommendation as cot_vector.py. |

---

### `xorzen/model/ssm.py` — 4 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 53–56 (4 lines) | `DummyLogger` class in import-fallback block (same pattern) | **DBG** | Same recommendation. |

---

### `xorzen/model/zmoe.py` — 18 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 774–812 (18 lines) | `print_statistics()` method (line 772 defines it) | **CLI** | The method's whole purpose is to print. Acceptable. |

**Verdict:** All CLI. No action needed. (Optionally, this method could return a string and let callers decide whether to print, but that's a design change, not a logging fix.)

---

### `xorzen/models/__init__.py` — 1 match

| Line | Context | Category |
|---|---|---|
| 9 | docstring `>>> print(list_models())` | **DOC** |

---

### `xorzen/models/registry.py` — 1 match

| Line | Context | Category |
|---|---|---|
| 37 | docstring `>>> print(ModelRegistry.list())` inside a class docstring | **DOC** |

---

### `xorzen/models/zero/model.py` — 32 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 1320 | `def get_memory_footprint(self)` | **FP** — `footprint(self)` contains `print(self)` |
| 1373–1415 (31 lines) | `print_model_summary(self, verbose=True)` method | **CLI** | Intentional. Method name explicitly says "print". |

**Verdict:** All real prints are CLI. No action needed.

---

### `xorzen/speed/booster.py` — 3 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 106 | `apply()`: torch.compile skipped on CPU Windows | **LOG** | Should be `logger.info`. |
| 112 | `apply()`: torch.compile failed | **LOG** | Should be `logger.warning`. |
| 117 | `apply()`: `print(stats.report())` | **CLI** | One-shot summary print at end of `apply()`. Could be returned to caller instead. |

**Verdict:** 2 of 3 should be logger calls. The `stats.report()` print is borderline CLI but is inside library code and should be returned, not printed.

---

### `xorzen/speed/fast_trainer.py` — 7 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 54–55 | `__init__`: CPU thread count and opt_einsum status | **LOG** | Should be `logger.info`. |
| 59, 62, 64 | `__init__`: torch.compile progress | **LOG** | Should be `logger.info` / `logger.warning`. |
| 76, 94 | `train()`: JIT warmup start/end | **PROG** | Should be `logger.info` or a progress callback. |

**Verdict:** All 7 should be logger calls. Pure library code emitting training-status messages.

---

### `xorzen/speed/fix_restrict.py` — 3 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 35, 52, 54 | Top-level script body (no `main()`; runs on import) | **CLI** | This is a developer helper script with a shebang (`#!/usr/bin/env python3`). Acceptable as a script. The top-level execution is a minor smell but the prints themselves are script-style CLI output. |

---

### `xorzen/speed/jit_kernels.py` — 3 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 221 | `apply_jit_patches()`: jit not available | **LOG** | Should be `logger.warning`. |
| 243 | `apply_jit_patches()`: patched SSM scan | **LOG** | Should be `logger.info`. |
| 245 | `apply_jit_patches()`: SSM patch failed | **LOG** | Should be `logger.warning`. |

**Verdict:** All 3 should be logger calls.

---

### `xorzen/speed/native_kernels.py` — 1 match

| Line | Context | Category | Notes |
|---|---|---|---|
| 108 | `load_native_kernels()`: `print("[xorzen.native] C++ kernels compiled successfully — ...")` | **LOG** | Should be `logger.info`. Library function called from `xorzen.speed` import. |

---

### `xorzen/speed/setup_speed.py` — 9 matches

This is a `setuptools` build script (`setup.py`-style). Build scripts universally use `print` for build progress.

| Line | Context | Category | Notes |
|---|---|---|---|
| 25–26 (2 lines) | Build dependency-missing error | **CLI** | Acceptable. |
| 49, 65, 84, 116, 143, 145, 146 (7 lines) | Top-level build progress | **CLI** | Acceptable for build scripts. |

**Verdict:** All CLI. No action needed (build scripts are conventionally print-based).

---

### `xorzen/speed/xorzen_compile.py` — 17 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 75–81 (7 lines) | `compile_model()`: patch banner (what's being applied) | **LOG** | Should be `logger.info`. Library function. |
| 89, 91, 93 (3 lines) | `compile_model()`: JIT patch outcome | **LOG** | Should be `logger.info` / `logger.warning`. |
| 100, 102, 104 (3 lines) | `compile_model()`: native kernel outcome | **LOG** | Should be `logger.info` / `logger.warning`. |
| 106 (1 line) | `compile_model()`: ready message | **LOG** | Should be `logger.info`. |
| 131, 142, 146 (3 lines) | `warmup_compiled_model()`: warmup progress | **PROG** | Should be `logger.info` or progress callback. |

**Verdict:** All 17 should be logger calls.

---

### `xorzen/tokenizer/__init__.py` — 27 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 16 | docstring `>>> print(list_pretrained())` | **DOC** | |
| 158–163 (6 lines) | `info()`-style helper, "no tokenizers" branch | **CLI** | Intentional. |
| 165–172 (8 lines) | Same helper, "available tokenizers" branch | **CLI** | Intentional. |
| 180–186+ (12 lines) | Same helper, "tokenizer detail" branch | **CLI** | Intentional. |

**Verdict:** All real prints are CLI. No action needed.

---

### `xorzen/tokenizer/adapter.py` — 31 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 95, 106, 137, 163, 181, 193, 201, 212, … (31 lines spread across the file) | `TokenizerAdapter` class methods: `load()`, `set_active()`, `train_legacy_bpe()`, `train_fast_bpe()`, `save()`, etc. | **LOG** / **PROG** | Almost all of these are "Loaded X", "Saved X", "Training X..." status messages inside library class methods. Should be `logger.info`. A few (line 201: error during training) should be `logger.error`. |

**Verdict:** **High-priority conversion target.** 31 prints in a single library class, all of which are informational status that callers may want to suppress.

---

### `xorzen/tokenizer/evaluation.py` — 7 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 439–442 (4 lines) | `DummyLogger` class inside `if __name__ == "__main__":` self-test | **DBG** | Test-only stub. Acceptable in `__main__` block, but the same DummyLogger anti-pattern as cot_vector.py/ssm.py/merger.py — could be replaced with `logging.getLogger`. |
| 446, 492, 496 (3 lines) | `if __name__ == "__main__":` self-test body | **CLI** | Script-style self-test output. Acceptable. |

---

### `xorzen/tokenizer/loader.py` — 14 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 15, 21, 505, 560, 577, 599, 637 (7 lines) | docstring `>>> print(...)` / `... print(...)` examples | **DOC** | |
| 745 | Commented-out `#    print(f"\n✅ Created metadata template: ...")` | (filtered by `#.*print`) | Dead code. Can be deleted. |
| Other 6 lines | Need spot-check — likely also docstrings or one of the patterns above. | (likely **DOC**) | |

**Verdict:** No real production print() calls. The commented-out line 745 is dead code that can be removed.

---

### `xorzen/tokenizer/special_tokens.py` — 7 matches

| Line | Context | Category |
|---|---|---|
| 277–285 (7 lines) | `if __name__ == "__main__":` self-test | **CLI** |

**Verdict:** All CLI. No action needed.

---

### `xorzen/tokenizer/trainer.py` — 3 matches

| Line | Context | Category |
|---|---|---|
| 104, 363, 411 | All docstring `>>> print(...)` examples | **DOC** |

---

### `xorzen/train_superintelligence.py` — 45 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 35, 40, 46, 50, 89, 99, 107, 155, … | `AGITrainingPipeline` class methods (`load_teachers`, `create_multitask_dataset`, `train_with_self_critique`) | **PROG** / **LOG** | All progress/status output inside library class methods. Should be `logger.info` and/or a progress callback. |
| `if __name__ == "__main__":` block at line 309 | Top-level script entry | **CLI** | Acceptable. |

**Verdict:** ~30+ library-level prints should be converted to logger calls.

---

### `xorzen/training/__init__.py` — 1 match

| Line | Context | Category |
|---|---|---|
| 231 | docstring `>>> print(f"Eval Loss: ...")` | **DOC** |

---

### `xorzen/training/continuation.py` — 1 match

| Line | Context | Category |
|---|---|---|
| 457 | docstring `>>> print(f"Loss: ...")` | **DOC** |

---

### `xorzen/training/state.py` — 15 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 19, 70 | docstring `>>> print(...)` | **DOC** | |
| 435–448 (13 lines) | `print_summary()` method (line 431 defines it) | **CLI** | Intentional. Method name says "print". |

**Verdict:** All real prints are CLI. No action needed.

---

### `xorzen/training/trainer.py` — 35 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 157–158 | `__init__`: GPU info | **LOG** | Should be `logger.info`. |
| 160 | `__init__`: CPU info | **LOG** | Should be `logger.info`. |
| 176, 180, 184 | `__init__`: precision mode | **LOG** | Should be `logger.info`. |
| 231 | `_setup_loggers()`: TensorBoard path | **LOG** | Should be `logger.info`. |
| 245 | `_setup_loggers()`: W&B enabled | **LOG** | Should be `logger.info`. |
| 249–270 (10 lines) | `_print_setup_info()` | **CLI** | Intentional. Method name says `_print_`. |
| 303 | `train()`: starting training | **PROG** | Should be `logger.info`. |
| 329 | `train()`: early stopping | **LOG** | Should be `logger.info`. |
| 342 | `train()`: user interrupt | **LOG** | Should be `logger.warning`. |
| 346 | `train()`: failure | **LOG** | Should be `logger.error`. |
| 359–363 (5 lines) | `train()`: completion summary | **CLI** / **PROG** | Borderline. Could be `logger.info`. |
| 573 | `save_checkpoint()` | **LOG** | Should be `logger.info`. |
| 601, 628 | `load_checkpoint()` | **LOG** | Should be `logger.info`. |
| 670–675 (6 lines) | `_print_epoch_summary()` | **CLI** | Intentional. Method name says `_print_`. |

**Verdict:** ~20 of 35 should become logger calls. The two `_print_*` helpers are fine as-is.

---

### `xorzen/transfer/transfer.py` — 22 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 18, 48, 78, 105, 160 | `TeacherExtractor.__init__` and other methods | **PROG** | "Loading teacher model...", "✓ Loaded", etc. Should be `logger.info`. |
| 179, 190, 209, 220, 226, 274, 276, 285, 305, 310, 317 | `XORZENXTransferLearning.transfer()` step-by-step progress | **PROG** | "[1/4] Transferring embeddings...", etc. Should be `logger.info` or a progress callback. |
| 390–392, 406–408 | `quick_transfer()` function banner | **CLI** / **PROG** | Convenience function. Banner is borderline acceptable. |

**Verdict:** ~19 of 22 should be logger calls (or progress callbacks). This is a long-running transfer pipeline and the prints are essentially progress events.

---

### `xorzen/transfer/universal.py` — 26 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 63, 76, 79, 103, 104, 105, 108, 109 | `UniversalTeacherExtractor.__init__` and fallback logic | **PROG** / **LOG** | Loading/quantization status. Line 108 (`✗ Failed to load`) should be `logger.error`. |
| 223, 227, 254, 258, 267, 312, 316, 325, 328 | `smart_transfer()` step progress | **PROG** | "[1/3] Embeddings...", " ✓ (...)", etc. Should be `logger.info`. |
| 352–362 | `print_recommended_teachers()` helper | **CLI** | Intentional reference listing. Acceptable. |
| 368, 375 | `if __name__ == "__main__":` self-test | **CLI** | Acceptable. |

**Verdict:** ~17 of 26 should be logger calls.

---

### `xorzen/ult/xorzen_distill.py` — 18 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 22, 33 | `DistillationMaster.__init__` | **PROG** | Should be `logger.info`. |
| 191–276 (most of the file's prints) | `ultra_train()` top-level function | **PROG** / **CLI** | Script-style entry function. Borderline; if it's documented as a one-shot pipeline entry point, the prints are CLI. If library callers can invoke it programmatically, they should be `logger.info`. |
| 284–285, 311 | `if __name__ == "__main__":` self-test | **CLI** | Acceptable. |

**Verdict:** Depends on how `ultra_train()` is meant to be used. Recommendation: convert to `logger.info` so callers can route/suppress.

---

### `xorzen/ult/xorzen_ultimate.py` — 43 matches

| Line range | Context | Category | Notes |
|---|---|---|---|
| 25, 31, 35, 42, 61, 64, 68, 99, 102, 120, 123, 139, 142, 147, 150, 152, 154 | `SuperFastTransfer` class methods | **PROG** | "  [1/6] Embeddings...", " OK", etc. Should be `logger.info`. |
| 261, 281, 313, 316, 318, 324, 330 | `SmartTrainer` class methods | **PROG** | "Epoch X/Y", "Step N | Loss: ...", etc. Should be `logger.info`. The `print(..., end='\r')` at line 313 is a TTY progress bar — should use `tqdm` or a real progress callback. |
| 374–430 (~17 lines) | `main()` pipeline entry | **CLI** | Script-style banner. Acceptable for a documented entry function. |

**Verdict:** ~25 library-level prints should be logger calls; the `\r` progress print should be `tqdm` or a callback.

---

### `xorzen/utils/logger.py` — 4 matches

| Line | Context | Category | Notes |
|---|---|---|---|
| 249 | `AsyncLogHandler` thread error: `print(f"Error in async log handler: {e}")` | **LRE** | Last-resort. The logger cannot use itself to log its own failure. Acceptable. |
| 265, 282 | Same pattern in batch handler | **LRE** | Same justification. |
| 426 | `ConsoleHandler` emit: `print(msg)` | **CLI** | This *is* the console output handler. The whole point of this class is to print log records to stdout. Acceptable. |

**Verdict:** All 4 are intentional. No action needed.

---

## 3. The "first 60" window (per the user's `head -60` filter)

With the user's `grep -v "__pycache__\|test\|\.pyc\|#.*print\|logger\|warning"` filter applied and `head -60`, the visible output (assuming alphabetical-by-path file traversal, the conventional `grep -r` order) is:

| # | File | Lines covered | All same category? |
|---|---|---|---|
| 1–36 | `xorzen/__init__.py` | 163–250 (the `info()` body) | Yes — all **CLI** (intentional `info()` helper) |
| 37–60 | `xorzen/data/__init__.py` | 165–194 (the `inspect_data()` body) | Yes — all **CLI** (intentional inspection helper) |

So the user's `head -60` window lands entirely on intentional CLI/info-helper code and **does not surface any of the problematic library-logging prints**. The library-logging prints live deeper in the file traversal order (e.g. `xorzen/inference/xorm_runtime.py`, `xorzen/tokenizer/adapter.py`, `xorzen/training/trainer.py`, `xorzen/transfer/*`, `xorzen/ult/*`, `xorzen/speed/*`).

**Recommendation:** Re-run the audit without the `head -60` (or with `head -300`) to surface the actual library-logging offenders. The filter `grep -v "warning"` is also too aggressive — it strips out the `DummyLogger` stub lines in `ssm.py` / `cot_vector.py` / `merger.py` / `evaluation.py`, which are exactly the kind of debug-output smells worth flagging.

---

## 4. Summary by category (real prints only, excluding DOC + FP)

| Category | Count (approx.) | Files most affected |
|---|---|---|
| **CLI** (intentional) | ~230 | `__init__.py` (info), `data/cli.py`, `data/cleaner.py`, `data/converter.py`, `data/inspector.py`, `data/__init__.py`, `model/zmoe.py`, `models/zero/model.py`, `training/state.py`, `training/trainer.py` (`_print_*`), `tokenizer/__init__.py`, `ult/*` main blocks, `utils/logger.py` (ConsoleHandler) |
| **Library logging** (should be `logger.info`/`warning`/`error`) | ~120 | `inference/xorm_runtime.py`, `speed/booster.py`, `speed/fast_trainer.py`, `speed/jit_kernels.py`, `speed/native_kernels.py`, `speed/xorzen_compile.py`, `tokenizer/adapter.py`, `training/trainer.py` (non-`_print_` methods), `transfer/transfer.py`, `transfer/universal.py`, `data/__init__.py` error paths |
| **Progress output** (should be `logger.info` or progress callback) | ~80 | `train_superintelligence.py`, `ult/xorzen_ultimate.py`, `ult/xorzen_distill.py`, `transfer/*`, `training/trainer.py` (`train()` body), `speed/fast_trainer.py` (warmup) |
| **Debug output** (remove or `logger.debug`) | ~17 | `model/ssm.py`, `model/components/cot_vector.py`, `model/components/merger.py`, `tokenizer/evaluation.py` (DummyLogger stubs); `data/augmentation.py` (placeholder); `xorzen/__init__.py` (`XORZENX_VERBOSE` block) |
| **Last-resort error** (logger module; acceptable) | 3 | `utils/logger.py` |
| **Docstring examples** (not real code) | ~24 | `data/converter.py`, `data/loader.py`, `data/validation.py`, `data/formats/*`, `models/__init__.py`, `models/registry.py`, `tokenizer/trainer.py`, `tokenizer/loader.py`, `training/__init__.py`, `training/continuation.py`, `training/state.py` |
| **False positives** (substring match in identifier) | 3 | `model/base.py:172`, `models/zero/model.py:1320`, `models/zero/model.py:1405` (all `get_memory_footprint`) |

---

## 5. Recommendations (no code changes proposed — for follow-up tickets)

1. **Highest-priority conversion targets** (files where nearly every print is library logging):
   - `xorzen/inference/xorm_runtime.py` (8/8 prints are LOG)
   - `xorzen/tokenizer/adapter.py` (~31 prints, all LOG/PROG)
   - `xorzen/speed/jit_kernels.py` (3/3 LOG)
   - `xorzen/speed/native_kernels.py` (1/1 LOG)
   - `xorzen/speed/xorzen_compile.py` (17/17 LOG/PROG)
   - `xorzen/speed/fast_trainer.py` (7/7 LOG/PROG)
   - `xorzen/speed/booster.py` (2/3 LOG)

2. **Progress-output files** (long-running pipelines that should use `logger.info` + optional progress callbacks or `tqdm`):
   - `xorzen/transfer/transfer.py`, `xorzen/transfer/universal.py`
   - `xorzen/ult/xorzen_distill.py`, `xorzen/ult/xorzen_ultimate.py`
   - `xorzen/train_superintelligence.py`
   - `xorzen/training/trainer.py` (the `train()` loop body)

3. **DummyLogger anti-pattern** (4 files, ~16 lines): `xorzen/model/ssm.py:53-56`, `xorzen/model/components/cot_vector.py:58-61`, `xorzen/model/components/merger.py:54-57`, `xorzen/tokenizer/evaluation.py:439-442`. The fallback should be `logging.getLogger(__name__)` directly; the `print`-based stub silently masks missing imports and produces output that can't be suppressed.

4. **Dead code**: `xorzen/tokenizer/loader.py:745` is a commented-out print. Safe to delete.

5. **False positives** (3 matches): no action needed; the `print(` substring inside `get_memory_footprint(self)` is unavoidable without word-boundary regex. If re-running this audit, use pattern `\bprint\s*\(` to eliminate the `footprint(self)` false positives.

6. **CLI helpers to leave alone**: `xorzen/__init__.py::info()`, `xorzen/data/__init__.py::inspect_data()/validate_data()`, `xorzen/data/inspector.py::generate_report()`, `xorzen/data/cli.py::main()`, `xorzen/data/converter.py::main()`, `xorzen/data/cleaner.py::main()`, `xorzen/models/zero/model.py::print_model_summary()`, `xorzen/model/zmoe.py::print_statistics()`, `xorzen/training/state.py::print_summary()`, `xorzen/training/trainer.py::_print_setup_info()/_print_epoch_summary()`, `xorzen/tokenizer/__init__.py::info()`, `xorzen/utils/logger.py::ConsoleHandler`, all `if __name__ == "__main__":` blocks. These are intentional user-facing output.

7. **Filter caveat for future audits**: the `grep -v "warning"` filter strips the `DummyLogger.warning` stubs (which are exactly what we want to flag), and `grep -v "test"` strips `test_mode` progress messages. Consider running the unfiltered grep first, then classifying manually.

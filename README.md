# XorZen Framework & AI Ecosystem

<p align="center">
  <strong>Advanced Hybrid Transformer × MoE × SSM Framework</strong><br>
  <em>Genuine per-token conditional compute across depth, width, pathway, and experts</em>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Core Principle](#core-principle)
  - [Forward Pass](#forward-pass)
  - [Key Components](#key-components)
- [Model Variants](#model-variants)
- [FRAZIYM Ecosystem](#fraziym-ecosystem)
  - [XorZen Core](#xorzen-core)
  - [Greed — Context Compression Models](#greed--context-compression-models)
  - [Jima — Agentic Model Family](#jima--agentic-model-family)
  - [xorvec-data — LLM Fine-Tuning Pipeline](#xorvec-data--llm-fine-tuning-pipeline)
  - [libcompact — C++ Deployment Library](#libcompact--c-deployment-library)
  - [Kage — Multimodal Reasoning (Planned)](#kage--multimodal-reasoning-planned)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Tokenizer](#tokenizer)
- [Data Pipeline](#data-pipeline)
- [Performance & Benchmarks](#performance--benchmarks)
  - [v0.4 Improvements](#v04-improvements)
  - [Conditional Compute Verification](#conditional-compute-verification)
  - [Scaling Law](#scaling-law)
  - [Ablation Study](#ablation-study)
- [Testing](#testing)
- [Audit & Verification](#audit--verification)
- [Reports](#reports)
- [Roadmap](#roadmap)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

**XorZen** is a hybrid deep learning framework that fuses Transformer attention, Mixture-of-Experts (MoE), and State-Space Models (SSM) into a single architecture with **genuine per-token conditional compute**. Rather than computing everything and masking outputs, XorZen dynamically routes each token through a selectively activated subset of layers, widths, pathways, and experts — saving FLOPs while maintaining (or improving) quality.

**Key design decision:** XORZEN uses **SSM fixed-size context**, not KV cache. The SSM pathway maintains a fixed-size recurrent state (`state_dim`, default 16) that carries information across the sequence with O(1) memory — no growing KV cache, no linear context cost. This enables bounded-memory inference regardless of sequence length.

The framework is the foundation for the broader **FRAZIYM AI Ecosystem**, which includes context compression models (Greed), agentic inference models (Jima), a data processing pipeline (xorvec-data), and a C++ deployment library (libcompact).

**Current version:** v1.0.1

---

## Architecture

### Core Principle

> *"Never compute something merely to multiply its output by zero later."*

Unlike architectures that run all computations and then blend or mask results (the "compute-then-mask" anti-pattern), XorZen uses **genuine sparse dispatch** — unselected pathways, experts, and width slices simply are never called. This produces real FLOP savings that scale with model size.

### Forward Pass

```
input_ids [B, T]
    |
token_embedding + position_embedding
    |
hidden_states [B, T, H]
    |
AdaptiveRouter -> RoutingDecision (depth_mask, path_probs, expert_indices, ...)
    |
for each HASS block:
    |  (inference:  gather active tokens via depth_mask, run block, scatter back)
    |  (training:    STE mask blend for differentiability)
    |  (pathway:     sparse_pathway_dispatch - only top-k pathways called)
    |  (FFN:         SlicedFFN with width_idx - genuine width slicing)
    |
hidden_states [B, T, H]
    |
flatten -> MoE (top-k expert dispatch, disk-sharded)
    |
MergerGate (fuses HASS + MoE + CoT)
    |
final_norm -> lm_head -> logits [B, T, V]
```

### Key Components

| Component | Description | File |
-----------|-------------|------|
| **AdaptiveRouter** | 4-axis per-token routing (depth, width, pathway, expert) with Gumbel-softmax at training, hard argmax at inference | `model/components/routing.py` |
| **HASSBlock** | Hybrid Attention-Shard Switch with 3 parallel pathways: Local Attention, Low-Rank Global, and SSM | `model/components/hass_block.py` |
| **SlicedFFN** | Genuine per-token width sparsity via nested weight slicing — FLOPs scale linearly with selected width | `model/components/sliced_ffn.py` |
| **SSMPathway** | Diagonal state-space model with ZOH discretization, input-dependent dt/B/C, and chunked scan | `model/components/ssm_scan.py` |
| **ShardedExpertFabric** | Disk-sharded MoE with LRU cache, top-k routing, and SwiGLU expert FFNs | `model/zmoe.py` |
| **MergerGate** | Gated fusion of HASS output + MoE output + latent CoT vector | `model/components/merger.py` |
| **SPPQ** | Scalar Post-Training Quantization engine | `utils/sppq.py` |
| **CoT Vector** | 6-component latent reasoning vector (frozen during pre-training, unfrozen for fine-tuning via `model.enable_cot()`) | `model/components/cot_vector.py` |

---

## Model Variants

XorZen provides 8 model sizes from tiny to 7B, all sharing the same architecture and differing only in hyperparameters. Parameter counts below are **measured from the actual implementation** (not estimated):

| Variant | Hidden | Layers | Vocab | Context | Experts | Top-K | Width Choices | Total Params | Trainable Params |
|---------|--------|--------|-------|---------|---------|-------|---------------|-------------|-----------------|
| `zero_tiny_23k` | 8 | 1 | 8 | 32 | 1 | 1 | (8,) | 34,728 | 30,531 |
| `zero_1M` | 64 | 3 | 6,765 | 128 | 2 | 1 | (64,) | 1,053,711 | 953,838 |
| `zero_10M` | 192 | 6 | 10,000 | 512 | 8 | 2 | (96, 192) | 10,865,977 | 10,099,928 |
| `zero_50M` | 256 | 10 | 10,000 | 1,024 | 43 | 2 | (128, 256) | 50,157,888 | 48,825,279 |
| **`zero_277M`** (flagship) | 512 | 13 | 33,898 | 1,024 | 64 | 2 | (256, 512) | 276,490,648 | 274,633,239 |
| `zero_500M` | 640 | 16 | 64,563 | 8,192 | 69 | 2 | (320, 640) | 499,230,752 | 491,230,175 |
| `zero_1_3B` | 896 | 24 | 68,003 | 8,192 | 64 | 2 | (448, 896) | ~1,000,000,886 | ~995,000,000 |
| `zero_7B` | 1,792 | 48 | 98,425 | 8,192 | 116 | 2 | (896, 1,792) | ~7,000,000,466 | ~6,950,000,000 |

> **Note:** `zero_1_3B` and `zero_7B` param counts are from config (too large to instantiate on CPU). All others are measured. Trainable params exclude frozen CoT parameters (unfrozen via `model.enable_cot()` for fine-tuning). Position encoding is learned absolute (not RoPE).

Additional model families built on XorZen Core:

| Model | Architecture | Purpose | Status |
|-------|-------------|---------|--------|
| **zero_agentic_nano** | FlashSSM + GatedLinearAttention + RecursiveRouter + memory_vault | Compact recursive inference (self-critique + memory) | Implemented (C++ port in `xorzen_agentic.h`) |
| **zero_agentic_micro** | FlashSSM + GatedLinearAttention + RecursiveRouter + memory_vault | Consumer recursive inference | Implemented |
| **GreedModel** | Subclass of zeroModel (continuous feature_proj + classification head, 5 classes) | Conversation quality grading | Implemented (C++ port in `xorzen_greed.h`) |
| **Jima** | Spec-only (not in source code) | Agentic reasoning | Spec only — NOT implemented |
| **Kage** | Spec-only (not in source code) | Multimodal reasoning | Spec only — NOT implemented |

---

## FRAZIYM Ecosystem

DevNet is the monorepo for the FRAZIYM AI ecosystem. These projects share XorZen as their foundation and integrate into a full pipeline from raw data to deployed inference.

```
XorZen Core (Foundation)
    +-> libcompact  (C++ inference engine)
    +-> Greed       (trained context compression models)
    +-> Jima        (trained agentic reasoning models)
    +-> Zero/Agentic  (compact model variants with recursive inference)
    +-> Kage        (future multimodal reasoning - planned)

xorvec-data (Data Pipeline)
    +-> Input:  raw conversations (Claude, OpenAI, ShareGPT, etc.)
    +-> Output: clean training datasets
    +-> Feeds -> XorZen fine-tuning
```

### XorZen Core

The foundational framework. This repository (`xorzen/`) contains the complete implementation: model architecture, training loop, tokenizers, data loaders, and benchmarks. See the sections above for architecture details and usage.

### Greed — Context Compression Models

Specialized models for compressing different types of context in LLM inference pipelines. All built on the XorZen-GREED architecture (~80M shared parameters).

| Model | Purpose | Compression Ratio | Semantic Preservation |
-------|---------|-------------------|----------------------|
| **SPC** (System Prompt Compactor) | Score and compress system prompts (0-4 importance scale) | 80-90% reduction (keep 10-20%) | 85-95% |
| **CCC** (Chat Context Compactor) | Summarize and compress chat history | 70-80% reduction | 75-85% |
| **CRLC** (Code Run Log Compactor) | Collapse repeating error traces | 90-95% reduction | 100% issue identification |
| **CoT** (Latent CoT) | Internal reasoning state across layers | Frozen during pre-training | — |

Greed models are saved in the `.xorm` (XorZen Model Format) and loaded by `libcompact` for deployment.

### Jima — Agentic Model Family

Lightweight agentic models with explicit reasoning pathways for agent-as-reasoner architectures. Each layer includes three specialized heads:

- **Thought Head**: maintains a reasoning state vector
- **Action Head**: selects actions from a discrete set
- **Critique Head**: self-evaluates the reasoning quality

Jima models have been trained and registered, with gradient flow verified. They are designed for autonomous agent systems, action selection, and self-critique loops.

### xorvec-data — LLM Fine-Tuning Pipeline

A universal data pipeline for LLM fine-tuning: auto-detects input formats (Claude Markdown, OpenAI JSON, ShareGPT, Alpaca, JSONL, Gemini), cleans PII/duplicates, and exports to multiple training formats. Provides a FastAPI service, MCP server, and CLI.

**Status:** ~35% complete. Core parsers, cleaners, exporters, and API endpoints are implemented. Testing (Phase 7) and quality filtering (Phase 3) are remaining critical blockers.

### libcompact — C++ Deployment Library

C++/LibTorch port of the XorZen-GREED model for high-performance deployment. Implements the three context compactors (SPC, CCC, CRLC) with PyBind11 Python bindings.

**Status:** ~40% complete. Structural port and build system are done. Critical remaining work: correct RMSNorm implementation, Straight-Through Estimator for training, and `.xorm` model loader.

### Kage — Multimodal Reasoning (Planned)

A planned multimodal model combining a Vision Transformer encoder with the XorZen-GREED reasoning backbone. Target: anime/manga visual understanding and reasoning. **Status:** 0% — roadmap item only, no code exists yet.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Akik-Forazi/XorZen.git
cd XorZen

# Install in editable mode (picks up changes to xorzen/ immediately)
pip install -e .

# Optional: install dev dependencies (pytest, black)
pip install -e ".[dev]"
```

**Runtime dependencies** (declared in `pyproject.toml`):
`torch`, `transformers`, `tokenizers`, `sentencepiece`, `einops`, `tqdm`, `numpy`, `pydantic`, `psutil`

**Requires:** Python 3.10+

---

## Quick Start

```python
import xorzen

# --- Models ---
# Create the flagship model in test mode (uses dummy experts for fast CPU smoke tests)
m = xorzen.zero_277M(test_mode=True)
print(f"Models available: {xorzen.list_models()}")

# Create any model variant
m_tiny = xorzen.zero_1M(test_mode=True)
m_10m = xorzen.zero_10M(test_mode=True)

# zero Agentic variants
m_agentic = xorzen.zero_agentic_nano()

# --- Tokenizer ---
# Load a pretrained tokenizer (65k BPE variant)
tk = xorzen.load_pretrained("xorzen_agi_tokenizer_65k")
print(f"Vocab size: {tk.get_vocab_size()}")  # -> 65000

# Or the 10k variant
tk_small = xorzen.load_pretrained("zero_bpe_10k")

# --- SPPQ Quantization ---
from xorzen.utils.sppq import SPPQQuantizer, QuantizationConfig, QuantizationType
import torch

cfg = QuantizationConfig(
    bits=8,
    quantization_type=QuantizationType.SYMMETRIC,
    observe_iterations=2
)
q = SPPQQuantizer(m, cfg)
x = torch.randint(0, 1000, (2, 32))
with torch.no_grad():
    for _ in range(2):
        _ = m(x)
q.calibrate()
q.apply_quantization()
print("Quantized OK")

# --- Environment Info ---
xorzen.info()
```

---

## Configuration

All model behavior is controlled through the `ConfigFactory`:

```python
from xorzen.config import ConfigFactory, ModelSize

# Get a config for any model size
cfg = ConfigFactory.get_config(ModelSize.NANO_10M)

# v0.4 Architecture Flags (all default True except where noted)
cfg.use_sliced_ffn       = True   # Genuine per-token width sparsity in HASSBlock
cfg.width_div_weight     = 0.1    # Entropy regularization for width router
cfg.path_div_weight      = 0.2    # Entropy regularization for path router (was 0.1 in v0.3)
cfg.unify_load_balance   = True   # Drop the double-counting model-level L2 loss
cfg.cost_aware_routing   = True   # Bias routing toward compute_budget
cfg.compute_budget       = 1.0    # 1.0 = full compute, 0.25 = quarter compute

# To restore v0.3 behavior for ablation comparison:
cfg.update(
    use_sliced_ffn=False,
    width_div_weight=0.0,
    path_div_weight=0.1,
    unify_load_balance=False,
    cost_aware_routing=False,
)
```

---

## Training

```python
import xorzen

# Create model and load data
model = xorzen.zero_10M(test_mode=False)
train_data = xorzen.load_from_bin("data/train.bin")

# Train
xorzen.train(model, train_data, epochs=10)

# Or use the Trainer class directly
from xorzen.training import Trainer

trainer = Trainer(model, lr=3e-4, warmup_steps=1000)
trainer.fit(train_data, epochs=10, val_data=val_data)

# Continue from checkpoint
xorzen.continue_train(model, train_data, checkpoint="checkpoints/step_5000")
```

The training infrastructure supports:
- Mixed precision training (FP16/BF16)
- Gradient accumulation and clipping
- Learning rate scheduling with warmup
- Checkpoint versioning and resumption
- Curriculum learning support

---

## Tokenizer

XorZen includes a custom BPE tokenizer with pretrained vocabularies:

```python
import xorzen

# Load pretrained
tk = xorzen.load_pretrained("xorzen_agi_tokenizer_65k")

# List available pretrained tokenizers
print(xorzen.list_pretrained())

# Encode / Decode
tokens = tk.encode("Hello, world!")
text = tk.decode(tokens)

# Round-trip verified: decode(encode(x)).startswith(x)

# Train a new tokenizer
xorzen.train_tokenizer(
    files=["data/corpus.txt"],
    vocab_size=32000,
    output_dir="tokenizers/my_tokenizer"
)
```

---

## Data Pipeline

XorZen supports multiple input formats and efficient binary storage:

```python
import xorzen

# Convert raw text to training-ready binary format
xorzen.txt_to_bin("data/corpus.txt", "data/train.bin")
xorzen.json_to_bin("data/conversations.json", "data/train.bin")
xorzen.jsonl_to_bin("data/conversations.jsonl", "data/train.bin")

# Load for training
dataset = xorzen.load_from_bin("data/train.bin")

# Load from directory (auto-detects format)
dataset = xorzen.load_from_dir("data/")

# Validate data
xorzen.validate_data("data/train.bin")
```

The data module also includes a sharded dataset system, data augmentation, sampling strategies, and an inspector for analyzing dataset statistics.

---

## Performance & Benchmarks

### v0.4 Improvements

v0.4 made major architectural improvements over v0.3 based on ablation findings and audit results:

| # | Change | Evidence | Effect |
|---|--------|----------|--------|
| A | Replace `AdaptiveFFN` with `SlicedFFN` | Phase 5: SlicedFFN gives proportional FLOPs scaling | Genuine model-level width sparsity |
| B | Add `width_diversity_loss` | Phase 6: width router collapses to largest | Width router now uses smaller widths for easy tokens |
| C | Unify double-counting load-balance losses | Phase 15: v0.3 had two LB losses | Only standard Switch formula used |
| D | Cost-aware routing in `AdaptiveRouter` | Phase 18: ComputeController was dead code | Per-axis compute cost estimates bias routing |
| E | Bump `path_div_weight` to 0.2 | Phase 6: 0.1 collapses to 1 pathway | Pathway collapse broken in training mode |
| F | Remove `adaptive_halting.py` | Phase 18: not wired into model | Cleaner codebase |

**Measured impact** (200 training steps, H=32, 3 layers, 4 experts/top-2):

| Metric | v0.3 | v0.4 | Delta |
--------|------|------|-------|
| Final loss | 0.0132 | 0.0064 | **2x lower** |
| FLOPs | 27.6M | 21.3M | **-22.8%** |
| Runtime | 7.93 ms | 7.58 ms | **-4.5%** |
| Path entropy | 0.405 | 0.672 | +66% more diverse |

### Conditional Compute Verification

All four axes of conditional compute have been independently verified:

| Axis | What Was Tested | Result |
------|----------------|--------|
| **Width sparsity** | Forcing width=16 vs 32 changes model-level FFN FLOPs by exactly 0.500x (16/32) | PASS |
| **Depth sparsity** | Gather-scatter at inference reduces FLOPs by 31.8% (all vs none active) | PASS |
| **Pathway sparsity** | Forcing all tokens to SSM: local=0, low_rank=0, ssm=3 calls | PASS |
| **MoE top-k** | Forcing top-1 expert: exactly 1 expert used, 0 gradient to others | PASS |

### Scaling Law

Real parameter counts from instantiated variants reveal a scaling pattern where active parameter percentage decreases with scale (51-54% at >=50M variants), enabling a hypothesis that a 12B Xorzen could outperform a 60B dense model at equal compute — though this remains unvalidated at production scale.

| Variant | Total Params | Active % | Compute Efficiency |
---------|-------------|----------|-------------------|
| zero_10M | ~7.8M | 66.1% | 1.51x |
| zero_50M | ~17.1M | 54.4% | 1.84x |
| zero_277M | ~78.3M | 53.5% | 1.87x |
| zero_500M | ~165M | 53.8% | 1.86x |
| zero_1_3B | ~391M | 43.8% | 2.28x |

### Ablation Study

8 ablations at tiny scale (50 steps each) reveal which components contribute:

| Ablation | Final Loss | vs Baseline | Verdict |
----------|-----------|-------------|---------|
| Base | ~0.92 | — | Baseline |
| no_ssm | 0.92 | Same | Neutral |
| no_local | 0.94 | Same | Neutral |
| **no_low_rank** | **0.80** | **Better** | Low-rank hurts at tiny scale (router collapse) |
| **no_adaptive_width** | **0.83** | **Better** | Width routing adds noise when collapsed |
| **no_moe** | **0.88** | **Better** | MoE needs scale to help |
| no_balancing | 0.93 | Same | Neutral at 50 steps |

---

## Testing

```bash
# Core regression tests (run directly, not via pytest — these are scripts)
python tests/test_golden_parity.py         # 3 tests — golden reference determinism
python tests/test_ssm_scan_parity.py       # 53 tests — SSM scan chunk-safety
python tests/test_beam_search.py           # 6 tests — beam search regression
python tests/test_kv_cache.py              # 8 tests — KV cache correctness
python tests/test_routing_consistency.py   # 7 tests — routing noise position-indexing
python tests/test_training_routing.py      # 9 tests — STE gradient flow + CoT lifecycle

# Phase tests (pytest-compatible)
pytest tests/test_phase1_correctness.py -v  # 14 tests — SSM/MoE/routing correctness
pytest tests/test_phase2_sparsity.py -v     # 8 tests — sparse dispatch
pytest tests/test_phase4_v04.py -v          # 20 tests — v0.4 architecture

# C++ parity tests (requires g++ + LibTorch)
cd tests/cpp_parity/cpp && ./build.sh && cd .. && python compare.py
```

Current test count: **86+ tests** across 6 Python suites + C++ parity.

| File | Tests | Coverage |
|------|-------|----------|
| `test_golden_parity.py` | 3 | Golden reference determinism (40 tensors, byte-reproducible) |
| `test_ssm_scan_parity.py` | 53 | SSM scan chunk-safety + JIT parity |
| `test_beam_search.py` | 6 | Beam search regression |
| `test_kv_cache.py` | 8 | KV cache: prefill, incremental, batch isolation, generation |
| `test_routing_consistency.py` | 7 | Router noise position-indexing, batch independence |
| `test_training_routing.py` | 9 | STE gradient flow, CoT freeze/unfreeze, training step |
| `test_phase1_correctness.py` | 14 | SSM/MoE/routing correctness |
| `test_phase2_sparsity.py` | 8 | Sparse dispatch unit tests |
| `test_phase4_v04.py` | 20 | v0.4 architecture + conditional compute |
| C++ `parity_harness` | 27 | Python↔C++ component parity (27/27 PASS) |
| C++ `test_sliced_ffn_widths` | 4 | SlicedFFN multi-width parity (narrow/intermediate/max/mixed) |

---

## Audit & Verification

An adversarial architecture audit (claim-by-claim re-derivation from source code) evaluated 24 architectural claims. Results:

| Status | v0.3 | v0.4 |
--------|------|------|
| PROVEN | 12 | **13** |
| EMPIRICALLY VERIFIED | 5 | **7** |
| PARTIALLY TRUE | 3 | 1 |
| FALSE | 2 | 1 |
| REMOVED (dead code) | 0 | 1 |

Key improvements from v0.3 to v0.4: SlicedFFN and sparse pathway dispatch went from standalone-only to proven at model level; the width diversity loss was added and verified; dead code (`adaptive_halting.py`) was removed.

See `XORZEN_v04_ADVERSARIAL_AUDIT_REPORT.md` for the full independent audit.

---

## Reports

Comprehensive verification, audit, and benchmark reports are available under `reports/`:

### v0.4 Reports (`reports/v04/`)

| File | Description |
------|-------------|
| `ARCHITECTURE.md` | Full architecture documentation with component details |
| `PERFORMANCE.md` | SSM timing, FLOPs reduction, compute budget, profiling |
| `ABLATIONS.md` | 8-component ablation study with analysis |
| `TRAINING_VALIDATION.md` | Training stability and convergence |
| `ARCHITECTURE_AUDIT.md` | Architecture-level audit findings |
| `phase4_overfit.json` | Minimal overfit test (loss 3.48 -> 0.006, 99.8% reduction) |
| `phase_v04_old_vs_new.json` | Head-to-head v0.3 vs v0.4 comparison |
| `phase_v04_model_level_sparse.json` | Model-level conditional compute (4/4 PASS) |
| `phase_v04_profiling.json` | Subsystem FLOPs breakdown |
| `phase_v04_final_audit.json` | Final 24-claim audit with v0.3 -> v0.4 deltas |
| `phase9_ssm_deep_validation.json` | SSM numerical equivalence (8 configs, fp32+bf16) |
| `phase10_moe_stress.json` | Disk-sharded MoE lifecycle (14/14 PASS) |
| `phase6_router_stability.json` | Router collapse experiments |
| `phase7_compute_budget.json` | Compute budget validation |

### Other Reports

| Path | Description |
------|-------------|
| `reports/scaling/scaling_law.md` | Parameter counts and scaling projections |
| `reports/audit/` | Parameter counts, FLOPs audit, verified proofs |
| `reports/xorzen_v0.2.4_fix_report.md` | v0.2.4 bug fix documentation |

---

## Roadmap

### In Progress

- [ ] **xorvec-data Phase 3-8**: Quality filtering, CLI expansion, testing, Docker/CI, PostgreSQL migration, HuggingFace integration
- [ ] **libcompact Phase 3**: Fix RMSNorm, implement STE for training, build `.xorm` loader, achieve numerical parity with Python reference
- [ ] **Jima benchmarks**: Inference latency measurement, reasoning quality evaluation, agent task datasets

### Planned

- [ ] **Kage (Phase 0-4)**: Vision encoder design, multimodal fusion, training pipeline, HuggingFace publishing
- [ ] **XorZen 12B scale validation**: Test the 12B > 60B dense hypothesis
- [ ] **HuggingFace model cards**: Publish Zero, Zero Agentic, Jima, and Greed model families
- [ ] **Eval-mode pathway diversity**: KL-to-uniform loss or temperature scheduling to break eval-mode collapse
- [ ] **ComputeController cleanup**: Remove dead `ComputeController.py` (functionality now in `AdaptiveRouter`)

See `FRAZIYM_REBUILD_ROADMAP_2MONTHS.md` for the detailed 8-week rebuild plan.

---

## Known Limitations

1. **No RoPE**: Position encoding is learned absolute (`nn.Embedding(context_length, hidden_size)`), not RoPE. Context extrapolation beyond `context_length` is impossible without retraining. RoPE config fields exist in C++ `types.h` but are ignored (documented as NOT IMPLEMENTED).

2. **No GQA**: `n_kv_heads` config exists per-variant but `LocalAttentionPathway` always uses full MHA (`Linear(hidden, hidden)` for Q/K/V). GQA is declared but not wired.

3. **Depth routing is training-only**: During training, all layers are computed with STE mask blending (no FLOP savings — dense compute). Genuine depth skipping only occurs at inference via gather-scatter.

4. **SSM state not carried across forward calls**: The SSM pathway uses a fixed-size recurrent state (`state_dim`=16) that IS the context mechanism — no KV cache is used. However, the state is not currently returned from `forward()` or carried across calls. For incremental generation, the SSM state should be passed as `init_state` to the scan. This is the intended context mechanism, not a bug — it needs implementation.

5. **SlicedFFN `ln_hidden` dead code**: `ln_hidden` is only applied when `w == max_width` (hidden×4), but `width_choices` never include `max_width` (max is `hidden_size`). So `ln_hidden` parameters are allocated and trainable but never receive gradients. [BUG-G002, medium]

6. **CoT frozen by design**: The Chain-of-Thought vector (766K params in zero_10M) is frozen during pre-training (zero signal). Unfreeze with `model.enable_cot()` for fine-tuning. All 40 CoT parameter tensors are frozen by default.

7. **Training nondeterministic**: Gumbel noise in the router's training path consumes RNG state, making checkpoint resume nondeterministic. Save/restore `torch.get_rng_state()` for exact resume. [BUG-R001, low]

8. **Loss small discrepancy**: `lm_loss` differs from manual cross-entropy by ~4.5e-4 (was 0.00 before merge `635f41e`). Likely FP accumulation in auxiliary loss terms. [BUG-L001, medium]

9. **`use_cache=True` is not the intended path**: XORZEN's context mechanism is the SSM recurrent state, not attention KV cache. The `use_cache` / `past_key_values` API was added experimentally but should be replaced with SSM state passing (`init_state` in `sequential_scan`). The current KV cache implementation forces `top_k=3` (disabling pathway sparsity) and is architecturally inconsistent with the SSM-first design.

10. **No KV cache (by design)**: XORZEN uses SSM fixed-size context, not KV cache. The Python `use_cache` / `past_key_values` API that was added is NOT the intended inference path — it should be replaced with SSM state passing. The C++ `DiskShardedKVCache` (540 lines) is dead code that should be removed or replaced with SSM state management. Generation currently recomputes the full prefix (O(T²)) — the fix is to carry the SSM state, not cache attention K/V.

11. **Native kernels not wired**: `xorzen/speed/csrc/` has 8 C++ files (~1156 lines) of AVX2 kernels that JIT-compile but are never called from any forward path. `auto_patch_model` always returns `False`.

12. **S4D dead code**: `xorzen/model/ssm.py` defines `S4DKernel` (complex diagonal A) but it is never imported. Replaced with a stub that raises `ImportError`. The actual SSM is `SSMPathway` in `hass_block.py` (Mamba-style real diagonal A).

13. **Width routing compute savings negligible**: FFN is only ~3.6% of total FLOPs. Even 87% FFN savings = 3.1% total FLOP savings. Width routing does not learn input-dependent specialization in tested configs (router collapses to one width). [Research finding, v2 experiment]

---

## License

XORZENX Proprietary License. See `LICENSE` for full terms.

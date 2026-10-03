# XORZEN v1.0.1 Architecture Documentation

**Version:** 1.0.1  
**Date:** 2026-10-01

---

## Table of Contents

1. [HASS Block](#1-hass-block)
2. [SlicedFFN](#2-slicedffn)
3. [MoE (ShardedExpertFabric)](#3-moe-sharedexpertfabric)
4. [SSM (State Space Model)](#4-ssm-state-space-model)
5. [GreedModel](#5-greedmodel)
6. [ZeroAgenticModel](#6-zeroagenticmodel)

---

## 1. HASS Block

**File:** `xorzen/model/components/hass_block.py`

### Overview

The HASS (Hybrid Attention-Shard Switch) Block is the core conditional-compute building block of XORZEN. It implements **3 parallel pathways** with adaptive routing:

1. **Local Attention Pathway** — Windowed causal attention for local context
2. **Low-Rank Global Pathway** — Low-rank approximation for global attention
3. **SSM Pathway** — State Space Model for sequential processing

### Architecture

```
Input → Router → Pathway Selection → 3 Parallel Pathways → Merger Gate → Output
                     ↓
            STE Blend (training)
            Hard Top-K (inference)
```

### Key Components

#### Pathway Routing (`sparse_pathway_dispatch`)

The router produces probabilities for each of the 3 pathways per token. During training, a Straight-Through Estimator (STE) is used:
- **Forward:** Hard top-K pathways selected per token
- **Backward:** Soft probabilities flow gradients

During inference: Hard top-K execution.

#### SlicedFFN Integration

Each HASS Block contains a `SlicedFFN` that enables **width-conditional computation**:
- Multiple width choices (e.g., 25%, 50%, 75%, 100% of max)
- Nested weight slicing: only `[:W_i]` columns of fc1 and `[:W_i]` rows of fc2 used
- Genuine FLOP reduction proportional to selected width

#### Conditional Depth

- `forward_with_depth()` enables inference-time depth skipping
- During training: full depth always computed with STE blend
- `max_depth` / `min_depth` config controls range

#### SSM Integration

SSM pathway uses `S4DKernel` with parallel scan for O(L log L) sequential processing.

### Configuration Flags

```python
# In ModelConfig:
max_depth: int = 24          # Max transformer depth
min_depth: int = 3           # Min transformer depth  
width_choices: Tuple[int]    # e.g., (384, 768, 1152, 1536, 2048)
path_choices: int = 3        # Local, Low-rank, SSM
pathway_top_k: int = 2       # Top-K pathways per token
use_sliced_ffn: bool = True  # Enable SlicedFFN (default True)
```

---

## 2. SlicedFFN

**File:** `xorzen/model/components/sliced_ffn.py`

### Overview

SlicedFFN implements **genuine per-token width selection** via nested weight slicing. This replaces the legacy `AdaptiveFFN` which computed ALL widths and blended them (dense computation).

### How Width Slicing Works

```
Weight matrices stored at max_width:
    fc1: [hidden, max_width]   # Input projection
    fc2: [max_width, hidden]   # Output projection

For selected width W_i (W_i <= max_width):
    fc1_sliced = fc1[:, :W_i]  # [hidden, W_i]
    fc2_sliced = fc2[:W_i, :]  # [W_i, hidden]
    
    hidden = activation(fc1_sliced(x))    # [B, T, W_i]
    output = fc2_sliced(hidden)           # [B, T, hidden]
```

**FLOPs reduction:** Proportional to `W_i / max_width`. At 25% width → ~75% FLOP savings on FFN.

### Width Choices

Default: 25%, 50%, 75%, 100% of max_width
Configurable via `width_choices` in ModelConfig.

### Training: Straight-Through Estimator (STE)

```python
# Router produces soft probs: width_probs [B, T, num_widths]
hard_idx = width_probs.argmax(dim=-1)        # Hard selection for forward
y_hard = forward_per_token_width(x, hard_idx)  # Compute at hard width
# Backward: gradient flows through width_probs for auxiliary losses
```

### Inference: Hard Argmax + Grouping

Tokens grouped by selected width, processed in batches per width for efficiency.

---

## 3. MoE (ShardedExpertFabric)

**File:** `xorzen/model/zmoe.py`

### Overview

Disk-sharded Mixture of Experts designed for **CPU training with massive expert counts**.

### Expert Registration (BUG-CRITICAL-2 Fix)

**Before:** Experts were NOT registered as `nn.Module` → no gradients, no device placement
**After:** All experts registered in `nn.ModuleList` → gradients flow correctly

```python
# Before (broken):
self.experts = [ExpertFFN(...) for _ in range(num_experts)]  # Plain list!

# After (fixed):
self.experts = nn.ModuleList([ExpertFFN(...) for _ in range(num_experts)])
```

### Architecture

- **Expert count:** Up to 192 (configurable via `expert_count`)
- **Top-K routing:** 2 experts per token (configurable via `top_k_experts`)
- **Disk sharding:** Experts saved to `experts/` directory
- **LRU cache:** `max_expert_cache` experts in RAM (default 24)
- **Expert hidden multiplier:** 4.0x hidden size for intermediate dim

### Load Balancing

Switch Transformer formula (unified, single loss):
```
L_aux = num_experts * sum(mean(expert_gates) * mean(expert_assignments))
```

### Disk Persistence

- `sync_to_disk()` — Save experts to sharded files
- `load_from_disk()` — Load experts on demand
- Automatic stale shard detection (checks hidden_dim compatibility)

### Warning

**After fix: ALL N experts live in RAM** (nn.ModuleList). For 277M with 64 experts (~8MB each) = 512MB RAM just for experts.

---

## 4. SSM (State Space Model)

**File:** `xorzen/model/ssm.py`

### Overview

Production-grade S4D (Diagonal State Space) kernel with parallel scan.

### Formulation

Continuous-time SSM:
```
h'(t) = A h(t) + B u(t)
y(t)  = C h(t) + D u(t)
```

Discretized via **Zero-Order Hold (ZOH)**:
```
A_bar = exp(dt * A)
B_bar = (A_bar - I) @ A^{-1} @ B
```

### S4D Parameterization

- **A matrix:** Complex diagonal (real + imaginary parts learned)
- **B, C:** Learned projections
- **dt:** Learnable step size per channel

### Parallel Scan

Custom numerically stable parallel scan:
- Sequential: O(L²) — impractical for long sequences
- Parallel: O(L log L) — enables long context

### Implementation Details

```python
class S4DKernel(nn.Module):
    # Parallel scan computes SSM recurrence in O(L log L)
    # Includes causality tests and numerical equivalence verification
```

---

## 5. GreedModel

**File:** `xorzen/models/greed/model.py`

### Overview

**XorZen-GREED** — Continuous-input conversation quality evaluator.

### Inheritance

`GreedModel(zeroModel)` — Reuses full XORZEN stack:
- AdaptiveRouter (depth/width/path/expert)
- HASSBlock × N
- ShardedExpertFabric
- XorzenMergerGate
- RMSNorm

### Greed-Specific Components

#### 1. Feature Projection (replaces token embedding)
```python
self.feature_proj = nn.Sequential(
    nn.Linear(input_dim, config.hidden_size),
    nn.LayerNorm(config.hidden_size),
)
```
Accepts continuous per-turn feature vectors `[B, T, input_dim]`.

#### 2. Active Internal CoT (unfrozen)
```python
self.enable_cot()  # CoT trained end-to-end, not frozen
```

#### 3. Greedy Gated CoT Fusion
```python
# Gate: decides how much CoT to fuse
self.greedy_cot_gate = nn.Sequential(
    nn.Linear(config.hidden_size + cot_total_dim, config.hidden_size),
    nn.SiLU(),
    nn.Linear(config.hidden_size, 1),
    nn.Sigmoid(),
)

# Projection: fuses CoT into hidden states
self.greedy_fusion_proj = nn.Linear(
    config.hidden_size + cot_total_dim, config.hidden_size
)
```
Amplifies turns with high reasoning density before pooling.

#### 4. Mean-Pooled Classification Head
```python
self.classification_head = nn.Sequential(
    nn.LayerNorm(config.hidden_size),
    nn.Linear(config.hidden_size, num_classes),
)
```
Outputs `[B, num_classes]` grade logits (not per-token LM logits).

### Config Presets (v1.0.1)

| Preset | hidden | layers | vocab | context | MoE |
|---|---|---|---|---|---|
| `greed_tiny` | 64 | 3 | 10K | 512 | 1 expert (disabled) |
| `greed_small` | 256 | 6 | 10K | 1024 | 1 expert (disabled) |

---

## 6. ZeroAgenticModel

**File:** `xorzen/models/zero/agentic_model.py`

### Overview

Agentic extension of zeroModel with **memory vault** and **action head**.

### Key Components

#### Memory Vault (Associative Retrieval)
- Stores and retrieves past hidden states
- Key-value associative memory
- Used for long-term context and tool use

#### Action Head
```python
class ActionHead(nn.Module):
    # 0: Output Token, 1: Search Memory, 2: Use Tool, 3: Internal Reasoning
    self.proj = nn.Linear(d_model, num_actions)
```

#### Internal Latent CoT
```python
class InternalLatentCoT(nn.Module):
    # GRU-like latent reasoning state evolution
    update_gate = nn.Linear(d_model + cot_dim, cot_dim)
    transform = nn.Sequential(nn.Linear(...), GELU, nn.Linear(...))
```

#### Critique Module
```python
class CritiqueModule(nn.Module):
    # Evaluates quality/uncertainty of latent state
    quality_eval = nn.Sequential(...)
```

#### Core Architecture
- **FlashSSM** — Optimized SSM with conv1d + gating
- **GatedLinearAttention** — Linear attention with gating

### Bug Fixes (v1.0.1)

1. **Memory vault crash** — Fixed indexing errors
2. **Action head zero grad** — Fixed gradient flow to action head

---

## Summary: Conditional Compute Flow

```
Input Tokens/Features
        ↓
Embedding / Feature Projection
        ↓
For each HASS Block (num_layers):
        ↓
    AdaptiveRouter → depth, width, pathway, expert decisions
        ↓
    Pathway Dispatch (sparse_pathway_dispatch)
        ├─ Local Attention → [B, T, H]
        ├─ Low-Rank Global → [B, T, H]
        └─ SSM (S4D) → [B, T, H]
        ↓
    Merger Gate (fuses pathway outputs)
        ↓
    SlicedFFN (width-conditional, nested slicing)
        ↓
    Residual + Norm
        ↓
Merger Gate (final HASS + MoE + CoT fusion)
        ↓
Output Head (LM / Classification / Action)
```

---

## Configuration Reference

```python
from xorzen.config import ConfigFactory, ModelSize

# Standard LM sizes
ConfigFactory.get_config(ModelSize('1M'))      # 1M params
ConfigFactory.get_config(ModelSize('10M'))     # 10M params
ConfigFactory.get_config(ModelSize('277M'))    # 277M params
ConfigFactory.get_config(ModelSize('1B'))      # 1B params
...

# GreedModel presets (NEW in v1.0.1)
ConfigFactory.get_config(ModelSize('greed_tiny'))   # 64 hidden, 3 layers
ConfigFactory.get_config(ModelSize('greed_small'))  # 256 hidden, 6 layers

# Lightweight test config
ConfigFactory.get_lightweight_config()
```

---

**End of Architecture Documentation**
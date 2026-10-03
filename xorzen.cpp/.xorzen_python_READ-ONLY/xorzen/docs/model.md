# XorZen Architecture & Components Documentation

The `model` sub-library contains the building blocks of the XorZen-Zero architecture. It is designed for extreme efficiency, enabling large-scale MoE (Mixture of Experts) and SSM (State Space Model) training on standard hardware.

---

## 1. Core Abstractions

### `base.py`
Defines the fundamental interfaces for all models in the framework.
- **`ModelOutput`**: A rich dataclass returned by the model's `forward` pass. It includes not just `logits` and `loss`, but also `cot_vector`, `routing_info`, and auxiliary losses (`load_balance_loss`, `routing_loss`).
- **`BaseModel`**: The abstract base class providing common utilities like `count_parameters()` and `get_memory_footprint()`.
- **`GenerationConfig`**: Manages sampling parameters (temperature, top-k, top-p) and generation constraints.

---

## 2. Advanced Pathways & Blocks

### `ssm.py` (State Space Model)
A production-grade implementation of structured diagonal SSMs (S4D).
- **Parallel Scan**: Implements an $O(L \log L)$ parallel prefix scan in PyTorch, avoiding slow sequential loops.
- **ZOH Discretization**: Uses Zero-Order Hold to convert continuous parameters ($A, B$) into discrete ones ($\bar{A}, \bar{B}$).
- **Performance**: Capable of handling massive sequence lengths with linear scaling.

### `zmoe.py` (Zero Mixture of Experts)
The "secret sauce" for high-parameter training on low-memory systems.
- **`ShardedExpertFabric`**: Implements a disk-based MoE system.
    - **Total Experts**: 192 experts (approx. 6GB).
    - **LRU Cache**: Only keeps a small fraction (e.g., 24 experts) in RAM (approx. 768MB).
    - **On-Demand Loading**: Automatically loads experts from disk when routed to them.
- **`ExpertFFN`**: The basic building block using SwiGLU activation.

### `hass_block.py` (Hybrid Attention-Shard Switch)
The primary computational unit of XorZen-Zero. It combines three distinct pathways:
1. **Local Attention**: Multi-head attention with a sliding window mask for local context.
2. **Global Pathway**: Low-rank global interaction.
3. **SSM Pathway**: Long-range sequence modeling via the S4D kernel.
- **Hybrid Routing**: Uses a learned router to blend these pathways based on token complexity.

---

## 3. Internal Intelligence Components (`components/`)

### `cot_vector.py` (Latent Chain-of-Thought)
Implements an internal, latent "thought process" that persists across tokens.
- **Multi-Component State**: Decomposes "thinking" into intention, decomposition, confidence, and contradiction.
- **Updater Mechanisms**: Supports both `CoTGRUUpdater` and `CoTTransformerUpdater`.
- **Consistency Loss**: Ensures the reasoning trace remains stable over time.

### `routing.py` (Adaptive Router)
The brain of the model that makes efficiency decisions.
- **Depth Routing**: Skips layers for simple tokens.
- **Width Routing**: Adjusts intermediate dimensions dynamically.
- **Path Routing**: Chooses between Attention and SSM pathways.
- **Expert Routing**: Routes tokens to the most relevant specialized experts in the MoE fabric.
- **Auxiliary Losses**: Includes `load_balance_loss` and `path_diversity_loss` to prevent routing collapse.

### `merger.py`
Orchestrates the fusion of different pathways and CoT vectors.
- Implements `ZAIMergerGate` which uses a learned gate or cross-attention to combine the outputs of Attention, SSM, and the Latent CoT state.

---
*Back to [README.md](../README.md)*

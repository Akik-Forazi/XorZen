# XorZen Model Variants Documentation

The `models` sub-library houses the specific implementations of XorZen's neural architectures, organized by family and scale. It uses a centralized registry for discovery and instantiation.

---

## 1. System Architecture

### `registry.py`
The `ModelRegistry` class acts as the central hub for the entire framework.
- **Dynamic Discovery**: Allows listing all available models via `ModelRegistry.list()`.
- **Case-Insensitive Retrieval**: Get model classes by name (e.g., `"zero_277m"`).
- **Metadata Management**: Stores parameter counts, descriptions, and configuration factories for every registered model.

---

## 2. The Zero Model Family (`models/zero/`)
The flagship architecture of XorZen, optimized for scaling and efficiency.

### `model.py`
Contains the core `zeroModel` class.
- **Hierarchical Construction**: Orchestrates the assembly of `HASSBlock` layers, `ZAIMoE` (Expert Fabric), and `InternalLatentCoT`.
- **Forward Pass**: Implements the complex routing logic where tokens are dynamically passed through Attention or SSM pathways, potentially skipping layers or routing to specialized experts.
- **Generation**: Includes optimized methods for autoregressive text generation.

### `variants.py`
Defines and registers concrete model sizes. All variants inherit from `zeroBase`.
- **Scaling Tiers**:
    - **TINY_23K**: Architectural floor for testing.
    - **NANO_1M / 10M**: Ultra-lightweight models for edge devices.
    - **MICRO_50M**: A balanced small-scale model.
    - **MINI_277M**: The **flagship variant**, providing the best performance-to-compute ratio for CPU training.
    - **SMALL_500M / MEDIUM_1B**: High-performance mid-scale models.
    - **LARGE_3B / XL_7B**: Large-scale variants for complex AGI tasks.
- **Registration**: Every variant is automatically added to the `ModelRegistry` upon import.

---

## 3. The Igris Model Family (`models/igris/`)
A secondary model architecture within the XorZen ecosystem.

### `config.py`
Defines `IGRISConfig`, which uses a different set of hyperparameters compared to the Zero family.

### `model.py`
The implementation of the Igris architecture. While it shares some components with Zero, it has a distinct structural layout and information flow.

### `variants.py`
Defines Igris-specific sizes:
- **IGRIS_Nano**
- **IGRIS_Micro**

---

## 4. Initialization & Aliases
The `__init__.py` files in each sub-directory handle the exposure of these models to the rest of the framework, ensuring that `from xorzen import zero_277M` works seamlessly.

---
*Back to [README.md](../README.md)*

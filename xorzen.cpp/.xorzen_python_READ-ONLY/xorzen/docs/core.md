# XorZen Core & Root Documentation

This document provides a detailed technical breakdown of the core files located at the root of the `xorzen` package. These files define the framework's identity, configuration system, and high-level training pipelines.

---

## 1. `__init__.py`
**Purpose**: The package entry point that exposes the public API and handles versioning.

### Key Components:
- **Versioning**: Defines `__version__ = "0.2.4"`.
- **Public API Exposure**:
    - **Models**: Exposes specific `zero` variants (from `zero_tiny_23k` to `zero_7B`) and `igris` variants.
    - **Tokenizer**: Exposes functions for loading and training tokenizers (`load_pretrained`, `train_tokenizer`).
    - **Data**: Exposes binary conversion and loading utilities (`txt_to_bin`, `BinaryDataset`).
    - **Training**: Exposes the main trainer and state management tools.
- **Legacy Support**: Provides aliases like `zero277M` for backward compatibility.

---

## 2. `config.py`
**Purpose**: A massive (2000+ lines), production-grade configuration system that acts as the "DNA" of every XorZen model.

### Key Classes & Enums:
- **`ModelSize` (Enum)**: Defines scaling laws from `TINY_23K` to `XXXL_70B`.
- **`ArchitectureVariant` (Enum)**: Supports `STANDARD`, `HASS` (Hybrid Attention-Shard Switch), `MOE` (Mixture of Experts), and the flagship `XORZENX_zero`.
- **`RouterType` (Enum)**: Manages MoE routing strategies like `DEPTH_WIDTH`, `ADAPTIVE`, and `CERTAINTY`.
- **`CoTType` (Enum)**: Configures Chain-of-Thought reasoning (None, Output, `LATENT`, or `PROVABLE`).
- **`QuantizationScheme` (Enum)**: Supports `STATIC`, `DYNAMIC`, `PROGRESSIVE`, and `AWARE` quantization.
- **`BaseConfig` (Dataclass)**: The foundation for all config objects, providing:
    - `to_dict()` / `to_json()` / `to_yaml()` serialization.
    - Automatic validation and hardware optimization logic.
- **`XorZenConfig`**: The primary configuration class that aggregates model hyperparameters, training settings, and optimization flags.

---

## 3. `train_superintelligence.py`
**Purpose**: The top-level orchestration script for training advanced AGI capabilities. It demonstrates how to integrate multiple sub-libraries (`ult`, `tokenizer`, `models`) into a single pipeline.

### Key Class: `AGITrainingPipeline`
- **Ensemble Distillation**: `load_teachers()` supports loading multiple models (e.g., GPT-2 family) to act as teachers for the student model.
- **Multi-Task Dataset Generation**: `create_multi_task_dataset()` automatically transforms base text into specialized formats:
    - **Reasoning**: `<|think|> ... <|/think|>` tags.
    - **Tool Use**: API call patterns.
    - **Multi-Agent**: Planner/Critic/Executor dialogues.
    - **Reflection**: Self-evaluation loops.
- **Self-Critique Training**: `train_with_critique()` implements a reinforcement loop where the model critiques its own outputs to improve quality.

---

## 4. `exceptions.py`
**Purpose**: Provides a standardized, hierarchy-based error handling system for the entire framework.

### Hierarchy:
- **`XORZENXError`**: The root exception class.
    - Includes `message`, `details` (dict), and `suggestion` (string) fields.
    - Automatically formats errors with a "💡 Suggestion" icon for better DX.
- **Domain-Specific Errors**:
    - **`ModelError`**: Base for `ModelNotFoundError`, `ModelConfigError`, and `ModelLoadError`.
    - **`DataError`**: Handles dataset loading and validation failures.
    - **`TokenizerError`**: Issues with vocabulary or encoding.
    - **`TrainingError` / `CheckpointError`**: Related to the optimization loop and state persistence.

---
*Back to [README.md](../README.md)*

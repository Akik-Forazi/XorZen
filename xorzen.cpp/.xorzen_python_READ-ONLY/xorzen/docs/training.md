# XorZen Training Ecosystem Documentation

The `training` sub-library provides the infrastructure for optimizing XorZen models, ranging from standard gradient descent to advanced curriculum-based learning and state persistence.

---

## 1. Core Trainer

### `trainer.py`
The `XORZENXTrainer` is a production-grade training engine.
- **Precision**: Supports FP32, FP16, and BF16 (Mixed Precision) for faster training.
- **Optimization**: Handles gradient accumulation, gradient clipping, and learning rate scheduling.
- **Integration**: Pluggable callbacks and support for TensorBoard and Weights & Biases (W&B).
- **Features**: Automatic evaluation passes, early stopping, and detailed metric logging.

### `state.py`
Manages the `TrainingState` object, which tracks the global progress of a training run.
- **Metrics**: Stores loss history, throughput (tokens/sec), and hardware utilization.
- **Formatting**: Provides utility functions like `format_time()` and `format_number()` for clean log output.
- **Persistence**: The state is automatically serialized into checkpoints to ensure seamless resumption.

---

## 2. Learning Strategies

### `curriculum.py`
Implements the `CurriculumTrainer`, which treats training data like a reading list.
- **File-by-File Learning**: Processes one text file (e.g., a book) at a time.
- **Evaluation Loop**: Tests the model on a held-out portion of the current file.
- **Revisit Logic**: If the loss on a file is above the `revisit_loss_threshold`, the file is added back to a queue to be "re-studied" later.
- **Use Case**: Ideal for training small-to-medium models on CPUs using structured datasets like Project Gutenberg.

### `continuation.py`
Provides high-level functions for resuming training.
- **`continue_train()`**: Automatically restores the model, optimizer, scheduler, and training state from a given checkpoint.
- **Safety Checks**: Validates that the model architecture and data configuration match the checkpoint metadata before starting.

---

## 3. Persistence

### `checkpoint.py`
Handles the complexity of saving and loading model states.
- **`CheckpointManager`**: Orchestrates the saving process, including:
    - **Versioning**: Keeps the last $N$ checkpoints to prevent data loss.
    - **Best-Model Tracking**: Automatically keeps the checkpoint with the lowest evaluation loss.
- **`CheckpointMetadata`**: Stores the framework version, configuration, and training state alongside the model weights.

---

## 4. High-Level API (`__init__.py`)
The module exposes simple entry points for common workflows:
- **`train(...)`**: A convenience function to quickly initialize a trainer with sensible defaults.
- **`Trainer`**: An alias for `XORZENXTrainer`.

---
*Back to [README.md](../README.md)*

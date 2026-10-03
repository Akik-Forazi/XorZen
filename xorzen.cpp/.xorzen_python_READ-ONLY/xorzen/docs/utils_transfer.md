# XorZen Utilities, Transfer & GUI Documentation

This document covers the supporting infrastructure of XorZen, including the mathematical utility library, model transfer tools, and the graphical monitoring interface.

---

## 1. Utilities (`utils/`)

### `logger.py`
The `XORZENXLogger` is a sophisticated logging system designed for production AI training.
- **Async Handling**: Uses `AsyncLogHandler` to prevent logging from blocking the training loop.
- **Metric Tracking**: Specialized `MetricsHandler` for recording loss, accuracy, and custom stats.
- **Performance Monitor**: Tracks system resources (CPU, RAM, Disk I/O) in real-time.

### `math_utils.py`
A comprehensive mathematical library for advanced neural modeling.
- **`TensorStability`**: Tools for detecting and correcting numerical instabilities (NaNs/Infs).
- **`InformationTheory`**: Implements entropy calculations used in routing diversity losses.
- **`RoutingMathematics`**: Specialized functions for load balancing and expert probability distributions.

### `sharding.py`
The engine behind XorZen's massive MoE scaling.
- **`ExpertShardManager`**: Manages the serialization and deserialization of experts to disk.
- **`LRUCache`**: A least-recently-used cache that swaps experts between RAM and disk based on routing frequency.
- **`XORZENXShardingSystem`**: Orchestrates the entire sharded fabric, ensuring zero-copy access where possible.

### `sppq.py`
Implements **Sharded Progressive Parameter Quantization (SPPQ)**.
- **Stability-Aware**: Only quantizes parameters after they have reached a "Stable" or "Very Stable" state during training.
- **Mixed Precision**: Allows different layers or experts to have different bit-widths (e.g., 4-bit, 8-bit) based on their sensitivity.

---

## 2. Model Transfer (`transfer/`)

### `gguf_transfer.py`
Enables knowledge transfer from local GGUF models (e.g., LLaMA, Mistral, Nemotron) into XorZen-Zero.
- **Local-First**: Works entirely offline using the `gguf` library.
- **Weight Adaptation**: Automatically crops or pads teacher weights to fit the student's dimensions.

### `universal.py` & `transfer.py`
Provides a universal framework for extracting features and knowledge from any PyTorch-based teacher model.

---

## 3. Graphical Interface (`gui/`)

### `bootloader.py`
The "BIOS" of the XorZen application. It handles:
- Dependency verification.
- Hardware auto-detection.
- Initialization of core services (Logging, Speed Booster).

### `dashboard.py`
A real-time monitoring dashboard for the training process.
- **Visualizations**: Plots loss curves, expert heatmaps, and routing entropy.
- **Controls**: Allows the user to pause, resume, or adjust hyperparameters (like learning rate) on the fly.

---
*Back to [README.md](../README.md)*

# XorZen Speed & Optimization Documentation

The `speed` sub-library provides the high-performance backbone of XorZen, enabling near-GPU training speeds on standard CPUs through custom C++ kernels and efficient Python wrappers.

---

## 1. Orchestration

### `booster.py`
The `SpeedBooster` is the central optimizer for XorZen models.
- **`SpeedProfile`**: Offers three optimization levels: `CONSERVATIVE`, `BALANCED`, and `LIGHTNING`.
- **Automatic Patching**: It recursively scans a model and replaces slow layers with their optimized counterparts (e.g., `LocalAttentionPathway` → `FlashLocalAttention`).
- **`torch.compile` Integration**: Automatically triggers PyTorch's Inductor compiler in `LIGHTNING` mode.

### `fast_trainer.py`
A specialized subclass of the main trainer that maxes out CPU utilization.
- **Thread Tuning**: Automatically sets `torch.set_num_threads` based on the system's core count.
- **JIT Warmup**: Performs fake forward/backward passes to "warm up" kernel caches and `torch.compile` before logging metrics.

---

## 2. Optimized Layer Wrappers

### `fast_attention.py`
Wraps the `window_attention_f32` C++ kernel.
- **Priority Logic**: Prefers the custom C++ kernel, then falls back to PyTorch's `scaled_dot_product_attention` (SDPA), and finally to a manual implementation.
- **Mask Caching**: Prevents redundant memory allocation for attention masks.

### `fast_ssm.py`
Wraps the `ssm_parallel_scan_f32` kernel.
- Provides a significant speedup for long-range sequential modeling by computing the SSM recurrence in parallel across the sequence dimension.

### `fast_moe.py`
Optimizes the Mixture of Experts (MoE) dispatch process.
- **Capacity Constraint**: Moves the token-sorting logic to C++.
- **Background Prefetching**: Uses a `ThreadPoolExecutor` to load the most likely next experts from disk into RAM before they are needed.

---

## 3. C++ Kernels (`csrc/`)
Low-level implementations using AVX2 SIMD and OpenMP parallelism.

- **`ssm_scan.cpp`**: Implements the Blelloch parallel prefix scan algorithm for SSMs.
- **`attention_ops.cpp`**: Optimized causal window attention with online softmax.
- **`expert_dispatch.cpp`**: Vectorized weighted sum for MoE expert accumulation.
- **`fused_ops.cpp`**: Fused kernels for `RMSNorm` and `SwiGLU` to minimize memory passes.
- **`math_utils.cpp`**: High-performance approximations for `exp`, `log`, and `sigmoid`.

---

## 4. Compilation & Bridge

### `xorzen_ext.pyx` (Cython)
The bridge between Python and C++.
- **Zero-Copy**: Passes raw data pointers directly to C++ kernels to avoid data duplication.
- **GIL Management**: Releases the Global Interpreter Lock (GIL) during kernel execution, allowing true multi-threaded performance.

### `setup_speed.py`
The build script for the C++ extension.
- **Compiler Detection**: Automatically configures flags for MSVC (Windows) or GCC/MinGW.
- **Architecture Tuning**: Enables `/arch:AVX2` and `/openmp:experimental` for maximum throughput.

---
*Back to [README.md](../README.md)*

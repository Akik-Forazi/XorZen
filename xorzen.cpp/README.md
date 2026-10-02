# XORZEN.CPP

**High-Performance C++ Backend for XORZEN AI Architecture**

> Current handoff: see [`PORTING_STATUS.md`](PORTING_STATUS.md) and [`BUILD_MSVC.md`](BUILD_MSVC.md) before continuing implementation. As of 2026-05-17, the MSVC + LibTorch build is verified, the backend DLL links, and CTest passes the train, infer, BEBPE tokenizer, and micro-benchmark smoke checks.

## Current Verified Build

Use MSVC Build Tools with the installed PyTorch/LibTorch wheel:

```powershell
cmd.exe /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S xorzen.cpp -B xorzen.cpp\build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:\Users\akikf\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\torch\share\cmake" && cmake --build xorzen.cpp\build-msvc -j 8'
ctest --test-dir xorzen.cpp\build-msvc --output-on-failure
```

The current C++ backend is buildable and runnable, but not yet full Python parity. Remaining parity work includes production checkpoint compatibility, quantization, full sharding metadata, GPU backends, optimized custom kernels, and layer-by-layer numeric tests against Python. Older roadmap sections below describe the intended destination and may be more ambitious than the verified current state.

This repository is intended to become a production-grade, hardware-optimized C++ implementation of the XORZEN transformer architecture with disk-sharded MoE.

---

## 🎯 Project Goals

Transform the XORZEN Python research codebase into a **fast, production-ready C++ inference and training engine** that:

- ✅ **Runs 10-50x faster** than PyTorch Python (native SIMD, cache-friendly memory layout, zero Python overhead)
- ✅ **Compiles to a single DLL** (`XorZen_BETA-v-0.2.5_win64_backend.dll`) for seamless integration with XORZEN Studio UI
- ✅ **Supports the full architecture**: 192-expert MoE, disk-based expert sharding, adaptive routing, latent CoT, progressive quantization
- ✅ **Cross-platform**: Windows (primary), Linux, macOS via CMake + modern C++20
- ✅ **Low memory footprint**: LRU expert caching allows 277M parameter models to train on 16GB RAM (CPU-friendly)
- ✅ **GPU-ready**: OpenCL/CUDA acceleration for forward passes (inference first, training later)

---

## 📂 Project Structure

```
xorzen.cpp/
├── include/xorzen/          # Public API headers
│   ├── tensor.h             # Lightweight tensor abstraction (no PyTorch dependency)
│   ├── model.h              # Model interface (XorzenX, config)
│   ├── moe.h                # MoE layer + expert routing
│   ├── expert.h             # Expert FFN implementations
│   ├── cache.h              # LRU expert cache (disk ↔ RAM)
│   ├── ssm.h                # State Space Model (Mamba-like)
│   ├── attention.h          # Multi-head attention variants
│   ├── cot.h                # Chain-of-Thought latent reasoning
│   ├── quant.h              # Progressive quantization (FP32 → BF16 → INT8)
│   └── api.h                # C ABI exports for DLL interface
│
├── src/
│   ├── model/               # Core model implementations
│   │   ├── xorzenx.cpp      # Main XorzenX transformer
│   │   ├── moe.cpp          # ShardedExpertFabric (disk-based MoE)
│   │   ├── expert.cpp       # ExpertFFN (SwiGLU)
│   │   ├── router.cpp       # Adaptive routing logic
│   │   ├── ssm.cpp          # SSM block
│   │   ├── attention.cpp    # Grouped-query attention
│   │   └── cot.cpp          # Latent CoT projection
│   │
│   ├── ops/                 # Low-level kernels (SIMD-optimized)
│   │   ├── matmul.cpp       # Matrix multiplication (AVX2/AVX-512)
│   │   ├── softmax.cpp      # Numerically stable softmax
│   │   ├── layernorm.cpp    # RMSNorm (faster than LayerNorm)
│   │   ├── silu.cpp         # SiLU activation (vectorized)
│   │   ├── rope.cpp         # Rotary Position Embedding
│   │   └── quantize.cpp     # INT8/BF16 quantization kernels
│   │
│   ├── utils/               # Infrastructure
│   │   ├── tensor.cpp       # Tensor memory management
│   │   ├── cache.cpp        # LRU cache (experts on disk)
│   │   ├── serialization.cpp # Checkpoint save/load
│   │   ├── logger.cpp       # Logging system
│   │   └── thread_pool.cpp  # CPU thread pool for parallel ops
│   │
│   └── api.cpp              # DLL entry points (C ABI)
│
├── build/                   # CMake build artifacts (generated)
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
└── LICENSE                  # PolyForm Noncommercial 1.0.0
```

---

## 🏗️ Architecture Overview

### Core Components

**1. XorzenX Transformer**
- 24-layer decoder-only architecture (277M params)
- Grouped-query attention (8 heads, 2 KV heads)
- RoPE positional embeddings
- RMSNorm (faster than LayerNorm)
- Context length: 8192 tokens

**2. Disk-Sharded MoE (192 Experts)**
- **Innovation**: Experts stored on disk, loaded on-demand via LRU cache
- Only 24 experts in RAM at once (768MB vs 6GB total)
- Enables massive MoE on consumer hardware
- Top-K routing (K=2)
- Adaptive router adjusts expert selection based on input

**3. Latent Chain-of-Thought**
- Projects hidden states into latent "reasoning" space
- No explicit CoT tokens (efficient for inference)
- Consistency loss ensures stable reasoning traces

**4. Progressive Quantization**
- Start training in FP32
- Progressively quantize to BF16 → INT8
- Maintains accuracy while reducing memory

**5. State Space Model (SSM)**
- Mamba-inspired selective state spaces
- Alternative to attention for long-range dependencies
- Lower memory footprint

---

## 🚀 Build Instructions

### Prerequisites

**Windows (Primary Platform)**
- **Compiler**: MinGW-W64 GCC 15.2+ (already installed on your system)
- **CMake**: 4.3+ (installed)
- **Optional**: CUDA Toolkit 12.x (for GPU acceleration)

**Linux/macOS**
- GCC 11+ / Clang 14+
- CMake 3.20+
- OpenCL headers (optional, for GPU)

### Build Steps

```bash
# Navigate to project root
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp

# Create build directory
mkdir build
cd build

# Configure (Windows MinGW)
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..

# Build the DLL
mingw32-make -j8

# Output: XorZen_BETA-v-0.2.5_win64_backend.dll
```

**Build Targets**:
- `xorzen_backend.dll` — Main inference/training DLL for XORZEN Studio
- `xorzen_cli.exe` — Standalone CLI for testing (optional)
- `xorzen_tests.exe` — Unit tests (optional)

---

## 📦 DLL API (C ABI)

The DLL exports a clean C API for integration with XORZEN Studio (C++ UI):

```c
// Initialize model from checkpoint
void* xorzen_load_model(const char* checkpoint_path);

// Forward pass (inference)
int xorzen_forward(void* model, const float* input_ids, int seq_len, float* logits);

// Training step
int xorzen_train_step(void* model, const float* input_ids, const float* labels, 
                      int batch_size, int seq_len, float* loss_out);

// Generate text
int xorzen_generate(void* model, const int* prompt, int prompt_len, 
                    int max_new_tokens, int* output, int* output_len);

// Save checkpoint
int xorzen_save_checkpoint(void* model, const char* path);

// Get metrics (loss, expert usage, cache hit rate, etc.)
int xorzen_get_metrics(void* model, float* metrics, int max_metrics);

// Free model
void xorzen_free_model(void* model);
```

**Thread Safety**: All API calls are thread-safe (internal mutex locks).

---

## 🔧 Integration with XORZEN Studio

The C++ UI (`xorzen_studio/`) loads this DLL dynamically:

```cpp
// In Studio UI (C++)
HMODULE backend = LoadLibrary("XorZen_BETA-v-0.2.5_win64_backend.dll");
auto xorzen_load = (xorzen_load_model_fn)GetProcAddress(backend, "xorzen_load_model");

void* model = xorzen_load("checkpoints/xorzen_277M.bin");

// Training loop
float loss = 0.f;
xorzen_train_step(model, batch_input, batch_labels, 4, 512, &loss);

// Update UI with metrics
float metrics[32];
xorzen_get_metrics(model, metrics, 32);
update_ui_metrics(metrics);
```

**Named Pipe IPC** (optional fallback):
- If Python backend is still running, Studio can talk to it via `\\.\pipe\XorzenStudio`
- Allows hybrid mode during migration (C++ inference, Python training)

---

## ⚡ Performance Targets

| Operation | Python (PyTorch) | C++ (Optimized) | Speedup |
|-----------|------------------|-----------------|---------|
| Forward pass (seq=512) | 180ms | 12ms | **15x** |
| Expert routing (192 experts) | 45ms | 3ms | **15x** |
| RMSNorm (batch=4) | 8ms | 0.4ms | **20x** |
| Matrix multiply (768x2048) | 22ms | 1.1ms | **20x** |
| Full training step | 850ms | 60ms | **14x** |

**Memory**:
- Python: 8.2GB RAM (full model + optimizer states)
- C++: 2.1GB RAM (LRU cache + gradients)

---

## 📊 Compatibility Matrix

| Feature | Python (Reference) | C++ (This Repo) | Status |
|---------|-------------------|-----------------|--------|
| XorzenX base model | ✅ | x | unImplemented |
| 192-expert MoE | ✅ | x | unImplemented |
| Disk sharding + LRU cache | ✅ | x | unImplemented |
| Adaptive routing | ✅ | x | unImplemented |
| Latent CoT | ✅ | x | unImplemented |
| SSM (Mamba) | ✅ | x | unImplemented |
| Progressive quantization | ✅ | x | UnImplemented |
| Checkpoint save/load | ✅ | x | Compatible format |
| Training loop | ✅ | x | UnImplemented |
| Inference only | ✅ | x | Optimized path |
| GPU acceleration | ❌ | 🚧 | Planned (CUDA/OpenCL) |
| Multi-GPU | ❌ | ❌ | Future |

---

## 🧪 Testing

```bash
# Run unit tests
cd build
./xorzen_tests.exe

# Benchmark against Python
./xorzen_cli.exe benchmark --checkpoint ../checkpoints/xorzen_277M.bin
```

**Test Coverage**:
- Tensor operations (matmul, softmax, layernorm)
- Expert loading/caching
- Forward pass correctness (vs Python outputs)
- Checkpoint serialization round-trip
- Memory leak detection (Valgrind on Linux)

---

## 📝 Migration Status (Python → C++)

**Phase 1: Inference** ✅ VERIFIED
- [x] Tensor abstraction (LibTorch)
- [x] Forward pass (all layers)
- [x] Expert loading from disk
- [x] LRU cache
- [x] Checkpoint deserialization (minimal)
- [x] BEBPE Tokenizer parity
- [x] DLL exports (C ABI)

**Phase 2: Training** 🚧 FUNCTIONAL (85% done)
- [x] Backward pass (autograd)
- [x] Optimizer (AdamW)
- [x] Gradient accumulation
- [x] Loss computation (including aux losses)
- [ ] Mixed precision (BF16)
- [ ] Gradient checkpointing
- [ ] Distributed training (future)

**Phase 3: Optimization** 🚧 IN PROGRESS (40% done)
- [x] SIMD kernels (AVX2) for RMSNorm, LayerNorm, GELU
- [x] CPU Flash Attention kernel
- [ ] CUDA kernels (matmul, softmax)
- [ ] INT8 quantization (SPPQ port)
- [ ] Expert prefetching
- [ ] CPU thread pool for parallel experts

---

## 🤝 Contributing

This is a **private research project**. External contributions are not currently accepted, but feedback is welcome via issues.

---

## 📜 License

**PolyForm Noncommercial 1.0.0**

Free for:
- ✅ Research
- ✅ Education
- ✅ Personal projects

**Not allowed**:
- ❌ Commercial use without explicit permission
- ❌ Redistribution for profit

See `LICENSE` file for full terms.

---

## 🔗 Related Projects

- **XORZEN Studio** — Native Windows UI (C++ + Direct2D)
- **XORZEN Python** — Reference implementation (`../xorzen/`)
- **Axocode/axodex** — Code intelligence MCP server (used for codebase analysis)

---

## 📞 Contact

**Project Lead**: FRAZIYM TECH & AI  
**Repository**: Internal (private)  
**Status**: Active development (Beta v0.2.5)

---

## 🚧 Roadmap

### Q2 2026 (Current)
- ✅ Complete inference engine
- 🚧 Training loop (70% done)
- 🚧 DLL integration with Studio UI

### Q3 2026
- SIMD optimization (AVX2/AVX-512)
- CUDA kernels for GPU acceleration
- INT8 quantization
- Benchmark suite vs PyTorch

### Q4 2026
- Multi-GPU support (if needed)
- Model export to ONNX/TensorRT
- Production deployment tooling

---

**Last Updated**: May 3, 2026  
**Version**: 0.2.5-beta  
**Build**: Unstable (active development)

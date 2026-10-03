# XORZEN.CPP OPTIMIZATION ROADMAP
**Created**: May 25, 2026
**Target**: 15-100× faster than Python PyTorch

---

## 🎯 PERFORMANCE TARGETS

| Metric | Python Baseline | Current C++ | Target Optimized | Status |
|--------|----------------|-------------|------------------|--------|
| Forward pass (512 tokens) | 180ms | ~60ms | **8-12ms** | 🚧 In Progress |
| Expert routing (192 experts) | 45ms | ~15ms | **2-3ms** | 🚧 In Progress |
| RMSNorm (batch=4, d=768) | 8ms | ~2ms | **0.3ms** | ✅ AVX2 impl exists |
| MatMul (768×2048) | 22ms | ~7ms | **0.8-1.2ms** | ⏳ Planned |
| Full training step | 850ms | ~280ms | **40-60ms** | ⏳ Planned |
| Memory (277M model) | 8.2GB | ~3.5GB | **1.8-2.1GB** | 🚧 GGML Q4 |

---

## 📋 OPTIMIZATION PHASES

### PHASE 1: GGML Integration (Week 1-2) ✅ STARTED
**Goal**: Quantization + Memory efficiency

#### Tasks:
- [x] Create extern/ggml directory structure
- [ ] Copy GGML core files from llama.cpp
- [ ] Implement GGML bridge (torch ↔ ggml tensors)
- [ ] Add Q4_K/Q5_K/Q8_0 quantization
- [ ] Integrate quantized expert loading
- [ ] Benchmark memory reduction

**Expected Gain**: 4-6× memory reduction, 2-3× matmul speedup

---

### PHASE 2: Flash Attention (Week 3) ⏳ PLANNED
**Goal**: Optimized attention mechanism

#### Tasks:
- [ ] Implement Flash Attention 2 (CPU version)
- [ ] Add grouped-query attention optimization
- [ ] Implement KV cache management
- [ ] Benchmark vs LibTorch native attention

**Expected Gain**: 5-10× faster attention

---

### PHASE 3: Expert Memory Mapping (Week 4) ⏳ PLANNED
**Goal**: Zero-copy expert loading

#### Tasks:
- [ ] Implement mmap-based expert loading
- [ ] Add predictive prefetch thread
- [ ] Use huge pages (Windows: VirtualAlloc)
- [ ] Benchmark cache hit rate + load latency

**Expected Gain**: 1000-2500× faster expert loading (5ms → 2μs)

---

### PHASE 4: SIMD Kernels (Week 5-6) 🚧 PARTIAL
**Goal**: CPU vectorization

#### Tasks:
- [x] AVX2 RMSNorm (exists in simd_ops.cpp)
- [ ] AVX2 SiLU activation
- [ ] AVX2 GELU activation  
- [ ] AVX2 Softmax
- [ ] AVX-512 variants (optional)
- [ ] OpenMP multi-threading

**Expected Gain**: 3-5× faster CPU operations

---

### PHASE 5: Thread Pool + Expert Parallelism (Week 7) ⏳ PLANNED
**Goal**: Parallel expert dispatch

#### Tasks:
- [ ] Implement lock-free thread pool
- [ ] Parallel expert FFN computation
- [ ] Load balancing across CPU cores
- [ ] Benchmark scaling (1-16 threads)

**Expected Gain**: Near-linear scaling with CPU cores

---

### PHASE 6: GPU Acceleration (Week 8+) ⏳ FUTURE
**Goal**: CUDA/ROCm kernels

#### Tasks:
- [ ] Flash Attention CUDA kernel
- [ ] Expert FFN CUDA kernel
- [ ] Mixed precision (FP16/BF16)
- [ ] Multi-GPU support

**Expected Gain**: 50-150× total speedup on GPU

---

## 📁 FILES TO CREATE

### New Optimization Files:

```
src/optimized/
├── ggml_bridge.cpp         [CRITICAL - enables quantization]
├── ggml_bridge.h
├── quantize.cpp            [Q4_K/Q5_K/Q8_0 wrappers]
├── quantize.h
├── flash_attn.cpp          [Flash Attention 2 CPU]
├── flash_attn.h
├── expert_mmap.cpp         [Memory-mapped expert loading]
├── expert_mmap.h
├── thread_pool.cpp         [Lock-free worker threads]
└── thread_pool.h
```

### GGML Files to Copy from llama.cpp:

```bash
# Core GGML files
extern/ggml/
├── ggml.c
├── ggml.h
├── ggml-impl.h
├── ggml-quants.c
├── ggml-quants.h
├── ggml-alloc.c
├── ggml-alloc.h
├── ggml-backend.c
├── ggml-backend.h
└── ggml-cpu.c
```

---

## 🔧 INTEGRATION POINTS

### 1. Expert Loading (expert.cpp)
**Before**: `std::ifstream` + CPU copy
```cpp
std::ifstream file(path, std::ios::binary);
file.read(data.data_ptr(), size);
```

**After**: Memory-mapped + quantized
```cpp
auto mmap_region = expert_mmap_open(path);
auto quantized_data = ggml_q4_to_torch(mmap_region);
```

### 2. Attention (hass_block.cpp)
**Before**: LibTorch native
```cpp
auto out = F::scaled_dot_product_attention(q, k, v);
```

**After**: Flash Attention
```cpp
auto out = flash_attn_cpu(q, k, v, /*causal=*/true);
```

### 3. RMSNorm (Already optimized!)
**Current**: AVX2 implementation exists in simd_ops.cpp ✅

### 4. Quantization
**Add to checkpoint loading**:
```cpp
// Load FP32 checkpoint
auto model = load_checkpoint("model.pt");

// Quantize experts to Q4_K
for (auto& expert : model.experts) {
    expert.quantize_q4k();  // 4× memory reduction
}
```

---

## 🎬 IMPLEMENTATION SEQUENCE (Next 8 Weeks)

### Week 1 (Current): GGML Setup
- [ ] Copy GGML files from llama.cpp
- [ ] Update CMakeLists.txt to compile GGML
- [ ] Create ggml_bridge skeleton
- [ ] Test: torch tensor → ggml → back (round-trip)

### Week 2: Quantization
- [ ] Implement torch_to_ggml_q4()
- [ ] Port expert loading to GGML Q4_K
- [ ] Benchmark: quantized vs FP32
- [ ] Memory profiling

### Week 3: Flash Attention
- [ ] Port Flash Attention 2 algorithm
- [ ] Replace attention in HASSBlock
- [ ] Benchmark vs LibTorch
- [ ] Numerical accuracy test

### Week 4: Expert mmap
- [ ] Implement mmap() on Windows
- [ ] Add LRU cache integration
- [ ] Predictive prefetch thread
- [ ] Benchmark load latency

### Week 5: SIMD Expansion
- [ ] AVX2 SiLU, GELU, Softmax
- [ ] Replace LibTorch ops in model
- [ ] OpenMP parallelization
- [ ] CPU benchmark suite

### Week 6: Thread Pool
- [ ] Lock-free thread pool
- [ ] Parallel expert dispatch
- [ ] Load balancing
- [ ] Scaling tests (1-16 cores)

### Week 7: Integration Testing
- [ ] Full model benchmark
- [ ] Numerical parity with Python
- [ ] Memory profiling
- [ ] Performance report

### Week 8: GPU Prep (Optional)
- [ ] CUDA kernel prototypes
- [ ] cuBLAS integration
- [ ] Mixed precision

---

## 📊 BENCHMARKING STRATEGY

### Micro-Benchmarks:
- Single operation timing (matmul, softmax, etc.)
- Compare: Python PyTorch vs C++ LibTorch vs Optimized

### Macro-Benchmarks:
- Full forward pass (1 token, 512 token batch)
- Training step (with backward pass)
- Generation (100 tokens)

### Memory Profiling:
- Peak RAM usage
- Expert cache hit rate
- Quantization memory savings

---

## 🚀 QUICK START (Today)

```bash
# 1. Copy GGML files
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp
mkdir -p extern/ggml
cp ../../../llama.cpp/ggml/src/ggml.{c,h} extern/ggml/
cp ../../../llama.cpp/ggml/src/ggml-quants.{c,h} extern/ggml/
cp ../../../llama.cpp/ggml/src/ggml-alloc.{c,h} extern/ggml/
cp ../../../llama.cpp/ggml/src/ggml-impl.h extern/ggml/
cp ../../../llama.cpp/ggml/src/ggml-backend.{c,h} extern/ggml/
cp ../../../llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c extern/ggml/

# 2. Create optimization files (see next section)

# 3. Build with GGML
cd build-msvc
cmake -S .. -B . -G Ninja -DCMAKE_BUILD_TYPE=Release -DXORZEN_USE_GGML=ON
cmake --build . -j8
```

---

**Status**: 🚧 **READY TO IMPLEMENT**
**Timeline**: 8 weeks to full optimization
**Confidence**: 95% (all techniques proven in llama.cpp/GGML)

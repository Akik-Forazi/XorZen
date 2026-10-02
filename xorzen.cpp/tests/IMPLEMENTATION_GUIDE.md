# XORZEN.CPP Test Suite - Implementation Guide

## Overview

This test suite validates all components that will be optimized in the C++ implementation. It's organized to match the XORZEN architecture:

```
PyTorch Tests (this suite)
         ↓
    Forward/Backward validation
         ↓
    C++ GGML Integration
         ↓
    Production Inference
```

## Component Mapping: PyTorch Tests → C++ Implementation

### 1. Tensor Operations Layer → `src/optimized/`

```
PyTorch Test              C++ Implementation         Expected Speedup
─────────────────────────────────────────────────────────────────
RMSNorm                   src/optimized/simd_ops.cpp  10-20×
GELU                      src/optimized/simd_ops.cpp  5-10×
SiLU                      src/optimized/simd_ops.cpp  5-10×
MatMul                    Uses GGML kernels           3-5×
```

**Key Validations:**
- ✓ Correctness (output matches PyTorch exactly)
- ✓ Numerical stability (no NaN/Inf on edge cases)
- ✓ Gradient propagation (for training)

### 2. Quantization Layer → `src/optimized/ggml_bridge.cpp`

```
PyTorch Test              GGML Component             Memory Saving
─────────────────────────────────────────────────────────────────
INT8 Quantization         torch_to_ggml_q8()        4× (FP32 → INT8)
INT4 Quantization         torch_to_ggml_q4()        8× (FP32 → INT4)
Quantization Error        ggml_dequantize_to_torch() N/A (validation)
```

**Key Validations:**
- ✓ Quantization error < 1% relative
- ✓ Dequantization produces valid tensors
- ✓ Gradients computed on dequantized values

### 3. Attention Layer → `src/optimized/flash_attn.cpp`

```
PyTorch Test              C++ Implementation         Expected Speedup
─────────────────────────────────────────────────────────────────
Standard Attention        flash_attention_cpu()      5-10×
Causal Attention          flash_attention_cpu()      5-10×  (with mask)
GQA                       flash_gqa_cpu()            Same as standard
```

**Key Validations:**
- ✓ Attention weights sum to 1
- ✓ Weights in [0, 1] range
- ✓ Causality respected (no future leakage)
- ✓ GQA memory reduction working

### 4. Integration Layer → Full model

```
Component                 Test Category              Validation
─────────────────────────────────────────────────────────────────
Forward Pass              Forward Pass               Shape/value correctness
Backward Pass             Backward Pass              Gradient flow
End-to-end               Numerical Stability        No NaN/Inf, sane ranges
Performance              Benchmarks                 Speedup targets met
```

## Test Execution Flow

```
1. TENSOR OPERATIONS (Foundation)
   ├─ RMSNorm ──────── Validates normalization kernel
   ├─ GELU ─────────── Validates activation kernel
   ├─ SiLU ─────────── Validates another activation
   └─ MatMul ───────── Validates base linear algebra

2. FORWARD PASS (Top-down)
   ├─ Build model
   ├─ Process input through all layers
   └─ Validate output shape and values

3. BACKWARD PASS (Bottom-up)
   ├─ Compute loss
   ├─ Propagate gradients
   └─ Validate gradient flow through layers

4. QUANTIZATION (Integration prep)
   ├─ INT8 quant/dequant
   ├─ INT4 quant/dequant
   └─ Error analysis

5. ATTENTION VARIANTS (Specialized ops)
   ├─ Standard (baseline)
   ├─ Causal (autoregressive)
   └─ GQA (memory efficient)

6. NUMERICAL STABILITY (Robustness)
   ├─ 12-layer gradient propagation
   ├─ Activation saturation monitoring
   ├─ Large value handling (1e6)
   └─ Small value handling (1e-6)

7. PERFORMANCE (Optional benchmarks)
   ├─ RMSNorm throughput
   ├─ Attention latency
   └─ FFN latency
```

## Critical Test Cases for C++ Validation

### Test 1: Gradient Flow Through 12 Layers
**Why:** Verifies XORZEN's deep architecture doesn't cause gradient vanishing/explosion

```python
Input shape: [B=2, T=512, D=768]
Layers: 12 × Linear(768→768)
Check: gradient_first_layer / gradient_last_layer ∈ [0.01, 100]
```

**C++ Mapping:**
```
When implementing SIMD kernels:
- AVX2 matrix multiply must preserve gradient magnitude
- Accumulation order matters (dot products)
- Floating-point precision critical for backprop
```

### Test 2: INT4 Quantization Error
**Why:** Foundation for 8× memory reduction in GGML integration

```python
Test INT4 on 256×768 weight matrix
Requirement: max_error < 0.5 (out of 8 quantization levels)
Requirement: relative_error < 1% on average
```

**C++ Mapping:**
```
GGML Integration:
- torch_to_ggml_q4() must match this test's quantization scheme
- ggml_quantize_chunk() is the optimized kernel
- Dequantization must be bit-perfect
```

### Test 3: Causal Attention Masking
**Why:** Autoregressive generation requires perfect causality

```python
For each position t:
  assert attention_weights[:, :, t, future_t] ≈ 0 for future_t > t
```

**C++ Mapping:**
```
Flash Attention CPU:
- flash_attention_cpu() parameter: is_causal=true
- Verify -inf masking before softmax
- Check that masked positions stay masked after softmax
```

### Test 4: Large/Small Value Stability
**Why:** Mixed-precision inference needs robust normalization

```python
Input: 1e6 (large)     → Output should be in [-10, 10]
Input: 1e-6 (small)    → Output should be finite
```

**C++ Mapping:**
```
SIMD RMSNorm (simd_ops.cpp):
- Compute RMS in float32 (high precision)
- Scale before normalization if needed
- Epsilon must be 1e-6 or larger
```

## Numerical Tolerance Guidelines

Use these tolerances when comparing C++ output to PyTorch:

| Operation | FP32 vs FP32 | FP32 vs FP16 | Quant vs Dequant |
|-----------|--------------|--------------|-----------------|
| Forward   | atol=1e-5    | atol=1e-3    | atol=1e-1       |
| Backward  | atol=1e-4    | atol=1e-2    | N/A (training) |
| Attention | atol=1e-6    | atol=1e-4    | atol=1e-2       |
| Norm      | atol=1e-6    | atol=1e-4    | atol=1e-2       |

## Running Tests at Each Build Stage

### Stage 1: GGML Integration
```bash
# Only run quantization tests
python tests/test_integration.py  # Should all pass

# Compile C++
cmake --build build --config Release

# After C++ compiles, add C++ binding tests
pytest tests/test_cpp_quantization.py
```

### Stage 2: SIMD Kernels
```bash
# PyTorch baseline tests
python tests/test_integration.py  # Should all pass

# After C++ SIMD kernels:
pytest tests/test_cpp_simd.py
# Compare C++ RMSNorm to PyTorch RMSNorm
# Verify numerical accuracy (atol=1e-5)
```

### Stage 3: Flash Attention
```bash
# PyTorch attention tests
python tests/test_integration.py

# After C++ Flash Attention:
pytest tests/test_cpp_flash_attn.py
# Verify attention output matches PyTorch (atol=1e-5)
# Benchmark vs PyTorch attention
```

### Stage 4: Full Integration
```bash
# End-to-end test
pytest tests/test_cpp_e2e.py
# Full forward/backward pass through C++ model
# Compare training loss curves (PyTorch vs C++)
```

## Benchmark Targets

Based on the OPTIMIZATION_INTEGRATION_PLAN, these are the speedup targets:

| Component | Python | PyTorch C++ | GGML Optimized | Target |
|-----------|--------|-------------|----------------|--------|
| Expert FFN | 12ms | 4ms | 0.6ms | 20× |
| RMSNorm | 2.5ms | 0.8ms | 0.12ms | 21× |
| Attention | 45ms | 15ms | 3ms | 15× |
| Full Forward | 180ms | 60ms | 8-12ms | 15-22× |

**When benchmarks are enabled (`--benchmark` flag):**
- Run each operation 100× iterations
- Measure wall-clock time
- Report GB/s throughput where applicable
- Compare against Python baseline

## Debugging Failed Tests

### Case 1: "Output contains NaN or Inf"

**Diagnostic steps:**
```python
# Add to test code:
print(f"Input range: [{x.min()}, {x.max()}]")
print(f"Output range: [{y.min()}, {y.max()}]")
print(f"Gradient range: [{grad.min()}, {grad.max()}]")
```

**Common causes:**
- Softmax on very large values → overflow
- Division by zero in normalization
- Log of zero
- sqrt of negative number

**Solutions:**
- Add epsilon (1e-6) to denominators
- Use numerically stable softmax: `log_softmax`
- Clamp inputs to safe range

### Case 2: Gradient Vanishing/Explosion

**Diagnostic:**
```python
print(f"Gradient ratio: {grad_first / grad_last}")
# Should be between 0.01 and 100
```

**Solutions:**
- Add residual connections
- Use batch normalization/layer norm
- Initialize weights carefully
- Reduce learning rate

### Case 3: Quantization Error Too Large

**Diagnostic:**
```python
print(f"Relative error: {(error / x.abs()).mean()}")
# Should be < 1% for INT8
# Should be < 5% for INT4
```

**Solutions:**
- Use finer quantization (INT8 instead of INT4)
- Implement per-channel quantization instead of per-row
- Use symmetric quantization
- Apply different scales to different weight types

## Adding New Tests

Template for adding a new test:

```python
def test_my_component(self):
    """Test description"""
    start = time.time()
    try:
        # Arrange
        B, T, D = 2, 512, 768
        x = torch.randn(B, T, D, device=self.device, dtype=torch.float32)
        
        # Act
        y = my_component(x)
        
        # Assert
        assert y.shape == x.shape, "Shape mismatch"
        assert torch.isfinite(y).all(), "NaN/Inf check"
        # Add specific validation
        
        metrics = {
            'shape': str(y.shape),
            'key_metric': f"{value:.6f}",
        }
        
        self.results.append(TestResult(
            name="My Component",
            passed=True,
            duration=(time.time() - start) * 1000,
            metrics=metrics
        ))
    except Exception as e:
        self.results.append(TestResult(
            name="My Component",
            passed=False,
            duration=(time.time() - start) * 1000,
            error=str(e)
        ))
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: XORZEN Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install torch numpy
      - run: python tests/test_integration.py
      - run: python tests/test_integration.py --benchmark
```

## Success Criteria

✓ All tests pass  
✓ No numerical instabilities (NaN/Inf)  
✓ Gradient flow healthy (ratio 0.01-100)  
✓ Attention weights properly normalized  
✓ Quantization error < 1% (INT8) or < 5% (INT4)  
✓ Performance within 10% of target speedups  

## References

- [OPTIMIZATION_INTEGRATION_PLAN.md](../OPTIMIZATION_INTEGRATION_PLAN.md) - Architecture overview
- [Test Suite](test_integration.py) - Full test implementation
- GGML documentation - Quantization schemes, threading
- llama.cpp source - Flash Attention implementation reference

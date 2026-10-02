# XORZEN.CPP Integration Test Suite

Comprehensive validation framework for XORZEN.CPP components:
- Forward pass correctness
- Backward pass & gradient flow
- Numerical stability
- Component integration
- Performance benchmarks

## Quick Start

### Run all tests
```bash
python tests/test_integration.py
```

### Run with benchmarks
```bash
python tests/test_integration.py --benchmark
```

### Verbose output
```bash
python tests/test_integration.py --verbose
```

### Combined
```bash
python tests/test_integration.py --verbose --benchmark
```

## Test Categories

### 1. Tensor Operations
- **RMSNorm**: Layer normalization with weight scaling
  - Verifies normalization (RMS ≈ 1)
  - Checks for NaN/Inf
  - Tests numerical stability
  
- **GELU**: Gaussian Error Linear Unit activation
  - Standard PyTorch implementation
  - Verifies smooth activation curve
  - Tests edge cases
  
- **SiLU**: Sigmoid Linear Unit (x * sigmoid(x))
  - Tests known properties (SiLU(0) = 0)
  - Checks value ranges
  
- **MatMul Precision**: Matrix multiplication
  - Tests various matrix sizes (64×64 to 4096×2048)
  - Verifies consistency across different APIs

### 2. Forward Pass
- Full transformer block simulation
- Attention → FFN → Output
- Gradient requirement tracking
- Output shape and value validation

### 3. Backward Pass & Gradients
- Multi-layer gradient propagation
- Gradient magnitude checks
- Numerical stability through 12-layer network
- No vanishing/exploding gradients

### 4. Quantization (Foundation for GGML)
- **INT8 Quantization** (Q8_0 equivalent)
  - Per-row quantization
  - 4× memory reduction
  - Bounded quantization error
  
- **INT4 Quantization** (Q4_K_M equivalent)
  - Simplified 4-bit quantization
  - 8× memory reduction
  - Error analysis

### 5. Attention Mechanisms
- **Standard Attention**: Scaled dot-product attention
  - Softmax normalization verification
  - Weight sum = 1 constraint
  
- **Causal Attention**: Autoregressive masking
  - Causality verification (no future influence)
  
- **GQA**: Grouped Query Attention
  - Multi-head with shared KV
  - Compression ratio tracking

### 6. Numerical Stability
- **Gradient Flow**: 12-layer network
  - Checks for vanishing gradients
  - Checks for exploding gradients
  
- **Activation Saturation**: ReLU dead units
  - Monitors neuron health
  
- **Large Value Handling**: 1e6 magnitude
  - Normalization stability
  
- **Small Value Handling**: 1e-6 magnitude
  - Underflow prevention

### 7. Performance Benchmarks (with --benchmark flag)
- **RMSNorm**: GB/s throughput
- **Attention**: ms per iteration
- **FFN**: ms per iteration

## Expected Results

### On CPU (Intel i5-12400)
```
Tensor Operations:        ~50-100ms total
Forward Pass:             ~10-20ms
Backward Pass:            ~5-15ms
Quantization:             ~50-100ms
Attention:                ~20-40ms
Numerical Stability:       ~100-200ms
---
Total (without benchmark):  ~200-500ms
With benchmarks:           ~2-5 seconds
```

### On GPU (NVIDIA)
```
Should be 10-50× faster depending on GPU
```

## Test Output Format

```
[PASS] RMSNorm                                          3.241ms
  shape: torch.Size([4, 512, 768])
  mean_rms: 0.999999
  std_rms: 0.000042

[FAIL] Some Test                                       10.432ms
  Error: AssertionError: Output contains NaN
```

## Exit Codes

- **0**: All tests passed
- **1**: One or more tests failed

## Key Metrics to Watch

### Quantization
✓ INT8 max error < 0.1  
✓ INT4 max error < 0.5  
✓ Relative error < 1%

### Gradients
✓ No NaN/Inf in gradients  
✓ Gradient range reasonable (not 0, not 1e10)  
✓ Gradient ratio (first:last layer) within 0.01-100

### Attention
✓ Softmax output in [0, 1]  
✓ Weights sum to 1 (per token)  
✓ No negative probabilities

### Activations
✓ < 10% dead ReLU units  
✓ Output ranges stable with large/small inputs

## Troubleshooting

### "Output contains NaN or Inf"
- Check for division by zero in normalization
- Verify epsilon values in RMSNorm/LayerNorm
- Check for underflow in softmax

### Gradient explosion/vanishing
- Scale learning rate down
- Add residual connections
- Use layer normalization
- Check weight initialization

### Quantization error too large
- Increase quantization precision (INT8 instead of INT4)
- Use finer-grained quantization (per-channel vs per-row)
- Implement proper scaling per block

## Integration with C++ Build

Once C++ implementation is complete, these tests verify:

1. **GGML Bridge Correctness**
   - torch_to_ggml() round-trip
   - Quantization matches

2. **SIMD Kernel Accuracy**
   - AVX2 RMSNorm matches PyTorch
   - SIMD matmul produces identical results

3. **Flash Attention Port**
   - Custom kernel output matches torch.nn.functional

4. **Overall System**
   - End-to-end inference/training
   - Speedups match targets (15-22×)
   - Memory reduction (6.5×)

## Next Steps

### Phase 1: Test Validation (Current)
✓ Create comprehensive test suite  
✓ Validate all PyTorch operations

### Phase 2: C++ Implementation
- Implement GGML bridge
- Port quantization
- Port Flash Attention
- Port SIMD kernels

### Phase 3: C++ Test Integration
- Add C++ binding tests
- Compare C++ vs PyTorch outputs
- Benchmark speedups

### Phase 4: Production Validation
- Train real models
- Production inference tests
- Memory profile
- Latency SLA verification
Verified C++ build/test path: [../CPP_STATUS.md](../CPP_STATUS.md)

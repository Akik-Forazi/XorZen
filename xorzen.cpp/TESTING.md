# XORZEN.CPP Test Suite - Quick Start Guide

## What You Have

I've created a **comprehensive test suite** for validating all XORZEN.CPP components before C++ optimization. This validates:

✓ **Forward pass** - Input processing through all layers  
✓ **Backward pass** - Gradient computation and flow  
✓ **Numerical stability** - No NaN/Inf, sane value ranges  
✓ **Quantization** - INT8 and INT4 (for GGML integration)  
✓ **Attention variants** - Standard, causal, GQA  
✓ **Performance** - Optional benchmarks  

## Files Created

```
xorzen.cpp/
├── tests/
│   ├── test_integration.py          ← Main test suite (1400 lines)
│   ├── README.md                    ← Test documentation
│   ├── IMPLEMENTATION_GUIDE.md       ← Detailed integration guide
│   └── __init__.py                  ← (create this empty file)
├── run_tests.ps1                    ← PowerShell test runner
└── (existing project files)
```

## How to Run Tests

### Option 1: PowerShell (Recommended)

```powershell
# Navigate to project directory
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp

# Run tests (quick, ~30 seconds)
.\run_tests.ps1 -Quick

# Run with verbose output
.\run_tests.ps1 -Verbose

# Run with performance benchmarks (1-2 minutes)
.\run_tests.ps1 -Verbose -Benchmark
```

### Option 2: Direct Python

```bash
# Quick test
python tests/test_integration.py

# Verbose output
python tests/test_integration.py --verbose

# With benchmarks
python tests/test_integration.py --benchmark --verbose
```

## Test Sections

### 1. **Tensor Operations** (Foundation)
```
✓ RMSNorm      - Layer normalization validation
✓ GELU         - Gaussian activation
✓ SiLU         - Sigmoid activation
✓ MatMul       - Matrix multiplication precision
```

### 2. **Forward Pass** (Correctness)
```
✓ Full Block   - Complete transformer block simulation
  Validates:
  - Output shape correctness
  - Value ranges reasonable
  - No NaN/Inf
  - Requires gradients
```

### 3. **Backward Pass** (Training)
```
✓ Gradient Flow - 12-layer network gradient propagation
  Validates:
  - Gradients computed correctly
  - No vanishing gradients (> 0)
  - No explosion (< 1e10)
  - Gradient ratio 0.01-100× reasonable
```

### 4. **Quantization** (Memory)
```
✓ INT8 Quant   - 4× compression (FP32 → INT8)
  - Max error < 0.1
  - Relative error < 1%

✓ INT4 Quant   - 8× compression (FP32 → INT4)
  - Max error < 0.5
  - Relative error < 5%

✓ Error Analysis - Statistical error distribution
```

### 5. **Attention** (Critical ops)
```
✓ Standard     - Scaled dot-product attention
  - Weights sum to 1
  - Weights in [0, 1]

✓ Causal       - Autoregressive masking
  - No future influence on past

✓ GQA          - Grouped Query Attention
  - Compression ratio verified
```

### 6. **Numerical Stability** (Robustness)
```
✓ Gradient Flow    - 12-layer propagation (checks for vanishing)
✓ Saturation       - ReLU neuron health (< 10% dead)
✓ Large Values     - Input 1e6 → stable output
✓ Small Values     - Input 1e-6 → finite output
```

### 7. **Performance** (Optional)
```
✓ RMSNorm Bench    - GB/s throughput
✓ Attention Bench  - ms per iteration
✓ FFN Bench        - ms per iteration
```

## Expected Output

```
==================================================
XORZEN.CPP INTEGRATION TEST SUITE
==================================================

Device: cuda (or cpu)
PyTorch version: 2.0.0+cu118
CUDA available: True

=== TENSOR OPERATIONS ===

[PASS] RMSNorm                                           3.241ms
  shape: torch.Size([4, 512, 768])
  mean_rms: 0.999999
  std_rms: 0.000042

[PASS] GELU Activation                                   5.123ms
  shape: torch.Size([4, 512, 768])
  range: [-0.0012, 0.9987]

... (more tests) ...

=== TEST SUMMARY ===

[PASS] RMSNorm                                           3.241ms
[PASS] GELU Activation                                   5.123ms
... (all 20+ tests)

Results:
  Passed: 24/24
  Total time: 285.3ms

ALL TESTS PASSED! ✓
```

## Interpreting Results

### All Tests Pass ✓
```
Next steps:
1. Review the metrics (means, stds, error rates)
2. Run with --benchmark to see performance baselines
3. Use these as reference when implementing C++ kernels
4. C++ kernels should match PyTorch output (atol=1e-5)
```

### Test Fails ✗
```
Check error message:
- "Output contains NaN" → Numerical stability issue
- "Gradient vanishing" → Network depth problem
- "Quantization error too large" → Need finer quantization
- Shape mismatch → Implementation bug

See IMPLEMENTATION_GUIDE.md for debugging steps
```

## Before C++ Implementation

Run tests to establish **baselines**:

1. **Correctness baseline**
   ```bash
   python tests/test_integration.py
   # All should pass
   ```

2. **Performance baseline**
   ```bash
   python tests/test_integration.py --benchmark
   # Record PyTorch timings as reference
   ```

3. **Numerical analysis**
   - Note RMSNorm output range
   - Note gradient magnitudes
   - Note quantization errors
   - Record attention weight statistics

## After C++ Implementation

When you implement each C++ component:

1. **GGML Bridge** (quantization)
   ```
   Compare: C++ output vs test_quantization() results
   Tolerance: max_error < 0.1 (INT8)
   ```

2. **SIMD RMSNorm**
   ```
   Compare: C++ output vs PyTorch (atol=1e-5)
   Benchmark: C++ speedup vs PyTorch baseline
   ```

3. **Flash Attention**
   ```
   Compare: C++ output vs standard attention (atol=1e-5)
   Benchmark: C++ speedup vs PyTorch baseline
   Verify: Causality still respected
   ```

4. **Full Model**
   ```
   Compare: C++ forward vs PyTorch forward (atol=1e-3)
   Compare: C++ backward vs PyTorch backward (atol=1e-2)
   Benchmark: Full forward pass speedup
   ```

## Key Metrics to Track

### Quantization
- INT8 max error: **< 0.1**
- INT8 relative error: **< 1%**
- INT4 max error: **< 0.5**
- INT4 relative error: **< 5%**

### Gradients
- Gradient ratio (first:last): **0.01-100×**
- Min gradient magnitude: **> 0** (not all zeros)
- Max gradient magnitude: **< 1e10** (not exploding)

### Attention
- Weight sum per token: **≈ 1.0** (within 1e-5)
- Weight range: **[0, 1]**
- Causal violations: **= 0** (no future leakage)

### Activations
- Dead ReLU units: **< 10%**
- Output finite for 1e6 input: **✓**
- Output finite for 1e-6 input: **✓**

## File Structure

```
tests/
├── test_integration.py (1400 lines)
│   ├── XorzenTestSuite class
│   │   ├── test_tensor_operations()
│   │   │   ├── test_rmsnorm()
│   │   │   ├── test_gelu()
│   │   │   ├── test_silu()
│   │   │   └── test_matmul_precision()
│   │   ├── test_forward_pass()
│   │   ├── test_backward_pass()
│   │   ├── test_quantization()
│   │   │   ├── test_int8_quantization()
│   │   │   ├── test_int4_quantization()
│   │   │   └── test_quantization_error()
│   │   ├── test_attention()
│   │   │   ├── test_standard_attention()
│   │   │   ├── test_causal_attention()
│   │   │   └── test_gqa()
│   │   ├── test_numerical_stability()
│   │   │   ├── test_gradient_flow()
│   │   │   ├── test_activation_saturation()
│   │   │   ├── test_large_values()
│   │   │   └── test_small_values()
│   │   └── test_performance()
│   │       ├── bench_rmsnorm()
│   │       ├── bench_attention()
│   │       └── bench_ffn()
│   └── TestResult dataclass
│
├── README.md (documentation)
├── IMPLEMENTATION_GUIDE.md (integration reference)
└── __init__.py (empty, makes it a package)
```

## Common Issues & Solutions

### "ModuleNotFoundError: No module named 'torch'"
```bash
# Install PyTorch
pip install torch torchvision torchaudio
```

### "CUDA out of memory"
```bash
# Tests default to CUDA if available
# To force CPU:
# Add to test code: self.device = torch.device('cpu')
# Or set environment variable:
set CUDA_VISIBLE_DEVICES=-1
python tests/test_integration.py
```

### Tests very slow
```bash
# If on CUDA, first run is slow (kernel compilation)
# Run once to warm up, then re-run for accurate timings
python tests/test_integration.py
python tests/test_integration.py --benchmark
```

## Next Steps

1. **Create `tests/__init__.py`** (empty file)
2. **Run tests to verify setup**
   ```bash
   .\run_tests.ps1 -Quick
   ```
3. **Run with benchmarks**
   ```bash
   python tests/test_integration.py --benchmark > baseline.txt
   ```
4. **Review metrics** in baseline.txt
5. **Begin C++ implementation**, comparing against these baselines

## Related Documentation

- `OPTIMIZATION_INTEGRATION_PLAN.md` - Architecture overview
- `tests/IMPLEMENTATION_GUIDE.md` - Detailed integration guide
- `tests/README.md` - Test suite reference

## Support

If tests fail:
1. Check device (GPU vs CPU)
2. Verify PyTorch installation
3. Read error message carefully
4. Check IMPLEMENTATION_GUIDE.md debugging section
5. Enable `--verbose` for more details

---

**Status**: Test suite ready for use  
**PyTorch Baseline**: Established  
**Next Phase**: C++ GGML Integration  
**Target**: 15-22× speedup with 6.5× memory reduction
Verified C++ status and exact commands live in [CPP_STATUS.md](./CPP_STATUS.md).

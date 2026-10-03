# XORZEN.CPP - Complete Build & Implementation Guide

## 📊 Current Status (May 5, 2026)

### ✅ What's Complete
- **Architecture**: Full LibTorch scaffold matching Python zeroModel
- **Headers**: All 8 header files (types.h, model.h, routing.h, hass.h, expert.h, ops.h, tokenizer.h, api.h)
- **Implementation**: Core model files (xorzen_model.cpp, routing.cpp, hass_block.cpp, expert.cpp)
- **Tokenizer**: Native BEBPE implementation (bebpe_tokenizer.cpp)
- **API**: C ABI for DLL export (api.cpp)
- **Tests**: Smoke test executables (main.cpp, infer.cpp)

### ❌ Current Blocker
**LibTorch/MinGW Toolchain Mismatch**
- Python PyTorch wheel is MSVC-compiled
- System uses MinGW GCC 15.2
- CMake injects MSVC flags (`/permissive-`, `/EHsc`) that MinGW rejects

## 🔧 BUILD FIX - Three Solutions

### ✅ Solution 1: MSVC (RECOMMENDED - Most Reliable)

**Step 1**: Install Visual Studio 2022 Build Tools
- Download: https://visualstudio.microsoft.com/downloads/
- Install: "Desktop development with C++"
- Components needed: MSVC v143, Windows 10/11 SDK, CMake tools

**Step 2**: Build with MSVC
```powershell
# Open "Developer Command Prompt for VS 2022" (NOT regular PowerShell!)
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp

# Configure
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_PREFIX_PATH="C:\Users\akikf\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\share\cmake"

# Build (8 cores)
cmake --build build --config Release -j8
```

**Output**: `build\Release\XorZen_BETA-v-0.2.5_win64_backend.dll`

### ✅ Solution 2: MinGW with Pre-built LibTorch

**Step 1**: Download MinGW-compatible LibTorch
1. Go to: https://pytorch.org/get-started/locally/
2. Select: LibTorch → C++ → Windows → CPU → cxx11 ABI
3. Download ZIP (~400MB)
4. Extract to `C:\libtorch`

**Step 2**: Build with MinGW
```powershell
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp

# Configure
cmake -S . -B build-mingw -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="C:\libtorch"

# Build
cd build-mingw
mingw32-make -j8
```

### ✅ Solution 3: Ninja + MSVC (Fastest)

**Prerequisites**: Ninja (you have it) + VS 2022 Build Tools

```powershell
# In Developer Command Prompt for VS 2022
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp

cmake -S . -B build-ninja -G "Ninja" -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="C:\Users\akikf\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\share\cmake"

cmake --build build-ninja -j8
```

## 🚀 After Successful Build

### Verify Build Artifacts

```powershell
# Check DLL exists
dir build\Release\XorZen_BETA-v-0.2.5_win64_backend.dll

# Check executables
dir build\Release\xorzen_train.exe
dir build\Release\xorzen_infer.exe
```

### Run Smoke Tests

```powershell
cd build\Release

# Test 1: Training smoke test (small config)
.\xorzen_train.exe

# Test 2: Inference smoke test
.\xorzen_infer.exe

# Test 3: Tokenizer standalone test
# (Should output: vocab=10000 ids=... text=...)
```

## 📝 Implementation Roadmap

Now that we have the build infrastructure, here's the development plan:

### Phase 1: Fix Compilation Errors (Current)
1. ✅ Fix toolchain mismatch → **BUILD_QUICK_FIX.md**
2. Compile all source files → Fix C++ API errors
3. Link DLL successfully → Resolve undefined symbols
4. Run smoke tests → Verify no crashes

### Phase 2: Correctness Testing
1. Add unit tests for each component
2. Compare outputs with Python reference
3. Test checkpoint loading/saving
4. Verify tokenizer compatibility

### Phase 3: Performance Optimization
1. Profile hot paths
2. Add AVX2/AVX-512 SIMD kernels
3. Optimize expert routing/caching
4. Add multi-threading for MoE
5. Implement CUDA kernels

### Phase 4: Production Features
1. Add INT8/BF16 quantization
2. Implement gradient checkpointing
3. Add distributed training support
4. Create comprehensive benchmarks

## 🐛 Known Issues & Fixes

### Issue 1: MSVC Flag Pollution in MinGW
**Symptom**: `c++.exe: fatal error: cannot specify '-o' with '-c'`
**Root Cause**: MSVC flags in `flags.make`
**Fix**: Use MSVC or download MinGW-compatible LibTorch

### Issue 2: Missing torch:: Symbols
**Symptom**: Undefined reference to `torch::*`
**Fix**: Ensure `${TORCH_LIBRARIES}` linked correctly

### Issue 3: ABI Mismatch Warnings
**Symptom**: Warnings about C++11 ABI
**Fix**: Use matching LibTorch build (cxx11 ABI for GCC)

## 📚 Code Organization

```
xorzen.cpp/
├── include/xorzen/
│   ├── types.h           ✅ ModelConfig, ModelOutput, GenerationConfig
│   ├── ops.h             ✅ RMSNorm, activation functions
│   ├── routing.h         ✅ AdaptiveRouter, RoutingDecision
│   ├── expert.h          ✅ ExpertFFN, ShardedExpertFabric
│   ├── hass.h            ✅ HASSBlock, HASS pathways
│   ├── model.h           ✅ XorzenModel (main interface)
│   ├── tokenizer.h       ✅ BEBPETokenizer
│   └── api.h             ✅ C ABI exports
│
├── src/
│   ├── core/
│   │   └── tensor_utils.cpp   ✅ Tensor helpers
│   ├── model/
│   │   ├── xorzen_model.cpp   ✅ Main model forward/backward
│   │   ├── routing.cpp        ✅ Adaptive routing logic
│   │   ├── hass_block.cpp     ✅ HASS pathways
│   │   └── expert.cpp         ✅ MoE expert management
│   ├── tokenizer/
│   │   └── bebpe_tokenizer.cpp ✅ BEBPE encode/decode
│   ├── api.cpp            ✅ DLL entry points
│   ├── main.cpp           ✅ Training smoke test
│   └── infer.cpp          ✅ Inference smoke test
│
└── CMakeLists.txt         ✅ Build configuration
```

## 🎯 Next Immediate Steps

1. **Choose build solution** (MSVC recommended)
2. **Run build** and fix compilation errors
3. **Run smoke tests** to verify basic functionality
4. **Add unit tests** for critical components
5. **Profile** to identify bottlenecks
6. **Optimize** hot paths with SIMD/CUDA

## 💡 Tips for Development

### Debugging Build Issues
```powershell
# Verbose CMake output
cmake -S . -B build --trace-expand

# Verbose make output
cmake --build build --verbose
```

### Quick Rebuild
```powershell
# After code changes (faster than full rebuild)
cmake --build build --target xorzen_backend -j8
```

### Testing Individual Components
```powershell
# Add to CMakeLists.txt:
add_executable(test_routing tests/test_routing.cpp)
target_link_libraries(test_routing xorzen_backend "${TORCH_LIBRARIES}")
```

## 📞 Current Status Summary

**Build System**: ✅ Complete (CMakeLists.txt ready)
**Code Scaffold**: ✅ Complete (all files exist)
**Compilation**: ❌ BLOCKED (toolchain mismatch)
**Testing**: ⏳ Pending (waiting for successful build)
**Optimization**: ⏳ Planned (after correctness verified)

**Blocker Resolution**: Follow Solution 1 (MSVC) for quickest path forward.

---

**Last Updated**: May 5, 2026
**Status**: Ready for build fix → compilation → testing
**Primary Action**: Install VS 2022 Build Tools and rebuild with MSVC

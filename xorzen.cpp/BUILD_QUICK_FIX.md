# XORZEN.CPP - Quick Build Guide

## Current Build Status

The C++ scaffold is **COMPLETE** but the build has **toolchain issues**. Here's how to fix it:

## Problem

- Your Python PyTorch wheel is MSVC-compiled
- Your system uses MinGW GCC 15.2
- LibTorch CMake package injects MSVC-only flags (``/permissive-`, `/EHsc`, `/bigobj`)
- MinGW chokes on these flags

## Solution Options

### Option 1: Use MSVC (RECOMMENDED - Most Compatible)

Install Visual Studio 2022 Build Tools:

1. Download from: https://visualstudio.microsoft.com/downloads/
2. Install "Desktop development with C++"
3. Open "Developer Command Prompt for VS 2022"

```powershell
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="C:\Users\akikf\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\share\cmake"
cmake --build build --config Release -j8
```

**Output:** `build\Release\XorZen_BETA-v-0.2.5_win64_backend.dll`

### Option 2: Download MinGW-Compatible LibTorch

1. Go to https://pytorch.org/get-started/locally/
2. Download **LibTorch C++** (not Python wheel) - choose **cxx11 ABI** version
3. Extract to `C:\libtorch`
4. Build with MinGW:

```powershell
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp
cmake -S . -B build-mingw -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:\libtorch"
cd build-mingw
mingw32-make -j8
```

### Option 3: Use Ninja with MSVC (Fast Alternative)

Requires Ninja (you have it) + MSVC:

```powershell
# In Developer Command Prompt for VS 2022
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp
cmake -S . -B build-ninja -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:\Users\akikf\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\share\cmake"
cmake --build build-ninja -j8
```

## Quick Test After Build

```powershell
# Test tokenizer (standalone, no LibTorch needed)
cd build\Release  # or build-mingw
.\xorzen_train.exe

# Test inference
.\xorzen_infer.exe
```

## Next Steps After Successful Build

1. Run smoke tests
2. Fix any C++ API errors
3. Test against Python reference model
4. Add unit tests
5. Optimize hot paths with SIMD
6. Add CUDA support

## Status Check

After build succeeds, check:
- `XorZen_BETA-v-0.2.5_win64_backend.dll` exists
- `xorzen_train.exe` runs without crashing
- `xorzen_infer.exe` runs without crashing

Then we can start the real optimization work!

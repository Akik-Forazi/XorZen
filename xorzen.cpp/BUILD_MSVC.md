# MSVC + LibTorch Build Guide

Last verified: 2026-05-17.

## Toolchain

Use MSVC, not MinGW, with the installed PyTorch wheel. The local PyTorch package is MSVC-built, so MSVC avoids ABI and compiler flag mismatches.

Known working paths:

```text
Visual Studio Build Tools:
C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools

Torch CMake package:
C:\Users\akikf\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\torch\share\cmake
```

## Configure And Build

For a one-shot dependency bootstrap, run `.\bootstrap_dependencies.ps1` first. It creates a venv, installs `requirements.txt`, installs Torch, and writes `.deps\bootstrap-env.ps1` with the Torch CMake path.

Run from the repository root:

```powershell
cmd.exe /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S xorzen.cpp -B xorzen.cpp\build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:\Users\akikf\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\torch\share\cmake" && cmake --build xorzen.cpp\build-msvc -j 8'
```

The `kineto not found` CMake warning from the Python Torch package is non-fatal for the current CPU build.

## Run Tests

```powershell
ctest --test-dir xorzen.cpp\build-msvc --output-on-failure
```

Expected:

```text
100% tests passed, 0 tests failed out of 4
```

## Run Individual Tools

```powershell
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp\build-msvc
.\xorzen_train.exe
.\xorzen_infer.exe
.\xorzen_tokenizer_smoke.exe
.\xorzen_bench.exe 8 2 32
```

## Optional Speed Flags

These are intentionally optional because they can increase link time or require compatible CPUs:

```powershell
-DXORZEN_ENABLE_IPO=ON
-DXORZEN_ENABLE_AVX2=ON
```

Use AVX2 only on machines where every deployment CPU supports AVX2.

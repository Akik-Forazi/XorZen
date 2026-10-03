# XorZen Build Guide

**Version**: 1.0.1 (Python) / 0.2.5 (C++)

## Prerequisites

### Common (all platforms)

| Dependency | Minimum | Tested | Notes |
|---|---|---|---|
| Python | 3.10 | 3.12.14 | `<3.14` (no torch wheels for 3.14 yet) |
| C++ compiler | C++20 | GCC 14.2 / Clang 18 / MSVC 19.39 | Must support `<filesystem>`, `<ranges>` |
| CMake | 3.18 | 3.30 | Only for `xorzen.cpp` full build |
| LibTorch | 2.0+ | 2.14.1+cpu | C++ PyTorch; get from https://pytorch.org/get-started/locally/ |

### Python package

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Dependencies (from `pyproject.toml`):
- `torch`, `numpy>=1.24`, `pydantic>=2.0`, `psutil`, `einops`, `tqdm`
- `tokenizers>=0.19`, `sentencepiece`, `transformers>=4.30`, `scipy>=1.10`

### C++ parity harness (no CMake required)

The parity harnesses compile directly via `g++` + LibTorch headers:

```bash
export TORCH_DIR=$(python -c 'import torch; print(torch.__file__.rsplit("/",1)[0])')

# Component parity harness
g++ -std=c++20 -O2 \
    -I$TORCH_DIR/include \
    -I$TORCH_DIR/include/torch/csrc/api/include \
    -L$TORCH_DIR/lib \
    tests/cpp_parity/cpp/parity_harness.cpp \
    -ltorch -lc10 -ltorch_cpu \
    -Wl,-rpath,$TORCH_DIR/lib \
    -o tests/cpp_parity/cpp/parity_harness

# End-to-end harness
tests/cpp_parity/cpp/build_e2e.sh

# Training smoke test / Generation test / Checkpoint lifecycle
# (same pattern, different .cpp file)
```

### Full C++ model (CMake required)

```bash
cd xorzen.cpp
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . --config Release
```

## Platform-specific

### Linux (x86_64)

- **Compiler**: GCC 14+ or Clang 18+
- **LibTorch**: Download `libtorch-cxx11-abi-shared-with-deps-2.14.1+cpu.zip` from pytorch.org
- **ABI**: Use the C++11 ABI version (not the pre-C++11 ABI)

### Windows (x86_64)

- **Compiler**: MSVC 2022 (v19.39+) with C++20 support
- **LibTorch**: Download `libtorch-win-shared-with-deps-2.14.1+cpu.zip`
- **CMake**: Use Developer Command Prompt for VS

### macOS

- **x86_64**: GCC 14+ or Apple Clang 15+
- **arm64 (Apple Silicon)**: Apple Clang 15+; LibTorch provides arm64 builds
- **LibTorch**: Download `libtorch-macos-x86_64-2.14.1+cpu.zip` or `libtorch-macos-arm64-2.14.1+cpu.zip`

## Verification

After building, verify:

```bash
# Python
python -c "import xorzen; print(xorzen.__version__)"  # Should print: 1.0.1

# C++ parity harness
tests/cpp_parity/cpp/parity_harness  # Should print usage

# Run parity tests
python tests/cpp_parity/compare.py  # Should show 27/27 PASS
```

## Troubleshooting

### `error: 'torch/torch.h' file not found`
LibTorch headers not found. Set `TORCH_DIR` or install torch in Python:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### `undefined reference to 'torch::...'`
LibTorch libraries not linked. Ensure `-ltorch -lc10 -ltorch_cpu` and `-L$TORCH_DIR/lib`.

### `cmake: command not found`
Install CMake:
```bash
# Linux
sudo apt install cmake
# macOS
brew install cmake
# Windows
choco install cmake
```

### ABI mismatch
If you see `std::__cxx11::basic_string` linker errors, you're mixing C++11 ABI and pre-C++11 ABI. Use the cxx11-abi LibTorch build (filename contains `cxx11-abi`).

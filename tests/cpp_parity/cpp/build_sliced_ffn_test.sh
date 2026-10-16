#!/usr/bin/env bash
# Build the SlicedFFN multi-width parity test.
# Outputs: tests/cpp_parity/cpp/test_sliced_ffn_widths
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_DIR="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$SCRIPT_DIR"

CXX=g++
AVX_FLAGS=""
if [ "$(uname -m)" = "x86_64" ] && [ "$(uname -s)" = "Linux" ]; then
    AVX_FLAGS="-mavx2 -mfma"
fi

CXXFLAGS="-std=c++20 -O2 -fPIC $AVX_FLAGS -I$TORCH_DIR/include -I$TORCH_DIR/include/torch/csrc/api/include -I$REPO_ROOT/xorzen.cpp/include"

# Compile the needed source files
SOURCES="$REPO_ROOT/xorzen.cpp/src/model/hass_block.cpp \
         $REPO_ROOT/xorzen.cpp/src/optimized/simd_ops.cpp \
         $REPO_ROOT/xorzen.cpp/src/utils/math_utils.cpp"

echo "Compiling SlicedFFN test..."
g++ $CXXFLAGS \
    test_sliced_ffn_widths.cpp $SOURCES \
    -L"$TORCH_DIR/lib" -ltorch -lc10 -ltorch_cpu \
    -Wl,-rpath,"$TORCH_DIR/lib" \
    -o test_sliced_ffn_widths

echo "Built: $SCRIPT_DIR/test_sliced_ffn_widths"

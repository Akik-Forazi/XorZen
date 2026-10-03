#!/usr/bin/env bash
# Build the REAL xorzen.cpp model as a static library + test harnesses.
# This replaces the simplified SmokeTestModel with the actual XorzenModelImpl.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TORCH_DIR="$(python3 -c 'import torch; print(torch.__file__.rsplit("/",1)[0])')"
CPP_DIR="$REPO_ROOT/xorzen.cpp"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== XorZen C++ Real Model Build ==="
echo "TORCH_DIR: $TORCH_DIR"
echo "CPP_DIR:   $CPP_DIR"
echo "BUILD_DIR: $BUILD_DIR"

mkdir -p "$BUILD_DIR"

CXXFLAGS="-std=c++20 -O2 -I$TORCH_DIR/include -I$TORCH_DIR/include/torch/csrc/api/include -I$CPP_DIR/include"
LDFLAGS="-L$TORCH_DIR/lib -ltorch -lc10 -ltorch_cpu -Wl,-rpath,$TORCH_DIR/lib"

# ─── Source files for the real model library ───
MODEL_SRCS=(
    $CPP_DIR/src/model/xorzen_model.cpp
    $CPP_DIR/src/model/hass_block.cpp
    $CPP_DIR/src/model/routing.cpp
    $CPP_DIR/src/model/merger.cpp
    $CPP_DIR/src/model/cot_vector.cpp
    $CPP_DIR/src/model/expert.cpp
    $CPP_DIR/src/model/zmoe.cpp
    $CPP_DIR/src/model/variants.cpp
    $CPP_DIR/src/model/ssm.cpp
    $CPP_DIR/src/model/igris.cpp
    $CPP_DIR/src/model/coherence_field.cpp
    $CPP_DIR/src/core/tensor_utils.cpp
    $CPP_DIR/src/utils/math_utils.cpp
    $CPP_DIR/src/utils/logger.cpp
    $CPP_DIR/src/optimized/simd_ops.cpp
    $CPP_DIR/src/optimized/thread_pool.cpp
)

# ─── Compile all model sources in parallel ───
echo ""
echo "=== Compiling ${#MODEL_SRCS[@]} model sources (parallel) ==="
for src in "${MODEL_SRCS[@]}"; do
    obj="$BUILD_DIR/$(basename ${src%.cpp}).o"
    if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
        echo "  CC $(basename $src)"
        g++ $CXXFLAGS -c "$src" -o "$obj" &
    fi
done
wait
echo "  All model objects compiled."

# ─── Create static library ───
LIB="$BUILD_DIR/libxorzen_model.a"
echo ""
echo "=== Creating static library: $LIB ==="
ar rcs "$LIB" "$BUILD_DIR"/*.o
echo "  $(ls -la $LIB | awk '{print $5}') bytes"

# ─── Build real model test harness ───
echo ""
echo "=== Building real model test harness ==="
g++ $CXXFLAGS -c "$SCRIPT_DIR/real_model_test.cpp" -o "$BUILD_DIR/real_model_test.o"
g++ $BUILD_DIR/real_model_test.o "$LIB" $LDFLAGS -o "$SCRIPT_DIR/real_model_test"
echo "  Built: $SCRIPT_DIR/real_model_test"

# ─── Build pybind11 bindings (if pybind11 available) ───
if python3 -c "import pybind11" 2>/dev/null; then
    echo ""
    echo "=== Building pybind11 bindings ==="
    PYINC="$(python3 -c 'import pybind11; print(pybind11.get_include())')"
    g++ $CXXFLAGS -I"$PYINC" -shared -fPIC \
        "$SCRIPT_DIR/xorzen_bindings.cpp" "$LIB" $LDFLAGS \
        -o "$SCRIPT_DIR/xorzen_cpp$(python3 -c 'import sys; print(".pyd" if sys.platform=="win32" else ".so")')"
    echo "  Built: xorzen_cpp.so"
else
    echo ""
    echo "=== pybind11 not available — skipping bindings ==="
    echo "  Install with: pip install pybind11"
fi

echo ""
echo "=== Build complete ==="

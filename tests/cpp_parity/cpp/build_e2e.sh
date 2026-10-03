#!/usr/bin/env bash
# Build the end-to-end C++ harness.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_DIR="$(python -c 'import torch; print(torch.__file__.rsplit("/",1)[0])')"
cd "$SCRIPT_DIR"
g++ -std=c++20 -O2 \
    -I"$TORCH_DIR/include" \
    -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -L"$TORCH_DIR/lib" \
    end_to_end_harness.cpp \
    -ltorch -lc10 -ltorch_cpu \
    -Wl,-rpath,"$TORCH_DIR/lib" \
    -o end_to_end_harness
echo "Built: $SCRIPT_DIR/end_to_end_harness"

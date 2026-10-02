# XORZEN.CPP Integration & Validation Plan

## Phase 1: Universal Numeric Parity (The Baseline)
- **Goal**: Guarantee the C++ backend is mathematically identical to the Python PyTorch reference.
- **Method**: "Golden Vector" baseline using static binary input/output tensors.
- **Scope**:
    - [ ] `RMSNorm` (Already Passed)
    - [ ] `AdaptiveRouter` (Forward/Backward)
    - [ ] `HASSBlock` (Attention Mechanism)
    - [ ] `GatedMerger`
- **Output**: A standalone C++ test suite `tests/cpp/test_parity_all.cpp` that verifies all components.

## Phase 2: Dual-Path Architecture (`pyxorzen` vs `fastxorzen`)
- **Goal**: Seamlessly swap between Python and C++ implementations in `train_greed.py`.
- **Method**: Create a Python package wrapper that dynamically detects and loads the C++ backend DLL.
- **Structure**:
    - `xorzen.py` (Wrapper)
    - `fastxorzen.pyd` (Compiled C++ extension via pybind11)
- **Validation**: Implement a `patch_model(model)` function that replaces standard `nn.Module` layers with their C++ counterparts.

## Phase 3: Integrated Hardware-Constrained Training
- **Goal**: Train the `Tiny_23K` model entirely using the C++ backend on your i5-8350U.
- **Validation**:
    - [ ] Training parity (loss curve match).
    - [ ] Performance benchmark (tok/s comparison Python vs C++).
    - [ ] Memory footprint check (ensure 6.5x reduction target is met).

## Immediate Next Actions:
1. Write `dump_layer_io.py` (generic script to extract IO tensors for any layer).
2. Update `src/api.cpp` to expose the remaining kernels (`Router`, `HASS`, `Merger`).
3. Execute and verify each parity test.

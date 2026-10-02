# XORZEN.CPP TODO List

## Immediate Tasks
- [ ] Debug and fix `AdaptiveRouter` numeric divergence (Max diff: 0.127).
- [ ] Implement robust `load_weights` for all components (HASS, SSM, Merger).
- [ ] Establish numeric parity test suite for `HASSBlock` and `GatedMerger`.
- [ ] Implement `pybind11` bridge for `fastxorzen` Python module.

## Future Tasks
- [ ] Full `Tiny_23K` training run verification on C++ backend.
- [ ] Integrate SIMD-optimized `Attention` pathway kernels.

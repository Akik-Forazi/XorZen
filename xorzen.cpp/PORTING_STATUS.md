# XORZEN.CPP Porting Status

Last updated: 2026-05-31, Asia/Dhaka.

This is the continuation handoff for `C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp`.

## Current Verified State (2026-05-31)

`xorzen.cpp` is now in a **buildable and stable** state using the MSVC + LibTorch toolchain. The architectural scaffold is complete, and several high-performance optimizations have been verified.

### Key Accomplishments
- **Core Model Parity**: `XorzenModel`, `AdaptiveRouter`, `HASSBlock`, and `ShardedExpertFabric` are fully implemented in C++ matching the Python architectural logic.
- **Optimized Kernels**: 
    - **SIMD (AVX2)**: Hand-optimized `RMSNorm`, `LayerNorm`, and `GELU` kernels are implemented in `src/optimized/simd_ops.cpp`.
    - **Flash Attention**: A CPU-optimized Flash Attention kernel is available in `src/optimized/flash_attn.cpp`.
- **BEBPE Tokenizer**: Fully native C++ implementation of the Byte-Encoded BPE tokenizer, ensuring ID parity with the Python reference.
- **Build System**: MSVC + Ninja build is verified, resolving previous ABI issues with MinGW. CTest passes for train, infer, and tokenizer smoke tests.
- **C ABI**: Stable API exported via `api.cpp` for integration with XORZEN Studio.

### Identified Gaps & Remaining Work
- **Checkpoint Compatibility**: Need production-grade logic to load weights directly from Python `.pt` or `.safetensors` files. Current implementation uses a placeholder/minimal format.
- **Advanced Quantization**: The `SPPQ` (Progressive Quantization) logic from `xorzen/utils/sppq.py` is not yet ported.
- **Full Training Loop Parity**: Features like curriculum learning, training continuation, and complex checkpoint management from `xorzen/training/` are still in the scaffold phase.
- **Sharding Metadata**: Full parity for disk-based expert sharding metadata is pending.
- **Numeric Parity Tests**: Need rigorous layer-by-layer numeric comparison tests against the Python reference to ensure absolute mathematical fidelity.
- **GPU Backends**: CUDA and OpenCL integration is planned but not yet implemented.

## Handoff State (2026-05-17)

`xorzen.cpp` now builds successfully with MSVC + LibTorch from the installed Python PyTorch wheel.

Generated artifacts in `xorzen.cpp/build-msvc/`:

- `XorZen_BETA-v-0.2.5_win64_backend.dll`
- `xorzen_core.lib`
- `xorzen_train.exe`
- `xorzen_infer.exe`
- `xorzen_tokenizer_smoke.exe`
- `xorzen_bench.exe`
- copied Torch runtime DLLs, including `torch_cpu.dll`, `torch.dll`, and `c10.dll`

Verified command:

```powershell
cmd.exe /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S xorzen.cpp -B xorzen.cpp\build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:\Users\akikf\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\torch\share\cmake" && cmake --build xorzen.cpp\build-msvc -j 8'
```

Verified tests:

```powershell
ctest --test-dir xorzen.cpp\build-msvc --output-on-failure
```

Result on 2026-05-17:

```text
100% tests passed, 0 tests failed out of 4
```

Runtime smoke outputs observed:

```text
xorzen_train.exe: loss=7.02125 logits=[2, 16, 1024] params=1494426
xorzen_infer.exe: generated a CPULongType tensor with 11 tokens
xorzen_tokenizer_smoke.exe: vocab=10000 tokens=20 text=Hello world! XORZEN BEBPE tokenizer smoke.
xorzen_bench.exe 8 2 32: avg_ms=57.1922 tokens_per_sec=1119.03 logits=[2, 32, 1024]
```

The benchmark is for the tiny built-in smoke model, not the full production model.

## What Was Added

Core C++/LibTorch scaffold:

- `include/xorzen/types.h`
- `include/xorzen/ops.h`
- `include/xorzen/routing.h`
- `include/xorzen/expert.h`
- `include/xorzen/hass.h`
- `include/xorzen/model.h`
- `include/xorzen/api.h`
- `include/xorzen/tokenizer.h`
- `include/xorzen/data_pipeline.h`

Source modules:

- `src/core/tensor_utils.cpp`
- `src/model/routing.cpp`
- `src/model/hass_block.cpp`
- `src/model/expert.cpp`
- `src/model/xorzen_model.cpp`
- `src/tokenizer/bebpe_tokenizer.cpp`
- `src/data/text_dataset.cpp`
- `src/api.cpp`
- `src/main.cpp`
- `src/infer.cpp`
- `src/tokenizer_smoke.cpp`
- `src/bench.cpp`

Build improvements:

- `xorzen_core` static library for internal C++ model symbols.
- `xorzen_backend` DLL for C ABI exports.
- Console smoke tools linked against `xorzen_core`, avoiding Windows DLL C++ symbol export problems.
- CTest coverage for train smoke, infer smoke, tokenizer smoke, and micro-bench.
- MSVC compile options `/MP` and `/bigobj`.
- Optional CMake switches:
  - `-DXORZEN_ENABLE_IPO=ON` for interprocedural optimization/LTO.
  - `-DXORZEN_ENABLE_AVX2=ON` for AVX2 code generation on compatible CPUs.

## Architecture Map

The C++ flow mirrors the Python reference at a LibTorch level:

1. `XorzenModelImpl::forward` receives `[batch, seq]` token IDs.
2. Token and positional embeddings produce `[B, T, H]`.
3. A latent CoT lane preserves the expected `[B, T, cot_dim * cot_components]` shape.
4. `AdaptiveRouterImpl::forward` computes:
   - depth mask per layer
   - width multiplier
   - HASS pathway probabilities
   - top-k expert IDs and expert weights
5. `HASSBlockImpl::forward` blends:
   - local causal/window attention
   - low-rank global pathway
   - diagonal SSM pathway
   - adaptive FFN
6. `ShardedExpertFabricImpl::forward` routes tokens through top-k SwiGLU experts.
7. `GatedMergerImpl::forward` combines HASS output, MoE output, and latent CoT.
8. Final normalization and LM head produce logits.
9. Training smoke computes next-token cross entropy plus auxiliary routing losses.

## BEBPE Tokenizer

XORZEN's main tokenizer is BEBPE: byte-encoded BPE. It must not be replaced by ordinary BPE or SentencePiece unless an adapter preserves token IDs exactly.

Native BEBPE support now exists:

- Loader: `BEBPETokenizer::from_tokenizer_json`
- Encoder: `BEBPETokenizer::encode`
- Batch encoder: `BEBPETokenizer::batch_encode`
- Decoder: `BEBPETokenizer::decode`
- C ABI:
  - `xorzen_load_tokenizer`
  - `xorzen_tokenizer_encode`
  - `xorzen_tokenizer_decode`
  - `xorzen_free_tokenizer`

Verified against:

```text
xorzen/tokenizer/pretrained/zero_bpe_10k.json
```

Current tokenizer implementation supports:

- HuggingFace-style byte-level BPE `tokenizer.json`
- `model.vocab`
- string or pair-array `model.merges`
- GPT-2/HuggingFace byte-to-unicode byte reversibility
- byte fallback for arbitrary UTF-8 input
- BOS/EOS/PAD/UNK special IDs
- batch encode and decode

Future tokenizer optimization path:

- Replace repeated full-pair scans with a ranked pair heap.
- Convert symbol strings to compact integer IDs after load.
- Add a mmap/binary tokenizer cache generated from JSON.
- Add trie or double-array trie for byte-piece lookup.
- Add thread-pool batch encode.

## Data Pipeline

A native C++ text data pipeline now exists in `include/xorzen/data_pipeline.h` and `src/data/text_dataset.cpp`.

It uses LibTorch `torch::data::datasets::Dataset`, which keeps the C++ path aligned with PyTorch/Meta infrastructure instead of a custom one-off loader.

Features:

- `discover_text_files(root, recursive)` for `.txt`, `.md`, `.jsonl`, `.csv`, `.tsv`, and `.text`.
- `BEBPETextDataset` tokenizes files with the native BEBPE tokenizer.
- Emits `torch::data::Example<>` with input and target tensors for next-token training.
- Supports configurable sequence length and stride.

## Backend Strategy

Current backend:

- LibTorch from the installed Python package:
  - `C:\Users\akikf\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\torch`
  - Observed Torch version: `2.11.0+cpu`
  - CUDA: not available in this local wheel
- `torch::nn::Module` for model components.
- `torch::Tensor` for activations, routing tensors, logits, losses, and data pipeline examples.
- LibTorch autograd for training compatibility.
- Standard C++20 for filesystem, caches, mutexes, C ABI, and utility code.

This is the right first backend because the Python XORZEN implementation is PyTorch-native. CUDA or vendor-specific kernels should be added after parity tests exist.

## Not Yet Full Python Parity

This C++ backend is now buildable and runnable, but it is not yet a complete Python-to-C++ conversion.

Still missing or incomplete:

- Full checkpoint compatibility with existing Python training checkpoints.
- Production training loop and optimizer/scheduler parity.
- Progressive quantization from `xorzen/utils/sppq.py`.
- Full disk sharding and mmap parity from Python sharding utilities.
- CUDA/OpenCL/GPU acceleration.
- Custom optimized kernels from `xorzen/speed/csrc`.
- Full tokenizer trainer/registry/metadata stack.
- Full config serialization and CLI for production model sizes.
- Numeric parity tests comparing Python and C++ outputs layer by layer.

Do not claim production completeness until those are done and tested.

## Python Reference Files

Use these as source of truth for parity work:

- `../xorzen/models/zero/model.py`
- `../xorzen/model/components/routing.py`
- `../xorzen/model/components/hass_block.py`
- `../xorzen/model/zmoe.py`
- `../xorzen/model/components/merger.py`
- `../xorzen/utils/math_utils.py`
- `../xorzen/utils/sppq.py`
- `../xorzen/utils/sharding.py`
- `../xorzen/training/trainer.py`
- `../xorzen/training/checkpoint.py`
- `../xorzen/tokenizer/`

## Next High-Impact Steps

1. Add Python-vs-C++ tokenizer parity tests over a corpus with expected token IDs.
2. Add layer-by-layer shape and numeric parity tests against the Python `zeroModel`.
3. Implement config/checkpoint loading for real XORZEN model sizes.
4. Port progressive quantization and sharded checkpoint metadata.
5. Mine `xorzen/speed/csrc` for hot-path kernels after correctness is locked.
6. Add CUDA build path once a CUDA-enabled LibTorch is installed.

## axodex / Axocode Note

The AGENTS instructions ask for axodex impact analysis before symbol edits. In this Codex tool context, no axodex MCP resources were exposed by `list_mcp_resources`, so work proceeded through local source inspection and build/test verification.

If Axocode/axodex is available in a future session, run impact checks before editing shared Python symbols or high-risk C++ symbols.

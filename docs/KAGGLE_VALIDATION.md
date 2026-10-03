# Kaggle Validation Procedure

**Status:** PREPARED (not yet verified on Kaggle)

## Environment

| Item | Value |
|------|-------|
| Platform | Kaggle Notebook |
| Accelerator | GPU T4 x1 (or T4 x2) |
| VRAM | 14.56 GB per T4 |
| RAM | ~30 GB |
| Disk | ~20 GB /kaggle/working |
| PyTorch | 2.10.0+cu128 or 2.11.0+cu128 |
| CUDA | 12.8 |
| Python | 3.12 or 3.13 |

## Dependencies

The notebook installs via:
```bash
pip install --no-deps git+https://github.com/Akik-Forazi/XorZen.git
pip install tokenizers>=0.21 sentencepiece einops pydantic>=2.0 psutil scipy datasets matplotlib numpy cryptography
```

**CRITICAL:** `--no-deps` on the xorzen install prevents pip from replacing
Kaggle's CUDA-enabled torch with a CPU-only build.

## Notebook

URL: https://github.com/Akik-Forazi/XorZen/raw/main/notebooks/XORZEN_Kaggle_T4.ipynb

## Configuration

| Setting | Value |
|---------|-------|
| Model | zero_50M (50M params, ~1.5M active) |
| Tokenizer | xorzen_python_10k (10K vocab) |
| Batch size | 16 |
| Gradient accumulation | 2 |
| Sequence length | 1024 |
| Mixed precision | bf16 |
| Multi-GPU | False |
| Gradient checkpointing | False |
| xorzen.compile | True (applies JIT SSM patch only) |
| Dataset | TinyStories (500K samples capped) |
| Max steps | 5000 |

## Validation Steps

### Cells 1-5: Config + Hardware Check
- Verify `torch.cuda.is_available() = True`
- Verify 1-2 T4 GPUs visible
- Verify ~30 GB RAM

### Cell 6: Install XORZEN
- Verify `torch.cuda.is_available() = True` AFTER install
- If False: the `--no-deps` fix was not applied

### Cell 7: Verify XORZEN
- Verify xorzen version 1.0.1
- Verify 3 tokenizers available (xorzen_agi_tokenizer_65k, zero_bpe_10k, xorzen_python_10k)
- Verify zero_50M config: vocab=10000, hidden=256, layers=10, experts=43, top_k=2

### Cells 9-15: Dataset + Tokenizer
- TinyStories downloads (~20s)
- Tokenization with xorzen_python_10k (500K stories → ~115M tokens)
- Cache saved to /kaggle/working/XORZEN/zero_50M/checkpoints/tokenizer_cache/

### Cell 17: Model Init
- Verify "Set gradient_checkpointing=False"
- Verify "[xorzen.compile] Applying optimization patches..."
- Verify "[xorzen.jit] Patched ssm_scan.sequential_scan → jit_sequential_scan"
- Verify forward pass test: loss ~9.x, forward time ~2-4s
- Verify backward pass test: loss ~9.x, backward time ~3-6s

### Cell 19: Pre-flight
- 8 sanity steps, loss should start ~7.5 and show slight decrease

### Cell 21: Benchmark
- Record avg step time (target: <5s/step)
- Record throughput (target: >2000 tok/s)
- Record peak VRAM (target: <10 GB)

### Cell 23: Training
- Loss should decrease from ~7.5 toward ~5.0
- Steps/s should be >0.2
- VRAM should stay <12 GB
- Training should complete 5000 steps within 9 hours

## Failure Diagnostics

### "module 'torch' has no attribute 'cuda'"
- Cause: pip replaced CUDA torch with CPU-only
- Fix: Ensure `--no-deps` in cell 6 install command

### "CUDA out of memory"
- Cause: Batch too large or gradient checkpointing off with large batch
- Fix: Set FORCE_GRADIENT_CHECKPOINTING=True or reduce BATCH_SIZE to 8

### "torch._dynamo hit config.recompile_limit"
- Cause: torch.compile was used (should be disabled)
- Fix: Ensure USE_TORCH_COMPILE calls xorzen_compile (which does JIT, not torch.compile)

### "TypeError: jit_sequential_scan() got an unexpected keyword argument 'init_state'"
- Cause: Old version of jit_kernels.py without init_state support
- Fix: Pull latest from GitHub

### "RuntimeError: a Tensor with 2 elements cannot be converted to Scalar"
- Cause: ModelOutputDataParallel not used / DataParallel gathering scalar losses
- Fix: Ensure USE_MULTI_GPU=False (single GPU mode)

## Expected Output Shapes

| Tensor | Shape |
|--------|-------|
| input_ids | [16, 1024] |
| labels | [16, 1024] |
| hidden_states | [16, 1024, 256] |
| logits | [16, 1024, 10000] |
| loss | scalar |

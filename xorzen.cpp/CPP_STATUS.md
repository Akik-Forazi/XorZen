# XORZEN.CPP Current Status

Last verified: 2026-07-05

## Verified State
- `RMSNorm`: Verified (PASS) via static golden vectors.
- `AdaptiveRouter`: In-Progress (Divergent: Max diff ~0.127).
- Tokenizer: Verified.
- Build system: Fixed `torch.h` linking issue.

## Outstanding Issues
- **AdaptiveRouter numeric divergence**: Investigation ongoing, likely due to activation/layer ordering differences.
- **HASSBlock/ICoT Parity**: Not yet started.

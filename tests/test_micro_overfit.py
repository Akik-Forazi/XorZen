"""
Micro-Overfit Regression Test for XORZEN v1.0.1.
Verifies that zero_tiny_23k can genuinely learn and overfit 4 synthetic sequences,
driving cross-entropy loss down from ~5.6 to < 0.5 within 100 optimizer steps.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from xorzen.models.zero.variants import zero_tiny_23k


def test_micro_overfit():
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Instantiate model with full MoE registration (test_mode=False)
    model = zero_tiny_23k(test_mode=False).to(device)
    model.train()

    vocab_size = model.config.vocab_size
    seq_len = 16
    batch_size = 4

    # 4 synthetic sequences
    input_ids = torch.randint(0, min(100, vocab_size), (batch_size, seq_len), device=device)
    # Target is next-token shift
    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

    initial_loss = None
    final_loss = None

    print(f"Starting micro-overfit test: {batch_size} samples, seq_len={seq_len}, vocab={vocab_size}")

    for step in range(1, 101):
        optimizer.zero_grad()
        out = model(input_ids)
        logits = out.logits  # [B, T, V]

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == 1 or step == 100:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")

        final_loss = loss.item()

    print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert final_loss < 0.5, f"Micro-overfit test failed! Final loss {final_loss:.4f} >= 0.5"
    print(">>> MICRO-OVERFIT TEST PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    test_micro_overfit()

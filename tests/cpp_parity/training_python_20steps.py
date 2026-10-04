"""Run 20 steps of deterministic Python training, recording per-step metrics."""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from pathlib import Path

torch.manual_seed(42)
from xorzen.models.zero import zero_tiny_23k
m = zero_tiny_23k(test_mode=False)
m.eval()  # we'll call m.train() below

cfg = m.config
B, T = 2, 16
input_ids = torch.randint(0, cfg.vocab_size, (B, T), generator=torch.Generator().manual_seed(42))
labels = torch.randint(0, cfg.vocab_size, (B, T), generator=torch.Generator().manual_seed(43))

opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)

results = []
m.train()
for step in range(20):
    opt.zero_grad()
    out = m(input_ids=input_ids, labels=labels, return_dict=True)
    loss = out.loss
    loss.backward()
    # Grad norm
    grad_norm = 0.0
    for p in m.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    # Param update magnitude (before vs after)
    params_before = [p.detach().clone() for p in m.parameters()]
    opt.step()
    param_delta = 0.0
    for i, p in enumerate(m.parameters()):
        param_delta += (p.data - params_before[i]).abs().sum().item()
    param_delta /= len(params_before)
    # NaN check
    has_nan = torch.isnan(loss).item() or any(torch.isnan(p).any().item() for p in m.parameters())
    results.append({
        "step": step,
        "loss": loss.item(),
        "lm_loss": out.lm_loss.item(),
        "routing_loss": out.routing_loss.item(),
        "load_balance_loss": out.load_balance_loss.item(),
        "total_loss": out.loss.item(),
        "grad_norm": grad_norm,
        "param_delta_mean": param_delta,
        "has_nan": has_nan,
    })
    print(f"  step {step:3d}: loss={loss.item():.6f}  lm={out.lm_loss.item():.6f}  "
          f"grad_norm={grad_norm:.4f}  delta={param_delta:.6e}  nan={has_nan}")

out = Path("tests/cpp_parity/reports/training_python_20steps.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {out}")
print(f"First loss: {results[0]['loss']:.6f}")
print(f"Last loss:  {results[-1]['loss']:.6f}")
print(f"Loss delta: {results[-1]['loss'] - results[0]['loss']:.6f}")

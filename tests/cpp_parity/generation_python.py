"""Run Python generation on identical prompts, save outputs for C++ comparison."""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from pathlib import Path

torch.manual_seed(42)
from xorzen.models.zero import zero_tiny_23k
m = zero_tiny_23k(test_mode=False)
m.eval()
cfg = m.config

# Test prompts — same as C++ generation_test
prompts = [
    torch.tensor([[6, 3, 4, 6]], dtype=torch.long),  # Test 1-6 prompt
    torch.tensor([[1, 2, 3]], dtype=torch.long),     # Shorter prompt
    torch.tensor([[0, 5, 7, 2, 1]], dtype=torch.long),  # 5-token prompt
]

results = {}

# Greedy
with torch.no_grad():
    out = m.generate(prompts[0], max_new_tokens=5, do_sample=False, num_beams=1)
    results["greedy"] = out.tolist()
    print(f"Python greedy: {out.tolist()}")

# Temperature (low temp = near-greedy)
with torch.no_grad():
    out = m.generate(prompts[0], max_new_tokens=5, do_sample=False, num_beams=1, temperature=0.5)
    results["temperature_05"] = out.tolist()
    print(f"Python temp=0.5: {out.tolist()}")

# Top-k (greedy with top-k filter)
with torch.no_grad():
    out = m.generate(prompts[0], max_new_tokens=5, do_sample=False, num_beams=1, top_k=3)
    results["top_k_3"] = out.tolist()
    print(f"Python top-k=3: {out.tolist()}")

# Top-p (greedy with top-p filter)
with torch.no_grad():
    out = m.generate(prompts[0], max_new_tokens=5, do_sample=False, num_beams=1, top_p=0.9)
    results["top_p_09"] = out.tolist()
    print(f"Python top-p=0.9: {out.tolist()}")

# Repetition penalty
with torch.no_grad():
    out = m.generate(prompts[0], max_new_tokens=5, do_sample=False, num_beams=1, repetition_penalty=1.5)
    results["rep_penalty_15"] = out.tolist()
    print(f"Python rep_penalty=1.5: {out.tolist()}")

# EOS
with torch.no_grad():
    out = m.generate(prompts[0], max_new_tokens=10, do_sample=False, num_beams=1, eos_token_id=0)
    results["eos_0"] = out.tolist()
    print(f"Python eos=0: {out.tolist()}")

# Beam search
with torch.no_grad():
    out = m.generate(prompts[0], max_new_tokens=5, do_sample=False, num_beams=3)
    results["beam_3"] = out.tolist()
    print(f"Python beam=3: {out.tolist()}")

# Save
out_file = Path("tests/cpp_parity/reports/generation_python.json")
out_file.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {out_file}")

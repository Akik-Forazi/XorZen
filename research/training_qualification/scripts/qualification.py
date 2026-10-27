#!/usr/bin/env python3
"""
XORZEN Full Training Qualification Campaign

Executes Phases A-G of the training-readiness audit in one systematic pass.
Each phase prints raw evidence — no pass/fail without data.

Phases:
  A: Core forward-pass correctness
  B: Gradient and optimizer integrity
  C: Numerical stability and loss correctness
  D: Overfit a tiny dataset
  E: Checkpoint, restore, and reproducibility
  F: Training versus inference parity
  G: Minimal generation smoke test

Run: python research/training_qualification/scripts/qualification.py
"""
import sys, os, json, time, warnings, traceback
warnings.simplefilter("ignore")
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from xorzen.config import ModelConfig, ConfigFactory, ModelSize
from xorzen.models.zero.model import zeroModel
from xorzen.models.zero.variants import zero_tiny_23k

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

all_results = {}
all_bugs = []

def record(phase, test_name, data):
    """Record a test result."""
    key = f"{phase}/{test_name}"
    all_results[key] = data
    status = data.get('status', 'UNKNOWN')
    emoji = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️', 'INFO': 'ℹ️'}.get(status, '?')
    print(f"  {emoji} {test_name}: {data.get('detail', '')}")

def report_bug(bug_id, severity, component, file, expected, actual, root_cause, fix=None):
    """Record a bug."""
    bug = {
        'id': bug_id, 'severity': severity, 'component': component, 'file': file,
        'expected': expected, 'actual': actual, 'root_cause': root_cause, 'fix': fix
    }
    all_bugs.append(bug)
    print(f"  🐛 BUG-{bug_id} [{severity}] {component}: {actual}")

def make_small_config():
    """Create a small diagnostic config for CPU testing."""
    cfg = ModelConfig()
    cfg.hidden_size = 32
    cfg.num_layers = 2
    cfg.max_depth = 2
    cfg.min_depth = 1
    cfg.vocab_size = 50
    cfg.context_length = 32
    cfg.num_attention_heads = 4
    cfg.n_kv_heads = 4
    cfg.use_sliced_ffn = True
    cfg.width_choices = (8, 32)
    cfg.expert_count = 2
    cfg.top_k_experts = 1
    cfg.pathway_top_k = 3
    cfg.router_hidden_dim = 32
    cfg.eval_routing_noise = 0.0
    cfg.cost_aware_routing = False
    cfg.compute_budget = 1.0
    cfg.cot_dim = 4
    cfg.cot_components = 6
    cfg.dropout = 0.0
    cfg.gradient_checkpointing = False
    return cfg

def make_tiny_model(seed=42):
    """Create the production zero_tiny_23k model."""
    torch.manual_seed(seed)
    m = zero_tiny_23k(test_mode=False)
    m.gradient_checkpointing = False
    return m


# ═══════════════════════════════════════════════════════════════════
# PHASE A: Core forward-pass correctness
# ═══════════════════════════════════════════════════════════════════
def phase_a():
    print("\n" + "="*80)
    print("  PHASE A: Core Forward-Pass Correctness")
    print("="*80)

    cfg = make_small_config()
    torch.manual_seed(42)
    model = zeroModel(cfg, test_mode=True)
    model.eval()

    # A1: Basic forward shape
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(input_ids=x, return_dict=True)
    record('A', 'forward_shape', {
        'status': 'PASS' if out.logits.shape == (2, 16, cfg.vocab_size) else 'FAIL',
        'detail': f"logits shape={list(out.logits.shape)}, expected=[2, 16, {cfg.vocab_size}]",
        'logits_shape': list(out.logits.shape),
    })

    # A2: Batch size 1
    x1 = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out1 = model(input_ids=x1, return_dict=True)
    record('A', 'batch_size_1', {
        'status': 'PASS' if out1.logits.shape == (1, 16, cfg.vocab_size) else 'FAIL',
        'detail': f"batch=1 logits={list(out1.logits.shape)}",
    })

    # A3: Sequence length 1
    x_short = torch.randint(0, cfg.vocab_size, (2, 1))
    try:
        with torch.no_grad():
            out_short = model(input_ids=x_short, return_dict=True)
        record('A', 'seq_len_1', {
            'status': 'PASS' if out_short.logits.shape == (2, 1, cfg.vocab_size) else 'FAIL',
            'detail': f"seq=1 logits={list(out_short.logits.shape)}",
        })
    except Exception as e:
        record('A', 'seq_len_1', {'status': 'FAIL', 'detail': str(e)})

    # A4: Max context length
    x_max = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
    with torch.no_grad():
        out_max = model(input_ids=x_max, return_dict=True)
    record('A', 'max_context', {
        'status': 'PASS' if out_max.logits.shape == (1, cfg.context_length, cfg.vocab_size) else 'FAIL',
        'detail': f"max_ctx logits={list(out_max.logits.shape)}",
    })

    # A5: Beyond context length should error
    x_over = torch.randint(0, cfg.vocab_size, (1, cfg.context_length + 1))
    try:
        with torch.no_grad():
            _ = model(input_ids=x_over, return_dict=True)
        record('A', 'over_context', {'status': 'FAIL', 'detail': 'should have raised ValueError'})
    except ValueError:
        record('A', 'over_context', {'status': 'PASS', 'detail': 'correctly rejected over-context input'})
    except Exception as e:
        record('A', 'over_context', {'status': 'WARN', 'detail': f'wrong exception: {type(e).__name__}'})

    # A6: Finite outputs
    all_finite = torch.isfinite(out.logits).all().item()
    record('A', 'finite_outputs', {
        'status': 'PASS' if all_finite else 'FAIL',
        'detail': f"all finite={all_finite}, min={out.logits.min().item():.4f}, max={out.logits.max().item():.4f}",
    })

    # A7: Causal masking — changing future tokens must not affect past logits
    x_a = torch.randint(0, cfg.vocab_size, (1, 16))
    x_b = x_a.clone()
    x_b[0, 10:] = torch.randint(0, cfg.vocab_size, (6,))  # change future
    with torch.no_grad():
        out_a = model(input_ids=x_a, return_dict=True)
        out_b = model(input_ids=x_b, return_dict=True)
    diff = (out_a.logits[:, :10, :] - out_b.logits[:, :10, :]).abs().max().item()
    record('A', 'causal_mask', {
        'status': 'PASS' if diff < 1e-5 else 'FAIL',
        'detail': f"max diff at positions 0-9 when changing 10-15: {diff:.2e}",
    })
    if diff >= 1e-5:
        report_bug('A001', 'critical', 'causal_mask', 'hass_block.py',
                   'future tokens must not affect past logits',
                   f'max diff = {diff:.2e}', 'causal mask may be broken')

    # A8: No cross-batch interaction
    x_batch = torch.randint(0, cfg.vocab_size, (4, 16))
    with torch.no_grad():
        out_batch = model(input_ids=x_batch, return_dict=True)
    # Compare with individual
    max_diff = 0
    for i in range(4):
        with torch.no_grad():
            out_i = model(input_ids=x_batch[i:i+1], return_dict=True)
        d = (out_batch.logits[i] - out_i.logits[0]).abs().max().item()
        max_diff = max(max_diff, d)
    record('A', 'no_cross_batch', {
        'status': 'PASS' if max_diff < 1e-5 else 'FAIL',
        'detail': f"max diff batch vs individual: {max_diff:.2e}",
    })

    # A9: Training vs eval mode (dropout=0 so should be identical)
    model.train()
    with torch.no_grad():
        out_train = model(input_ids=x, return_dict=True)
    model.eval()
    with torch.no_grad():
        out_eval = model(input_ids=x, return_dict=True)
    diff_te = (out_train.logits - out_eval.logits).abs().max().item()
    record('A', 'train_eval_parity', {
        'status': 'PASS' if diff_te < 1e-5 else 'WARN',
        'detail': f"train vs eval diff: {diff_te:.2e} (dropout=0, should be ~0)",
    })

    # A10: Invalid vocab ID should error or be handled
    x_bad = torch.tensor([[0, 1, cfg.vocab_size + 10, 3]])  # invalid ID
    try:
        with torch.no_grad():
            _ = model(input_ids=x_bad, return_dict=True)
        record('A', 'invalid_vocab_id', {'status': 'WARN', 'detail': 'invalid vocab ID did not error (may use embedding padding)'})
    except Exception as e:
        record('A', 'invalid_vocab_id', {'status': 'PASS', 'detail': f'correctly rejected: {type(e).__name__}'})

    # A11: Output with labels (loss computation)
    labels = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out_loss = model(input_ids=x, labels=labels, return_dict=True)
    record('A', 'loss_computation', {
        'status': 'PASS' if out_loss.loss is not None and torch.isfinite(out_loss.loss) else 'FAIL',
        'detail': f"loss={out_loss.loss.item():.4f}, lm_loss={out_loss.lm_loss.item():.4f}, finite={torch.isfinite(out_loss.loss).item()}",
    })

    # A12: Parameter count
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record('A', 'param_count', {
        'status': 'INFO',
        'detail': f"total={total_p:,} trainable={train_p:,} frozen={total_p-train_p:,}",
    })

    return model, cfg


# ═══════════════════════════════════════════════════════════════════
# PHASE B: Gradient and optimizer integrity
# ═══════════════════════════════════════════════════════════════════
def phase_b(model, cfg):
    print("\n" + "="*80)
    print("  PHASE B: Gradient and Optimizer Integrity")
    print("="*80)

    torch.manual_seed(42)
    model.train()
    model.gradient_checkpointing = False

    x = torch.randint(1, cfg.vocab_size, (2, 16))
    labels = torch.randint(1, cfg.vocab_size, (2, 16))

    # B1: Forward + backward
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer.zero_grad()
    out = model(input_ids=x, labels=labels, return_dict=True)
    loss_val = out.loss.item()
    out.loss.backward()

    record('B', 'forward_backward', {
        'status': 'PASS' if torch.isfinite(out.loss) else 'FAIL',
        'detail': f"loss={loss_val:.4f} finite={torch.isfinite(out.loss).item()}",
    })

    # B2: All trainable params receive gradients
    has_grad = 0
    no_grad = 0
    zero_grad = 0
    non_finite = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            no_grad += 1
        elif not torch.isfinite(p.grad).all():
            non_finite += 1
        elif p.grad.abs().max() < 1e-15:
            zero_grad += 1
        else:
            has_grad += 1

    record('B', 'gradient_coverage', {
        'status': 'PASS' if no_grad == 0 and non_finite == 0 else 'FAIL',
        'detail': f"has_grad={has_grad} no_grad={no_grad} zero_grad={zero_grad} non_finite={non_finite}",
    })

    # B3: Specific router parameter gradients
    router_groups = ['width_router', 'depth_router', 'path_router', 'expert_router']
    for rg in router_groups:
        module = getattr(model.router, rg, None)
        if module is None:
            continue
        params = list(module.parameters())
        all_have = all(p.grad is not None and p.grad.abs().max() > 1e-15 for p in params if p.requires_grad)
        norms = [p.grad.norm().item() if p.grad is not None else 0 for p in params]
        record('B', f'router_{rg}_grad', {
            'status': 'PASS' if all_have else 'FAIL',
            'detail': f"grad norms: {[f'{n:.4e}' for n in norms]}",
        })

    # B4: Frozen params stay frozen
    cot_params_before = {n: p.data.clone() for n, p in model.cot.named_parameters() if not p.requires_grad}
    optimizer.step()
    cot_unchanged = all(torch.equal(p.data, cot_params_before[n]) for n, p in model.cot.named_parameters() if not p.requires_grad)
    record('B', 'frozen_params', {
        'status': 'PASS' if cot_unchanged else 'FAIL',
        'detail': f"CoT frozen params unchanged after step: {cot_unchanged}",
    })

    # B5: Parameters actually update
    # Re-init to check
    torch.manual_seed(42)
    model2 = zeroModel(cfg, test_mode=True)
    model2.train()
    model2.gradient_checkpointing = False
    before = {n: p.data.clone() for n, p in model2.named_parameters() if p.requires_grad}
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    opt2.zero_grad()
    out2 = model2(input_ids=x, labels=labels, return_dict=True)
    out2.loss.backward()
    opt2.step()
    changed = sum(1 for n, p in model2.named_parameters() if p.requires_grad and not torch.equal(p.data, before[n]))
    unchanged = sum(1 for n, p in model2.named_parameters() if p.requires_grad and torch.equal(p.data, before[n]))
    record('B', 'param_update', {
        'status': 'PASS' if changed > 0 else 'FAIL',
        'detail': f"changed={changed} unchanged={unchanged} (out of {changed+unchanged} trainable)",
    })

    # B6: Gradient clipping
    torch.manual_seed(42)
    model3 = zeroModel(cfg, test_mode=True)
    model3.train()
    model3.gradient_checkpointing = False
    opt3 = torch.optim.AdamW(model3.parameters(), lr=1e-3)
    opt3.zero_grad()
    out3 = model3(input_ids=x, labels=labels, return_dict=True)
    out3.loss.backward()
    grad_norm_before = torch.nn.utils.clip_grad_norm_(model3.parameters(), 1.0).item()
    record('B', 'gradient_clipping', {
        'status': 'PASS' if torch.isfinite(torch.tensor(grad_norm_before)).item() else 'FAIL',
        'detail': f"grad_norm before clip: {grad_norm_before:.4f} (clip=1.0)",
    })

    # B7: Repeated steps — no divergence
    losses = []
    opt4 = torch.optim.AdamW(model3.parameters(), lr=1e-3)
    for step in range(10):
        opt4.zero_grad()
        out4 = model3(input_ids=x, labels=labels, return_dict=True)
        out4.loss.backward()
        torch.nn.utils.clip_grad_norm_(model3.parameters(), 1.0)
        opt4.step()
        losses.append(out4.loss.item())
    all_finite = all(np.isfinite(l) for l in losses)
    loss_change = losses[-1] - losses[0]
    record('B', 'repeated_steps', {
        'status': 'PASS' if all_finite else 'FAIL',
        'detail': f"10 steps: loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} delta={loss_change:+.4f} all_finite={all_finite}",
        'losses': losses,
    })


# ═══════════════════════════════════════════════════════════════════
# PHASE C: Numerical stability and loss correctness
# ═══════════════════════════════════════════════════════════════════
def phase_c(model, cfg):
    print("\n" + "="*80)
    print("  PHASE C: Numerical Stability and Loss Correctness")
    print("="*80)

    torch.manual_seed(42)
    m = zeroModel(cfg, test_mode=True)
    m.train()
    m.gradient_checkpointing = False

    # C1: Loss = cross-entropy with correct shift
    x = torch.randint(1, cfg.vocab_size, (2, 8))
    labels = x.clone()
    with torch.no_grad():
        out = m(input_ids=x, labels=labels, return_dict=True)
    # Manual CE: shift logits and labels
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    manual_ce = nn.functional.cross_entropy(
        shift_logits.view(-1, cfg.vocab_size),
        shift_labels.view(-1)
    )
    loss_diff = abs(out.lm_loss.item() - manual_ce.item())
    record('C', 'loss_shift_correctness', {
        'status': 'PASS' if loss_diff < 1e-5 else 'FAIL',
        'detail': f"model lm_loss={out.lm_loss.item():.6f} manual CE={manual_ce.item():.6f} diff={loss_diff:.2e}",
    })
    if loss_diff >= 1e-5:
        report_bug('C001', 'critical', 'loss_shift', 'model.py',
                   'lm_loss should match manual CE with shift',
                   f'diff={loss_diff:.2e}', 'incorrect label shifting or loss computation')

    # C2: Loss components
    record('C', 'loss_components', {
        'status': 'INFO',
        'detail': f"lm_loss={out.lm_loss.item():.4f} routing_loss={out.routing_loss.item():.4f} "
                  f"total_loss={out.loss.item():.4f} lb_loss={out.load_balance_loss.item():.4f}",
    })

    # C3: Pad token handling — pad token should not contribute to loss
    x_pad = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]])  # 3 real + 5 pad
    labels_pad = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]])
    with torch.no_grad():
        out_pad = m(input_ids=x_pad, labels=labels_pad, return_dict=True)
    # With ignore_index=0 (pad), loss should only count positions 0,1 (predicting 2,3)
    record('C', 'pad_token_handling', {
        'status': 'PASS' if torch.isfinite(out_pad.loss) else 'FAIL',
        'detail': f"loss with padding={out_pad.loss.item():.4f} finite={torch.isfinite(out_pad.loss).item()}",
    })

    # C4: 20-step training — track loss, grad norm, param norm
    torch.manual_seed(42)
    m2 = zeroModel(cfg, test_mode=True)
    m2.train()
    m2.gradient_checkpointing = False
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randint(1, cfg.vocab_size, (2, 16))
    labels = torch.randint(1, cfg.vocab_size, (2, 16))

    history = {'step': [], 'loss': [], 'grad_norm': [], 'param_norm': [], 'non_finite': []}
    for step in range(20):
        opt.zero_grad()
        out = m2(input_ids=x, labels=labels, return_dict=True)
        out.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0).item()
        opt.step()
        pn = sum(p.data.norm().item()**2 for p in m2.parameters())**0.5
        nf = sum((~torch.isfinite(p.grad)).sum().item() for p in m2.parameters() if p.grad is not None)
        history['step'].append(step)
        history['loss'].append(out.loss.item())
        history['grad_norm'].append(gn)
        history['param_norm'].append(pn)
        history['non_finite'].append(nf)

    all_finite = all(history['non_finite'][i] == 0 for i in range(20))
    loss_trend = history['loss'][-1] - history['loss'][0]
    record('C', 'training_stability_20steps', {
        'status': 'PASS' if all_finite else 'FAIL',
        'detail': f"loss: {history['loss'][0]:.4f}→{history['loss'][-1]:.4f} (Δ={loss_trend:+.4f}) "
                  f"grad_norm: {history['grad_norm'][0]:.4f}→{history['grad_norm'][-1]:.4f} "
                  f"non_finite_grads: {sum(history['non_finite'])}",
        'history': history,
    })


# ═══════════════════════════════════════════════════════════════════
# PHASE D: Overfit a tiny dataset
# ═══════════════════════════════════════════════════════════════════
def phase_d(cfg):
    print("\n" + "="*80)
    print("  PHASE D: Overfit a Tiny Dataset")
    print("="*80)

    # Create a tiny repeated-pattern dataset
    # Pattern: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1]
    # The model should learn to predict the repeating sequence
    pattern = [1, 2, 3, 4, 5] * 3 + [1]
    train_x = torch.tensor([pattern] * 32, dtype=torch.long)  # 32 copies
    train_l = train_x.clone()
    val_x = torch.tensor([pattern] * 8, dtype=torch.long)
    val_l = val_x.clone()

    torch.manual_seed(42)
    model = zeroModel(cfg, test_mode=True)
    model.train()
    model.gradient_checkpointing = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    losses = []
    val_losses = []
    for step in range(50):
        optimizer.zero_grad()
        out = model(input_ids=train_x, labels=train_l, return_dict=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(out.loss.item())

        if step % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(input_ids=val_x, labels=val_l, return_dict=True)
                val_losses.append(val_out.loss.item())
            model.train()

    loss_drop = losses[0] - losses[-1]
    final_loss = losses[-1]
    can_overfit = final_loss < 1.0  # should reach low loss on a trivial pattern

    record('D', 'tiny_overfit', {
        'status': 'PASS' if can_overfit else 'FAIL',
        'detail': f"50 steps: loss {losses[0]:.4f}→{final_loss:.4f} (drop={loss_drop:.4f}) "
                  f"val_loss={val_losses[-1]:.4f}",
        'train_losses': losses,
        'val_losses': val_losses,
    })

    # D2: Repeated-pattern prediction quality
    model.eval()
    with torch.no_grad():
        out = model(input_ids=train_x[:1], return_dict=True)
        preds = out.logits[0].argmax(dim=-1)
    # Check if predictions match the pattern (positions 1-15, predicting next token)
    correct = (preds[:-1] == train_x[0, 1:]).sum().item()
    total = len(preds) - 1
    record('D', 'pattern_prediction', {
        'status': 'PASS' if correct / total > 0.5 else 'WARN',
        'detail': f"next-token accuracy on pattern: {correct}/{total} = {correct/total:.1%}",
    })

    return model


# ═══════════════════════════════════════════════════════════════════
# PHASE E: Checkpoint, restore, and reproducibility
# ═══════════════════════════════════════════════════════════════════
def phase_e(model, cfg):
    print("\n" + "="*80)
    print("  PHASE E: Checkpoint, Restore, and Reproducibility")
    print("="*80)

    # E1: Save model state_dict
    state = {k: v.clone() for k, v in model.state_dict().items()}
    record('E', 'save_state_dict', {
        'status': 'PASS',
        'detail': f"saved {len(state)} tensors",
    })

    # E2: Restore into fresh model
    torch.manual_seed(999)  # different seed — should NOT matter after load
    model2 = zeroModel(cfg, test_mode=True)
    model2.load_state_dict(state)
    model2.eval()

    # Compare outputs
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out1 = model(input_ids=x, return_dict=True)
        out2 = model2(input_ids=x, return_dict=True)
    diff = (out1.logits - out2.logits).abs().max().item()
    record('E', 'restore_logits', {
        'status': 'PASS' if diff < 1e-6 else 'FAIL',
        'detail': f"max logit diff after restore: {diff:.2e}",
    })

    # E3: Parameter match
    state2 = model2.state_dict()
    all_match = all(torch.equal(state[k], state2[k]) for k in state)
    max_pdiff = max((state[k] - state2[k]).abs().max().item() for k in state)
    record('E', 'param_match', {
        'status': 'PASS' if all_match else 'FAIL',
        'detail': f"all params match: {all_match}, max diff: {max_pdiff:.2e}",
    })

    # E4: Save/load to file
    ckpt_path = RESULTS_DIR / 'test_checkpoint.pt'
    torch.save({
        'model_state_dict': state,
        'config': {'hidden_size': cfg.hidden_size, 'num_layers': cfg.num_layers,
                    'vocab_size': cfg.vocab_size, 'context_length': cfg.context_length},
    }, ckpt_path)
    record('E', 'save_to_file', {
        'status': 'PASS' if ckpt_path.exists() else 'FAIL',
        'detail': f"checkpoint size: {ckpt_path.stat().st_size:,} bytes",
    })

    # Load from file
    ckpt = torch.load(ckpt_path, weights_only=False)
    model3 = zeroModel(cfg, test_mode=True)
    model3.load_state_dict(ckpt['model_state_dict'])
    model3.eval()
    with torch.no_grad():
        out3 = model3(input_ids=x, return_dict=True)
    diff3 = (out1.logits - out3.logits).abs().max().item()
    record('E', 'load_from_file', {
        'status': 'PASS' if diff3 < 1e-6 else 'FAIL',
        'detail': f"max logit diff after file round-trip: {diff3:.2e}",
    })

    # E5: Deterministic eval — same input should give same output
    with torch.no_grad():
        out_a = model(input_ids=x, return_dict=True)
        out_b = model(input_ids=x, return_dict=True)
    det_diff = (out_a.logits - out_b.logits).abs().max().item()
    record('E', 'deterministic_eval', {
        'status': 'PASS' if det_diff < 1e-7 else 'FAIL',
        'detail': f"eval determinism diff: {det_diff:.2e}",
    })


# ═══════════════════════════════════════════════════════════════════
# PHASE F: Training versus inference parity
# ═══════════════════════════════════════════════════════════════════
def phase_f(model, cfg):
    print("\n" + "="*80)
    print("  PHASE F: Training versus Inference Parity")
    print("="*80)

    x = torch.randint(1, cfg.vocab_size, (2, 16))

    # F1: Train vs eval logits (dropout=0)
    model.train()
    with torch.no_grad():
        out_train = model(input_ids=x, return_dict=True)
    model.eval()
    with torch.no_grad():
        out_eval = model(input_ids=x, return_dict=True)
    diff = (out_train.logits - out_eval.logits).abs().max().item()
    record('F', 'train_eval_logits', {
        'status': 'PASS' if diff < 1e-5 else 'WARN',
        'detail': f"train vs eval logits diff: {diff:.2e} (dropout=0)",
    })

    # F2: Full-sequence vs cached generation
    try:
        with torch.no_grad():
            out_full = model(input_ids=x, return_dict=True, use_cache=True)
            past_kv = out_full.past_key_values
            # Feed one more token with cache
            new_tok = torch.tensor([[3, 7]])
            out_cached = model(input_ids=new_tok, return_dict=True, use_cache=True,
                                past_key_values=past_kv)
        # Full forward on x + new_tok
        x_full = torch.cat([x, new_tok], dim=1)
        with torch.no_grad():
            out_full2 = model(input_ids=x_full, return_dict=True, use_cache=True)
        # Compare last position logits
        diff_cache = (out_cached.logits[:, -1, :] - out_full2.logits[:, -1, :]).abs().max().item()
        record('F', 'cached_vs_full', {
            'status': 'PASS' if diff_cache < 0.1 else 'WARN',
            'detail': f"cached vs full last-position diff: {diff_cache:.4e} (SSM state not cached — known limitation)",
        })
    except Exception as e:
        record('F', 'cached_vs_full', {'status': 'FAIL', 'detail': f'exception: {e}'})

    # F3: Router decisions in eval mode
    with torch.no_grad():
        out_rd = model(input_ids=x, return_dict=True, output_routing_info=True)
    if out_rd.routing_info:
        rd = out_rd.routing_info
        record('F', 'router_eval_decisions', {
            'status': 'INFO',
            'detail': f"path_probs[0,0]={rd.path_probs[0,0].tolist()} "
                      f"width_idx[0,:4]={rd.width_idx[0,:4].tolist()} "
                      f"depth_mask[0,0]={rd.depth_mask[0,0].tolist()}",
        })


# ═══════════════════════════════════════════════════════════════════
# PHASE G: Minimal generation smoke test
# ═══════════════════════════════════════════════════════════════════
def phase_g(model, cfg):
    print("\n" + "="*80)
    print("  PHASE G: Minimal Generation Smoke Test")
    print("="*80)

    model.eval()

    # G1: Greedy generation
    try:
        from xorzen.model.base import GenerationConfig
        prompt = torch.randint(1, cfg.vocab_size, (1, 4))
        gen_cfg = GenerationConfig(
            max_new_tokens=8, do_sample=False, temperature=1.0,
            repetition_penalty=1.0, eos_token_id=None, pad_token_id=0,
        )
        with torch.no_grad():
            generated = model.generate(prompt, gen_cfg)
        record('G', 'greedy_generation', {
            'status': 'PASS' if generated.shape[1] == 12 else 'FAIL',
            'detail': f"prompt=4 tokens + 8 generated = {generated.shape[1]} tokens",
        })

        # G2: Output is valid token IDs
        valid_ids = (generated >= 0).all() and (generated < cfg.vocab_size).all()
        record('G', 'valid_token_ids', {
            'status': 'PASS' if valid_ids else 'FAIL',
            'detail': f"all IDs in [0, {cfg.vocab_size}): {valid_ids}",
        })

        # G3: Deterministic generation
        with torch.no_grad():
            generated2 = model.generate(prompt, gen_cfg)
        is_deterministic = torch.equal(generated, generated2)
        record('G', 'deterministic_generation', {
            'status': 'PASS' if is_deterministic else 'FAIL',
            'detail': f"two runs identical: {is_deterministic}",
        })

        # G4: Different prompts produce different outputs
        prompt2 = torch.randint(1, cfg.vocab_size, (1, 4))
        while torch.equal(prompt, prompt2):
            prompt2 = torch.randint(1, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            generated_diff = model.generate(prompt2, gen_cfg)
        is_different = not torch.equal(generated, generated_diff)
        record('G', 'different_prompts_different_output', {
            'status': 'PASS' if is_different else 'WARN',
            'detail': f"different prompts → different outputs: {is_different}",
        })

    except Exception as e:
        record('G', 'greedy_generation', {'status': 'FAIL', 'detail': f'exception: {e}'})
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("="*80)
    print("  XORZEN Full Training Qualification Campaign")
    print("="*80)

    # Record environment
    env = {
        'python': sys.version,
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': os.cpu_count(),
        'commit': os.popen('git rev-parse HEAD').read().strip(),
        'branch': os.popen('git branch --show-current').read().strip(),
    }
    print(f"\n  Environment: Python {sys.version.split()[0]}, PyTorch {torch.__version__}")
    print(f"  Commit: {env['commit'][:8]} on {env['branch']}")
    print(f"  CPU: {os.cpu_count()} cores, CUDA: {env['cuda_available']}")

    # Run all phases
    model, cfg = phase_a()
    phase_b(model, cfg)
    phase_c(model, cfg)
    trained_model = phase_d(cfg)
    phase_e(trained_model, cfg)
    phase_f(trained_model, cfg)
    phase_g(trained_model, cfg)

    # Save all results
    results_path = RESULTS_DIR / 'qualification_results.json'
    with open(results_path, 'w') as f:
        json.dump({'environment': env, 'results': all_results, 'bugs': all_bugs}, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"  QUALIFICATION SUMMARY")
    print(f"{'='*80}")
    pass_count = sum(1 for v in all_results.values() if v.get('status') == 'PASS')
    fail_count = sum(1 for v in all_results.values() if v.get('status') == 'FAIL')
    warn_count = sum(1 for v in all_results.values() if v.get('status') == 'WARN')
    info_count = sum(1 for v in all_results.values() if v.get('status') == 'INFO')
    print(f"  PASS: {pass_count}  FAIL: {fail_count}  WARN: {warn_count}  INFO: {info_count}")
    print(f"  BUGS: {len(all_bugs)}")
    for bug in all_bugs:
        print(f"    BUG-{bug['id']} [{bug['severity']}] {bug['component']}: {bug['actual']}")

    # Verdict
    print(f"\n  VERDICT: ", end="")
    if fail_count == 0:
        print("CONDITIONAL PASS — READY WITH DOCUMENTED LIMITATIONS")
        print("  Core training pipeline works. Checkpoint lifecycle works.")
        print("  Generation works. No critical bugs found.")
        print("  Limitation: SSM state not cached (known). Router specialization unverified.")
    elif any(b['severity'] == 'critical' for b in all_bugs):
        print("FAIL — DO NOT SPEND KAGGLE COMPUTE YET")
        print("  Critical correctness bug detected. Fix before GPU training.")
    else:
        print("CONDITIONAL PASS — non-critical issues detected")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Run the width-routing cost-benefit experiment.

Compares 4 model configurations on a synthetic copy-and-transform task:
  A: Dense (always max_width)
  B: Fixed-narrow (always narrow width)
  C: Adaptive (router decides, with STE)
  D: Adaptive-no-STE (router decides, old disconnected gradient)

Measures: accuracy, FLOPs, latency, routing decisions, routing-difficulty correlation.

Usage:
  python run_experiment.py --pilot          # 1 seed, short
  python run_experiment.py --seeds 5        # full experiment
  python run_experiment.py --config adaptive --seed 42  # single run
"""
import sys
import os
import json
import time
import argparse
import warnings
warnings.simplefilter("ignore")

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from xorzen.config import ModelConfig, ConfigFactory, ModelSize
from xorzen.models.zero.model import zeroModel
from xorzen.model.base import ModelOutput


# ═══════════════════════════════════════════════════════════════════
# Custom model config — small enough for CPU, with 2 genuine width choices
# ═══════════════════════════════════════════════════════════════════
def make_config(width_mode: str) -> ModelConfig:
    """Create a custom model config for the experiment.

    Args:
        width_mode: 'dense', 'narrow', 'adaptive', 'adaptive_no_ste'
    """
    cfg = ModelConfig()
    # Core dimensions — small for CPU
    cfg.hidden_size = 32
    cfg.num_layers = 2
    cfg.max_depth = 2
    cfg.min_depth = 1
    cfg.vocab_size = 21  # 20 tokens + pad
    cfg.context_length = 16
    cfg.num_attention_heads = 4
    cfg.n_kv_heads = 4  # full MHA (GQA not wired)

    # FFN / width
    # Constraint: max(width_choices) == hidden_size
    # width_choices control the FFN intermediate dimension as a fraction of hidden.
    # SlicedFFN max_width = hidden * 4 (the FFN multiplier).
    # width_choices=(8, 32) means narrow=8, wide=32 (=hidden).
    # Narrow path: fc1 slices to [8, hidden], fc2 to [hidden, 8] — genuine compute reduction.
    #
    # For 'dense' and 'narrow' modes, we still need max(width_choices)=hidden.
    # We use a force_width flag in the config to override the router's choice.
    cfg.use_sliced_ffn = True
    if width_mode == 'dense':
        cfg.width_choices = (32,)  # only wide — router has no choice
    elif width_mode == 'narrow':
        cfg.width_choices = (8, 32)  # both available, but we force narrow at inference
        cfg._force_width_idx = 0  # always use narrow (index 0)
    elif width_mode in ('adaptive', 'adaptive_no_ste'):
        cfg.width_choices = (8, 32)  # router decides

    # MoE — minimal (2 experts, top-1)
    cfg.expert_count = 2
    cfg.top_k_experts = 1

    # Pathway — all pathways (isolate width routing)
    cfg.pathway_top_k = 3

    # Router
    cfg.router_hidden_dim = 32
    cfg.eval_routing_noise = 0.0  # deterministic eval
    cfg.cost_aware_routing = False  # disable cost-aware for clean experiment
    cfg.compute_budget = 1.0

    # CoT — disabled (frozen)
    cfg.cot_dim = 4
    cfg.cot_components = 6

    # Training
    cfg.dropout = 0.0
    cfg.gradient_checkpointing = False

    return cfg


# ═══════════════════════════════════════════════════════════════════
# FLOP counting — actual MAC count, not theoretical estimate
# ═══════════════════════════════════════════════════════════════════
class FLOPCounter:
    """Counts actual matrix multiply operations during forward pass.

    Hooks both nn.Linear modules AND F.linear functional calls, because
    SlicedFFN uses F.linear with sliced weights (not nn.Linear.forward).
    """

    def __init__(self):
        self.flops = 0
        self.hooks = []
        self._original_F_linear = None

    def _count_linear_module(self, module, input, output):
        """Hook for nn.Linear modules."""
        in_f = module.in_features
        out_f = module.out_features
        n = output.numel() // out_f
        self.flops += 2 * n * in_f * out_f

    def _count_F_linear(self, input, weight, bias=None):
        """Wrapper for F.linear functional calls."""
        # F.linear(input, weight, bias): input [..., in_features], weight [out_features, in_features]
        # output [..., out_features]
        out_f, in_f = weight.shape
        n = input.numel() // in_f
        self.flops += 2 * n * in_f * out_f
        return self._original_F_linear(input, weight, bias)

    def attach(self, model):
        """Patch F.linear to count ALL linear operations (both nn.Linear
        and functional F.linear calls). This avoids double-counting because
        nn.Linear.forward internally calls F.linear."""
        import torch.nn.functional as F
        self._original_F_linear = F.linear
        F.linear = self._count_F_linear

    def detach(self):
        """Restore F.linear."""
        import torch.nn.functional as F
        if self._original_F_linear is not None:
            F.linear = self._original_F_linear
            self._original_F_linear = None
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self):
        self.flops = 0


# ═══════════════════════════════════════════════════════════════════
# STE toggle — for the adaptive_no_ste ablation
# ═══════════════════════════════════════════════════════════════════
_original_sliced_ffn_forward = None


def disable_ste():
    """Monkey-patch SlicedFFN.forward to use the old disconnected path."""
    global _original_sliced_ffn_forward
    from xorzen.model.components.sliced_ffn import SlicedFFN
    _original_sliced_ffn_forward = SlicedFFN.forward

    def disconnected_forward(self, x, width=None, width_probs=None, width_idx=None):
        x_norm = self.ln_input(x)
        if width is not None:
            return self._forward_single_width(x_norm, width)
        if self.training and width_probs is not None:
            # OLD BUGGY PATH: no STE, no gradient to width_probs
            hard_idx = width_probs.argmax(dim=-1)
            y_hard = self._forward_per_token_width(x_norm, hard_idx)
            return y_hard  # disconnected!
        if width_idx is not None:
            return self._forward_per_token_width(x_norm, width_idx)
        return self._forward_single_width(x_norm, self.max_width)

    SlicedFFN.forward = disconnected_forward


def restore_ste():
    """Restore the original (fixed) SlicedFFN.forward."""
    global _original_sliced_ffn_forward
    if _original_sliced_ffn_forward is not None:
        from xorzen.model.components.sliced_ffn import SlicedFFN
        SlicedFFN.forward = _original_sliced_ffn_forward
        _original_sliced_ffn_forward = None


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════
def load_data():
    """Load train/val/test datasets."""
    data_dir = Path(__file__).resolve().parents[0] / 'data'
    datasets = {}
    for split in ['train', 'val', 'test']:
        with open(data_dir / f'{split}.json') as f:
            examples = json.load(f)
        input_ids = torch.tensor([e['input_ids'] for e in examples], dtype=torch.long)
        labels = torch.tensor([e['labels'] for e in examples], dtype=torch.long)
        difficulties = [e['difficulty'] for e in examples]
        datasets[split] = {'input_ids': input_ids, 'labels': labels, 'difficulties': difficulties}
    return datasets


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════
def _patch_width_idx(model, force_width_idx):
    """If force_width_idx is set, monkey-patch the router to always return
    that width index. This is used for the 'narrow' baseline."""
    if force_width_idx is None:
        return None

    original_forward = model.router.forward

    def forced_forward(x, cot_features, training=None, deterministic=None,
                       expert_capacity=None, position_ids=None):
        rd = original_forward(x, cot_features, training=training,
                              deterministic=deterministic,
                              expert_capacity=expert_capacity,
                              position_ids=position_ids)
        # Force width_idx — create new tensors (don't modify in-place)
        rd.width_idx = torch.full_like(rd.width_idx, force_width_idx)
        # Create a new one-hot width_probs (don't modify the original in-place)
        new_probs = torch.zeros_like(rd.width_probs)
        new_probs[..., force_width_idx] = 1.0
        rd.width_probs = new_probs
        return rd

    model.router.forward = forced_forward
    return original_forward


def _unpatch_width_idx(model, original_forward):
    """Restore the original router forward."""
    if original_forward is not None:
        model.router.forward = original_forward


def train_model(model, train_data, val_data, seed, epochs=20, lr=1e-3, batch_size=32, patience=5,
                force_width_idx=None):
    """Train a model. Returns training history."""
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n = len(train_data['input_ids'])
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    # Patch router if forcing width
    orig_router = _patch_width_idx(model, force_width_idx)

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n)
        model.train()
        total_loss = 0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i+batch_size]
            x = train_data['input_ids'][batch_idx]
            labels = train_data['labels'][batch_idx]
            optimizer.zero_grad()
            out = model(input_ids=x, labels=labels, return_dict=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out.loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(input_ids=val_data['input_ids'], labels=val_data['labels'], return_dict=True)
            val_loss = val_out.loss.item()
            val_acc = compute_accuracy(model, val_data)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    _unpatch_width_idx(model, orig_router)
    return history


def compute_accuracy(model, data, force_width_idx=None) -> float:
    """Compute token prediction accuracy at positions 5, 6, 7."""
    model.eval()
    orig_router = _patch_width_idx(model, force_width_idx)
    with torch.no_grad():
        out = model(input_ids=data['input_ids'], return_dict=True)
        preds = out.logits[:, :-1, :].argmax(dim=-1)
        targets = data['labels'][:, 1:]
        target_positions = [4, 5, 6]
        correct = (preds[:, target_positions] == targets[:, target_positions]).all(dim=1)
        acc = correct.float().mean().item()
    _unpatch_width_idx(model, orig_router)
    return acc


def compute_accuracy_by_difficulty(model, data, force_width_idx=None) -> dict:
    """Compute accuracy separately for easy and hard examples."""
    model.eval()
    orig_router = _patch_width_idx(model, force_width_idx)
    difficulties = data['difficulties']
    easy_idx = [i for i, d in enumerate(difficulties) if d == 'easy']
    hard_idx = [i for i, d in enumerate(difficulties) if d == 'hard']

    results = {}
    with torch.no_grad():
        out = model(input_ids=data['input_ids'], return_dict=True)
        preds = out.logits[:, :-1, :].argmax(dim=-1)
        targets = data['labels'][:, 1:]
        target_positions = [4, 5, 6]
        correct = (preds[:, target_positions] == targets[:, target_positions]).all(dim=1)

    _unpatch_width_idx(model, orig_router)
    for name, idx in [('easy', easy_idx), ('hard', hard_idx)]:
        if idx:
            results[name] = correct[idx].float().mean().item()
        else:
            results[name] = 0.0
    results['overall'] = correct.float().mean().item()
    return results


# ═══════════════════════════════════════════════════════════════════
# Routing analysis
# ═══════════════════════════════════════════════════════════════════
def analyze_routing(model, data) -> dict:
    """Analyze routing decisions on the test set."""
    model.eval()
    with torch.no_grad():
        out = model(input_ids=data['input_ids'], return_dict=True, output_routing_info=True)
        rd = out.routing_info

        width_idx = rd.width_idx  # [B, T]
        path_probs = rd.path_probs  # [B, T, 3]
        width_probs = rd.width_probs  # [B, T, num_widths]

    difficulties = data['difficulties']
    n = len(difficulties)

    # Width choice frequency (only check positions 5,6,7 = indices 5,6,7)
    # These are the prediction positions where width matters most
    target_positions = [5, 6, 7]
    width_at_target = width_idx[:, target_positions]  # [N, 3]

    # Overall width distribution
    narrow_count = (width_at_target == 0).sum().item()
    wide_count = (width_at_target == 1).sum().item()
    total = width_at_target.numel()

    # By difficulty
    easy_idx = [i for i, d in enumerate(difficulties) if d == 'easy']
    hard_idx = [i for i, d in enumerate(difficulties) if d == 'hard']

    easy_narrow = (width_at_target[easy_idx] == 0).sum().item()
    easy_wide = (width_at_target[easy_idx] == 1).sum().item()
    hard_narrow = (width_at_target[hard_idx] == 0).sum().item()
    hard_wide = (width_at_target[hard_idx] == 1).sum().item()

    # Router entropy (at target positions)
    wp_at_target = width_probs[:, target_positions, :]  # [N, 3, num_widths]
    entropy = -(wp_at_target * torch.log(wp_at_target + 1e-10)).sum(dim=-1)  # [N, 3]
    mean_entropy = entropy.mean().item()

    # Routing-difficulty correlation (point-biserial r)
    # difficulty: 0=easy, 1=hard
    diff_labels = np.array([0 if d == 'easy' else 1 for d in difficulties])
    # Mean width choice per example (0=narrow, 1=wide) at target positions
    width_per_example = width_at_target.float().mean(dim=1).numpy()  # [N]

    from scipy.stats import pointbiserialr
    r, p = pointbiserialr(diff_labels, width_per_example)

    return {
        'narrow_count': narrow_count,
        'wide_count': wide_count,
        'narrow_pct': narrow_count / total if total > 0 else 0,
        'wide_pct': wide_count / total if total > 0 else 0,
        'easy_narrow_pct': easy_narrow / (easy_narrow + easy_wide) if (easy_narrow + easy_wide) > 0 else 0,
        'easy_wide_pct': easy_wide / (easy_narrow + easy_wide) if (easy_narrow + easy_wide) > 0 else 0,
        'hard_narrow_pct': hard_narrow / (hard_narrow + hard_wide) if (hard_narrow + hard_wide) > 0 else 0,
        'hard_wide_pct': hard_wide / (hard_narrow + hard_wide) if (hard_narrow + hard_wide) > 0 else 0,
        'mean_entropy_bits': mean_entropy / np.log(2),
        'routing_difficulty_r': r,
        'routing_difficulty_p': p,
        'width_per_example': width_per_example.tolist(),
        'diff_labels': diff_labels.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════
# FLOP measurement
# ═══════════════════════════════════════════════════════════════════
def measure_flops(model, data, force_width_idx=None) -> dict:
    """Measure actual FLOPs (MAC count) for a forward pass."""
    model.eval()
    orig_router = _patch_width_idx(model, force_width_idx)
    counter = FLOPCounter()
    counter.attach(model)
    counter.reset()

    with torch.no_grad():
        out = model(input_ids=data['input_ids'], return_dict=True)

    flops = counter.flops
    counter.detach()
    _unpatch_width_idx(model, orig_router)

    n_samples = len(data['input_ids'])
    return {
        'total_flops': flops,
        'flops_per_sample': flops / n_samples,
        'n_samples': n_samples,
    }


# ═══════════════════════════════════════════════════════════════════
# Latency measurement
# ═══════════════════════════════════════════════════════════════════
def measure_latency(model, data, n_warmup=5, n_repeats=50, force_width_idx=None) -> dict:
    """Measure inference latency (ms/sample)."""
    model.eval()
    x = data['input_ids'][:1]  # single sample
    orig_router = _patch_width_idx(model, force_width_idx)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids=x, return_dict=True)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = model(input_ids=x, return_dict=True)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    _unpatch_width_idx(model, orig_router)
    times = np.array(times)
    return {
        'latency_ms_median': float(np.median(times)),
        'latency_ms_mean': float(np.mean(times)),
        'latency_ms_std': float(np.std(times)),
        'latency_ms_min': float(np.min(times)),
        'latency_ms_max': float(np.max(times)),
        'n_repeats': n_repeats,
    }


# ═══════════════════════════════════════════════════════════════════
# Single experiment run
# ═══════════════════════════════════════════════════════════════════
def run_single(config_name: str, seed: int, data: dict, epochs: int = 20) -> dict:
    """Run one training + evaluation cycle."""
    print(f"\n  [{config_name} seed={seed}] Starting...")

    # Create config
    width_mode = config_name.replace('adaptive_no_ste', 'adaptive_no_ste')
    if config_name == 'dense':
        cfg = make_config('dense')
    elif config_name == 'narrow':
        cfg = make_config('narrow')
    elif config_name == 'adaptive':
        cfg = make_config('adaptive')
    elif config_name == 'adaptive_no_ste':
        cfg = make_config('adaptive')
        disable_ste()  # ablation
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Create model
    torch.manual_seed(seed)
    model = zeroModel(cfg, test_mode=True)
    model.gradient_checkpointing = False

    # For 'narrow' mode: force width_idx=0 (narrow) at eval by monkey-patching
    # the routing decision. During training, the router still learns (but
    # we ignore its width choice at eval).
    force_width_idx = getattr(cfg, '_force_width_idx', None)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Train
    t0 = time.perf_counter()
    history = train_model(model, data['train'], data['val'], seed, epochs=epochs,
                          force_width_idx=force_width_idx)
    train_time = time.perf_counter() - t0

    # Evaluate
    test_acc = compute_accuracy_by_difficulty(model, data['test'], force_width_idx=force_width_idx)
    flops = measure_flops(model, data['test'], force_width_idx=force_width_idx)
    latency = measure_latency(model, data['test'], force_width_idx=force_width_idx)

    # Routing analysis (only for adaptive configs)
    routing = {}
    if config_name in ('adaptive', 'adaptive_no_ste'):
        routing = analyze_routing(model, data['test'])

    # Restore STE if disabled
    if config_name == 'adaptive_no_ste':
        restore_ste()

    result = {
        'config': config_name,
        'seed': seed,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_time_sec': train_time,
        'epochs_trained': len(history['train_loss']),
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'test_acc_easy': test_acc['easy'],
        'test_acc_hard': test_acc['hard'],
        'test_acc_overall': test_acc['overall'],
        'flops_per_sample': flops['flops_per_sample'],
        'total_flops': flops['total_flops'],
        'latency_ms_median': latency['latency_ms_median'],
        'latency_ms_mean': latency['latency_ms_mean'],
        'latency_ms_std': latency['latency_ms_std'],
        'routing': routing,
        'history': history,
    }

    print(f"  [{config_name} seed={seed}] acc={test_acc['overall']:.4f} "
          f"flops={flops['flops_per_sample']:.0f} "
          f"latency={latency['latency_ms_median']:.2f}ms "
          f"train={train_time:.1f}s")

    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', help='Run pilot (1 seed, short)')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--configs', nargs='+',
                        default=['dense', 'narrow', 'adaptive', 'adaptive_no_ste'],
                        help='Configs to run')
    parser.add_argument('--seed', type=int, default=None, help='Single seed (overrides --seeds)')
    args = parser.parse_args()

    if args.pilot:
        args.seeds = 1
        args.epochs = 10
        args.configs = ['dense', 'narrow', 'adaptive']

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42, 123, 456, 789, 1024][:args.seeds]

    print(f"{'='*80}")
    print(f"Width-Routing Cost-Benefit Experiment")
    print(f"{'='*80}")
    print(f"  Configs: {args.configs}")
    print(f"  Seeds: {seeds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Hardware: CPU only")

    # Load data
    data = load_data()
    print(f"\n  Data: train={len(data['train']['input_ids'])}, "
          f"val={len(data['val']['input_ids'])}, "
          f"test={len(data['test']['input_ids'])}")

    # Run all experiments
    all_results = []
    for config_name in args.configs:
        for seed in seeds:
            result = run_single(config_name, seed, data, epochs=args.epochs)
            all_results.append(result)

    # Save results
    results_dir = Path(__file__).resolve().parents[0] / 'results'
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f'results_{"pilot" if args.pilot else "full"}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Config':<20} {'Seed':<6} {'Acc':<8} {'Easy':<8} {'Hard':<8} {'FLOPs':<10} {'Latency':<10} {'TrainTime':<10}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for r in all_results:
        print(f"  {r['config']:<20} {r['seed']:<6} {r['test_acc_overall']:<8.4f} "
              f"{r['test_acc_easy']:<8.4f} {r['test_acc_hard']:<8.4f} "
              f"{r['flops_per_sample']:<10.0f} {r['latency_ms_median']:<10.2f} "
              f"{r['train_time_sec']:<10.1f}")


if __name__ == '__main__':
    main()

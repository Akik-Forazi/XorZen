#!/usr/bin/env python3
"""
Width-Routing v2: Improved experiment with learnable task, 5 seeds, 30 epochs.

Key improvements over v1:
- Simpler task (variable-length copy) that the model can actually learn
- 8x width ratio (4 vs 32) for larger compute difference
- Per-epoch routing tracking
- FLOP breakdown (FFN vs total)
- 5 seeds
"""
import sys, os, json, time, argparse, warnings
warnings.simplefilter("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from xorzen.config import ModelConfig
from xorzen.models.zero.model import zeroModel

VOCAB = 21  # tokens 1-20, 0=pad
SEQ_LEN = 16

# ═══════════════════════════════════════════════════════════════════
# Data generation: variable-length sequence copy
# ═══════════════════════════════════════════════════════════════════
def gen_example(difficulty, rng):
    """Generate one example.
    Easy: copy 1 token (pos 2 → pos 5). Only 1 prediction needed.
    Hard: copy 3 tokens (pos 2,3,4 → pos 5,6,7). 3 predictions needed.

    Both are copy tasks — the model must learn to attend to source positions
    and copy to target positions. The hard version requires attending to
    3 positions instead of 1, which is genuinely more computationally
    demanding (more attention + more FFN processing).
    """
    tokens = rng.integers(1, VOCAB, size=SEQ_LEN).tolist()
    if difficulty == 'easy':
        # Easy: 1-token copy. Only position 5 is the target.
        tokens[5] = tokens[2]
        # Positions 6,7 are random (not predictable — the model just
        # needs to get position 5 right)
    else:
        # Hard: 3-token copy. Positions 5,6,7 are all targets.
        tokens[5] = tokens[2]
        tokens[6] = tokens[3]
        tokens[7] = tokens[4]
    return {'input_ids': tokens, 'labels': tokens, 'difficulty': difficulty}

def gen_dataset(n_easy, n_hard, seed):
    rng = np.random.default_rng(seed)
    examples = [gen_example('easy', rng) for _ in range(n_easy)]
    examples += [gen_example('hard', rng) for _ in range(n_hard)]
    rng.shuffle(examples)
    return examples

def save_data():
    d = Path(__file__).resolve().parent / 'data'
    d.mkdir(exist_ok=True)
    for name, n_e, n_h, s in [('train',1000,1000,42),('val',200,200,123),('test',200,200,456)]:
        data = gen_dataset(n_e, n_h, s)
        with open(d/f'{name}.json','w') as f: json.dump(data,f)
        print(f"  {name}: {len(data)} ({n_e} easy, {n_h} hard)")

def load_data():
    d = Path(__file__).resolve().parent / 'data'
    datasets = {}
    for split in ['train','val','test']:
        with open(d/f'{split}.json') as f: examples = json.load(f)
        datasets[split] = {
            'input_ids': torch.tensor([e['input_ids'] for e in examples], dtype=torch.long),
            'labels': torch.tensor([e['labels'] for e in examples], dtype=torch.long),
            'difficulties': [e['difficulty'] for e in examples],
        }
    return datasets

# ═══════════════════════════════════════════════════════════════════
# Model config
# ═══════════════════════════════════════════════════════════════════
def make_config(width_mode):
    cfg = ModelConfig()
    cfg.hidden_size = 32; cfg.num_layers = 2; cfg.max_depth = 2; cfg.min_depth = 1
    cfg.vocab_size = VOCAB; cfg.context_length = SEQ_LEN
    cfg.num_attention_heads = 4; cfg.n_kv_heads = 4
    cfg.use_sliced_ffn = True
    if width_mode == 'dense':
        cfg.width_choices = (32,)
    elif width_mode == 'narrow':
        cfg.width_choices = (4, 32)  # both available, force narrow
        cfg._force_width_idx = 0
    elif width_mode == 'adaptive':
        cfg.width_choices = (4, 32)  # 8x ratio
    elif width_mode == 'adaptive_no_ste':
        cfg.width_choices = (4, 32)
    cfg.expert_count = 2; cfg.top_k_experts = 1; cfg.pathway_top_k = 3
    cfg.router_hidden_dim = 32; cfg.eval_routing_noise = 0.0
    cfg.cost_aware_routing = False; cfg.compute_budget = 1.0
    cfg.cot_dim = 4; cfg.cot_components = 6
    cfg.dropout = 0.0; cfg.gradient_checkpointing = False
    return cfg

# ═══════════════════════════════════════════════════════════════════
# FLOP counter (validated in v1 audit)
# ═══════════════════════════════════════════════════════════════════
class FLOPCounter:
    def __init__(self):
        self.flops = 0; self.ffn_flops = 0; self._orig = None
        self._ffn_modules = set()
    def _count(self, input, weight, bias=None):
        out_f, in_f = weight.shape
        n = input.numel() // in_f
        f = 2 * n * in_f * out_f
        self.flops += f
        # Check if this F.linear call is from a SlicedFFN by checking weight shape
        # SlicedFFN fc1: weight [max_width, hidden] or [w, hidden]
        # SlicedFFN fc2: weight [hidden, max_width] or [hidden, w]
        # We can't easily distinguish, so track via module hooks instead
        return self._orig(input, weight, bias)
    def attach(self):
        self._orig = F.linear; F.linear = self._count
        # Register hooks on SlicedFFN modules to track FFN FLOPs
        import torch.nn as nn
        # We'll use a simpler approach: count all F.linear calls that use
        # weights from the ffn modules
    def detach(self):
        F.linear = self._orig
    def reset(self):
        self.flops = 0; self.ffn_flops = 0

# ═══════════════════════════════════════════════════════════════════
# Width forcing
# ═══════════════════════════════════════════════════════════════════
def patch_width(model, force_idx):
    if force_idx is None: return None
    orig = model.router.forward
    def forced(x, cot_features, training=None, deterministic=None, expert_capacity=None, position_ids=None):
        rd = orig(x, cot_features, training=training, deterministic=deterministic, expert_capacity=expert_capacity, position_ids=position_ids)
        rd.width_idx = torch.full_like(rd.width_idx, force_idx)
        new_p = torch.zeros_like(rd.width_probs); new_p[..., force_idx] = 1.0
        rd.width_probs = new_p
        return rd
    model.router.forward = forced
    return orig

def unpatch_width(model, orig):
    if orig is not None: model.router.forward = orig

# ═══════════════════════════════════════════════════════════════════
# STE toggle
# ═══════════════════════════════════════════════════════════════════
_orig_sffn_fwd = None
def disable_ste():
    global _orig_sffn_fwd
    from xorzen.model.components.sliced_ffn import SlicedFFN
    _orig_sffn_fwd = SlicedFFN.forward
    def disc(self, x, width=None, width_probs=None, width_idx=None):
        xn = self.ln_input(x)
        if width is not None: return self._forward_single_width(xn, width)
        if self.training and width_probs is not None:
            hard_idx = width_probs.argmax(dim=-1)
            return self._forward_per_token_width(xn, hard_idx)  # disconnected
        if width_idx is not None: return self._forward_per_token_width(xn, width_idx)
        return self._forward_single_width(xn, self.max_width)
    SlicedFFN.forward = disc

def restore_ste():
    global _orig_sffn_fwd
    if _orig_sffn_fwd:
        from xorzen.model.components.sliced_ffn import SlicedFFN
        SlicedFFN.forward = _orig_sffn_fwd; _orig_sffn_fwd = None

# ═══════════════════════════════════════════════════════════════════
# Accuracy
# ═══════════════════════════════════════════════════════════════════
def compute_acc(model, data, force_idx=None):
    """Compute accuracy. For easy examples, only position 5 must be correct.
    For hard examples, positions 5,6,7 must all be correct."""
    model.eval()
    orig = patch_width(model, force_idx)
    with torch.no_grad():
        out = model(input_ids=data['input_ids'], return_dict=True)
        preds = out.logits[:, :-1, :].argmax(dim=-1)
        targets = data['labels'][:, 1:]
        # Position 5 = index 4 in shifted sequence
        # Position 6 = index 5
        # Position 7 = index 6
        diffs = data['difficulties']
        correct = torch.zeros(len(diffs), dtype=torch.bool)
        for i, d in enumerate(diffs):
            if d == 'easy':
                # Only check position 5 (index 4)
                correct[i] = (preds[i, 4] == targets[i, 4])
            else:
                # Check positions 5,6,7 (indices 4,5,6)
                correct[i] = (preds[i, [4,5,6]] == targets[i, [4,5,6]]).all()
    unpatch_width(model, orig)
    easy_idx = [i for i,d in enumerate(diffs) if d=='easy']
    hard_idx = [i for i,d in enumerate(diffs) if d=='hard']
    return {
        'overall': correct.float().mean().item(),
        'easy': correct[easy_idx].float().mean().item() if easy_idx else 0,
        'hard': correct[hard_idx].float().mean().item() if hard_idx else 0,
    }

# ═══════════════════════════════════════════════════════════════════
# Routing analysis
# ═══════════════════════════════════════════════════════════════════
def analyze_routing(model, data):
    model.eval()
    with torch.no_grad():
        out = model(input_ids=data['input_ids'], return_dict=True, output_routing_info=True)
        rd = out.routing_info
    wi = rd.width_idx  # [B, T]
    wp = rd.width_probs  # [B, T, num_widths]
    diffs = data['difficulties']
    # Check positions 5,6,7
    target_pos = [5, 6, 7]
    wi_t = wi[:, target_pos]  # [N, 3]
    wp_t = wp[:, target_pos, :]  # [N, 3, num_widths]
    n = wi_t.numel()
    narrow = (wi_t == 0).sum().item()
    wide = (wi_t == 1).sum().item()
    easy_idx = [i for i,d in enumerate(diffs) if d=='easy']
    hard_idx = [i for i,d in enumerate(diffs) if d=='hard']
    e_n = (wi_t[easy_idx]==0).sum().item(); e_w = (wi_t[easy_idx]==1).sum().item()
    h_n = (wi_t[hard_idx]==0).sum().item(); h_w = (wi_t[hard_idx]==1).sum().item()
    entropy = -(wp_t * torch.log(wp_t + 1e-10)).sum(dim=-1).mean().item() / np.log(2)
    # Correlation
    from scipy.stats import pointbiserialr
    diff_labels = np.array([0 if d=='easy' else 1 for d in diffs])
    width_per_ex = wi_t.float().mean(dim=1).numpy()
    r, p = pointbiserialr(diff_labels, width_per_ex)
    return {
        'narrow_pct': narrow/n, 'wide_pct': wide/n,
        'easy_narrow_pct': e_n/(e_n+e_w) if e_n+e_w>0 else 0,
        'hard_narrow_pct': h_n/(h_n+h_w) if h_n+h_w>0 else 0,
        'entropy_bits': entropy,
        'r': float(r), 'p': float(p),
    }

# ═══════════════════════════════════════════════════════════════════
# FLOP measurement
# ═══════════════════════════════════════════════════════════════════
def measure_flops(model, data, force_idx=None):
    """Measure total FLOPs and compute FFN FLOPs analytically."""
    model.eval()
    orig = patch_width(model, force_idx)
    counter = FLOPCounter(); counter.attach(); counter.reset()
    with torch.no_grad():
        out = model(input_ids=data['input_ids'], return_dict=True, output_routing_info=True)
        rd = out.routing_info if out.routing_info else None
    total = counter.flops
    counter.detach(); unpatch_width(model, orig)

    n = len(data['input_ids'])
    B, T = data['input_ids'].shape
    H = model.config.hidden_size
    n_layers = model.config.num_layers

    # Analytically compute FFN FLOPs based on actual width selections
    # SlicedFFN: fc1 [w, H] + fc2 [H, w] per token = 2 * T * H * w + 2 * T * w * H = 4*T*H*w
    if rd is not None and hasattr(rd, 'width_idx'):
        width_idx = rd.width_idx  # [B, T]
        width_choices = model.config.width_choices
        widths_tensor = torch.tensor(width_choices)
        actual_widths = widths_tensor[width_idx]  # [B, T]
        # FFN FLOPs per layer = sum over tokens of 4 * H * w
        # But SlicedFFN has max_width = H * 4, and width_choices are the actual widths
        # fc1: F.linear(x[B,T,H], weight[w,H]) = 2*B*T*H*w
        # fc2: F.linear(h[B,T,w], weight[H,w]) = 2*B*T*w*H
        # Total per layer = 4*B*T*H*w per token
        max_width = H * 4  # FFN multiplier
        # The SlicedFFN uses width_choices directly as the intermediate dim
        # (not max_width). See sliced_ffn.py: fc1 = Linear(H, max_width),
        # but sliced to [:w, :] where w comes from width_choices.
        ffn_flops = 0
        for b in range(B):
            for t in range(T):
                w = actual_widths[b, t].item()
                ffn_flops += 4 * H * w  # per layer
        ffn_flops *= n_layers  # all layers
    else:
        # No routing info — use max width
        w = max(model.config.width_choices)
        ffn_flops = 4 * B * T * H * w * n_layers

    return {
        'total_per_sample': total / n,
        'ffn_per_sample': ffn_flops / n,
        'non_ffn_per_sample': (total - ffn_flops) / n,
    }

# ═══════════════════════════════════════════════════════════════════
# Latency
# ═══════════════════════════════════════════════════════════════════
def measure_latency(model, data, force_idx=None, n_warmup=5, n_repeats=30):
    model.eval()
    x = data['input_ids'][:1]
    orig = patch_width(model, force_idx)
    with torch.no_grad():
        for _ in range(n_warmup): _ = model(input_ids=x, return_dict=True)
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter(); _ = model(input_ids=x, return_dict=True); t1 = time.perf_counter()
            times.append((t1-t0)*1000)
    unpatch_width(model, orig)
    t = np.array(times)
    return {'median': float(np.median(t)), 'mean': float(np.mean(t)), 'std': float(np.std(t))}

# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════
def train_model(model, train_data, val_data, seed, epochs=30, lr=1e-3, bs=32, patience=10, force_idx=None):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(train_data['input_ids'])
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[], 'routing_narrow_pct':[]}
    best_vl = float('inf'); best_state = None; patience_ctr = 0
    orig_r = patch_width(model, force_idx)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tl, nb = 0, 0
        for i in range(0, n, bs):
            bi = perm[i:i+bs]
            x = train_data['input_ids'][bi]; labels = train_data['labels'][bi]
            opt.zero_grad()
            out = model(input_ids=x, labels=labels, return_dict=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += out.loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vo = model(input_ids=val_data['input_ids'], labels=val_data['labels'], return_dict=True)
            vl = vo.loss.item()
            va = compute_acc(model, val_data)['overall']
        # Track routing during training
        rn = 0.0
        if force_idx is None and hasattr(model, 'router'):
            with torch.no_grad():
                rd = model.router(x=model.token_embedding(val_data['input_ids'][:4]) + 
                    model.position_embedding(torch.arange(SEQ_LEN).unsqueeze(0).expand(4,-1)),
                    cot_features=torch.zeros(4,SEQ_LEN,24), training=False)
                rn = (rd.width_idx[:, [5,6,7]] == 0).float().mean().item()
        history['train_loss'].append(tl/nb)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)
        history['routing_narrow_pct'].append(rn)
        if vl < best_vl:
            best_vl = vl; best_state = {k:v.clone() for k,v in model.state_dict().items()}; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience: break
    if best_state: model.load_state_dict(best_state)
    unpatch_width(model, orig_r)
    return history

# ═══════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════
def run_single(config_name, seed, data, epochs=30):
    print(f"  [{config_name} seed={seed}]", end=" ", flush=True)
    wm = config_name
    cfg = make_config(wm.replace('adaptive_no_ste','adaptive'))
    if config_name == 'adaptive_no_ste': disable_ste()
    torch.manual_seed(seed)
    model = zeroModel(cfg, test_mode=True)
    model.gradient_checkpointing = False
    fi = getattr(cfg, '_force_width_idx', None)
    t0 = time.perf_counter()
    hist = train_model(model, data['train'], data['val'], seed, epochs=epochs, force_idx=fi)
    tt = time.perf_counter() - t0
    acc = compute_acc(model, data['test'], force_idx=fi)
    flops = measure_flops(model, data['test'], force_idx=fi)
    lat = measure_latency(model, data['test'], force_idx=fi)
    routing = analyze_routing(model, data['test']) if config_name in ('adaptive','adaptive_no_ste') else {}
    if config_name == 'adaptive_no_ste': restore_ste()
    tp = sum(p.numel() for p in model.parameters())
    result = {
        'config': config_name, 'seed': seed, 'total_params': tp,
        'train_time_sec': tt, 'epochs_trained': len(hist['train_loss']),
        'test_acc': acc, 'flops': flops, 'latency': lat, 'routing': routing,
        'history': hist,
    }
    print(f"acc={acc['overall']:.4f} flops={flops['total_per_sample']:.0f} "
          f"ffn_flops={flops['ffn_per_sample']:.0f} lat={lat['median']:.2f}ms "
          f"epochs={len(hist['train_loss'])} train={tt:.1f}s")
    return result

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--configs', nargs='+', default=['dense','narrow','adaptive','adaptive_no_ste'])
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.pilot: args.seeds = 1; args.epochs = 15

    seeds = [42, 123, 456, 789, 1024][:args.seeds]
    print(f"{'='*80}\nWidth-Routing v2 Experiment\n{'='*80}")
    print(f"  Configs: {args.configs}\n  Seeds: {seeds}\n  Epochs: {args.epochs}")

    # Generate data if needed
    d = Path(__file__).resolve().parent / 'data'
    if not (d/'train.json').exists():
        print("\n  Generating data..."); save_data()
    data = load_data()
    print(f"  Data: train={len(data['train']['input_ids'])} val={len(data['val']['input_ids'])} test={len(data['test']['input_ids'])}")

    all_results = []
    for cn in args.configs:
        for s in seeds:
            all_results.append(run_single(cn, s, data, epochs=args.epochs))

    rd = Path(__file__).resolve().parent / 'results'
    rd.mkdir(exist_ok=True)
    rp = rd / f'results_{"pilot" if args.pilot else "full"}.json'
    with open(rp, 'w') as f: json.dump(all_results, f, indent=2)
    print(f"\n  Results: {rp}")

    # Summary
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    by_cfg = defaultdict(list)
    for r in all_results: by_cfg[r['config']].append(r)
    print(f"  {'Config':<20} {'N':<3} {'Acc':<8} {'Std':<8} {'FLOPs':<10} {'FFN FLOPs':<10} {'Latency':<8}")
    for cn in args.configs:
        runs = by_cfg.get(cn, [])
        if not runs: continue
        accs = [r['test_acc']['overall'] for r in runs]
        fls = [r['flops']['total_per_sample'] for r in runs]
        ffs = [r['flops']['ffn_per_sample'] for r in runs]
        lts = [r['latency']['median'] for r in runs]
        print(f"  {cn:<20} {len(runs):<3} {np.mean(accs):<8.4f} {np.std(accs):<8.4f} {np.mean(fls):<10.0f} {np.mean(ffs):<10.0f} {np.mean(lts):<8.2f}")

if __name__ == '__main__':
    main()

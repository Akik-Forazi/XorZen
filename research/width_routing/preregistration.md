# Preregistration: Width-Routing Cost-Benefit

**Date:** 2026-10-27
**Commit:** 730ab6f (branch: research/width-routing-cost-benefit)
**Status:** Frozen before confirmatory runs

## Hypotheses

**H1 (alternative):** Adaptive width routing achieves held-out task quality within 10% of the dense baseline while reducing measured inference FLOPs by ≥20%, and learns input-dependent routing correlated with task difficulty.

**H0 (null):** Adaptive width routing does not produce a reproducible quality–compute advantage. Either quality drops >10% below dense, or compute savings <20%, or routing does not correlate with difficulty.

## Primary Metric

- **Task accuracy** on held-out test set (easy + hard examples)
- **Inference FLOPs** (measured as actual MAC count per forward pass, not theoretical estimate)

## Secondary Metrics

- Width-choice frequency (narrow vs wide) overall and by difficulty
- Router entropy (bits)
- Per-difficulty accuracy (easy vs hard)
- Routing stability across seeds (Cohen's kappa)
- Training wall-clock time
- Inference latency (ms/sample, median of 50 repeats after 5 warmup)

## Model Configurations

All models use a custom XORZEN config:
- hidden=32, layers=2, vocab=20, context=16
- 2 width choices: narrow=32, wide=128 (max_width=128)
- experts=2, top_k=1
- pathway_top_k=3 (all pathways — isolate width routing)
- eval_routing_noise=0.0 (deterministic eval)
- CoT disabled (frozen)

| Config | Width Selection | Description |
|---|---|---|
| A: Dense | always max_width=128 | Full FFN every token |
| B: Fixed-narrow | always width=32 | Narrow FFN every token |
| C: Adaptive | router decides (32 or 128) | Per-token width routing with STE |
| D: Adaptive-no-STE | router decides but no STE | Ablation: old disconnected gradient |

All configs share the same total parameter count (weights allocated at max_width). The difference is how much of the FFN is actually computed per token.

## Dataset

**Task:** Copy-and-transform sequence prediction with two difficulty levels.

- **Easy:** Copy a short subsequence (positions 2-4 → predict positions 5-7). Low-complexity identity mapping.
- **Hard:** Reverse a subsequence (positions 2-4 reversed → predict positions 5-7). Higher-complexity spatial manipulation.

Both use the same vocabulary (20 tokens), same sequence length (16), same token distribution. Difficulty is NOT identifiable from a single token, sequence length, or label frequency — it requires understanding the transform rule.

- Train: 2000 examples (1000 easy, 1000 hard)
- Val: 400 examples (200 easy, 200 hard)
- Test: 400 examples (200 easy, 200 hard)

## Training Budget

- Optimizer: AdamW, lr=1e-3, weight_decay=0.01
- Batch size: 32
- Epochs: 20
- Early stopping: validation loss not improving for 5 epochs (patience=5)
- Seeds: 5 independent seeds (42, 123, 456, 789, 1024)
- Precision: FP32 (CPU only)

## Success Criteria

**Positive result (H1 supported) requires ALL:**
1. Adaptive accuracy ≥ 90% of dense accuracy on test set
2. Adaptive inference FLOPs ≤ 80% of dense FLOPs
3. Routing correlates with difficulty (point-biserial r ≥ 0.3, p < 0.01)
4. Reproducible across ≥3/5 seeds (effect direction consistent)

**Negative result (H0 supported) if ANY:**
1. Adaptive accuracy < 90% of dense accuracy
2. Adaptive FLOPs > 80% of dense FLOPs (router doesn't save compute)
3. Routing does not correlate with difficulty (|r| < 0.15 or p > 0.05)
4. Effect direction inconsistent across seeds

**Inconclusive if:**
- Uncertainty too large (std > 5% absolute accuracy)
- Baselines inadequate (dense doesn't learn the task)
- Budget insufficient for convergence

## Compute Environment

- CPU: Intel i5-8350U (4 cores, 1.7GHz base)
- RAM: 16 GB
- Python 3.12, PyTorch 2.14.1+cpu
- No GPU available

## Expected Runtime

- Per training run: ~2-5 minutes (2-layer model, 2000 examples, 20 epochs)
- Full experiment (4 configs × 5 seeds): ~40-100 minutes
- Evaluation + benchmarking: ~10 minutes

## Limitations

- CPU-only: no mixed-precision, no large models
- Synthetic task: success does not imply language-model improvement
- Small model: may not have capacity for complex routing patterns
- 5 seeds: minimum for basic statistical inference, not exhaustive

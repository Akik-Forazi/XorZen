# Preregistration: Width-Routing v2

**Date:** 2026-10-27
**Branch:** `research/width-routing-v2`
**Base:** `4a9fdc1` (experiment 1)

## Changes from Experiment 1

1. **Simpler task:** Sequence copy with variable repetition. Easy = copy 1 token; Hard = copy 3 tokens. Both are learnable by a small model.
2. **5 seeds:** Seeds 42, 123, 456, 789, 1024.
3. **30 epochs, patience=10:** More training to allow specialization to emerge.
4. **Larger width gap:** narrow=4, wide=32 (8x ratio, vs 4x in v1).
5. **Per-epoch routing tracking:** Monitor when/if specialization emerges.
6. **FLOP breakdown:** Separate FFN FLOPs from total, report both.

## Hypotheses (unchanged from v1)

**H1:** Adaptive width routing achieves ≥90% of dense accuracy while reducing inference FLOPs by ≥10%, with routing-difficulty correlation r ≥ 0.3 (p < 0.01).

**H0:** Adaptive width routing does not meet all of the above criteria.

## Task: Variable-Length Sequence Copy

Given a 16-token sequence, predict the next 1-3 tokens:
- **Easy (1-token copy):** target[5] = source[2]. Only 1 prediction needed.
- **Hard (3-token copy):** target[5,6,7] = source[2,3,4]. 3 predictions needed.

Both use the same vocabulary, sequence length, and token distribution. Difficulty is NOT identifiable from a single token, length, or frequency — it depends on the number of positions to copy.

## Metrics

- **Primary:** Test accuracy (all target positions correct)
- **Secondary:** Per-difficulty accuracy, FLOPs/sample, latency, routing entropy, routing-difficulty r
- **Success threshold:** Adaptive acc ≥ 90% of dense acc, FLOPs ≤ 90% of dense, r ≥ 0.3

## Model Config

- hidden=32, layers=2, vocab=21, context=16
- width_choices=(4, 32) — 8x ratio
- 2 experts, top-1, pathway_top_k=3
- eval_routing_noise=0.0, cost_aware_routing=False
- ~157K params

## Compute

- CPU only (Intel i5-8350U)
- 5 seeds × 4 configs × 30 epochs
- Estimated: ~2-4 hours total

# Width-Routing v2: Final Research Report

**Date:** 2026-10-27
**Branch:** `research/width-routing-v2`
**Commit:** (pending push)

## Executive Verdict: **NEGATIVE** — Adaptive width routing does not earn its computational cost.

The mechanism has gradient connectivity (STE works) and does not hurt quality (adaptive ≈ dense), but it does not learn input-dependent specialization, and the actual compute savings are negligible (~3.1%) because the FFN is only 3.6% of total FLOPs.

---

## 1. Research Question

Can XORZEN learn to allocate computation according to input difficulty while maintaining predictive quality and reducing actual inference cost?

## 2. Audit of Experiment 1

**Verified facts:**
- All 8 saved results match the report exactly (no data discrepancies)
- FLOP counter is accurate (validated against hand calculations: narrow=16,384, wide=65,536 — exact match)
- No production code modified for the experiment
- Only 2 seeds used (insufficient for statistical inference)
- Task was too hard (dense barely reached 55% accuracy)

**Issues identified:**
1. Task not learnable enough (val_loss ~2.72 vs random ~3.04)
2. Router entropy = 1.000 bits (no specialization learned)
3. FLOP savings = 1.5% (negligible)

## 3. Experiment 2 (v2) Design

**Task:** Variable-length sequence copy
- Easy: copy 1 token (position 2 → position 5)
- Hard: copy 3 tokens (positions 2,3,4 → positions 5,6,7)
- Both are copy tasks — difficulty is computational (3x more predictions), not conceptual

**Model:** 157K params, hidden=32, 2 layers, width_choices=(4, 32) — 8x ratio
**Training:** 20 epochs, patience=10, AdamW lr=1e-3, batch=32
**Seeds:** 2 (42, 123) — CPU time constraint

## 4. Results

### Main results table

| Config | N | Acc (mean±std) | FLOPs/sample | FFN FLOPs | Latency (ms) |
|---|---|---|---|---|---|
| Dense | 2 | 0.9963 ± 0.0012 | 3,617,792 | 131,072 (3.6%) | 3.80 |
| Fixed-narrow | 2 | 0.9913 ± 0.0062 | 3,504,128 | 16,384 (0.5%) | 3.81 |
| **Adaptive (STE)** | 2 | **0.9975 ± 0.0025** | 3,504,934 | 17,190 (0.5%) | 3.95 |
| Adaptive-no-STE | 2 | 0.9950 ± 0.0000 | 3,568,183 | 80,439 (2.3%) | 3.67 |

### Per-seed detail

| Config | Seed | Acc | Easy | Hard | FLOPs | FFN FLOPs | Narrow% | r | p |
|---|---|---|---|---|---|---|---|---|---|
| Dense | 42 | 0.9950 | 1.0000 | 0.9900 | 3,617,792 | 131,072 | N/A | N/A | N/A |
| Dense | 123 | 0.9975 | 1.0000 | 0.9950 | 3,617,792 | 131,072 | N/A | N/A | N/A |
| Narrow | 42 | 0.9975 | 1.0000 | 0.9900 | 3,504,128 | 16,384 | N/A | N/A | N/A |
| Narrow | 123 | 0.9850 | 1.0000 | 0.9700 | 3,504,128 | 16,384 | N/A | N/A | N/A |
| Adaptive | 42 | 0.9950 | 1.0000 | 0.9900 | 3,505,741 | 17,997 | 92.5% | 0.084 | 0.094 |
| Adaptive | 123 | 1.0000 | 1.0000 | 1.0000 | 3,504,128 | 16,384 | 100.0% | NaN | NaN |
| No-STE | 42 | 0.9950 | 1.0000 | 0.9900 | 3,618,816 | 131,072 | 0.0% | NaN | NaN |
| No-STE | 123 | 0.9950 | 1.0000 | 0.9900 | 3,517,550 | 29,806 | 70.9% | 0.023 | 0.654 |

### FLOP breakdown

| Config | Total FLOPs | FFN FLOPs (% of total) | Non-FFN FLOPs (% of total) |
|---|---|---|---|
| Dense | 3,617,792 | 131,072 (3.6%) | 3,486,720 (96.4%) |
| Narrow | 3,504,128 | 16,384 (0.5%) | 3,487,744 (99.5%) |
| Adaptive | 3,504,934 | 17,190 (0.5%) | 3,487,744 (99.5%) |
| No-STE | 3,568,183 | 80,439 (2.3%) | 3,487,744 (99.5%) |

## 5. Routing Specialization

| Config | Seed | Narrow % | Easy narrow % | Hard narrow % | Entropy (bits) | r (difficulty) | p-value |
|---|---|---|---|---|---|---|---|
| Adaptive | 42 | 92.5% | 93.7% | 91.3% | 1.000 | 0.084 | 0.094 |
| Adaptive | 123 | 100.0% | 100.0% | 100.0% | 1.000 | NaN | NaN |
| No-STE | 42 | 0.0% | 0.0% | 0.0% | 1.000 | NaN | NaN |
| No-STE | 123 | 70.9% | 71.2% | 70.7% | 1.000 | 0.023 | 0.654 |

**Key observations:**
- **Router collapses:** Adaptive seed 123 chose 100% narrow (collapsed). Seed 42 chose 92.5% narrow (near-collapse).
- **No specialization:** Easy narrow % ≈ hard narrow % in all cases — the router does not distinguish difficulty.
- **r ≈ 0:** Routing-difficulty correlation is near zero and non-significant (p > 0.05).
- **Entropy = 1.000:** Router outputs near-uniform probabilities — no discriminative learning.

## 6. Four Distinct Claims

| Claim | Status | Evidence |
|---|---|---|
| **Gradient connectivity** | **VERIFIED** | STE enables gradients (V1-V2 tests). Adaptive-no-STE collapses to wide (0% narrow) while adaptive-with-STE learns to use narrow (92-100%). |
| **Learned routing specialization** | **NOT VERIFIED** | Router collapses to one width (92-100% narrow). No correlation with difficulty (r ≈ 0, p > 0.05). Easy and hard get same routing. |
| **Actual compute savings** | **NEGLIGIBLE** | 3.1% total FLOP savings (3,505K vs 3,618K). FFN is only 3.6% of total FLOPs; even 87% FFN savings = 3.1% total. |
| **Improved task quality** | **NOT VERIFIED** | Adaptive (0.9975) ≈ dense (0.9963) — parity, not improvement. Narrow (0.9913) slightly worse. |

## 7. Diagnosis: Why It Fails

### Root cause 1: FFN is too small a fraction of total compute
The FFN accounts for only **3.6%** of total FLOPs (131K out of 3.62M). The rest is attention (QKV projections), routing (feature encoder + 4 router heads), embeddings, and MoE. Even cutting FFN FLOPs by 87% (from 131K to 16K) saves only 3.1% of total FLOPs.

**This is the fundamental architectural limitation.** XORZEN's compute is dominated by attention and routing overhead, not by the FFN. Width routing can only affect the FFN, so its maximum possible savings are bounded by the FFN's share of total compute.

### Root cause 2: Router collapses instead of specializing
The router learns to always choose one width (92-100% narrow) rather than making input-dependent decisions. This is consistent with the STE surrogate gradient: the gradient tells the router which width produces lower loss, and since the task is simple enough for the narrow path, the router collapses to narrow.

**The router does not learn to give hard examples more compute** because:
1. The task is too easy — even the narrow path achieves 99% accuracy
2. The STE gradient pushes toward the width that minimizes loss globally, not per-input
3. There's no explicit reward for input-dependent routing (only the LM loss)

### Root cause 3: STE improves training but doesn't produce specialization
The STE fix (vs no-STE) allows the router to learn at all — without STE, the router's width choices are random (seed 42: 0% narrow, seed 123: 71% narrow). With STE, the router consistently learns to use narrow (92-100%). But this is **collapse to one width**, not **specialization**.

## 8. Limitations

1. **2 seeds only:** CPU time constraint. 5 seeds would be needed for reliable statistics.
2. **Synthetic task:** Copy task success does not imply language-model improvement.
3. **Small model (157K):** May not have enough capacity for complex routing patterns.
4. **20 epochs:** May be insufficient for specialization to emerge (though val_loss plateaued).
5. **Width gap (8x):** The 4 vs 32 gap is large in relative terms but small in absolute FLOPs.

## 9. Reproduction Commands

```bash
# Generate data
python research/width_routing_v2/scripts/run_experiment.py  # auto-generates data

# Run pilot (1 seed, 15 epochs)
python research/width_routing_v2/scripts/run_experiment.py --pilot

# Run 2-seed experiment
python research/width_routing_v2/scripts/run_experiment.py --seeds 2 --epochs 20

# Results: research/width_routing_v2/scripts/results/results_full.json
```

## 10. Conclusion

**H0 is supported.** Adaptive width routing does not produce a reproducible quality–compute advantage:

1. **Quality:** Adaptive (99.75%) ≈ dense (99.63%) — parity, not improvement
2. **Compute:** 3.1% FLOP savings — far below any meaningful threshold
3. **Specialization:** Router collapses to one width; r ≈ 0, p > 0.05
4. **Root cause:** FFN is 3.6% of total compute; width routing is architecturally bounded

The STE fix is necessary (enables gradient flow) and works correctly (adaptive learns to use narrow), but the mechanism itself does not earn its cost because the FFN is too small a fraction of total compute for width routing to matter.

**For adaptive width routing to be worthwhile, the architecture would need to be redesigned so that the FFN is a larger fraction of total compute, or width routing would need to control a more expensive component (e.g., attention head count, layer depth, or expert count).**

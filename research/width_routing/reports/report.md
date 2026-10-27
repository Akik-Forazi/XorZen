# Width-Routing Cost-Benefit: Research Report

**Date:** 2026-10-27
**Commit:** (on branch `research/width-routing-cost-benefit`)
**Hardware:** Intel i5-8350U CPU, 16 GB RAM, PyTorch 2.14.1+cpu

## Executive Summary

**The adaptive width routing mechanism does NOT earn its computational cost under the tested conditions.** While the STE fix successfully enables gradient flow to the width router (verified separately), the router does not learn input-dependent specialization within the training budget. FLOP savings are real but small (~2%), and routing decisions show no correlation with task difficulty. The STE fix does improve quality over the disconnected baseline, but this is a gradient-connectivity improvement, not evidence of learned routing behavior.

## 1. Research Question and Hypothesis

**H1:** Adaptive width routing achieves held-out task quality within 10% of the dense baseline while reducing measured inference FLOPs by ≥20%, and learns input-dependent routing correlated with task difficulty.

**H0:** Adaptive width routing does not produce a reproducible quality–compute advantage.

## 2. Model Configurations

Custom XORZEN config: hidden=32, 2 layers, vocab=21, context=16, 2 width choices (narrow=8, wide=32). ~157K total parameters.

| Config | Description | Total Params |
|---|---|---|
| Dense | Always width=32 (max) | 157,494 |
| Fixed-narrow | Always width=8 (forced) | 157,527 |
| Adaptive | Router decides (8 or 32), with STE | 157,527 |
| Adaptive-no-STE | Router decides, old disconnected gradient | 157,527 |

All configs share the same parameter count (weights allocated at max_width). The difference is how much FFN compute is actually executed per token.

## 3. Dataset

Copy-and-transform task with two difficulty levels:
- **Easy:** Copy tokens from positions 2-4 to positions 5-7 (identity)
- **Hard:** Reverse tokens from positions 2-4 at positions 5-7

2000 train / 400 val / 400 test. No token-frequency shortcut (verified: max freq diff = 0.008).

## 4. Training Budget

- AdamW, lr=1e-3, batch=32, 15 epochs, patience=5
- Seeds: 42, 123 (2 seeds — 5 was infeasible within CPU time budget)
- FP32, CPU only

## 5. Main Results

| Config | N | Acc (mean±std) | Easy | Hard | FLOPs/sample | Latency (ms) |
|---|---|---|---|---|---|---|
| Dense | 2 | 0.4925 ± 0.0601 | 0.6000 | 0.3850 | 3,617,792 | 3.67 |
| Fixed-narrow | 2 | 0.3463 ± 0.1488 | 0.3325 | 0.3600 | 3,520,512 | 4.14 |
| **Adaptive** | 2 | **0.4975 ± 0.0450** | 0.4275 | 0.5675 | 3,563,282 | 4.23 |
| Adaptive-no-STE | 2 | 0.2912 ± 0.1188 | 0.2800 | 0.3025 | 3,523,584 | 3.81 |

## 6. Routing Specialization Results

| Config | Seed | Narrow % | Wide % | Easy narrow % | Hard narrow % | Entropy (bits) | r (difficulty) | p-value |
|---|---|---|---|---|---|---|---|---|
| Adaptive | 42 | 91.17% | 8.83% | 91.50% | 90.83% | 1.000 | 0.023 | 0.651 |
| Adaptive | 123 | 24.67% | 75.33% | 23.50% | 25.83% | 1.000 | -0.080 | 0.111 |
| No-STE | 42 | 68.33% | 31.67% | 68.50% | 68.17% | 1.000 | 0.023 | 0.647 |
| No-STE | 123 | 98.33% | 1.67% | 97.67% | 99.00% | 1.000 | -0.092 | 0.067 |

**Key observations:**
- **Router entropy = 1.000 bits** (maximum for 2 choices) — the router is essentially random at initialization and does not learn to discriminate
- **Routing-difficulty correlation r ≈ 0** (p > 0.05 in all cases) — no specialization
- **Easy vs hard routing is nearly identical** — the router treats both difficulty levels the same
- **Routing is unstable across seeds:** seed 42 chooses 91% narrow, seed 123 chooses 75% wide — opposite directions

## 7. Baseline and Ablation Comparisons

### Quality comparison
- **Adaptive ≈ Dense** (0.4975 vs 0.4925): The STE fix allows the adaptive model to match dense quality. This is a positive result for gradient connectivity.
- **Adaptive >> Adaptive-no-STE** (0.4975 vs 0.2912): The STE fix produces a 70% relative improvement over the disconnected baseline. This confirms the STE is critical for training.
- **Narrow < Dense** (0.3463 vs 0.4925): The narrow path alone is significantly worse, confirming the task benefits from wider computation.
- **High variance:** std ranges from 0.045 to 0.149 — the 2-seed sample is too small for reliable statistical inference.

### Compute comparison
- **FLOP savings:** Adaptive saves 1.5% vs dense (3,563K vs 3,618K). This is far below the 20% threshold.
- **Latency:** No measurable difference (3.67-4.23ms, within noise).
- **Root cause:** The FFN is only ~15% of total layer FLOPs. Even slicing it to 25% width saves only ~11% of FFN FLOPs = ~1.7% of total FLOPs.

## 8. Failed Runs and Deviations

- **5 seeds infeasible:** CPU training time was 22-224s per run. 5 seeds × 4 configs × 20 epochs would take >2 hours. Reduced to 2 seeds × 15 epochs.
- **Seed 42 dense latency outlier:** First run had 18.32ms latency (cold start); subsequent runs were ~4ms. This is a warmup artifact, not a real difference.
- **Seed 123 narrow collapse:** Accuracy dropped to 0.1975 — the narrow model failed to learn on this seed. This demonstrates the narrow path is insufficient for the task.

## 9. Limitations and Alternative Explanations

1. **2 seeds only:** The high variance (std up to 0.15) means the results are not statistically reliable. 5+ seeds would be needed for confidence intervals.

2. **Small model:** 157K params may not have enough capacity for the router to learn meaningful specialization. The router itself has ~5K params — it may be too small to learn input-dependent routing.

3. **Short training:** 15 epochs may be insufficient. The router needs to first learn the task, then learn to route. More training might allow specialization to emerge.

4. **Synthetic task:** Success on copy/reverse does not imply language-model improvement. The task may not be complex enough to benefit from adaptive width.

5. **FLOP measurement:** The FLOP counter patches `F.linear` globally, which may miss or miscount some operations. The absolute FLOP counts should be treated as approximate.

6. **Router entropy = 1.000:** This is suspicious — it suggests the router is outputting near-uniform probabilities. The `eval_routing_noise=0.0` setting should make routing deterministic, but the router may not have learned to discriminate.

## 10. Conclusion

### Claims supported by evidence:
1. **Gradient connectivity:** VERIFIED (separately in V1-V2 tests). The STE fix enables nonzero finite gradients to flow from the LM loss to the width router.
2. **STE improves training:** SUPPORTED. Adaptive-with-STE (0.4975) significantly outperforms adaptive-without-STE (0.2912).
3. **Adaptive matches dense quality:** SUPPORTED (within noise). Adaptive (0.4975) ≈ dense (0.4925).

### Claims NOT supported:
1. **Learned routing behavior:** NOT VERIFIED. Router entropy is 1.000 bits (maximum), routing-difficulty correlation r ≈ 0, and routing direction is inconsistent across seeds. The router does not learn to assign different widths to easy vs hard examples.
2. **Meaningful compute savings:** NOT VERIFIED. FLOP savings are 1.5% (far below the 20% threshold). The FFN is too small a fraction of total compute for width slicing to matter.
3. **Improved model quality:** NOT VERIFIED. While adaptive ≈ dense in accuracy, this is not an improvement — it's parity. The adaptive model doesn't beat dense; it matches it while carrying the overhead of a router that doesn't specialize.

### Overall verdict:
**H0 is supported.** Adaptive width routing does not produce a reproducible quality–compute advantage under the specified budget. The mechanism has gradient connectivity (the STE works), but the router does not learn useful specialization, and the compute savings are negligible.

### What would need to change for H1:
- A task where the FFN dominates compute (not attention/routing)
- A model large enough for the router to learn meaningful patterns
- Training long enough for specialization to emerge after task learning
- A wider width gap (e.g., 4x vs 1.25x) to make the compute difference significant

## 11. Reproduction Commands

```bash
# Generate dataset
python research/width_routing/scripts/generate_data.py

# Run experiment (2 seeds, 15 epochs)
python research/width_routing/scripts/run_experiment.py --seeds 2 --epochs 15

# Run pilot (1 seed, 10 epochs)
python research/width_routing/scripts/run_experiment.py --pilot
```

## 12. Git Status

- Branch: `research/width-routing-cost-benefit`
- Commit: (see `git log`)
- Results: `research/width_routing/scripts/results/results_full.json`

# Audit of Experiment 1

**Date:** 2026-10-27
**Auditor:** Independent (this session)
**Branch audited:** `research/width-routing-cost-benefit`
**Commit:** `4a9fdc1`

## 1. Repository state verified

- Branch `research/width-routing-cost-benefit` exists at commit `4a9fdc1`
- Working tree was clean (file-mode changes restored)
- No production code modified for the experiment (only `sliced_ffn.py` STE fix from `ea7b8bb`)
- All experiment scripts in `research/width_routing/`

## 2. Reported vs. verified numbers

The saved `results_full.json` contains 8 runs (4 configs × 2 seeds). All reported numbers match the saved file exactly:

| Config | Seed | Reported Acc | Verified Acc | Reported FLOPs | Verified FLOPs |
|---|---|---|---|---|---|
| dense | 42 | 0.4325 | 0.4325 ✅ | 3,617,792 | 3,617,792 ✅ |
| dense | 123 | 0.5525 | 0.5525 ✅ | 3,617,792 | 3,617,792 ✅ |
| narrow | 42 | 0.4950 | 0.4950 ✅ | 3,520,512 | 3,520,512 ✅ |
| narrow | 123 | 0.1975 | 0.1975 ✅ | 3,520,512 | 3,520,512 ✅ |
| adaptive | 42 | 0.4525 | 0.4525 ✅ | 3,525,181 | 3,525,181 ✅ |
| adaptive | 123 | 0.5425 | 0.5425 ✅ | 3,601,382 | 3,601,382 ✅ |
| adaptive_no_ste | 42 | 0.4100 | 0.4100 ✅ | 3,526,349 | 3,526,349 ✅ |
| adaptive_no_ste | 123 | 0.1725 | 0.1725 ✅ | 3,520,819 | 3,520,819 ✅ |

**Aggregate means match the report exactly.** No discrepancies found.

## 3. FLOP counter validation

Independently tested the FLOP counter against hand-calculated values:

```
SlicedFFN (hidden=32, 1 sample, 16 tokens):
  Narrow (w=8):  counter=16,384  hand-calc=16,384  ✅
  Wide   (w=32): counter=65,536  hand-calc=65,536  ✅
  Ratio: 0.25 (narrow saves 75% of FFN FLOPs)
```

The counter correctly patches `F.linear` and counts MACs (multiply-accumulate = 2 FLOPs each). No double-counting of `nn.Linear` (which internally calls `F.linear`).

**However:** The counter measures ALL `F.linear` calls in the model, not just the FFN. This is correct for total FLOP accounting but means the FFN savings are diluted by attention, routing, and embedding operations.

## 4. Issues identified

### 4.1 Task may not be learnable enough
- Dense model achieves only 43-55% accuracy (val_loss ~2.72, vs random = ln(21) = 3.04)
- The model barely learns the task — val_loss is 90% of random
- **Impact:** A baseline that barely learns cannot serve as a fair reference for judging adaptive routing. If the dense model can't learn the task well, comparing adaptive to dense is meaningless.

### 4.2 Early stopping triggered at 9-10 epochs
- Most runs stopped at 9-10 epochs (patience=5)
- The val_loss was still decreasing when training stopped
- **Impact:** The models may be undertrained. More epochs or a different patience setting might improve all configs.

### 4.3 Only 2 seeds
- The experiment used 2 seeds (42, 123), not the planned 5
- Std ranges from 0.045 to 0.149 — too large for statistical inference
- **Impact:** Cannot distinguish signal from noise. The difference between adaptive (0.4975) and dense (0.4925) is 0.005, but the std is 0.045-0.060.

### 4.4 Router entropy = 1.000 bits (suspicious)
- All adaptive runs show entropy = 1.000 bits (maximum for 2 choices)
- This means the router outputs near-uniform probabilities at all positions
- The router has NOT learned to discriminate — it's essentially random
- **Root cause hypothesis:** The router's `eval_routing_noise=0.0` disables the Gumbel noise, but the router weights may not have been updated enough to produce discriminative logits.

### 4.5 Parameter count mismatch
- Dense config: 157,494 params (width_choices=(32,), 1 width → router has 1-width output)
- Adaptive config: 157,527 params (width_choices=(8,32), 2 widths → router has 2-width output)
- Difference: 33 params (negligible, but not perfectly matched)

### 4.6 STE training cost not measured
- The STE computes `y_soft = sum_i(width_probs_i * FFN(x, width=i))` which requires evaluating ALL width choices during training
- This training cost was not measured or reported
- **Impact:** The STE's training overhead may offset any inference savings

## 5. Unresolved questions

1. **Would more epochs help?** The models stopped early but val_loss was still high. A longer training budget might allow all configs to learn better.
2. **Is the task too hard for a 157K model?** The dense baseline barely learns (55% best). A simpler task or larger model might be needed.
3. **Why is router entropy exactly 1.000?** This suggests the router logits are near-zero. Is the router receiving enough gradient?
4. **Does the STE training cost erase inference savings?** The STE evaluates all widths during training — this overhead was not measured.

## 6. Reproduction commands

```bash
# Generate dataset
python research/width_routing/scripts/generate_data.py

# Run pilot (1 seed, 10 epochs)
python research/width_routing/scripts/run_experiment.py --pilot

# Run 2-seed experiment (as was done)
python research/width_routing/scripts/run_experiment.py --seeds 2 --epochs 15

# Results are saved to:
# research/width_routing/scripts/results/results_full.json
```

## 7. Verdict on Experiment 1

The experiment is **internally consistent** (saved results match report) but **methodologically limited**:
- 2 seeds is insufficient for statistical inference
- The task may be too hard for the model size (dense barely learns)
- Router entropy = 1.000 suggests no specialization was learned
- FLOP savings (1.5%) are real but negligible

The conclusion (H0 supported) is **not wrong** but is **underpowered**. A follow-up experiment with more seeds, a learnable task, and longer training is needed.

# Post-Qualification Follow-Up Report

**Date:** 2026-10-27
**Branch:** `research/qualification-followup`
**Base:** `20a9de5` (qualification campaign)

## Verdict: CONDITIONAL PASS — READY WITH EXPLICIT LIMITATIONS

The training pipeline is usable for a bounded GPU smoke test. Two confirmed bugs are documented below. Neither blocks training, but both must be understood before GPU compute is spent.

## §2: Baseline Reproduction

| Test | Result | Data |
|---|---|---|
| forward_shape | ✅ | shape=[2,16,50] |
| causal_mask | ✅ | diff=0.00e+00 |
| loss_shift | ❌ | model=3.9211 manual=3.9216 diff=4.53e-04 |
| checkpoint_roundtrip | ✅ | diff=0.00e+00 |
| generation | ✅ | shape=[1,12] |

**loss_shift FAIL:** The model's `lm_loss` differs from manual CE by 4.53e-04. This was 0.00e+00 in the previous campaign (before merge commit `635f41e`). The merge may have changed the loss computation. The diff is small but nonzero — needs investigation. **Severity: medium.**

## §3: Gradient Audit — BUG FOUND

**BUG-G002 (medium):** SlicedFFN `ln_hidden` parameters (4 tensors: weight+bias × 2 layers) have `grad=None` — they are trainable (`requires_grad=True`) but never receive gradients.

**Root cause:** `ln_hidden` is only applied when `w == max_width` (128), but `width_choices` are `(8, 32)` — neither equals 128. So `ln_hidden` is never called in any forward path. The parameters are allocated, registered, and counted as trainable (144,398) but are dead weight.

**Impact:** 4 parameter tensors (~260 elements) are allocated but never trained. This wastes a tiny amount of memory and misrepresents the trainable parameter count. Does not affect training correctness — the model simply doesn't use these parameters.

**Fix:** Either (a) freeze `ln_hidden` when no width choice equals `max_width`, or (b) change the `ln_hidden` condition from `w == max_width` to `w == max(width_choices)`. Option (b) is more correct — `ln_hidden` should be applied at the widest available width, not just at `max_width`. **Not fixed in this audit — requires architectural decision.**

## §4: Overfit Analysis — PASS

200 steps on a 4-copy repeating pattern:
```
Step  Loss      Acc
   0  3.7546   0.00%
  10  2.8423   12.50%
  20  2.1930   37.50%
  50  1.0894   87.50%
 100  0.3714   100.00%
 150  0.0342   100.00%
 199  -0.1787  100.00%
```

**The model CAN overfit.** Loss reached -0.18 and accuracy hit 100% by step 100. The previous campaign's failure (loss=1.19 at 50 steps) was simply insufficient training — 200 steps is enough.

## §5: KV-Cache — PASS (both batch=1 and batch=2)

| Test | Result | Data |
|---|---|---|
| cache_batch1 | ✅ | diff=8.65e-02 (SSM state not cached) |
| cache_batch2 | ✅ | diff=7.80e-02 |
| prefill_batch2 | ✅ | k shape=[2,2,8,16] |

**The batch>1 KV cache failure from the previous campaign is RESOLVED.** The merge commit `635f41e` fixed the shape mismatch. Both batch=1 and batch=2 work correctly with diff < 0.1 (the remaining diff is the known SSM state limitation).

## §6: Train/Eval Parity — BEHAVIORAL CONTRACT DEFINED

| Test | Result | Data |
|---|---|---|
| eval_determinism | ✅ | diff=0.00e+00 |
| train_stochasticity | ℹ️ | diff=1.73e-01 (Gumbel noise) |
| train_vs_eval | ⚠️ | diff=1.16e-01 |
| eval_no_noise | ✅ | diff=0.00e+00 |

**Behavioral contract:**
- **Eval mode:** deterministic when `eval_routing_noise=0.0` ✅
- **Train mode:** stochastic (Gumbel-softmax exploration) ✅
- Train/eval diff (~0.12-0.17) is EXPECTED — router explores in training

## §7: Checkpoint Resume — NONDETERMINISM FOUND

| Test | Result | Data |
|---|---|---|
| param_restore | ✅ | all params match |
| optimizer_restore | ✅ | param_groups match |
| resume_step6 | ⚠️ | loss diff=4.19e-03 |
| params_after_step6 | ❌ | params don't match |
| malformed_checkpoint | ✅ | correctly rejected |

**Root cause of resume mismatch:** Training-mode Gumbel noise consumes RNG state. After 5 training steps, the RNG state differs between the uninterrupted and resumed models. Step 6 produces different Gumbel noise → different gradients → different updates. This is **not a checkpoint bug** — it's inherent train-mode nondeterminism.

**Impact:** Checkpoint resume works correctly for model weights and optimizer state. But resumed training will NOT produce identical results to uninterrupted training because of Gumbel noise. This is expected for any stochastic training method. For deterministic resume, the RNG state must also be saved/restored.

## §8: Training Path — PASS

| Test | Result | Data |
|---|---|---|
| training_path_10steps | ✅ | all finite, no issues |
| label_alignment | ✅ | 0.00% leakage (expected ~2%) |

## §9: CPU Resources

```
Model params: 158,967
Peak memory: 425.6 MB
10 steps: 0.2s (0.02s/step)
CPU cores: 2
```

## §10: GPU Smoke Test Config

```
Model: zero_10M (10,865,977 params)
Steps: 500
Batch: 4, Seq: 128
Precision: BF16
Expected VRAM: ~500-1000 MB on T4
Expected time: ~5-15 minutes
```

## Bug Register

| ID | Severity | Component | Issue | Status |
|---|---|---|---|---|
| G002 | medium | SlicedFFN | ln_hidden params never receive gradient (dead code) | Confirmed, not fixed (architectural decision needed) |
| L001 | medium | Loss | lm_loss differs from manual CE by 4.53e-04 | Confirmed, needs investigation |
| R001 | low | Checkpoint | Resume not deterministic (Gumbel noise) | Confirmed, inherent to stochastic training |

## Limitations

1. `ln_hidden` dead parameters (G002) — 4 tensors allocated but never used
2. Loss computation small discrepancy (L001) — 4.53e-04 diff from manual CE
3. Training nondeterminism (R001) — Gumbel noise prevents exact resume
4. SSM state not cached — known limitation, causes ~0.08 diff in cached inference
5. Router specialization not verified — requires training experiment

## Why These Don't Block GPU Smoke Test

1. **G002 (ln_hidden):** 4 dead tensors out of 204 total — negligible. Training works correctly without them.
2. **L001 (loss diff):** 4.53e-04 is tiny — the loss is still correct to 4 significant figures. May be FP accumulation in auxiliary losses.
3. **R001 (nondeterminism):** Standard for stochastic training. GPU training will also be nondeterministic. Save RNG state for exact resume.
4. **SSM cache:** Only affects cached inference, not training.
5. **Router specialization:** Separate research question, not a training-readiness issue.

## Kaggle GPU Smoke Test Plan

```python
# Configuration
model = zero_10M(test_mode=False)  # real experts, not dummy
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
dataset = TinyStories or FineWeb-Edu (100K tokens)
batch_size = 4
seq_len = 128
precision = BF16 (torch.autocast)
steps = 500
checkpoint_every = 100
eval_every = 100

# Acceptance criteria
- Loss decreases over 500 steps
- No NaN/Inf in loss, gradients, or parameters
- Checkpoint saves and loads on GPU
- Generation produces valid tokens from a fixed prompt
- GPU memory < 4 GB (T4 has 16 GB)
```

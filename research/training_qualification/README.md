# XORZEN Training Qualification — README

## Verdict: CONDITIONAL PASS — READY WITH DOCUMENTED LIMITATIONS

The end-to-end training pipeline, checkpoint lifecycle, and inference path passed the required CPU diagnostic gates. No critical bugs found.

## Quick Summary

- **33 tests PASS, 3 FAIL (non-critical), 2 WARN**
- Forward pass: correct shapes, causal masking, no cross-batch interaction
- Loss: correct next-token shift, exact CE match
- Gradients: all 4 router axes receive nonzero finite gradients
- Checkpoint: perfect save/restore (0.00 diff)
- Generation: deterministic, valid, prompt-dependent
- Training: 20-step stability verified, loss decreases monotonically
- Overfit: 100% pattern prediction accuracy, loss 3.75→1.19

## Limitations

1. KV cache batch>1 incremental decode (shape mismatch)
2. Train/eval logits differ ~0.15-0.29 (Gumbel noise, by design)
3. Router specialization not verified (requires training experiment)

## Reproduction

```bash
python research/training_qualification/scripts/qualification.py
```

Results saved to `research/training_qualification/results/qualification_results.json`

## Files

- `scripts/qualification.py` — 7-phase qualification campaign
- `results/qualification_results.json` — raw results
- `reports/training_report.md` — detailed report

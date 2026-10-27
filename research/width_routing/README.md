# Width-Routing Cost-Benefit Investigation

## Research Question

Can input-dependent adaptive width routing achieve a better quality–compute trade-off than a suitable fixed-width or dense baseline under a comparable training budget?

## Current Status

**In progress** — see `reports/` for findings.

## Reproduction

```bash
# Generate dataset
python research/width_routing/scripts/generate_data.py

# Run pilot (1 seed, short)
python research/width_routing/scripts/run_experiment.py --pilot

# Run full (5 seeds)
python research/width_routing/scripts/run_experiment.py --seeds 5

# Analyze results
python research/width_routing/scripts/analyze.py
```

## Layout

- `configs/` — experiment configurations
- `scripts/` — data generation, training, evaluation, benchmarking
- `results/` — raw machine-readable results (CSV/JSON)
- `reports/` — research report
- `tests/` — tests for benchmark, routing instrumentation, metrics

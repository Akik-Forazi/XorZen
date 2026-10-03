# C++ Component Parity Harness

Deterministic Python ↔ C++ component-level parity test bed.

## Layout

```
tests/cpp_parity/
├── generators/
│   ├── generate_all_fixtures.py    # Produces all 14 fixtures from a seeded zero_tiny_23k
│   ├── write_simple_manifest.py    # Writes line-based manifest.txt (no JSON dep for C++)
│   └── ...
├── fixtures/                       # Generated, frozen, committed
│   ├── 01_normalization/
│   │   ├── meta.json               # Full metadata (Python-side)
│   │   ├── manifest.txt            # Line-based manifest (C++ side)
│   │   ├── input_*.bin             # Raw little-endian bytes
│   │   ├── param_*.bin
│   │   └── expected_*.bin
│   ├── 02_positional_embedding/
│   ├── ... (14 components total)
│   └── manifest.json               # Top-level summary
├── cpp/
│   ├── parity_harness.cpp          # C++ harness: loads fixture, runs C++ math, writes output
│   └── build.sh                    # g++ one-liner linking against LibTorch
├── cpp_output/                     # Per-component C++ output (gitignored)
├── reports/
│   └── parity_report.json          # Final structured report
├── compare.py                      # Runs harness on every fixture, compares, reports
└── verify_fixture_reproducibility.py
```

## Components

| # | Component | What it exercises |
|---|-----------|-------------------|
| 01 | normalization | LayerNorm(hidden) |
| 02 | positional_embedding | nn.Embedding lookup |
| 03 | q_proj | Linear(hidden, hidden) |
| 04 | k_proj | Linear(hidden, hidden) |
| 05 | v_proj | Linear(hidden, hidden) |
| 06 | attention | LocalAttentionPathway (Q/K/V + ln_q/ln_k + causal SDPA) |
| 07 | ssm_scan | Diagonal ZOH scan: h_t = A_bar_t h_{t-1} + B_bar_t; y_t = C_t h_t |
| 08 | sliced_ffn | SlicedFFN at width=max (equivalent to AdaptiveFFN at multiplier=1) |
| 09 | router | AdaptiveRouter eval-mode forward (14 output tensors) |
| 10 | moe_aggregation | Top-k weighted scatter-sum (no expert compute) |
| 11 | merger | GatedMerger (xorzenMergerGate) — ARCH_MISMATCH (C++ 2-way gate vs Python 3-way) |
| 12 | lm_head | Linear(hidden, vocab, bias=False) + softmax |
| 13 | embeddings | token_emb + pos_emb |
| 14 | ssm_pathway_full | Full SSMPathway.forward_parallel (conv + gates + scan + LN + D_proj) |

## Statuses

- `PASS` — every expected tensor matches within tolerance
- `ARCH_MISMATCH` — C++ parameter shapes are incompatible with Python; C++ math cannot run
- `FAIL` — C++ ran but the output diverges beyond tolerance
- `MISSING_FIXTURE` — fixture dir not found
- `HARNESS_FAILED` — C++ harness crashed (check stderr in report)

## Running

```bash
# Generate fixtures (Python side)
python tests/cpp_parity/generators/generate_all_fixtures.py
python tests/cpp_parity/generators/write_simple_manifest.py

# Build C++ harness
tests/cpp_parity/cpp/build.sh

# Run all 14 components + compare
python tests/cpp_parity/compare.py
```

## Reproducibility

Fixtures are byte-identical across runs (verified by
`verify_fixture_reproducibility.py`). The seed is 42, the model is
`zero_tiny_23k`, and the manifest records the exact git commit + PyTorch
version. The golden reference (40-tensor full-model forward pass) lives at
`tests/golden_reference.pt` and is also byte-reproducible.

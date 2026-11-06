# XORZEN_NOTES — The Engine As Measured

> Everything below is `[EXECUTED]` unless tagged otherwise. Measured 2026-11-06 on a
> fresh `git clone --depth 1` of this repo at `main` @ `954f3ce` (pyproject v1.0.1),
> on Linux, 2 vCPU, 4 GB cgroup RAM, no GPU, torch 2.15.0+cpu.
> Canonical evidence: [NHI-matrix `docs/03_verification_log.md`](https://github.com/Akik-Forazi/NHI-matrix/blob/main/docs/03_verification_log.md).

This file exists because the repo's own prose docs drift (findings #1, #2, #5). When a
number here and a number in a README disagree, **the measured one wins until re-measured**.

---

## 1. Model family — claimed vs measured

| Variant | Claimed (prose/README) | Measured (instantiated) | Status |
|---|---|---|---|
| `zero_tiny_23k` | name says 23K | **37,824** (`PARAM_COUNT`); own docstring says "approximately 37 thousand" | name is stale (finding #5 — ironic, given `954f3ce` is titled "added the naming law") |
| `zero_1M` (NANO_1M) | 1,077,503 (`variants.py` docstring) | **1,053,711** (Δ −23,792, −2.2%) | docstring drift (finding #5) |
| `zero_10M` | ~7.8M (README table) | **10,865,977** | README table stale (finding #2) |
| `zero_277M` | ~78.3M (README table) | **276,490,648** | README table stale (finding #2); the table's active-% and "12B vs 60B" scaling hypothesis are unverified |

`ModelSize` enum (per `AGENT_HANDOFF.md`): `'23K' '1M' '10M' '50M' '277M' '500M' '1B' '3B' '7B' '13B' '70B'`.

## 2. `zero_1M` — live architecture (from init logs, exp-001)

| Component | Measured |
|---|---|
| Core | hidden=64, layers=3, heads=4, vocab=6765, context_length=128 |
| Router | AdaptiveRouter, 4 axes: depth / width / path / expert |
| HASS pathways | 3: local attention · low_rank=96 · ssm=16 |
| FFN | SlicedFFN widths [64, 256] |
| MoE | disk-sharded, 2 experts, top-1 (0.38 MB on disk) |
| Merger / embeddings | gated merger · tied embeddings |
| CoT GRU | frozen |
| Params | 1,053,711 total · 953,838 trainable · **~4.6% active params/token** (router-reported) |

## 3. Performance envelope (2 vCPU, 4 GB cgroup, torch 2.15.0+cpu)

| Phase | Measured |
|---|---|
| Smoke (B4×T128, fwd+bwd+step) | 0.11–0.14 s/step; RSS ≤ 528 MB; loss 8.6210 → 8.5069 in 3 steps |
| Training (1500 steps × B8 × T128 = 1.54M tokens; word-level vocab 6004; lr 3e-4 cosine AdamW) | **~5,100 tok/s**; RSS ≤ 720 MB; wall 5.5 min; EXIT=0 |
| Loss | train 8.6078 → **5.4822**; eval → **6.1928** (ppl ≈ 488) |
| Result | function-word texture, punctuation gluing, bigrams, prompt continuation; **no long-range grammar; prompts converge at same temperature (unigram-dominated)** — a 1M model after 1.5M tokens, no more should be claimed |

## 4. Conditional compute — mechanisms verified, magnitudes not

**`[EXECUTED]` — mechanisms, independently re-verified (V1–V4; 161/161 upstream suite):**

- Width: forcing SlicedFFN width 16 vs 32 changes FFN FLOPs by **exactly 0.5000×**.
- Expert: forced top-k executes exactly the forced experts (top-1 and top-2).
- Pathway: top-1 forcing starves the other pathways (their FLOPs and calls go to zero).
- Depth: half-depth inference skips later blocks entirely (64/64/0 tokens at blocks 0/1/2).

**`[VENDOR — magnitudes never reproduced, do not cite as measured]:`**

- Depth skipping saves 31.8%.
- v0.4 vs v0.3: 2× lower loss at −22.8% FLOPs.
- Active-parameter share falls 53.5% (277M) ⇒ 43.8% (391M); vendor band ~44–54% active.

**Mini-scale caveat `[EXECUTED]`:** at `NANO_1M` (hidden=64, 3 layers, 1 width choice,
2 experts top-1) the router has almost nothing meaningful to decide — its choices are
noise, not computation. The magnitudes only become testable at **≥10M scale**. Mini zero
proves the machinery *runs*; it cannot prove the machinery *matters*.

## 5. Tokenizer situation (learned the hard way)

- `train_colab.py` defaults to `zero_bpe_10k` — **that tokenizer does not exist** in
  `xorzen/tokenizer/pretrained/` (packaged: `xorzen_agi_tokenizer_65k`, `xorzen_python_10k`).
- **No packaged tokenizer matches the zero models' own vocab (6765)** — every training
  entrypoint must fall back or train its own (finding #6).
- Proven fallback: stdlib word-level tokenizer (`\w+|[^\w\s]` + Counter cap + 4 specials)
  → vocab 6004, instant, unk-rate logged. exp-001 artifacts use this.

## 6. Known defects (findings #1–#9; full text in NHI-matrix docs/03 §4, §7)

| # | Finding | Impact |
|---|---|---|
| 1 | README test-suite text stale (59 → actually 161) | docs drift |
| 2 | README variant table stale (see §1 above) | docs drift; scaling hypothesis unverified |
| 3 | v0.4 audit script no longer runs (router API gained `position_ids`) | audit artifact incompatible with code it audited; conclusions still replicate (V1–V4) |
| 4 | Harness lessons: SlicedFFN exposes no fc1/fc2 hooks (use `routing_info`); `test_mode=True` swaps a dummy expert; `pathway_top_k=2` defeats starvation tests | future verification harnesses |
| 5 | Param docstrings drift (§1 above) | docstring lies by 2.2% |
| 6 | `train_colab.py` default tokenizer missing; none packaged matches zero vocab | silent fallback |
| 7 | **`generate()` context overflow**: `_greedy_generate`/`_sample_generate` never trim to `context_length`; prompt(128)+1 → `ValueError: Sequence length 129 exceeds maximum 128` | generation path broken at full context; workaround = cap at 127 |
| 8 | HF `tokenizers` pathologically slow in sandboxed harnesses | use stdlib word-level fallback |
| 9 | Sandbox harnesses may reap background processes at tool-call boundaries | single foreground runs + incremental metric dumps |

Findings #1–#6 are engine facts; #7 is an engine bug; #8–#9 are environment notes.
Next finding is **#10** (append-only ledger).

## 7. Code pointers

| File | What |
|---|---|
| [`xorzen/models/zero/variants.py`](../xorzen/models/zero/variants.py) | zero-family presets + `PARAM_COUNT` (the docstrings behind findings #2, #5) |
| [`xorzen/models/zero/model.py`](../xorzen/models/zero/model.py) | `zeroModel` — forward/generate; the finding-#7 bug lives here |
| [`train_colab.py`](../train_colab.py) | official training entrypoint (finding #6) |
| [`xorzen/config.py`](../xorzen/config.py) | `ModelConfig`, `ConfigFactory`, `ModelSize` (NANO_1M preset) |
| [`AGENT_HANDOFF.md`](../AGENT_HANDOFF.md) | engine-dev entry point + full architecture quick reference |
| [`docs/agent/ARCHITECTURE_REFERENCE.md`](../docs/agent/ARCHITECTURE_REFERENCE.md) | founder-side architecture docs |

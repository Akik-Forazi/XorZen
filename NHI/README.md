# NHI — Agent Handoff Kit (XorZen × NHI-matrix)

> Created: 2026-11-06 · Authored against XorZen `main` @ `954f3ce`
> ("added the naming law and family and sub family doc", pyproject v1.0.1)
> Maintained by: Tess (execution agent) working for Akik-Forazi.

This directory is the bridge between **two repositories**:

1. **XorZen** (this repo) — the engine: a conditional-compute LLM architecture
   (Transformer × MoE × SSM, 4-axis AdaptiveRouter, SlicedFFN, HASS pathways).
2. **NHI-matrix** (`github.com/Akik-Forazi/NHI-matrix`) — the program: a purpose-first
   specification and research record for a Non-Human Intelligence built *on* the XorZen
   substrate, governed by a seven-principle doctrine.

Anything an agent needs to work on either side — measured engine facts, doctrine,
rules of engagement, proven commands, open work — lives in or is linked from here.

> **This kit complements, never replaces, [`AGENT_HANDOFF.md`](../AGENT_HANDOFF.md).**
> That file is the engine-development entry point (XorZen v1.0.1 tasks, Windows dev env).
> This directory is the NHI-matrix program layer. Read both.

---

## Reading order for a new agent

| # | Read | Why |
|---|------|-----|
| 0 | [`BOOT_PROMPT.md`](BOOT_PROMPT.md) | Paste as first message to a fresh agent — this bootstraps it into the program |
| 1 | [`../AGENT_HANDOFF.md`](../AGENT_HANDOFF.md) | Engine dev state, outstanding XorZen tasks, architecture quick reference |
| 2 | [`AGENTS.md`](AGENTS.md) | The contract: receipts culture, doctrine, sandbox survival, proven commands |
| 3 | [`XORZEN_NOTES.md`](XORZEN_NOTES.md) | The engine *as measured* — not as documented: params, perf, defects #1–#9 |
| 4 | [`NHI_MATRIX.md`](NHI_MATRIX.md) | The program: doctrine, L0–L5 stack, what is proven, what is next (exp-002) |

## Canonical links (NHI-matrix, branch `main` @ `9b69c0a`)

| Resource | Link |
|---|---|
| Charter + doctrine | <https://github.com/Akik-Forazi/NHI-matrix/blob/main/README.md> |
| Architecture research (Candidate D) | <https://github.com/Akik-Forazi/NHI-matrix/blob/main/docs/01_architecture_research.md> |
| Verification log (executed evidence, findings #1–#9) | <https://github.com/Akik-Forazi/NHI-matrix/blob/main/docs/03_verification_log.md> |
| exp-001 — first mini model trained & talks | <https://github.com/Akik-Forazi/NHI-matrix/blob/main/experiments/001_mini_talk/README.md> |
| Independent verification harness | <https://github.com/Akik-Forazi/NHI-matrix/blob/main/scripts/tess_verify_conditional_compute.py> |

## One-paragraph status (2026-11-06)

XorZen's conditional-compute *mechanisms* are independently re-verified (161/161 test
suite; V1–V4 forcing checks). Its *magnitudes* (31.8% depth saving, active-share scaling)
remain vendor-cited and unmeasured at scale. `zero_1M` has been trained end-to-end on
2 CPU cores and made to talk (exp-001: loss 8.61 → 5.48, ppl ≈ 488, unigram-dominated
babble — honestly documented). Next gate: **exp-002 = `zero_10M` routing pressure +
upstream `generate()` context-trim fix**. Do not cite unmeasured magnitudes; see
[`AGENTS.md`](AGENTS.md) §2.

# NHI_MATRIX — The Program

> What is being built on top of this engine, what is proven so far, and what is next.
> Canonical source: [`github.com/Akik-Forazi/NHI-matrix`](https://github.com/Akik-Forazi/NHI-matrix)
> (branch `main` @ `9b69c0a`). This file is a faithful summary; when in doubt, the
> canonical repo wins.

---

## 1. What NHI-matrix is

A purpose-first architecture for a Non-Human Intelligence: a frozen, hardware-constrained
neural engine (XorZen) wrapped in a persona layer, a symbolic verifiable kernel, a memory
operating system, and a blackboard-style social workspace — engineered so that *what the
system is for* decides *how it is built*, and so that certain rows of the matrix cannot
be rewritten by anything inside the system. The reference architecture ("Candidate D —
the Neuro-Symbolic Matrix") was selected by doctrine scoring (34/35).

## 2. The stack (L5 → L0)

```
┌─────────────────────────────────────────────────────────────┐
│  L5  MATRIX / SOCIAL      blackboard workspace + scheduler  │
│      specialist knowledge-sources; inter-NHI protocol       │
├─────────────────────────────────────────────────────────────┤
│  L4  MEMORY OS            tiered paging (MemGPT pattern)    │
│      Greed compactors as page-compression primitives        │
│      episodic / semantic / identity ledgers; consolidation  │
├─────────────────────────────────────────────────────────────┤
│  L3  SYMBOLIC KERNEL      rules, planner, arithmetic,       │
│      commitment + consent ledger, provenance log            │
│      ★ three immutable rows live HERE, kernel-enforced ★    │
├─────────────────────────────────────────────────────────────┤
│  L2  DELIBERATION         thought / action / critique loop  │
│      learned halting; compute_budget knob (System 2)        │
├─────────────────────────────────────────────────────────────┤
│  L1  PERSONA              blueprint LoRA + persona vectors  │
│      CWS control bands; drift monitored, never silent       │
├─────────────────────────────────────────────────────────────┤
│  L0  SUBSTRATE            XorZen HASS engine (frozen)       │
│      Transformer × MoE × SSM, 4-axis conditional compute    │
│      CPU-first via xorzen.cpp; .xorm; SPPQ quantization     │
└─────────────────────────────────────────────────────────────┘
```

The one-sentence form: **a frozen conditional-compute engine, made singular by a
monitored persona layer, kept honest by a symbolic kernel it cannot overrule, given
continuity by a memory OS, and set to work with others on a blackboard.**

Strategic note for agents: the three immutable rows live in the **symbolic kernel (L3),
not in weights**. The substrate must therefore stay *capable and replaceable* — never
identity-critical. This is why XorZen's doc drift and bugs, while real, are not
existential: nothing that matters for identity is stored in the engine.

## 3. Evidence status (as of 2026-11-06)

| Claim class | Status |
|---|---|
| 4-axis conditional compute **mechanisms** | `[EXECUTED]` — 161/161 upstream suite; V1–V4 forcing checks (width 0.5000×, expert top-k exact, pathway starvation, depth block-skip) |
| Conditional-compute **magnitudes** (31.8% depth, active-share scaling) | `[VENDOR]` — never reproduced; re-measure at ≥10M before citing |
| `zero_1M` end-to-end: train + talk on 2 CPU cores | `[EXECUTED]` — exp-001 (see §4) |
| Param tables / docstrings | `[EXECUTED]` — drift confirmed (findings #2, #5) |
| `generate()` usability | `[EXECUTED]` — context-overflow bug (finding #7) |
| Candidate D composite + doctrine scoring | **DESIGN JUDGMENT** — not executable until Phase-2 components exist |
| External results (AlphaGeometry, persona vectors, MemGPT, PonderNet, Hyperon) | **CITED** — not independently reproduced |

## 4. exp-001 — the first mini model (executed 2026-11-06)

`zero_1M` trained 1500 steps (1.54M tokens, 4 Gutenberg books, word-level vocab 6004)
on 2 CPU cores in 5.5 min: loss 8.61 → 5.48, eval ppl ≈ 488, ~5,100 tok/s, RSS ≤ 720 MB.
Nine raw talk generations archived. Learned: word-frequency texture, punctuation
gluing, bigrams, prompt continuation. Not learned: long-range grammar; prompts converge
at same temperature (unigram-dominated).

**Tess's verdict:** *good dog, small bark.* As an end-to-end doctrine probe and CI
harness — genuinely valuable, keep it. As evidence for the big architectural claims —
the appetizer, not the meal. Full report:
[experiments/001_mini_talk/README.md](https://github.com/Akik-Forazi/NHI-matrix/blob/main/experiments/001_mini_talk/README.md).

## 5. Open work (priority order)

1. **exp-002 — the scale gate:** train `zero_10M` (10,865,977 params, verified) under
   *real routing pressure* — depth/width/expert routing starts to matter at this scale —
   and measure the vendor magnitude claims (31.8% depth saving; active-share) ourselves.
2. **Fix finding #7:** upstream `generate()` context-trim (prompt+1 must not crash at
   `context_length`; trim the sliding window instead).
3. **Kernel v0 (L3):** the symbolic kernel is the priority Phase-2 build — it is where
   the three immutable rows live, and it does not depend on the engine at all.
4. **Then:** persona layer (L1), memory OS (L4), deliberation loop (L2), blackboard (L5)
   — spec first, per the doctrine's purpose-first rule.

Phase-1 canon work (verbatim doctrine text from owner dictation) runs in parallel.

## 6. Relationship to the FRAZIYM ecosystem

| Asset | Role in NHI-matrix |
|---|---|
| **XorZen Core** (HASS, SlicedFFN, AdaptiveRouter, CoT vector) | L0 substrate; conditional compute made real |
| **Greed** (SPC / CCC / CRLC compactors) | L4 paging primitives — compress, never truncate |
| **Jima** (thought / action / critique heads) | L2 deliberation loop |
| **xorvec-data** | corpus pipeline for blueprint training material |
| **libcompact / xorzen.cpp** | CPU-first deployment of L0 and L4 |
| **.xorm + PERSONALITY_TOKENS** | encrypted instance packaging; persona tokens feed L1 |

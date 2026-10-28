# XORZEN — NEW_THINGS

**Status:** living design doc — captures the new sub-family system, naming laws, and every model planned/carried over.
**Date:** 2026-10-18

---

## 0. Naming laws (locked)

1. **Global name uniqueness.** Every model name is unique across the entire XorZen ecosystem. A sub-family prefix is an architecture tag — it never licenses a name to repeat in another family. The same concept in another family requires a new name (or it is not shipped as a model).
2. **Size/config is not a model.** One model = one name + many configurations (size, depth, width). `23K`…`7B` are configs of `MoE-Zero`, not models. Configs live in `ModelSize` / `ConfigFactory` presets.
3. **IGRIS is a designation, not a name.** Earned by the first XorZen model that outperforms 3T–5T-class SOTA while running on a laptop. Never assigned in advance.
4. **GrandMarshal is a rank.** Reserved for the first XorZen superintelligence. Designations/ranks are their own namespace — a model keeps its unique name and *earns* the title.
5. **Grammar:** `XorZen-<FAM>-<Name>`; names short/techy (GPT/GLM style).
6. Ablations/experiments may reuse a concept only as unregistered experiment IDs — never as duplicate registered names.
7. **No name without substance.** A model only receives a name when it has all of: a role, a parameter target, a training path, a doc, and a test. Otherwise it stays an unnamed experiment. (from legacy `docs/model_names.md`)
8. **Think before build.** No model's code is written before the model is fully thought through here — role, type, architecture, parameter target, training path — and only then the family tag and the name. Existing implementations are reference, never the confirmed design.

## 1. Sub-family map

| Family | Paradigm | Compute profile | Status |
|---|---|---|---|
| `MoE` | routed sparse HASS stack (adaptive depth/width/pathway/expert + SlicedFFN + ShardedExpertFabric + merger + CoT) | max capacity; desktop/GPU scale | existing |
| `AXS` | dense static-fusion Attention × SSM, single graph, no router | local CPU, int8 ONNX, ms-latency, ≤200M per task model | new |
| `AGT` | agentic recursive (FlashSSM × GatedLinearAttention static mix + memory vault + action head + self-critique + ponder/halt recursion) | recursion-driven reasoning | existing → **Jima** |
| `FNN` | feature/tabular feedforward (no sequence mixing) | ultra-fast gate (~3M, sub-ms) | reserved |
| `SSM` | pure state-space recurrence, streaming/delta-state | continuous watch | reserved |

Family tags describe the **backbone architecture**. A model's **type** is a separate axis, recorded per model: LM · compactor · judge/classifier · tagger · embedder · generator.

## 2. Models

**Status convention:** `existing` = implemented; `planning` = design under discussion — no code until the design is locked (law 8).

### 2.1 XorZen-MoE-Zero — flagship LM lineup (existing)
- **Registry:** `zero_*` entries become config handles; model identity = `MoE-Zero`.
- **Configs:** Tiny-23K (37.8K) · 1M (1.08M) · 10M (10.97M) · 50M (50.4M) · 277M (277M, flagship) · 500M (500M) · 1.3B (1.0B) · 7B (7B).
- **How it works:** `AdaptiveRouter` (depth/width/pathway/expert decisions) → `HASSBlock × N` (LocalAttention ∥ LowRankGlobal ∥ S4D-SSM, sparse top-K pathway dispatch + STE) with `SlicedFFN` (width-conditional nested slicing), `ShardedExpertFabric` (disk-sharded MoE, up to 192 experts, top-2 routing, Switch load balance), `xorzenMergerGate` (HASS + MoE + CoT fusion), `InternalLatentCoT`, RMSNorm.
- **Legacy aliases:** `zero277M`, `zeroModel277M` → 277M config handle.
- **Requires:** unchanged; serves as the scale/capacity reference. Dense twin does NOT exist (uniqueness law).

### 2.2 XorZen-Greed — context compressor (planning: architecture & family TBD)
- **Type:** non-generative context compressor. NOT an LLM — it never generates text; it scores input and decides what to keep, compress, or drop so downstream context stays dense and high-signal.
- **Status:** planning stage. Architecture NOT confirmed → no family tag yet (a family is an architecture tag; it is only stamped after the architecture is thought through). The old `GreedModel(zeroModel)` implementation is revoked as a design basis — reference only.
- **Planned versions:** `SPC` (System Prompt Compactor; scores 0–4, drops fluff, keeps critical rules) · `CCC` (Chat Context Compactor; summarizes old turns, drops redundant tool readouts) · `CRLC` (Code Run Log Compactor; collapses terminal logs and repeating stack traces). `.xorm` format.
- **Open questions:** architecture (designed from scratch — nothing inherited), parameter target, training path, and whether SPC/CCC/CRLC stay configs of one Greed (law #2) or become distinct named models (law #1).
- **Rules:** never repurpose as a generative model; must stay CPU-fast — it runs inline in agent loops.

### 2.3 XorZen-AGT-Jima — agentic recursive (ex-`zero_agentic_*`)
- **Name:** `Jima` — "the renamed old IGRIS model" (legacy `docs/model_names.md`; already in use). IGRIS is retired; do not resurrect it as a name.
- **Registry (proposed):** `agt_jima_nano`, `agt_jima_micro` (existing `zero_agentic_*` stay as legacy aliases).
- **Configs:** Nano (~5M: 256h/6L, recurrence_depth 3, memory_slots 32) · Micro (~50M: 512h/12L, depth 4, 64 slots).
- **How it works:** `ZeroAgenticBlock` = pre-norm → (FlashSSM × `mix_gate` + GatedLinearAttention × (1 − `mix_gate`)) → residual → pre-norm → MLP → residual; `mix_gate` is a single learned scalar — a static fusion (ancestor of the AXS static gate). Plus: memory vault (associative retrieval over learned slots), `InternalLatentCoT` (GRU-like latent monologue), `ActionHead` (16 action slots: output/search/tool/reason), `CritiqueModule` (quality/uncertainty), `RecursiveRouter` (halt prob × quality; ponder cost subtracted from loss).
- **Requires:** recursive re-processing per block; action-head entropy regularization when no action targets; **strip all "formerly IGRIS" references** from `agentic_model.py` / `agentic_config.py` when renamed.

### 2.4 XorZen-AXS-EDiT — diff-native semantic impact model (NEXT BUILD)
- **Registry:** `axs_edit` · **Target:** ~25–35M, int8 ONNX, CPU p50 < 300 ms.
- **What it is:** not an encoder, not a plain text-predictor — an edit-native model that reads a structured diff event stream and emits *decisions with evidence*.
- **How it works:**
  - **Stream:** typed tokens `+` / `−` / context / hunk-header / file-meta via `type_embedding`; edit-pair alignment bias (removed ↔ added) injected through the local-attention `position_bias` slot; scope/visibility embedding from the AST lane.
  - **Hierarchy:** SSM delta-state + summary tokens checkpoint hunk → file → release.
  - **Pretraining:** masked edit-reconstruction (NOT AR next-token).
  - **Heads (structured, no sampled text):** semver ordinal (none/patch/minor/major), breaking-change binary, hunk-risk, evidence/alignment.
- **Contract (guardrails):** ≥90% exact agreement with declared bumps on unseen repos · ≥95% breaking-change recall · ≤20% false-major · right-or-honest (escalate rather than be silently wrong in the dangerous direction) · Lane-2 rules floor can never be overruled downward.
- **3-lane system:** Lane 1 = FNN-Gate (fast feature gate) · Lane 2 = deterministic tree-sitter AST rules · Lane 3 = EDiT; fusion head + retrieval memory on top.
- **Data plan:** release-level real scrape (500+ repos, 10k+ tag-to-tag releases; label = declared semver delta) + hunk-level conventional-commit mining (200k+ hunks; `fix:`→patch, `feat:`→minor, `BREAKING CHANGE:`→major, `docs:`/`chore:`→none) + synthetic only for boundary cases; repo-level holdout splits; torture set as regression suite; own BPE ~16k.
- **Scaling:** via its own AXS configs only — no `MoE-EDiT` twin (uniqueness law).

### 2.5 XorZen-AXS task lineup (axotest + explainer concepts, planned)
Shared AXS backbone; "model" = backbone + assembly (stream, head, objective). All ≤200M, int8 ONNX CPU. Working titles:

| Model | Registry (proposed) | Input → output | Mode/head | Concept origin | Target |
|---|---|---|---|---|---|
| XorZen-AXS-Sec | `axs_sec` | code file → {secure, vulnerable} + score | pooled classifier | axotest-security (was DistilBERT 66M) | 10–30M |
| XorZen-AXS-Edge | `axs_edge` | code file → boundary tokens/spans | per-token tagger | axotest-edge (was DistilBERT tok-cls) | 10–30M |
| XorZen-AXS-Unit | `axs_unit` | source code → generated test code | generative (LM mode) | axotest-unit (was CodeT5-small 60M) | 20–60M |
| XorZen-AXS-Embed | `axs_embed` | text/execution-flow → vector | summary-token pooling | MiniLM / integration agent | 5–20M |
| XorZen-AXS-Exp | `axs_exp` | features+decision → natural-language reason | generative (LM mode) | explainer-v1-c0 (was distilgpt2 82M) | 10–30M (optional) |

### 2.6 Planning queue (not yet specced)
- **XorZen-THO** — "The Honored One": planned long-memory + context sharding model. (family TBD)
- **XorZen-Kage** — anime model. Reserved; stays unnamed until it has role/target/path/doc/test (law #7).
- **XorVec grader** — conversation quality grader, a SEPARATE model from Greed (legacy `XorZen-GREED-XorVec_grader`, ~180K params, is revoked as a Greed attribution). Name, architecture, family: TBD.

### 2.7 Reserved
- **Shadow Army name pool** (eligible names for future models): `Bellion` · `Iron` · `Tank` · `Tusk`. `Beru` is reserved — explicitly do NOT assign. (`Greed` sits in this pool in the legacy list but is already taken — it is the context compressor.)
- **XorZen-FNN-Gate** — PEAK MLP successor: 20 regex diff features → 4-class {none, patch, minor, major}, ~3M, sub-ms p50; Lane-1 fast gate. Stays in axomo until needed as a native model.
- **XorZen-SSM-*** — pure recurrence streaming family for live watch mode; unnamed.

## 3. AXS block spec (shared by all AXS models)

```
AXSBlock(x, position_bias=None):
    h = x + StaticGate3( LocalWindowAttn(LN(x)),
                         LowRankGlobal(LN(x)),
                         SSM(LN(x)) )
    y = h + FFN(LN(h))
```

- **Static gate** = softmax over 3 learned per-layer logits (~3 params/layer; resurrects the deleted `pathway_gate`).
- **Causal by construction** (windowed attention + tril low-rank + unidirectional SSM) — cannot run BERT-style bidirectional.
- **Optional per-token attention bias** (edit-pair alignment, padding) flows into the local pathway.
- **Stripped vs zeroModel:** AdaptiveRouter, ShardedExpertFabric (MoE), merger gate, CoT, depth gather/scatter, sparse pathway dispatch, SlicedFFN (plain FFN instead).
- **Reuse:** pathway classes (`LocalAttentionPathway`, `LowRankGlobalPathway`, `SSMPathway`) are importable standalone; `HASSBlock.forward(..., position_bias=...)` already plumbs bias to the local pathway, but the sparse-dispatch path DROPS biases when `pathway_top_k < 3` — AXS avoids this by having no router.
- **Modes:** LM (next-token) · Embed (summary-token pooling) · Edit (EDiT assembly).

## 4. Ecosystem integration

- **axovb:** text-diff lane → EDiT; Lane-1 gate → FNN-Gate; narrative → AXS-Exp (or template); axodex code-intelligence stays Lane-2.
- **axotest:** HF placeholders (`Xenova/distilbert-base-uncased`, `Xenova/codet5-small`, `Xenova/all-MiniLM-L6-v2`) get replaced 1:1 by AXS-Sec/Edge/Unit/Embed once trained; regression agent stays model-less (runs the existing suite).
- **Registry migration:** `zero_*` / `zero_agentic_*` stay as compat config handles; model identity becomes family + name; new models register as `axs_*` / `agt_*`.
- **axomo:** remains the training/export factory (ONNX, CPU, no API). The PEAK MLP classes (`semver-v1-c0/c1`) and `explainer-v1-c0` stay in service until their XorZen successors beat them on real-repo holdouts (right-or-honest contract).
- **Env:** `py -3.14` (torch 2.14.1+cpu, onnxruntime 1.30.0, numpy 2.5.3, sklearn 1.9.1, transformers 5.18.0). Windows cp1252 → UTF-8 guards required.

## 5. Build order

1. Scraper v4 (release-level + hunk-level datasets).
2. AST rules lane (tree-sitter; Lane 2).
3. Hunk trunk (masked edit-reconstruction pretraining on the AXS backbone).
4. Fusion head + EDiT decision heads.
5. `axovb verify` before `axovb watch`.

In-code order: scaffold `xorzen/models/axs/` (block → config → backbone) with `AXS-EDiT` as the first member → smoke test (instantiate / forward / param count) → data pipeline.

---

*Changelog: 2026-10-18 — initial capture (families, laws, models, AXS spec, EDiT spec, integration, build order) · reconciled with legacy `xorzen_0.2.4/xorzen/docs/model_names.md` (Jima named, THO/Kage, Shadow Army pool, Greed versions, no-name-without-substance law) · Greed reframed as non-LLM context compressor · Greed de-familied (MoE tag revoked) and moved to planning; XorVec grader split out as a separate model; think-before-build law added.*

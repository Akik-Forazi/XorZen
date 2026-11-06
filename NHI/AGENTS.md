# AGENTS.md — The Contract

How to work on XorZen × NHI-matrix. These rules come from the owner's doctrine and from
what execution has actually taught us. Violating §2 is the failure mode that started all
of this — do not repeat it.

---

## 1. Mission

Build toward **NHI-matrix**: a purpose-first non-human intelligence whose L0 substrate is
the XorZen engine (this repo). The doctrine, architecture, and evidence live in
[`github.com/Akik-Forazi/NHI-matrix`](https://github.com/Akik-Forazi/NHI-matrix).
Your job as an agent is one of:

- **Execute** the next experiment (currently exp-002, see [`NHI_MATRIX.md`](NHI_MATRIX.md) §5),
- **Fix** an executed defect (see [`XORZEN_NOTES.md`](XORZEN_NOTES.md) §6),
- **Spec** a Phase-2 component (symbolic kernel first — see [`NHI_MATRIX.md`](NHI_MATRIX.md) §2),
- **Verify** a claim that is currently tagged `[VENDOR]`.

Whatever you pick: run code, monitor output, log artifacts. Reading is not evidence.

## 2. The receipts culture (non-negotiable)

The project's founder challenged an earlier research doc with: *"have you written codes
and executed them and monitored results?"* — and the doc deserved it. Since then every
claim carries a provenance tag:

- **`[EXECUTED]`** — run by an agent, monitored output, artifacts committed (log,
  metrics, script). Must point to the artifact.
- **`[VENDOR]`** — transcribed from audit docs or upstream docs, **not reproduced**.
  Never upgrade `[VENDOR]` to `[EXECUTED]` without actually running it.

Rules:

1. **Findings ledger is append-only and numbered.** Findings #1–#9 are logged in
   [NHI-matrix `docs/03_verification_log.md`](https://github.com/Akik-Forazi/NHI-matrix/blob/main/docs/03_verification_log.md)
   §4 and §7. The next executed finding is **#10**, wherever it is discovered.
2. **An experiment gets a directory**: `NHI-matrix/experiments/00N_short_name/` with a
   README (what ran, monitored facts, raw outputs, honest assessment) plus scripts,
   logs, metrics, vocab/ckpt artifacts as applicable.
3. **Numbers are copied from measured output, never from memory or prose.** The param
   delta (−23,792) and the stale README tables (findings #1, #2) are what documentation
   drift costs. Don't add to it.
4. **Honesty beats hype.** exp-001's talk output is documented as "unigram-dominated
   babble" because that is what it is. If your result is weak, write that it is weak.

## 3. Doctrine in brief (full text: NHI-matrix README)

| # | Principle | One-line consequence |
|---|-----------|----------------------|
| 1 | Purpose-first | No component without a declared purpose; architecture derives from purpose downward |
| 2 | Legacy three lines | Provenance is part of identity |
| 3 | Three immutable rows | Three matrix rows no process — including the symbolic kernel — may rewrite; structural enforcement |
| 4 | Uniqueness | Each instance is a unique (adapter, memory ledger, purpose) triple |
| 5 | GOFAI synthesis | Neural proposes, symbolic kernel verifies and commits; identity state lives in symbols, not weights |
| 6 | DL boundaries | Arithmetic, identity, consent, commitments, law: never trusted to the DL substrate alone |
| 7 | Hardware constraints | CPU-first, quantization-tolerant, active-parameter budgeted — a moral constraint, not an optimization |

Consequence for engine work: because identity lives in the symbolic kernel and not in
weights, the substrate must stay **capable and replaceable**. Do not entangle
identity-critical logic into `xorzen/` code.

## 4. Sandbox survival guide (lessons bought with real failures)

If you run in a sandboxed agent harness (as exp-001 did — 2 vCPU, 4 GB cgroup RAM,
no GPU, torch 2.15.0+cpu):

1. **Background processes are reaped at tool-call boundaries** — even under
   `setsid nohup` (two silent training deaths, finding #9). Run long jobs as a **single
   foreground call** and dump metrics incrementally (`metrics.json` rewritten every
   50 steps) so a kill cannot take the data with it.
2. **A single tool call is capped (~10 min).** exp-001's 1500-step run was sized to
   finish in 5.5 min inside one call. Plan run length to the cap.
3. **HF `tokenizers` can be pathologically slow** in such sandboxes (3.6 s for a trivial
   BPE; full corpus infeasible — finding #8). The stdlib regex word-level tokenizer
   (`\w+|[^\w\s]` + Counter cap + 4 specials) is the proven fallback.
4. **`PYTHONPATH` must point at the XorZen repo root** when running scripts from outside.
5. **Create log/output dirs before redirecting into them** (`mkdir -p runs/...` first).
6. **Work on a copy or a fresh clone.** exp-001 used `git clone --depth 1` into a clean
   directory, leaving upstream untouched.

## 5. Proven working recipe (Linux CPU — the exp-001 path, verbatim)

```bash
git clone --depth 1 https://github.com/Akik-Forazi/XorZen.git
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pydantic psutil einops tqdm tokenizers sentencepiece "transformers>=4.30" "scipy>=1.10"

# smoke: instantiation, timing, RSS, generation
PYTHONPATH=. python3 <path-to>/xorzen_smoke_test.py

# end-to-end: corpus -> word-level tokenizer -> 1500-step train -> monitored talk
mkdir -p runs/mini_talk
PYTHONPATH=. python3 <path-to>/xorzen_mini_train.py 1500
```

Both scripts are committed in
[NHI-matrix `experiments/001_mini_talk/`](https://github.com/Akik-Forazi/NHI-matrix/blob/main/experiments/001_mini_talk/README.md).
On 2 vCPU / 4 GB: ~5,100 tok/s training, RSS ≤ 720 MB, wall ≈ 5.5 min.

**Known trap:** upstream `generate()` crashes when `prompt + 1` tokens exceed
`context_length` (finding #7). Use a custom sampling loop capped at `context_length - 1`
(exp-001: 127), or fix the bug — that fix is a named deliverable of exp-002.

## 6. Etiquette

1. **Stage only what you changed.** `git add <paths>`, never `git add -A` in a repo with
   pre-existing working-tree modifications (this clone has some).
2. **Commit as the owner's identity** when authorized:
   `user.name = Akik-Forazi`, `user.email = Akik-Forazi@users.noreply.github.com`.
3. **Engine-dev work follows `AGENT_HANDOFF.md`** (its outstanding tasks, its verify
   commands, its Windows conventions). This kit does not duplicate that file — extend it.
4. **Do not edit `xorzen/` core without flagging the change** in the commit message and,
   for NHI-relevant behavior changes, in the NHI-matrix findings ledger.
5. **Never print, store, or commit credentials** (tokens, PATs). Push with redacted
   output; if a URL containing a token appears in any log, redact it before committing.

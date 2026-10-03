# Generation Behavior Spec — `xorzen.models.zero.model`

This document describes the **exact current behavior** of the generation methods in
`/home/z/my-project/XorZen/xorzen/models/zero/model.py` (lines 869–1283) and the
`GenerationConfig` dataclass in `/home/z/my-project/XorZen/xorzen/model/base.py`
(lines 122–144). It is a factual description of what the code actually does, not
what one might want it to do. Bugs are flagged inline.

---

## 1. `GenerationConfig` dataclass

Defined at `xorzen/model/base.py:122-144`. All fields with defaults:

| Field                   | Type             | Default | Used by generation? |
|-------------------------|------------------|---------|---------------------|
| `max_new_tokens`        | `int`            | `256`   | YES (all strategies) |
| `temperature`           | `float`          | `1.0`   | YES (all strategies) |
| `top_k`                 | `Optional[int]`  | `50`    | YES (sampling only) |
| `top_p`                 | `Optional[float]`| `0.9`   | YES (sampling only) |
| `repetition_penalty`    | `float`          | `1.0`   | YES (greedy + sampling; **NOT beam**) |
| `num_beams`             | `int`            | `1`     | YES (dispatch + beam) |
| `do_sample`             | `bool`           | `True`  | YES (dispatch only) |
| `bos_token_id`          | `Optional[int]`  | `None`  | **NEVER read** |
| `eos_token_id`          | `Optional[int]`  | `None`  | YES (all strategies) |
| `pad_token_id`          | `Optional[int]`  | `None`  | YES (greedy + sampling; **NOT beam**) |
| `early_stopping`        | `bool`           | `True`  | **NEVER read** |
| `min_length`            | `int`            | `10`    | **NEVER read** |
| `no_repeat_ngram_size`  | `int`            | `0`     | **NEVER read** |
| `length_penalty`        | `float`          | `1.0`   | **NEVER read** |

**Dead config fields (declared but never consulted by any generation code path):**
`bos_token_id`, `early_stopping`, `min_length`, `no_repeat_ngram_size`,
`length_penalty`.

---

## 2. `generate()` (lines 871–915)

```python
@torch.no_grad()
def generate(self, input_ids, generation_config=None, **kwargs) -> torch.LongTensor
```

### Parameters read from `GenerationConfig`
- `num_beams` — for dispatch.
- `do_sample` — for dispatch.

(All other config fields are passed through to the chosen sub-method.)

### Control flow
1. If `generation_config is None`, instantiate `GenerationConfig()` (all defaults).
2. **kwargs override**: iterate `kwargs.items()`; for each `(key, value)`, if
   `hasattr(generation_config, key)`, call `setattr(generation_config, key, value)`.
   - **Note**: kwargs whose key is not an attribute of `GenerationConfig` are
     silently dropped (no error, no warning). This includes any typo'd field
     names and any field that exists on the model but not on the config.
3. **Dispatch** (first-match priority):
   - `if config.num_beams > 1:` → `_beam_search_generate(...)`
   - `elif config.do_sample:` → `_sample_generate(...)`
   - `else:` → `_greedy_generate(...)`

### EOS handling
None at this level; delegated to sub-methods.

### Batching
Passed through unchanged to sub-methods. `input_ids` shape is `(batch_size, seq_len)`.

### Output shape
Whatever the sub-method returns. All three sub-methods return a `(batch_size, ?)`
tensor, but the second dimension is **not guaranteed** to be
`initial_length + max_new_tokens` (see per-method notes below).

### Bugs / issues
- **`num_beams > 1` silently overrides `do_sample=True`.** There is no sampled
  beam search; setting both `num_beams > 1` and `do_sample=True` simply runs
  deterministic beam search and ignores `do_sample`.
- **`@torch.no_grad()`** is applied only to `generate`, not to the sub-methods.
  In practice this is fine because the sub-methods are only ever called from
  `generate`, but it means calling `_greedy_generate` etc. directly would build
  a graph.
- The docstring claims the output shape is always
  `(batch_size, initial_sequence_length + max_new_tokens)` — this is **not
  guaranteed** for any sub-method.

---

## 3. `_greedy_generate()` (lines 917–980)

### Parameters read from `GenerationConfig`
- `max_new_tokens` — loop count.
- `temperature` — applied if `!= 1.0`.
- `repetition_penalty` — applied if `!= 1.0`.
- `eos_token_id` — EOS handling (if not None).
- `pad_token_id` — pad for finished sequences (defaults to `0` if None).

### Control flow
```
unfinished_sequences = ones(batch_size, dtype=long, device)
for _ in range(max_new_tokens):
    outputs = self(input_ids, return_dict=True)            # FULL forward pass every step
    next_token_logits = outputs.logits[:, -1, :]           # [B, V]
    if temperature != 1.0:
        next_token_logits = next_token_logits / temperature
    if repetition_penalty != 1.0:
        next_token_logits = _apply_repetition_penalty(next_token_logits, input_ids, repetition_penalty)
    next_tokens = argmax(next_token_logits, dim=-1)        # [B]
    if eos_token_id is not None:
        pad = pad_token_id if pad_token_id is not None else 0
        next_tokens = next_tokens * unfinished + pad * (1 - unfinished)
        unfinished = unfinished * next_tokens.ne(eos_token_id).long()
    input_ids = cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
    if unfinished.max() == 0:
        break
return input_ids
```

### EOS handling
- An `unfinished_sequences` mask (`shape=[B]`, dtype `long`, values 0/1) tracks
  which sequences have already produced EOS.
- For sequences already finished (`unfinished=0`), the sampled token is
  overwritten with `pad_token_id` (or `0` if `pad_token_id is None`).
- For sequences still unfinished (`unfinished=1`), the argmax token is kept as-is.
- After appending, `unfinished` is multiplied by `next_tokens.ne(eos_token_id)`.
  This means: **the EOS token IS appended to the output** on the step that
  produces it; only subsequent steps inject pad.
- Early stop: `if unfinished_sequences.max() == 0: break`.

### Batching
- `batch_size = input_ids.size(0)`. All sequences run in lockstep.
- Finished sequences keep receiving forward passes (with pad tokens appended)
  until ALL sequences are finished or `max_new_tokens` is hit.
- **No `attention_mask` is passed to `forward`**, so pad tokens in finished
  sequences participate in attention and influence the unfinished sequences'
  next-token logits. (Pads are not masked.)

### Output shape
- `(batch_size, initial_length + num_steps_actually_run)`.
- If the loop early-breaks, `num_steps_actually_run < max_new_tokens`, so the
  output is **shorter** than the docstring-claimed
  `initial_length + max_new_tokens`.
- If at least one sequence never emits EOS, the loop runs the full
  `max_new_tokens` and the shape matches the docstring.
- **No padding-to-fixed-length step at the end** — rows can be of unequal
  effective length only because finished rows contain pad tokens in their tail
  (the tensor itself is rectangular).

### Bugs / issues
- **No KV cache.** `self(input_ids, return_dict=True)` recomputes the full
  forward pass over the entire growing sequence every step. Generation is
  O(T²) in compute and the model's `use_cache=True` flag is never set.
- **No `attention_mask` passed.** Pad tokens appended for finished sequences
  are attended to by unfinished sequences, which can subtly shift their logits.
- **Temperature applied even though it cannot change argmax.** Harmless but
  pointless for greedy decoding; only matters if `temperature` is extreme (e.g.
  `0.0` → division by zero → `inf`/`NaN` logits → `argmax` returns 0 for all).
- **No `min_length` enforcement.** EOS can be emitted on step 0.
- Output length is variable (per the early break), contradicting the docstring.

---

## 4. `_sample_generate()` (lines 982–1060)

### Parameters read from `GenerationConfig`
- `max_new_tokens` — loop count.
- `temperature` — **always** applied (unconditional division, no `!= 1.0` guard).
- `repetition_penalty` — applied if `!= 1.0`.
- `top_k` — applied if `not None and > 0`.
- `top_p` — applied if `not None and < 1.0`.
- `eos_token_id`, `pad_token_id` — same as greedy.

### Control flow
```
unfinished = ones(batch_size, long, device)
for _ in range(max_new_tokens):
    outputs = self(input_ids, return_dict=True)
    next_token_logits = outputs.logits[:, -1, :]            # [B, V]
    next_token_logits = next_token_logits / temperature      # ALWAYS, no guard
    if repetition_penalty != 1.0:
        next_token_logits = _apply_repetition_penalty(...)
    if top_k is not None and top_k > 0:
        next_token_logits = _top_k_filtering(next_token_logits, top_k)
    if top_p is not None and top_p < 1.0:
        next_token_logits = _top_p_filtering(next_token_logits, top_p)
    probs = softmax(next_token_logits, dim=-1)
    next_tokens = multinomial(probs, num_samples=1).squeeze(1)   # [B]
    if eos_token_id is not None:
        pad = pad_token_id if pad_token_id is not None else 0
        next_tokens = next_tokens * unfinished + pad * (1 - unfinished)
        unfinished = unfinished * next_tokens.ne(eos_token_id).long()
    input_ids = cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
    if unfinished.max() == 0:
        break
return input_ids
```

### EOS handling
Identical to `_greedy_generate` (see above).

### Batching
Identical to `_greedy_generate`. One `multinomial` call per step draws a
single sample per row.

### Output shape
Same considerations as `_greedy_generate`: rectangular
`(batch_size, initial_length + steps_actually_run)`, possibly shorter than
`initial_length + max_new_tokens` due to early break.

### Bugs / issues
- **`temperature` is divided unconditionally.** If `temperature == 0.0` this is
  a division by zero → `inf`/`-inf` logits → `softmax` produces `NaN` →
  `multinomial` errors or returns garbage. No clamping.
- **No `attention_mask` passed.** Same pad-attention bleed-through as greedy.
- **No KV cache.** Same O(T²) recompute as greedy.
- **Default `top_k=50` and `top_p=0.9`** means even basic sampling filters the
  distribution. To get pure sampling the user must explicitly set `top_k=0`
  and `top_p=1.0`.
- **Ordering of filters**: temperature → repetition penalty → top-k → top-p
  → softmax → multinomial. `top_p_filtering` runs `softmax` internally on the
  post-top-k logits, so the cumulative-prob threshold is computed against the
  already-top-k-filtered distribution. This is the intended composition.
- If `top_k=1` is set, after `_top_k_filtering` only one logit is finite and
  all others are `-inf`. The subsequent `_top_p_filtering` then computes
  `softmax` over a distribution with one finite entry and many `-inf` entries,
  which yields `1.0` for the kept token and `0.0` elsewhere — this works, but
  it is a fragile chain.
- Same variable-length output / docstring mismatch as greedy.
- `torch.multinomial` without a manual seed means **results are not
  reproducible** across runs even with `torch.manual_seed` set on the parent
  thread, unless the user manages the global RNG state themselves.

---

## 5. `_beam_search_generate()` (lines 1062–1183)

### Parameters read from `GenerationConfig`
- `max_new_tokens` — loop count.
- `num_beams` — beam width.
- `temperature` — **always** applied (unconditional division).
- `eos_token_id` — for the finished check.

### Parameters IGNORED by beam search
- `top_k`, `top_p`, `repetition_penalty` — **never applied** to beam search.
- `pad_token_id` — finished beams are not padded.
- `length_penalty` — never used; final beam score is the raw sum of log-probs
  (no length normalization).
- `early_stopping`, `min_length`, `no_repeat_ngram_size` — never used.

### Control flow
```
batch_size = input_ids.size(0)
num_beams  = config.num_beams

# Expand input: (B, T) -> (B, K, T) -> (B*K, T)
input_ids = input_ids.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1)

# Initial beam scores: beam 0 gets 0.0, beams 1..K-1 get -1e9
beam_scores = zeros(B, K); beam_scores[:, 1:] = -1e9; beam_scores = beam_scores.view(-1)  # [B*K]

unfinished = ones(B*K, long, device)

for _ in range(max_new_tokens):
    outputs = self(input_ids, return_dict=True)
    next_token_logits = outputs.logits[:, -1, :]                # [B*K, V]
    next_token_logits = next_token_logits / temperature          # ALWAYS
    next_token_scores = log_softmax(next_token_logits, dim=-1)   # [B*K, V]
    next_token_scores = next_token_scores + beam_scores[:, None] # [B*K, V]
    V = next_token_scores.size(-1)
    next_token_scores = next_token_scores.view(B, K * V)         # [B, K*V]
    next_scores, next_tokens = topk(next_token_scores, 2*K, dim=1, largest=True, sorted=True)  # [B, 2K]
    next_indices = next_tokens // V     # parent beam id in [0, K)
    next_tokens  = next_tokens  %  V     # actual token id in [0, V)

    # Python loops over batch and beam candidates to assemble new beams
    beam_outputs = []
    beam_scores_new = []
    for batch_idx in range(B):
        beams = []
        for beam_idx in range(K):                  # <-- beam_idx is dead code
            for idx in range(2 * K):
                score   = next_scores[batch_idx, idx]
                token   = next_tokens[batch_idx, idx]
                beam_id = next_indices[batch_idx, idx]
                orig_idx = batch_idx * K + beam_id
                new_seq  = cat([input_ids[orig_idx], token.unsqueeze(0)], dim=0)
                beams.append((score, new_seq))
                if len(beams) >= K:
                    break
            if len(beams) >= K:
                break
        beams = sorted(beams, key=lambda x: x[0], reverse=True)[:K]   # redundant re-sort
        for score, seq in beams:
            beam_outputs.append(seq)
            beam_scores_new.append(score)

    input_ids   = torch.stack(beam_outputs, dim=0)               # [B*K, T+1]
    beam_scores = torch.tensor(beam_scores_new, device=device)   # [B*K]

    if eos_token_id is not None:
        unfinished = unfinished * input_ids[:, -1].ne(eos_token_id).long()
    if unfinished.max() == 0:
        break

# Pick best beam per batch
input_ids   = input_ids.view(B, K, -1)
beam_scores = beam_scores.view(B, K)
best_beam_idx = beam_scores.argmax(dim=1)                        # [B]
best_ids = input_ids[arange(B, device), best_beam_idx]           # [B, T]
return best_ids
```

### EOS handling
- After assembling the new beams, `unfinished` is multiplied by
  `input_ids[:, -1].ne(eos_token_id).long()`.
- A beam that produces EOS is marked finished but **stays in the beam pool**.
  It is not frozen, not padded, and not removed — it continues to be expanded
  in subsequent steps, and its children continue to compete for top-K slots.
- Early stop only when `unfinished.max() == 0` (i.e., **every beam in every
  batch** has produced EOS at some point).

### Batching
- The batch dimension is folded into the leading dim as `B*K`, and the model
  is called once per step over all `B*K` beams.
- The Python reassembly loop iterates `batch_idx in range(B)` and within each
  batch independently selects `K` new beams from the top `2*K` candidates.

### Output shape
- Returns `best_ids` of shape `(batch_size, initial_length + steps_actually_run)`.
- The output is one row per batch element (the best-scoring beam), not
  `num_beams` rows.
- Length may be shorter than `initial_length + max_new_tokens` due to the
  global early break.
- There is **no guarantee** the best beam's sequence actually ends with EOS
  (a beam that produced EOS early and was then out-scored by a non-EOS
  continuation may lose).

### Bugs / issues
1. **`for beam_idx in range(num_beams):` is dead code.** The variable
   `beam_idx` is never read inside the loop body. The loop is effectively
   `for idx in range(2*K): ... if len(beams) >= K: break` — it just collects
   the top `K` candidates (already sorted by `topk`).
2. **The `sorted(beams, ...)` step is redundant.** `topk` already returns
   candidates in descending score order, and the inner loop appends them in
   that order, so re-sorting changes nothing (modulo ties, which `topk`
   resolves deterministically).
3. **No diversity enforcement across parent beams.** The standard beam-search
   invariant — "at most one continuation per parent beam per round" — is not
   enforced. All `K` new beams may be continuations of the same parent,
   collapsing beam diversity.
4. **`unfinished_sequences` is never remapped when beams are reordered.**
   When `input_ids` is rebuilt as `torch.stack(beam_outputs, dim=0)`, the
   beam in slot `i` may now be a continuation of a *different* parent than
   the one that previously occupied slot `i`. But `unfinished` is updated as
   `unfinished * input_ids[:, -1].ne(eos)` — it never transfers "finished"
   state along with the beam identity. This means:
   - A finished beam whose slot is taken by a fresh continuation is
     incorrectly kept marked finished.
   - A fresh continuation whose slot was previously occupied by a finished
     beam inherits the finished state.
   In practice the bug is partly masked because the only consumer of
   `unfinished` is the global `max() == 0` early-stop check, but the state
   is still semantically wrong.
5. **Finished beams are never frozen or padded.** A beam that emits EOS
   keeps being expanded (its children appear in the next round's candidate
   pool). Its sequence grows past the EOS token. The final returned "best"
   sequence may therefore contain tokens after an EOS token.
6. **No length normalization.** `length_penalty` is in `GenerationConfig`
   but never used. Final selection is by raw sum-of-log-probs, which
   systematically favors shorter sequences (fewer negative log-probs summed).
7. **`temperature` divided unconditionally.** `temperature=0.0` produces
   `inf`/`NaN` and breaks `log_softmax`. No clamping.
8. **Python-loop reassembly with `torch.cat` per beam** allocates a new
   tensor of length `T+1` for every beam every step — O(B · K · T) memory
   allocations per step. Slow.
9. **`beam_scores = torch.tensor(beam_scores_new, device=device)`** rebuilds
   a tensor from a Python list every step. Slow and breaks any autograd
   graph (irrelevant under `no_grad`, but still wasteful).
10. **`beam_scores` dtype is inferred** from the Python floats in
    `beam_scores_new`, which on CPU defaults to `float64`. This can cause
    silent dtype mismatches with `next_token_scores` (float32) when they
    are added on the next iteration.
11. **No `top_k`, `top_p`, `repetition_penalty`** even though the config
    fields exist. Beam search is "pure" beam search.
12. **No `attention_mask`** passed to `forward`.
13. **No KV cache.** Full forward over `B*K` sequences every step.

---

## 6. `_apply_repetition_penalty()` (lines 1187–1222)

### Signature
```python
_apply_repetition_penalty(logits: Tensor, input_ids: Tensor, penalty: float) -> Tensor
```
- `logits`: shape `(B, V)`.
- `input_ids`: shape `(B, T)` (the full sequence so far, prompt + generated).
- `penalty`: scalar float.

### Exact algorithm
```python
B, V = logits.shape
for i in range(B):
    for token_id in set(input_ids[i].tolist()):
        if logits[i, token_id] < 0:
            logits[i, token_id] *= penalty
        else:
            logits[i, token_id] /= penalty
return logits
```

This is the CTRL-style repetition penalty (Keskar et al. 2019):
- For `penalty > 1.0`: negative logits get more negative (multiplied by `>1`);
  non-negative logits get smaller (divided by `>1`). Both operations reduce
  the logit value, hence reduce the probability of repeated tokens.
- For `penalty < 1.0`: the inverse — repeated tokens are *boosted*.
- For `penalty == 1.0`: no-op (caller guards with `if penalty != 1.0:`).

### Issues with the Python loop
1. **Forces a GPU→CPU sync.** `input_ids[i].tolist()` materializes the row on
   CPU. If `logits`/`input_ids` are on GPU, this serializes the pipeline and
   blocks the stream every call.
2. **O(B · U) Python-level iterations** where `U` is the number of unique
   tokens per row. For long sequences (large `U`) this is slow compared to a
   vectorized `scatter_`-based implementation.
3. **In-place modification of the input `logits` tensor.** The function
   returns the same tensor it was given; callers that hold a reference to the
   pre-call logits will see the modified values. (In the current generation
   code this is harmless because the caller re-binds the variable, but the
   side-effect is a footgun.)
4. **Penalizes prompt tokens as well as generated tokens.** The set is built
   from the entire `input_ids` row, which includes the original prompt. Many
   implementations only penalize generated tokens.
5. **No clamping on `penalty`.** `penalty == 0.0` produces `inf`/`NaN`
   (division by zero on the `else` branch). The caller only guards
   `!= 1.0`, so `0.0` slips through.
6. **`set(...)` ordering is non-deterministic across Python runs** (CPython
   set iteration order is hashed and depends on insertion order and PYTHONHASHSEED).
   Because each unique token is visited exactly once, this does not change the
   final logits, but it does change the order of the in-place writes — which
   can matter for bit-exact reproducibility on some backends.
7. **The penalty is applied to `logits` AFTER temperature scaling** (in both
   `_greedy_generate` and `_sample_generate`). The relative effect of the
   penalty therefore changes with temperature — at high temperature the same
   `penalty` has a smaller effect on probabilities than at low temperature.

---

## 7. `_top_k_filtering()` (lines 1224–1247)

### Signature
```python
_top_k_filtering(logits: Tensor, top_k: int) -> Tensor
```
- `logits`: shape `(B, V)`.
- `top_k`: int.

### Exact algorithm
```python
top_k = min(top_k, logits.size(-1))
indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
logits[indices_to_remove] = -float('inf')
return logits
```

Step by step:
1. Clamp `top_k` to `V` (no-op if `top_k <= V`).
2. `torch.topk(logits, top_k)` returns named tuple `(values, indices)` where
   `values` is shape `(B, top_k)` sorted descending along dim=-1.
3. `[0]` selects the `values` tensor.
4. `[..., -1, None]` selects the smallest value within each row's top-k
   (i.e., the k-th largest logit per row), shape `(B, 1)`. This is the
   threshold.
5. `indices_to_remove = logits < threshold` — boolean mask `(B, V)` that is
   `True` for every logit strictly less than the row's k-th largest value.
6. `logits[indices_to_remove] = -inf` — in-place set of all non-top-k
   entries to `-inf`.

### Behavior notes
- Ties at the threshold value are **kept** (because the comparison is strict
  `<`). If multiple tokens share the k-th largest logit, more than `top_k`
  tokens survive.
- The function modifies `logits` in place and returns the same tensor.
- If `top_k == 1`, only the single argmax per row survives.
- If `top_k >= V`, no entries are removed (threshold = min of all = the
  smallest logit; nothing is strictly less than itself; only ties with the
  min are kept and the min itself is kept).

### Bugs / issues
- **No guard for `top_k <= 0`.** If called directly with `top_k=0`,
  `torch.topk(logits, 0)` raises `RuntimeError: selected index k is out of range`.
  (The caller `_sample_generate` guards with `top_k > 0`, so this is not
  reachable through normal `generate()` usage.)
- In-place modification of input tensor (same footgun as repetition penalty).
- Ties can produce a survivor set larger than `top_k` — usually harmless but
  technically not "exactly top-k".

---

## 8. `_top_p_filtering()` (lines 1249–1283)

### Signature
```python
_top_p_filtering(logits: Tensor, top_p: float) -> Tensor
```
- `logits`: shape `(B, V)`.
- `top_p`: float, intended range `(0.0, 1.0]`.

### Exact algorithm
```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)         # (B, V), (B, V)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)    # (B, V)

sorted_indices_to_remove = cumulative_probs > top_p                          # (B, V) bool
sorted_indices_to_remove[..., 0] = False                                     # always keep top-1

indices_to_remove = sorted_indices_to_remove.scatter(
    1, sorted_indices, sorted_indices_to_remove
)
logits[indices_to_remove] = -float('inf')
return logits
```

Step by step:
1. Sort each row of `logits` descending; obtain `sorted_logits` and the
   permutation `sorted_indices` that maps original positions → sorted
   positions.
2. Compute `softmax` over the sorted logits (still descending), then cumulatively
   sum along the last dim. `cumulative_probs[i, j]` is the sum of the top
  `(j+1)` probabilities in row `i`.
3. `sorted_indices_to_remove = cumulative_probs > top_p` — `True` for sorted
   positions whose cumulative probability already exceeds `top_p`.
4. Force `sorted_indices_to_remove[:, 0] = False` so the highest-probability
   token is always kept (even if its own probability already exceeds `top_p`).
5. Scatter the boolean mask back to the original (unsorted) vocabulary
   positions: `indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)`.
   - `scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)`:
     for each row `i`, for each sorted position `j`, writes
     `sorted_indices_to_remove[i, j]` into `indices_to_remove[i, sorted_indices[i, j]]`.
6. `logits[indices_to_remove] = -inf`.

### Behavior notes
- The threshold test is **strict `>`**. A token whose cumulative probability
  equals `top_p` exactly is kept; a token whose cumulative probability is
  strictly greater is removed.
- This is **different from HF's convention**. HF's `TopPLogitsWarper` shifts
  the mask right by one so that the token that *first crosses* the threshold
  is also kept. This implementation removes that crossing token. The
  practical effect: this impl keeps a slightly smaller nucleus than HF would
  for the same `top_p`.
- The top-1 token is always kept (forced `False` at position 0), even when
  its probability alone exceeds `top_p`.
- If `top_p == 0.0`: every cumulative prob (except possibly the first) is
  `> 0`, so all but the top-1 token are removed. Top-p=0 degenerates to
  greedy.
- If `top_p == 1.0`: caller (`_sample_generate`) gates with `top_p < 1.0`, so
  this function is not called. If called directly, no entries would be
  removed (cumulative max is 1.0; `1.0 > 1.0` is False; mask is all False;
  nothing removed).

### Bugs / issues
- **`softmax` is computed inside this function** on the (possibly
  temperature-scaled, post-top-k) logits. This is correct in the current
  call order (temperature applied first by caller, then top-k, then top-p),
  but it means the `top_p` threshold is applied to the post-temperature
  distribution. At high temperature the cumulative probability mass spreads
  out and more tokens survive; at low temperature fewer survive.
- **Composed with `_top_k_filtering`**: if `_top_k_filtering` already set
  many entries to `-inf`, then `_top_p_filtering` computes `softmax` over
  them. `softmax(-inf) = 0`, so the cumulative-prob computation is
  numerically safe (the `-inf` entries contribute 0). However, after
  `_top_p_filtering`'s sort, the `-inf` entries go to the end of the sorted
  order with probability 0 — they will be marked for removal (cumulative
  prob already at 1.0 > `top_p`), but their original `logits` entries are
  already `-inf` so the assignment is a no-op. Safe in practice.
- **Edge case**: if all `logits` in a row are `-inf` (e.g., a degenerate
  upstream filter), `softmax` produces `NaN` (0/0). This is not reachable
  through normal `generate()` usage because `_top_k_filtering` always keeps
  at least the top-1 entry finite, but it is a hazard for direct callers.
- **In-place modification** of the input `logits` tensor (same footgun as
  the other helpers).
- The `scatter` call uses `sorted_indices_to_remove` as **both the source
  and the input** to scatter onto. This works because `scatter` writes
  `src[i, j]` into `self[i, index[i, j]]`, and since every position is
  written exactly once (the `sorted_indices` are a permutation), there are
  no race conditions. But it is subtle code.
- **Deviation from HF convention** (described above) — for the same `top_p`,
  the surviving nucleus is a strict subset of what HF's `top_p` would keep.

---

## 9. Cross-cutting issues

1. **No KV cache.** Every generation step calls
   `self(input_ids, return_dict=True)` with `use_cache` left at its default
   (`False`). Each step recomputes attention over the entire sequence so
   far. Generation is O(T²) in compute.

2. **No `attention_mask` ever passed.** All three strategies append pad
   tokens to finished sequences (greedy + sampling) or never pad (beam),
   but none of them construct or pass an `attention_mask` to `forward`.
   Pad tokens are therefore attended to and influence subsequent logits.

3. **Variable output length.** All three sub-methods can early-break,
   producing a tensor whose second dim is `initial_length + steps_run`,
   which may be less than `initial_length + max_new_tokens`. The
   `generate()` docstring's claimed output shape is incorrect.

4. **`@torch.no_grad()` only on `generate`.** The sub-methods are not
   decorated. Calling them directly will build an autograd graph (slow,
   memory-leaky).

5. **Dead `GenerationConfig` fields.** `bos_token_id`, `early_stopping`,
   `min_length`, `no_repeat_ngram_size`, `length_penalty` are declared
   with defaults but never read by any generation code path. Setting them
   has no effect.

6. **Silent kwargs dropping.** `generate(**kwargs)` drops any kwarg whose
   key is not an attribute of `GenerationConfig`. Typos are not caught.

7. **`num_beams > 1` + `do_sample=True`** runs deterministic beam search;
   `do_sample` is ignored.

8. **No seeding discipline.** `_sample_generate` calls `torch.multinomial`
   without a generator argument; reproducibility depends on the global
   PyTorch RNG state.

9. **No min-length enforcement anywhere.** EOS can be emitted on the very
   first generated step.

10. **In-place modification is pervasive.** `_apply_repetition_penalty`,
    `_top_k_filtering`, and `_top_p_filtering` all mutate the input
    `logits` tensor and return it. Callers that alias the pre-call logits
    will see modified values.

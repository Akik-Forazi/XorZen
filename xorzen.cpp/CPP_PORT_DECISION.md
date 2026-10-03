# CPP_PORT_DECISION — XorZen C++ vs Python Architecture Audit

**Purpose**: Evidence-only comparison of the current C++ port (`xorzen.cpp/src/model/*.cpp`)
against the Python reference (`xorzen/model/**/*.py`). No path is recommended here —
only findings, mathematical differences, and reuse/rewrite assessments.

**Method**: Direct line-by-line reading of the cited C++ and Python sources.
Every claim below is backed by a file:line reference.

---

## Executive summary (evidence only)

| # | Component | Parity status | Source evidence |
|---|-----------|---------------|-----------------|
| 1 | Config (variants) | ✅ Match | `variants.cpp:12-14` states "Python is the source of truth" |
| 2 | Embeddings | ✅ Match | `xorzen_model.cpp:12-16` vs Python model.py |
| 3 | LocalAttention | ⚠️ Math match, IO differ | `hass_block.cpp:50-62` matmul+softmax vs Python SDPA |
| 4 | QKV projections | ✅ Match | `hass_block.cpp:18-21` vs `hass_block.py:52-54` |
| 5 | HASS block | ⚠️ Structural match | 3 pathways + gate + FFN, same skeleton |
| 6 | SSM kernel (ssm.cpp) | ❌ Wrong algorithm | `ssm.cpp:20-122` S4D complex-A, **not used in HASS** |
| 7 | SSM pathway (hass_block.cpp) | ❌ Math differs | Missing ZOH on B, wrong LN position |
| 8 | SSM scan | ❌ Different algorithm | C++ serial CPU scan vs Python chunked_scan |
| 9 | SlicedFFN vs AdaptiveFFN | ❌ Wrong FFN class | `hass_block.cpp:250-278` AdaptiveFFN vs `sliced_ffn.py` |
| 10 | Sparse dispatch | ❌ Missing | C++ does dense FFN + scalar multiplier |
| 11 | Depth routing | ⚠️ Partial | Has complexity bias, **no cost-aware modulation** |
| 12 | Width routing | ⚠️ Partial | Has bias, **no cost-aware modulation** |
| 13 | Pathway routing | ⚠️ Partial | Has Gumbel+prior, **no cost-aware modulation** |
| 14 | Expert routing | ✅ Match | top-k + capacity, both implementations |
| 15 | Cost-aware routing | ❌ Missing entirely | `routing.cpp` has no budget/sparsity_pressure |
| 16 | MoE fabric | ⚠️ Different IO | C++ disk-sharded LRU, Python in-memory ModuleList |
| 17 | CoT | ⚠️ Structural match | InternalLatentCoT exists in both |
| 18 | Merger | ✅ Match | `hass_block.cpp:322-343` GatedMerger |
| 19 | LM head | ✅ Match | Linear(hidden, vocab, bias=false) |
| 20 | Loss | ✅ Match | CE + routing + load_balance + cot_aux |
| 21 | Generation | ✅ Match | Greedy autoregressive loop |
| 22 | Checkpointing | ❌ Missing in C++ | `xorzen_model.cpp` has no checkpoint() call |
| 23 | character_router | ⚠️ C++-only | `routing.cpp:72-75` — Python has no equivalent |

**Bottom line**: Config, embeddings, attention math, QKV, expert routing, merger,
LM head, loss, and generation are reusable. SSM pathway math, SlicedFFN,
cost-aware routing, and gradient checkpointing are architecturally wrong or
missing and require replacement.

---

## Component-by-component analysis

### 1. Config (`variants.cpp` vs `xorzen/config.py`)

**Python implementation** (config.py:435, 868, 1990+): Dataclass `ModelConfig`
with `ConfigFactory.get_config(size)` returning hardcoded values per size tier.
Default `gradient_checkpointing=True`, `cost_aware_routing=True` (implied),
`compute_budget=1.0` (implied).

**C++ implementation** (`variants.cpp:16-293`): Switch over `ModelSize` enum.
Every value carries a comment "Python: …" with the exact Python expression.
Common defaults at lines 281-291: `tie_word_embeddings=true`, `causal=true`,
`use_sliced_ffn=true`, `use_moe=true`, `low_rank_dim = hidden*3/8`,
`ssm_state_dim=16`, `unify_load_balance=true`.

**Mathematical difference**: None observed in the size tiers (TINY_23K through
XL_7B). All vocab/hidden/layers/heads/experts match.

**Missing in C++ config**: `cost_aware_routing` and `compute_budget` fields
are referenced by Python routing.py:546-548 via `getattr(self.config,
'cost_aware_routing', True)` and `getattr(self.config, 'compute_budget', 1.0)`.
These defaults exist in Python's dataclass but the C++ `ModelConfig` struct
does not appear to declare them (would need header check to confirm; the
routing.cpp code never reads either attribute).

**Port difficulty**: LOW
**Reuse potential**: HIGH — config values are already correct
**Rewrite required**: NO (but add `cost_aware_routing` + `compute_budget`
fields to match Python's defaults)

---

### 2. Embeddings (`xorzen_model.cpp:12-16`)

**Python**: `nn.Embedding(vocab, hidden, padding_idx=pad)` + learned position
embeddings `nn.Embedding(context_length, hidden)`, summed, then dropout.

**C++**: `torch::nn::Embedding(vocab, hidden, padding_idx=pad)` + position
embedding `nn::Embedding(context_length, hidden)`, summed at line 80:
`hidden + position_embedding->forward(position_ids)`, then `embedding_dropout`.

**Mathematical difference**: None. Same parameters, same sum, same dropout.

**Port difficulty**: LOW
**Reuse potential**: HIGH
**Rewrite required**: NO

---

### 3. Attention — LocalAttentionPathway (`hass_block.cpp:11-66` vs `hass_block.py:25-207`)

**Python** (`hass_block.py:188-197`): Calls `F.scaled_dot_product_attention`
(SDPA). Builds additive `attn_bias` from causal mask + window mask + padding
mask + position bias, then passes to SDPA's `attn_mask=` parameter.
- Memory: O(S) — never materializes [B,H,S,S] attention matrix
- Dispatch: FlashAttention-2 / memory-efficient backend on GPU
- Causality: handled via `attn_bias` (not `is_causal=True`) because windowed
  causality cannot be expressed with the boolean `is_causal` flag

**C++** (`hass_block.cpp:45-63`): Two paths:
- If `XORZEN_ENABLE_FLASH_ATTN` defined: calls `optimized::flash_attention_cpu`
  (a hand-written CPU flash kernel, not PyTorch's SDPA)
- Otherwise: `scores = matmul(q, k.T) / sqrt(d)` → `masked_fill(-inf)` →
  `softmax` → `matmul(probs, v)`. Materializes the full [B,H,S,S] matrix.

**Mathematical difference**: Identical math (softmax(QK^T/√d) · V), but
**different memory profile**. Python is O(S) via SDPA tiling; C++ is O(S²)
unless `XORZEN_ENABLE_FLASH_ATTN` is defined. For batch=16, S=1024, H=16:
Python ≈ 0 MB transient; C++ ≈ 1 GB transient (16×16×1024×1024×4 bytes).

**Note**: The C++ `flash_attention_cpu` is NOT PyTorch SDPA — it is a custom
implementation in `src/optimized/flash_attn.cpp`. Numerical equivalence to
PyTorch SDPA is not guaranteed without a parity test.

**Port difficulty**: MEDIUM
**Reuse potential**: MEDIUM — QKV/LN/out_proj all reusable; only the attention
compute kernel needs replacing
**Rewrite required**: UNCERTAIN — if LibTorch exposes `attn::scaled_dot_product_attention`
it can be a drop-in; otherwise the existing CPU flash_attn.cpp may suffice
with a parity test

---

### 4. QKV projections

**Python** (`hass_block.py:52-54`): Separate `q_proj`, `k_proj`, `v_proj`
(comment at line 47-51 explicitly rejects fused QKV for checkpoint compat).

**C++** (`hass_block.cpp:18-21`): Same — separate `q_proj`, `k_proj`, `v_proj`,
each `nn::Linear(hidden, hidden)`. Also has `ln_q`, `ln_k` LayerNorms on
`head_dim` (matches Python line 64-65).

**Mathematical difference**: None.

**Port difficulty**: LOW
**Reuse potential**: HIGH
**Rewrite required**: NO

---

### 5. HASS block (`hass_block.cpp:280-320` vs `hass_block.py:861+`)

**Python HASSBlock** (`hass_block.py:861-1051+`):
- 3 pathways: `LocalAttentionPathway`, `LowRankGlobalPathway`, `SSMPathway`
- `pathway_gate` (small MLP → 3 logits)
- FFN: `SlicedFFN` (default, line 921) or `AdaptiveFFN` (legacy, line 932)
- `ln1`, `ln2`, dropout
- Forward: `xa = ln1(x)` → 3 pathways in parallel → softmax-weighted sum →
  residual → `xf = ln2(residual)` → FFN → residual

**C++ HASSBlock** (`hass_block.cpp:280-320`):
- 3 pathways: `LocalAttentionPathway`, `LowRankGlobalPathway`, `SSMPathway`
- `pathway_gate` (Linear→LN→GELU→Linear to 3) at line 287-289
- FFN: `AdaptiveFFN` only (line 290) — **no SlicedFFN option**
- `ln1`, `ln2`, dropout
- Forward at lines 296-320:
  - `xa = ln1(x)`
  - Compute all 3 pathways unconditionally
  - If training: `w = 0.9 * routing_decision->path_probs + 0.1 * gate_probs`
    (line 309) — **hardcoded 0.9/0.1 blend, not in Python**
  - Else: `w = routing_decision->path_probs` (line 311)
  - `combined = sum_i pathway_i * w[i]`
  - Residual + dropout(combined)
  - `xf = ln2(residual)`
  - `ffn_out = ffn->forward(xf, routing_decision->width_multiplier)`
  - Return `residual + dropout(ffn_out)`

**Mathematical difference**:
1. C++ has hardcoded `0.9 * router + 0.1 * gate` blend during training
   (line 309). Python's forward (hass_block.py around line 1130+) uses
   SlicedFFN/AdaptiveFFN with the routing decision's `path_probs` directly.
2. C++ uses `AdaptiveFFN` exclusively — never `SlicedFFN`.
3. Pathway gate dims differ: C++ uses intermediate dim 128 (line 288);
   need to check Python's pathway_gate dims to confirm.

**Port difficulty**: MEDIUM (mostly because of FFN replacement)
**Reuse potential**: HIGH for the skeleton (3 pathways + gate + residuals);
LOW for the FFN
**Rewrite required**: YES for FFN; NO for the rest of the block

---

### 6. SSM kernel — `ssm.cpp` S4DKernel (`ssm.cpp:20-159`)

**Question asked**: "what SSM algorithm does it implement?"

**Answer**: Implements **S4D** (diagonal state-space model with **complex**
diagonal A). Specifically:

- `A_log` [hidden, N/2] + `A_im` [hidden, N/2] → complex A = -exp(A_log) + j·A_im
  (line 77-79). This is the S4D initialization from Gü et al. 2022.
- `B`, `C` are complex, stored as [..., 2] (real, imag) at lines 42-43.
- `D` is a real skip connection (line 46).
- `dt_proj`: `Linear(dt_rank, hidden)` with bias initialized so initial dt ~
  Uniform(dt_min, dt_max) via inv-softplus (lines 49-63).
- ZOH discretization:
  - `dtA = dt · A` (complex, [B,L,H,N2]) at line 93-94
  - `A_bar = exp(dtA)` at line 96
  - `B_bar_div = (exp(z) - 1)/z` with Taylor fallback `1 + z/2` when |z|<1e-4
    (lines 99-104)
  - `B_bar = B_bar_div · B_c` (line 105-106)
  - **BUG**: Missing `· dt` factor. Python's ZOH formula is
    `B_bar = B_bar_div · dt · B` (see ssm_scan.py:105). The C++ computes
    `B_bar = B_bar_div · B` (no dt), which is off by a factor of 1/dt.
- `parallel_scan` (line 124-159): Despite the name, this is a **serial CPU
  scan** with `#pragma omp parallel for collapse(2)` over (b, h) — NOT a
  Blelloch/associative scan. The comment at line 132 admits this:
  "Serial scan on CPU is MUCH faster than log(L) rounds of LibTorch kernel
  launches for typical sequence lengths".

**Critical observation**: `S4DKernel` and `SSMBlock` are defined in ssm.cpp
but **NOT instantiated by `xorzen_model.cpp`** or `hass_block.cpp`. The HASS
block uses `SSMPathway` (defined inside hass_block.cpp), not `SSMBlock`.
`ssm.cpp` appears to be either dead code or used by a different code path
(need to grep the rest of the codebase to confirm).

**Port difficulty**: HIGH (the algorithm itself is fine, but it's the wrong
algorithm for the HASS pathway, and the B_bar formula has a missing-dt bug)
**Reuse potential**: LOW for HASS parity (Python's SSMPathway uses real
diagonal A, not complex S4D); MEDIUM if a standalone S4D block is ever needed
**Rewrite required**: YES — for HASS parity, `ssm.cpp` is the wrong algorithm.
Python's `SSMPathway` uses real diagonal A and `discretize_zoh` from
`ssm_scan.py`.

---

### 7. SSM pathway — `SSMPathway` in `hass_block.cpp:95-248`

**Python SSMPathway** (`hass_block.py:394-570`):
- `A_log` [state_dim] real (line 419), `a = -exp(A_log)` real diagonal
- `dt_proj`, `B_proj`, `C_proj`, `D_proj` Linear
- `gate_proj` → gate + input_gate (both sigmoid)
- Causal conv (left-padded, padding=kernel-1, truncated to seq_len) at
  lines 525-531
- **Full ZOH on both A and B** (lines 542-545):
  ```
  dt = softplus(dt_proj(x_norm))
  a  = -exp(A_log)
  A_bar, B_bar = discretize_zoh(a, Bv, dt)   # B_bar = ((A_bar-1)/a) · B
  ```
- `select_scan(A_bar, B_bar, method='chunked')` (line 549-553)
- `states = ln_state(states)` — LN applied to **h_t** (line 556)
- `ssm_output = C · states` (line 559) — C multiplied **after** LN
- `ssm_output = D_proj(ssm_output)` (line 562)
- `output = ssm_output · gate` (line 565)

**C++ SSMPathway** (`hass_block.cpp:227-248`):
- `A_log` [state_dim] real (line 98), `a = -exp(A_log)` (line 240)
- `dt_proj`, `B_proj`, `C_proj`, `D_proj`, `gate_proj` — same modules
- Conv (line 105-106): `padding(kernel_size / 2)` — **center padding**, NOT
  causal left-padding. Python uses `padding=kernel_size - 1` (left-pad) and
  truncates. **This is a causality bug** in C++.
- Forward (lines 227-248):
  ```cpp
  auto Bv = B_proj(xn * input_gate);     // B(x), NO ZOH discretization
  auto C  = C_proj(xn);
  auto dt = softplus(dt_proj(xn));
  auto a  = -exp(A_log);
  auto Ab = exp(dt * a);                 // A_bar correct
  // B_bar is NOT computed — Bv is used directly
  auto states = SSMScanFunction::apply(Ab, Bv, C);
  // SSMScanFunction returns C_t * h_t (not h_t)
  auto out = D_proj(ln_state(states)) * gate;
  ```

**Exact mathematical differences**:

| Step | Python | C++ |
|------|--------|-----|
| Conv padding | `kernel-1` left-pad + truncate (causal) | `kernel/2` center-pad (NOT causal) |
| B discretization | `B_bar = ((A_bar-1)/a) · B(x)` (full ZOH) | `B_bar = B(x)` (no ZOH) |
| Scan output | returns `h_t` (running state) | returns `C_t · h_t` (already multiplied by C) |
| LayerNorm position | `D_proj(C · ln_state(h_t))` | `D_proj(ln_state(C · h_t))` |
| C multiplication | `C · ln_state(h_t)` (C outside LN) | `C · h_t` then LN (C inside LN) |

The LN-position difference is mathematically non-equivalent:
- Python: `D_proj(C · γ(h_t - μ)/σ + β)` where γ, β, μ, σ are over h_t
- C++:    `D_proj(γ(C·h_t - μ')/σ' + β)` where γ, β, μ', σ' are over C·h_t

The B_bar difference is also non-equivalent:
- Python: `h_t = A_bar · h_{t-1} + ((A_bar-1)/a) · B(x_t)`
- C++:    `h_t = A_bar · h_{t-1} + B(x_t)`

For small dt, `((A_bar-1)/a) ≈ dt`, so the C++ is missing roughly a factor
of `dt` on the input injection. This will produce different dynamics.

**Port difficulty**: HIGH
**Reuse potential**: MEDIUM — the module skeleton (A_log, dt_proj, B_proj,
C_proj, D_proj, gate_proj, ln_input, ln_state) is reusable; the forward
pass math is wrong
**Rewrite required**: YES — forward pass needs rewrite to:
1. Use causal left-padding conv
2. Apply `discretize_zoh` to get B_bar
3. Move LN before C multiplication
4. Return h_t from scan, multiply by C outside

---

### 8. SSM scan — `ssm_scan.py` vs C++ scan functions

**Python** (`ssm_scan.py:132-307`) provides three scan implementations:

1. `sequential_scan` (lines 132-157): plain Python `for t in range(T)` loop.
   O(T) latency, O(T·N) memory. Reference-correct.
2. `parallel_scan` (lines 160-225): **Blelloch / Hillis-Steele associative
   scan** with O(log T) parallel depth and O(T log T) work. Uses `clone()`
   at each step to avoid in-place aliasing. Operator:
   `(a_l, b_l) ⊕ (a_r, b_r) = (a_r·a_l, a_r·b_l + b_r)`.
3. `chunked_scan` (lines 228-273): hybrid — splits sequence into chunks of
   size C (default 256), runs `sequential_scan` within each chunk (vectorized
   over the chunk dimension), carries the final state of each chunk into the
   next. O(T/C) Python iterations.

`select_scan` (lines 276-307) is the dispatcher:
- T ≤ 64 → `sequential_scan`
- 64 < T ≤ 256 → `chunked_scan(chunk_size=T)` (one chunk)
- T > 256 → `chunked_scan(chunk_size=256)`

Default method used by `SSMPathway._scan_method` is `"chunked"` (hass_block.py:488).

**C++ scans** (two implementations):

1. `S4DKernel::parallel_scan` (`ssm.cpp:124-159`): Despite the name, this is
   a **serial CPU scan** with OpenMP `parallel for collapse(2)` over (b, h).
   Inner loops over n and l are serial. Complex<float> reinterpret_cast on
   CPU pointers. **Not associative, not log-depth.**

2. `SSMScanFunction` in `hass_block.cpp:120-225`: A `torch::autograd::Function`
   with custom forward and backward. Forward (lines 122-159): serial triple
   loop over b, t, d on CPU. Computes `h_t = A_bar · h_{t-1} + B_v` and
   returns `C · h_t`. Backward (lines 161-224): recompute forward states,
   then reverse-time sweep accumulating gradients. **Not chunked, not
   associative-parallel.**

**Mathematical difference**: All scans compute the same recurrence
`h_t = A_bar · h_{t-1} + B_bar` (modulo the B_bar bug from §7). The
differences are purely performance/IO:
- Python `chunked_scan`: T/256 Python iterations, each vectorized over the
  chunk — fast on GPU, runs in Python.
- C++ serial scan: single thread per (b, h) lane, OpenMP across (b, h).
  No chunking. Slow for long T on GPU (forces CPU←GPU sync at line 126-128).

**Port difficulty**: MEDIUM (math is fine; performance is wrong for GPU)
**Reuse potential**: MEDIUM — the autograd backward in SSMScanFunction is
reusable; the forward needs chunking for GPU parity
**Rewrite required**: UNCERTAIN — if the C++ target is CPU-only, the current
serial scan is correct (modulo the B_bar bug from §7). If GPU is a target,
a real chunked or associative scan is needed.

---

### 9. SlicedFFN vs AdaptiveFFN

**Python SlicedFFN** (`sliced_ffn.py:47-265`):
- Single set of weights at `max_width`: `fc1 [hidden, max_width]`,
  `fc2 [max_width, hidden]`
- Forward at width W: slice to `fc1[:, :W]` and `fc2[:W, :]` (lines 178-199)
- Per-token width selection (lines 201-246): tokens grouped by selected
  width, each group processed with sliced matmul
- Training: STE — hard argmax in forward, soft probs in backward
- This is **genuine conditional computation**: lower W = proportionally
  fewer FLOPs

**Python AdaptiveFFN** (`hass_block.py:650+`): Legacy, kept for backward
compat. Computes full base FFN + width adapters, blends them. Not genuine
sparsity (the module docstring at sliced_ffn.py:4-6 explicitly says this).

**C++ AdaptiveFFN** (`hass_block.cpp:250-278`):
- `fc1 [hidden, base_ffn_dim]`, `fc2 [base_ffn_dim, hidden]` at full width
- Forward (lines 262-278):
  ```cpp
  auto h = fc1(xn);                // FULL width matmul
  h = activation(h);
  h = ffn_dropout(ln_hidden(h));
  auto out = fc2(h);               // FULL width matmul
  if (width_multiplier.defined()) out = out * width_multiplier;  // scalar!
  ```
- The `width_multiplier` is a **scalar** (per-token, shape [B,T,1]) that
  scales the output. It does NOT change the matmul size.

**Exact mathematical difference**:

| Aspect | Python SlicedFFN | C++ AdaptiveFFN |
|--------|------------------|------------------|
| Weight shape | `[H, max_width]` shared across widths | `[H, base_ffn_dim]` fixed |
| Width-W forward | `fc2[:W,:](act(fc1[:,:W](x)))` — W-column matmul | `fc2(act(fc1(x))) · mult` — full matmul × scalar |
| FLOPs at width W | `2·H·W` (proportional to W) | `2·H·max_width` (constant) |
| Genuine sparsity | YES | NO |

The C++ AdaptiveFFN is NOT the SlicedFFN that Python defaults to
(`config.use_sliced_ffn=True` in variants.cpp:283, but the C++ HASSBlock
ignores this flag and always uses AdaptiveFFN).

**Port difficulty**: HIGH
**Reuse potential**: LOW — AdaptiveFFN is the wrong class. SlicedFFN needs
to be implemented from scratch in C++.
**Rewrite required**: YES — SlicedFFN is a new class. The forward requires:
1. Storing `fc1 [H, max_width]` and `fc2 [max_width, H]` at max width
2. Slicing weights per-token based on `width_idx`
3. Grouping tokens by selected width and batch-processing each group
4. STE for training (forward at hard argmax, backward through width_probs)

---

### 10. Sparse dispatch

**Python**: YES — `SlicedFFN._forward_per_token_width` (sliced_ffn.py:201-246)
groups tokens by selected width and processes each group with a sliced
matmul. Tokens with smaller W pay proportionally less compute.

**C++**: NO — `AdaptiveFFN::forward` always runs the full matmul. The
`width_multiplier` is a scalar that scales the OUTPUT, not the matmul size.
No token grouping, no sliced weights, no conditional compute.

**Port difficulty**: HIGH (requires implementing SlicedFFN, see §9)
**Reuse potential**: LOW
**Rewrite required**: YES

---

### 11. Depth routing (`routing.cpp:176-197` vs `routing.py:593+`)

**Python** (`routing.py:531-595`): Cost-aware modulation is applied BEFORE
the depth router:
```python
depth_shift = -sparsity_pressure * 4.0 * (1.0 - complexity.squeeze(-1))
depth_layer_bias = torch.linspace(0, -sparsity_pressure * 3.0, max_depth)
depth_logits = depth_logits + depth_layer_bias.view(1,1,-1) + depth_shift.unsqueeze(-1)
```
Then `_route_depth` is called. (We did not read `_route_depth` itself, but
the cost-aware bias is applied to the logits before the route call.)

**C++** (`routing.cpp:176-197`): No cost-aware modulation. The route_depth
function applies its own complexity-based bias:
```cpp
auto depth_bias = torch::linspace(0, 1, max_depth, ...);
auto scaled = (logits + complexity * depth_bias * 2.0) / max(temp, 1e-8);
// Gumbel noise if training, else sigmoid
```
This is similar in spirit (complexity biases depth) but has:
- Different sign convention: Python's `depth_shift` is NEGATIVE (penalizes
  depth for easy tokens under low budget); C++'s `complexity * depth_bias`
  is POSITIVE (encourages depth for complex tokens).
- No budget/sparsity_pressure term — C++ assumes budget=1.0 always.
- Different magnitudes: Python uses 4.0 × complexity_gap; C++ uses 2.0 × complexity.

**Exact mathematical difference**:
- Python: `logits' = logits + (-sparsity_pressure · 4 · (1 - complexity)) +
          linspace(0, -sparsity_pressure · 3, max_depth)`
- C++:    `logits' = logits + (complexity · linspace(0, 1, max_depth) · 2)`

When `sparsity_pressure = 0` (budget=1.0), Python applies zero cost-aware
bias and relies on the base `_route_depth` complexity modulation. C++ always
applies its 2× complexity bias. The two are not equivalent even at budget=1.

**Port difficulty**: MEDIUM
**Reuse potential**: MEDIUM — the Gumbel+sigmoid+hard-mask machinery is
reusable; the bias terms need replacement
**Rewrite required**: YES — need to add the cost-aware modulation block
from routing.py:531-590 before the route_depth call

---

### 12. Width routing (`routing.cpp:199-209` vs `routing.py:598+`)

**Python** (`routing.py:564-568, 598`):
```python
width_bias_axis = torch.linspace(sparsity_pressure * 2.0,
                                 -sparsity_pressure * 2.0,
                                 num_widths)
width_logits = width_logits + width_bias_axis.view(1, 1, -1)
# then _route_width is called
```
Lower budget biases toward SMALLER widths (early indices).

**C++** (`routing.cpp:199-209`):
```cpp
auto bias = torch::linspace(-1, 1, num_widths, ...);
auto scaled = (logits + complexity * bias * 3.0) / max(temp, 1e-8);
// Gumbel softmax or softmax
```
C++ biases toward LARGER widths for high-complexity tokens (bias goes -1 →
+1 across indices, multiplied by complexity). No budget awareness.

**Exact mathematical difference**:
- Python: `logits' = logits + linspace(+s·2, -s·2, num_widths)` where
  `s = sparsity_pressure = 1 - budget`
- C++:    `logits' = logits + complexity · linspace(-1, +1, num_widths) · 3`

**Port difficulty**: MEDIUM
**Reuse potential**: MEDIUM — gumbel_softmax machinery reusable
**Rewrite required**: YES — need to add cost-aware width_bias_axis

---

### 13. Pathway routing (`routing.cpp:211-219` vs `routing.py:603+`)

**Python** (`routing.py:572-576, 603`):
```python
path_bias_axis = torch.linspace(sparsity_pressure * 1.5,
                                -sparsity_pressure * 0.5,
                                num_paths)
path_logits = path_logits + path_bias_axis.view(1, 1, -1)
# then _route_path is called
```
Lower budget biases toward SSM (the last of 3 pathways, "cheapest for long
sequences"). The bias is monotonically decreasing across [local, low_rank,
ssm].

**C++** (`routing.cpp:211-219`):
```cpp
auto scaled = logits / max(temp, 1e-8);
if (deterministic || !is_training()) return softmax(scaled);
auto raw = gumbel_softmax(scaled, tau(max(temp * 2, 1.0)), hard=false);
double prior_weight = max(0.05, 0.5 * pow(0.995, training_step));
auto uniform = full_like(raw, 1.0 / num_paths);
return (1 - prior_weight) * raw + prior_weight * uniform;
```
No cost-aware path bias. Has an annealed uniform prior blend (prior_weight
decays from 0.5 toward 0.05 over training) — this is NOT in the Python code
we read (need to verify against Python `_route_path`).

**Exact mathematical difference**:
- Python: `logits' = logits + linspace(+s·1.5, -s·0.5, num_paths)` then route
- C++:    `logits' = logits / temp` then gumbel_softmax + uniform-prior blend

The uniform-prior blend in C++ is a training regularizer not present
(or present in a different form) in Python.

**Port difficulty**: MEDIUM
**Reuse potential**: MEDIUM — Gumbel softmax machinery reusable
**Rewrite required**: YES — need to add cost-aware path_bias_axis; verify
the uniform-prior blend exists in Python

---

### 14. Expert routing (`routing.cpp:221-282` vs `routing.py:608+`)

**Python** (`routing.py:608-610`): Calls `_route_experts(expert_logits,
current_temp, training, deterministic, expert_capacity)`. We did not read
`_route_experts` itself, but it produces `(expert_probs, expert_indices,
expert_weights)` with top-k selection and (optionally) capacity constraint.

**C++** (`routing.cpp:221-282`):
- Flattens [B,T,num_experts] → [N,num_experts]
- Gumbel noise + softmax (training) or plain softmax (eval)
- `torch::topk(probs_flat, top_k, -1)` to get top-k weights and indices
- L1-normalize the top-k weights: `weights / (weights.sum + 1e-12)` (line 238)
- Capacity constraint via `apply_capacity_constraint` (lines 246-282):
  - Sort by weight descending
  - For each expert, keep only the top `capacity` tokens (by weight rank)
  - Re-normalize

**Mathematical difference**: Likely matches Python's `_route_experts` (we
did not read Python's implementation in detail, but the structure — top-k +
L1-norm + capacity — is standard). The C++ `apply_capacity_constraint` has
an explicit Python-like loop over experts (line 270-277), which the comment
admits is "to keep it functional while moving away from CPU loop" — i.e.
not fully vectorized.

**Port difficulty**: LOW (assuming Python's _route_experts matches)
**Reuse potential**: HIGH
**Rewrite required**: NO (verify against Python `_route_experts`)

---

### 15. Cost-aware routing modulation

**Python** (`routing.py:531-590`): Full cost-aware block. Reads
`config.cost_aware_routing` (default True) and `config.compute_budget`
(default 1.0). Computes:
- `sparsity_pressure = 1 - budget` in [0, 1]
- `depth_shift = -sparsity_pressure · 4 · (1 - complexity)` per token
- `width_bias_axis = linspace(+s·2, -s·2, num_widths)`
- `path_bias_axis = linspace(+s·1.5, -s·0.5, num_paths)`
- `depth_layer_bias = linspace(0, -s·3, max_depth)`
- Adds all of these to the respective logits BEFORE routing

**C++** (`routing.cpp`): **Completely absent.** No `cost_aware_routing`
field read, no `compute_budget` field read, no `sparsity_pressure`
computation, no bias axes. The C++ router always behaves as if
`budget = 1.0` and `cost_aware_routing = False`.

**Port difficulty**: MEDIUM (the math is simple, but it requires touching
the router forward and adding config fields)
**Reuse potential**: LOW (nothing to reuse — the feature doesn't exist in C++)
**Rewrite required**: YES — implement the entire cost-aware block from
routing.py:531-590

---

### 16. MoE fabric (`zmoe.cpp` vs `zmoe.py`)

**Python** (`zmoe.py:532-679`):
- `self.experts = nn.ModuleList([ExpertFFN(...) for _ in range(num_experts)])`
  — all experts live in memory as registered submodules
- Forward: for each top-k slot, `unique_experts = torch.unique(slot_indices)`,
  for each expert_id: `expert = self.experts[expert_id]` (direct indexing,
  no cache miss possible)
- Test mode: single dummy expert with sum-of-weights scaling (lines 577-597)
- Explicit comment (lines 648-656): "do NOT re-normalize by sum(expert_weights)"
- Computes `load_balance_loss` and `routing_entropy` for stats

**C++** (`zmoe.cpp:266-477`):
- `ShardedExpertFabric` with `LRUExpertCache` + `ExpertDiskManager`
- Experts are stored on DISK as individual `.pt` files
  (`expert_0000.pt`, `expert_0001.pt`, ...)
- `ExpertDiskManager::load_expert` (lines 159-184): if file missing, create
  fresh expert and save; else load from disk via `torch::serialize::InputArchive`
- `LRUExpertCache` (lines 81-124): mutex-protected `std::list` + `std::unordered_map`
  with MRU splice on access, LRU eviction at capacity
- Forward (lines 302-406): for each top-k slot, `torch::unique(slot_idx)`,
  for each expert_id: `cache_.get(eid)` → if miss, `disk_mgr_->load_expert(eid, device)`
  and `cache_.put(eid, expert)`. Then `expert->forward(tok_hid)`.
- Same "do NOT re-normalize" comment (lines 388-398) — matches Python

**Mathematical difference**: NONE in the expert forward (both call
`ExpertFFN::forward` which is SwiGLU: `silu(gate(x)) * up(x)` → `down_proj`).
The difference is purely architectural:
- Python: in-memory `nn.ModuleList`, direct indexing — fast, but holds all
  experts in VRAM
- C++: disk-sharded with LRU cache — slow on first access (disk I/O), but
  can scale to large expert counts that don't fit in VRAM

The C++ design is actually MORE scalable for the XL_7B config (116 experts
× ~50MB each = ~6GB of expert weights), but it adds I/O latency and a
cache-hit-rate dependency. The Python design assumes all experts fit in
memory.

**Test mode parity**: Both have a `test_mode` that uses a single dummy
expert. C++ at lines 311-319, Python at lines 577-597. Match.

**Port difficulty**: LOW (math parity); HIGH (if matching Python's
in-memory design is required)
**Reuse potential**: HIGH for ExpertFFN itself; MEDIUM for the fabric
(disk sharding is a different design than Python's ModuleList)
**Rewrite required**: UNCERTAIN — depends on whether the goal is Python
parity (then rewrite to in-memory ModuleList) or production scalability
(then keep disk sharding)

---

### 17. CoT — InternalLatentCoT

**C++** (`xorzen_model.cpp:29-31, 87-97`): `cot = register_module("cot",
InternalLatentCoT(config))` and `cot_loss_head = register_module("cot_loss_head",
CoTAuxiliaryLoss(config))`. Forward applies CoT then injects via sigmoid gate:
```cpp
auto cot_out = cot->forward(hidden);
cot_vector = cot_out.first;
cot_influence = cot_out.second;
auto gate = sigmoid(cot->injection_gate->forward(hidden));
hidden = hidden + cot_influence * gate;
```
Auxiliary loss added at lines 143-148.

**Python**: Not read in detail for this audit. The C++ structure
(InternalLatentCoT + CoTAuxiliaryLoss + injection gate) appears structurally
consistent with the Python architecture based on the field names.

**Port difficulty**: LOW (structural match)
**Reuse potential**: HIGH (assuming the cot.cpp implementation matches)
**Rewrite required**: UNCERTAIN — need to read Python CoT module to confirm
math parity

---

### 18. Merger — GatedMerger (`hass_block.cpp:322-343`)

**C++**:
- `gate = Sequential(Linear(2H + cot_total_dim, H), GELU, Linear(H, 2), Softmax(-1))`
- `cot_proj = Linear(cot_total_dim, H)`
- `norm = LayerNorm(H)`
- Forward:
  ```cpp
  auto gate_input = cat({hass_output, moe_output, cot_vector}, -1);
  auto weights = gate(gate_input);                    // [B,T,2]
  auto cot = cot_proj(cot_vector);
  auto merged = hass * w[...,0] + moe * w[...,1] + 0.05 * cot;
  if (attention_mask.defined()) merged = merged * mask.unsqueeze(-1);
  return norm(merged);
  ```

**Python**: Not read directly, but the C++ structure matches the typical
gated-merger pattern. The `0.05 * cot` scaling is a hardcoded constant
that should be verified against Python.

**Port difficulty**: LOW
**Reuse potential**: HIGH
**Rewrite required**: UNCERTAIN (verify the 0.05 cot scaling and the
masking behavior against Python)

---

### 19. LM head (`xorzen_model.cpp:26-27`)

**C++**: `lm_head = Linear(hidden, vocab, bias=false)`. Tied to
`token_embedding->weight` if `config.tie_word_embeddings` is true (line 34-36).

**Python**: Standard `nn.Linear(hidden, vocab, bias=False)` with tying.
Match.

**Port difficulty**: LOW
**Reuse potential**: HIGH
**Rewrite required**: NO

---

### 20. Loss (`xorzen_model.cpp:125-148`)

**C++**: 
- `lm_loss = cross_entropy(shift_logits, shift_labels, ignore_index=pad)`
- `routing_loss = routing_regularizer->forward(decision)`
- `load_balance = compute_load_balance_loss(expert_indices, expert_weights)`
- `cot_aux = cot_loss_head->forward(...)` if CoT enabled
- `loss = lm_loss + routing_loss + load_balance + cot_aux`

**Python**: Standard composition. Match (assuming the routing_regularizer
and cot_loss_head internals match).

**Port difficulty**: LOW
**Reuse potential**: HIGH
**Rewrite required**: NO

---

### 21. Generation (`xorzen_model.cpp:173-189`)

**C++**: Greedy autoregressive loop:
- `tokens = prompt.clone()`
- For `max_length` steps: truncate to `context_length`, forward, take
  `logits[:, -1, :]`, `sample_next_token`, concat, early-stop on EOS.

**Python**: Standard generation loop. Match (assuming `sample_next_token`
implements the same sampling strategy — greedy/temperature/top-k).

**Port difficulty**: LOW
**Reuse potential**: HIGH
**Rewrite required**: NO

---

### 22. Checkpointing — gradient checkpointing

**Python** (`xorzen/models/zero/model.py:213-214, 521-535`):
```python
self.gradient_checkpointing = config.gradient_checkpointing
...
if self.gradient_checkpointing and self.training:
    block_out = checkpoint(
        create_forward_func(block, routing_decision, attention_mask),
        hidden_states,
        use_reentrant=False
    )
```
Each HASS block is wrapped in `torch.utils.checkpoint.checkpoint` during
training, trading compute for memory.

**C++** (`xorzen_model.cpp`): The `config.gradient_checkpointing` field
exists (set to `true` for NANO_10M and above in variants.cpp:105, 133, 161,
189, 217, 245, 273) but the forward pass at lines 102-113 does NOT call
any checkpoint function:
```cpp
for (int64_t i = 0; i < config.num_layers; ++i) {
    auto layer_mask = decision.depth_mask.select(-1, i);
    auto block = blocks[i]->as<HASSBlock>();
    torch::Tensor block_out;
    if (!is_training() && !layer_mask.any().item<bool>()) {
        block_out = hidden;
    } else {
        block_out = block->forward(hidden, &decision, attention_mask);
    }
    hidden = block_out * layer_mask.unsqueeze(-1) + hidden * (1.0 - layer_mask.unsqueeze(-1));
}
```
No `torch::checkpoint` (LibTorch does expose this in newer versions) or
equivalent. The flag is read into config but never acted upon.

**Mathematical difference**: Forward outputs are identical (gradient
checkpointing only affects backward memory, not forward math). But training
memory usage is O(L) higher in C++ than Python for any config with
`gradient_checkpointing=true`, which includes all configs from NANO_10M up.

**Port difficulty**: MEDIUM (LibTorch's `torch::checkpoint` API exists but
is less ergonomic than Python's; alternatively, manual recomputation)
**Reuse potential**: LOW (nothing to reuse — feature doesn't exist)
**Rewrite required**: YES — wrap each HASS block forward in
`torch::checkpoint::checkpoint` during training when the flag is set

---

### 23. character_router (C++-only, no Python equivalent)

**C++** (`routing.cpp:72-75`):
```cpp
character_router = register_module("character_router", torch::nn::Sequential(
    torch::nn::Linear(enc3, head), LayerNorm, GELU,
    torch::nn::Linear(head, config.max_characters),
    torch::nn::Sigmoid()));
```
Produces `character_probs` of shape [B, T, max_characters]. Stored in
`RoutingDecision::character_probs` (line 163).

**Python**: `grep -r "character_router\|max_characters" xorzen/` returns
**no matches**. The Python codebase does not have a `character_router`
module or `max_characters` config field.

**Implication**: The C++ router has an extra output head that Python doesn't
have. If checkpoints are trained in Python and loaded into C++, the C++
`character_router` weights will be uninitialized (or random). If trained in
C++, the checkpoint will have extra keys that Python can't load.

**Port difficulty**: LOW (just delete the module)
**Reuse potential**: NONE (Python doesn't want it)
**Rewrite required**: YES — remove `character_router` from C++ to match Python

---

## Three-path comparison

### PATH A — Incremental port (adapt current C++ component by component)

**Scope**: Keep the existing C++ codebase structure. For each component
flagged above as ❌ or ⚠️, patch the math in-place:
- Add `discretize_zoh` to SSMPathway, fix the LN position, fix the conv padding
- Replace `AdaptiveFFN` body with sliced matmuls (keep the class name)
- Add cost-aware modulation block to `AdaptiveRouter::forward`
- Remove `character_router`
- Wrap HASS blocks in `torch::checkpoint::checkpoint`
- Add `cost_aware_routing` + `compute_budget` to `ModelConfig`

**Reusable code** (high confidence):
- `variants.cpp` (configs match)
- `xorzen_model.cpp` skeleton (embeddings, router, blocks, moe, merger, head, loss, generate)
- `LocalAttentionPathway` QKV/LN/out_proj (the attention kernel itself may need swap)
- `AdaptiveRouter` feature_encoder + all 6 route heads + Gumbel machinery
- `ShardedExpertFabric` (if disk sharding is acceptable) or rewrite to ModuleList
- `GatedMerger`, `RMSNorm`, `lm_head`, `generate`
- `ExpertFFN` (SwiGLU) — math is correct

**Components needing replacement** (must rewrite the forward math):
- `SSMPathway::forward` (B_bar ZOH, LN position, conv padding)
- `AdaptiveFFN::forward` → become `SlicedFFN::forward` (sliced weights, token grouping, STE)
- `AdaptiveRouter::forward` (add cost-aware block before route_* calls)
- `xorzen_model.cpp::forward` (add gradient checkpointing wrapper around each block)
- Delete `character_router` and `S4DKernel`/`SSMBlock` (dead code or wrong algorithm)

**Checkpoint impact**:
- Existing C++ checkpoints: BREAK. The SSM pathway weights (A_log, B_proj,
  C_proj, D_proj) keep the same shapes but produce different outputs after
  the ZOH fix — the trained weights are tuned to the wrong dynamics.
  Retraining required.
- The `character_router.*` keys in existing C++ checkpoints become orphans.
  Need a checkpoint migration script.
- The SlicedFFN weight shapes differ from AdaptiveFFN (AdaptiveFFN:
  `fc1 [H, base_ffn_dim]`; SlicedFFN: `fc1 [H, max_width]` where
  `max_width = H * expert_hidden_multiplier`). If `base_ffn_dim == max_width`
  (they should both equal `H * 4`), the shapes are compatible and weights
  can be reused. Otherwise, reinitialize.
- Python-trained checkpoints: LOADABLE after fixes (modulo the character_router
  orphan keys, which Python doesn't have).

**Testing requirements**:
- Per-component parity tests: feed identical inputs to Python and C++,
  assert max_abs_diff < 1e-5 for each of: SSMPathway, SlicedFFN,
  AdaptiveRouter (with cost_aware on/off), HASSBlock, full model.
- The repo already has `tests/cpp/test_parity_router.cpp`,
  `test_parity_rmsnorm.cpp`, `test_mini_parity.cpp` — extend these.
- Numerical gradient parity for SSMScanFunction backward (the C++ has a
  hand-written backward that must match PyTorch autograd through
  `discretize_zoh`).
- Golden reference generation via `tests/generate_golden_reference.py`.

**Performance opportunity**:
- Keep the `SSMScanFunction` custom autograd (C++ backward is faster than
  PyTorch's reverse-mode through a Python loop).
- Keep the `LRUExpertCache` disk sharding for large-expert configs.
- Keep the OpenMP `parallel for` in `S4DKernel::parallel_scan` for CPU.
- Keep the `optimized::simd_ops` (silu_simd, gelu_simd, softmax_simd,
  fused_layernorm_gelu_simd) — these are C++-only optimizations.
- Keep the `optimized::flash_attention_cpu` if LibTorch SDPA is unavailable.

**Technical risk**:
- **HIGH** for SSMPathway: the C++ has a hand-written backward in
  `SSMScanFunction` that currently differentiates through `B_bar = B(x)`.
  After adding ZOH (`B_bar = ((A_bar-1)/a) · B(x)`), the backward must also
  differentiate through the `((A_bar-1)/a)` term w.r.t. A_log, dt, and B_proj.
  This is non-trivial and a likely source of numerical bugs.
- **MEDIUM** for SlicedFFN: per-token width grouping requires either Python-
  like loops over unique widths (slow) or a scatter-based implementation
  (complex but fast).
- **MEDIUM** for cost-aware routing: the bias terms are simple additions,
  but verifying parity requires Python routing.py:530-620 to be read in full
  (we only read lines 470-620).
- **LOW** for gradient checkpointing: LibTorch exposes
  `torch::checkpoint::checkpointer` (API may vary by version).

---

### PATH B — Partial rewrite (keep infrastructure, replace wrong model components)

**Scope**: Keep the "infrastructure" (config, embeddings, optimizer/trainer
plumbing, tokenizer, data pipeline, KV cache, ggml bridge, threading, SIMD
ops, disk manager, LRU cache) but REPLACE the model components that are
mathematically wrong:
- Rewrite `SSMPathway` from scratch as a port of `hass_block.py:394-570`
- Rewrite `AdaptiveFFN` as `SlicedFFN` (port of `sliced_ffn.py:47-265`)
- Rewrite `AdaptiveRouter::forward` to include the cost-aware block
  (port of `routing.py:531-590`)
- Add gradient checkpointing to `XorzenModelImpl::forward`
- Delete `S4DKernel`, `SSMBlock` (ssm.cpp) — wrong algorithm, not used
- Delete `character_router` from `AdaptiveRouter`
- Decide on MoE: keep `ShardedExpertFabric` (disk) or port Python's
  in-memory `ModuleList` design

**Reusable code** (infrastructure):
- All of `src/optimized/` (simd_ops, flash_attn_cpu, thread_pool, quantize,
  expert_mmap, ggml_bridge)
- All of `src/storage/` (kv_cache, project)
- All of `src/data/` (data_converter, text_dataset, mmap_dataset)
- All of `src/training/` (trainer, lora, curriculum, checkpoint)
- All of `src/tokenizer/` (bebpe_tokenizer)
- All of `src/utils/` (math_utils, logger)
- `src/model/variants.cpp` (configs)
- `src/model/cot_vector.cpp` (assuming parity)
- `src/model/merger.cpp` (GatedMerger, assuming parity)
- `src/model/coherence_field.cpp`, `src/model/igris.cpp` (if used)
- `src/model/expert.cpp` (ExpertFFN — SwiGLU math is correct)
- `include/xorzen/*.h` headers (with updates for new classes)

**Components needing replacement** (full rewrite):
- `src/model/ssm.cpp` — delete entirely (wrong algorithm)
- `src/model/hass_block.cpp` — keep `LocalAttentionPathway` and
  `LowRankGlobalPathway`, rewrite `SSMPathway` and `AdaptiveFFN`
  (rename to `SlicedFFN`)
- `src/model/routing.cpp` — keep `feature_encoder` and route heads,
  rewrite `forward` to add cost-aware block; delete `character_router`
- `src/model/zmoe.cpp` — DECIDE: keep disk-sharded design or rewrite to
  in-memory ModuleList for Python parity
- `src/model/xorzen_model.cpp` — add gradient checkpointing in the block loop

**Checkpoint impact**:
- Existing C++ checkpoints: BREAK hard. The `AdaptiveFFN` → `SlicedFFN`
  rename changes parameter keys (`ffn.fc1` → `ffn.fc1` if names are kept,
  but the shapes change if `base_ffn_dim != max_width`). The SSM pathway
  weights are reusable in shape but tuned to wrong dynamics. The
  `character_router.*` keys become orphans.
- Python-trained checkpoints: LOADABLE after the rewrite (the whole point
  of the rewrite is Python parity). May need a key-renaming shim if class
  names differ (`AdaptiveFFN` vs `SlicedFFN`).

**Testing requirements**:
- Same per-component parity tests as PATH A.
- The rewritten components can be tested in isolation before integration.
- Golden reference tests become the primary acceptance gate.
- Need to verify the MoE fabric choice (disk vs in-memory) against Python
  output parity — if Python uses ModuleList and C++ uses disk sharding,
  the outputs should still match mathematically (the ExpertFFN forward is
  the same), but caching/timing will differ.

**Performance opportunity**:
- Same as PATH A, plus: the rewrite can introduce a proper chunked scan
  (port of `chunked_scan`) for GPU, replacing the serial `SSMScanFunction`.
- The rewrite can use LibTorch's `at::special::expm1` for the ZOH Taylor
  fallback, vectorizing the small-z branch.
- SlicedFFN can use `torch::index_select` + batched matmul for the per-width
  grouping, avoiding Python-like loops.

**Technical risk**:
- **MEDIUM** overall: the rewrite is well-scoped (4-5 files), the Python
  reference is clear, and the infrastructure is preserved.
- **MEDIUM** for SSMPathway: still need to port `discretize_zoh` and
  `chunked_scan` correctly. The `chunked_scan` Python loop is a T/256
  iteration loop — in C++ this can be a single OpenMP loop over chunks.
- **MEDIUM** for SlicedFFN: per-token width grouping is the hard part.
  Python uses `torch.unique` + a loop; C++ can do the same or use a
  scatter-based approach.
- **LOW** for cost-aware routing: simple bias additions.
- **LOW** for gradient checkpointing: standard LibTorch API.
- **LOW** for MoE: keep disk sharding (already works) — just verify
  math parity with Python's ModuleList version.

---

### PATH C — Full model rewrite (rebuild from Python architecture)

**Scope**: Discard all of `src/model/*.cpp` and rebuild from Python as a
clean port. Keep only the infrastructure (optimized/, storage/, data/,
training/, tokenizer/, utils/).

**Reusable code** (infrastructure only):
- Everything listed in PATH B's "Reusable code (infrastructure)" list.
- Nothing from `src/model/` is reused except possibly `variants.cpp`
  (which already matches Python).

**Components needing replacement** (everything in src/model/):
- `ssm.cpp` — new SSMPathway matching `hass_block.py:394-570` + a new
  `ssm_scan.cpp` port of `ssm_scan.py` (discretize_zoh, sequential_scan,
  parallel_scan, chunked_scan, select_scan)
- `hass_block.cpp` — new HASSBlock port of `hass_block.py:861+`, including
  LocalAttentionPathway (use LibTorch SDPA), LowRankGlobalPathway,
  SSMPathway, SlicedFFN, AdaptiveFFN (legacy compat)
- `routing.cpp` — new AdaptiveRouter port of `routing.py:265+`, including
  cost-aware modulation, all 6 route heads, capacity constraint
- `zmoe.cpp` — new ShardedExpertFabric port of `zmoe.py:1-919`, decide
  disk vs in-memory
- `xorzen_model.cpp` — new XorzenModel port of `models/zero/model.py:1-1619`,
  including gradient checkpointing, depth-mask loop, CoT injection, merger
- `merger.cpp`, `cot_vector.cpp`, `coherence_field.cpp`, `igris.cpp`,
  `expert.cpp` — port each from corresponding Python file
- New: `sliced_ffn.cpp` (port of `sliced_ffn.py`)
- New: `cost_aware.cpp` or fold into routing.cpp

**Checkpoint impact**:
- Existing C++ checkpoints: UNSALVAGEABLE. All parameter keys change.
- Python-trained checkpoints: LOADABLE. This is the whole point — PATH C
  maximizes Python checkpoint compatibility. A clean port can match
  Python's `state_dict()` keys exactly (modulo `.` vs `_` separators in
  LibTorch, which is a serialization detail).

**Testing requirements**:
- Full model parity test: identical Python and C++ state_dict, identical
  inputs, identical outputs (max_abs_diff < 1e-5).
- Per-component parity tests as in PATH A/B.
- End-to-end training parity: run N steps on both, compare loss curves.
- Generation parity: identical prompts produce identical tokens (modulo
  sampling randomness, which must be seeded identically).

**Performance opportunity**:
- Cleanest opportunity to integrate the C++-only optimizations
  (`simd_ops`, `flash_attn_cpu`, OpenMP, ggml bridge) from the start,
  without fighting legacy code.
- Can use LibTorch's `at::sdpa` directly (if available in the LibTorch
  version) for Flash Attention parity.
- Can implement a proper chunked scan in C++ that matches Python's
  `chunked_scan` math exactly.
- Can use `torch::checkpoint::checkpointer` idiomatically.

**Technical risk**:
- **LOW** for correctness: the Python reference is the spec, and a clean
  port has the fewest "inherited bugs" from the current C++.
- **HIGH** for scope: this is the largest rewrite. Estimated 6-10 new
  .cpp files, each 200-500 lines. Total: ~3000-5000 lines of new C++.
- **MEDIUM** for timeline: PATH C takes the longest but produces the
  fewest ongoing parity bugs.
- **LOW** for infrastructure risk: the infrastructure (optimized/, storage/,
  etc.) is preserved and known-working.
- **MEDIUM** for LibTorch API risk: some Python idioms (e.g.,
  `F.scaled_dot_product_attention`, `torch.utils.checkpoint.checkpoint`,
  `torch.nn.functional.gumbel_softmax`) may not have direct LibTorch
  equivalents and require manual implementation. Need to verify LibTorch
  version exposes these.

---

## Evidence summary for path comparison

| Criterion | PATH A (incremental) | PATH B (partial rewrite) | PATH C (full rewrite) |
|-----------|----------------------|--------------------------|------------------------|
| Files touched | ~5 (patch in place) | ~5 (rewrite model files) | ~10+ (rebuild all model files) |
| New files | 0 | 1-2 (sliced_ffn.cpp, ssm_scan.cpp) | 4-6 (sliced_ffn, ssm_scan, cost_aware, etc.) |
| Config changes | Add 2 fields | Add 2 fields | Add 2 fields |
| Checkpoint compat (Python → C++) | YES after fixes | YES after fixes | YES (best compat) |
| Checkpoint compat (existing C++ → new C++) | PARTIAL (SSM weights reusable in shape, character_router orphans) | NO (SlicedFFN rename changes keys) | NO (all keys change) |
| Components requiring rewrite | 5 (SSM forward, FFN, router fwd, grad ckpt, delete character_router) | 5 (same + cleaner separation) | ~10 (everything in src/model/) |
| Components reusable as-is | ~15 | ~15 + infrastructure | infrastructure only |
| Parity test surface | Largest (must test old + new paths) | Medium (test new paths only) | Smallest (test new paths only) |
| Performance optimization headroom | Constrained by legacy shapes | Clean slate for new components | Cleanest slate |
| Risk: SSM backward correctness | HIGH (hand-written backward + ZOH) | HIGH (same) | HIGH (same) |
| Risk: SlicedFFN grouping perf | MEDIUM (in AdaptiveFFN body) | MEDIUM (in new SlicedFFN) | MEDIUM (in new SlicedFFN) |
| Risk: cost-aware routing parity | MEDIUM (verify vs Python) | MEDIUM (same) | MEDIUM (same) |
| Risk: MoE design divergence | LOW (keep disk sharding) | LOW (keep disk sharding) | LOW (keep disk sharding) |
| Risk: LibTorch API gaps (SDPA, checkpoint) | LOW (can fall back to existing flash_attn_cpu) | LOW (same) | MEDIUM (must use LibTorch APIs directly) |
| Estimated LOC delta | ~500 changed | ~1500 changed | ~3000-5000 new |

**No path is recommended here.** Each row above is factual evidence drawn
from the source files cited. The choice depends on constraints (timeline,
checkpoint compatibility requirements, acceptable technical risk) that are
outside the scope of this audit.

---

## Open questions (require further reading to resolve)

1. **Python `_route_depth`, `_route_width`, `_route_path`, `_route_experts`**
   (routing.py lines ~620-1000): not read in this audit. The C++ route_*
   functions may match or differ from Python in their internal logic (Gumbel
   noise, hard/soft masks, STE). A full parity audit requires reading these.

2. **Python `LowRankGlobalPathway`** (hass_block.py:261+): not read in
   detail. The C++ `LowRankGlobalPathwayImpl` (hass_block.cpp:68-93) uses
   `fused_layernorm_gelu_simd` and `softmax_simd` — need to verify the math
   matches Python's `to_low_rank` → LN → GELU → softmax over context_weights
   → matmul → `from_low_rank`.

3. **Python `AdaptiveFFN`** (hass_block.py:650+): not read in detail. The
   C++ AdaptiveFFN may or may not match Python's legacy AdaptiveFFN. If
   they match, PATH A can keep C++ AdaptiveFFN as a legacy fallback while
   adding SlicedFFN as the default.

4. **Python `InternalLatentCoT` and `CoTAuxiliaryLoss`**: not read. The
   C++ `cot_vector.cpp` was not read either. Parity unverified.

5. **Python `GatedMerger`** (merger.py or wherever defined): not read.
   The C++ GatedMerger at hass_block.cpp:322-343 has a hardcoded `0.05 * cot`
   scaling — need to verify against Python.

6. **Python `HASSBlock` forward** (hass_block.py:1051+): not read in full.
   The C++ HASSBlock has a hardcoded `0.9 * router + 0.1 * gate` blend
   during training (hass_block.cpp:309) — need to verify if Python does
   the same or uses a different blend.

7. **`S4DKernel` usage**: is `ssm.cpp` dead code? `xorzen_model.cpp` and
   `hass_block.cpp` don't instantiate `SSMBlock`. Need to grep the rest of
   the codebase (igris.cpp, anime_model.cpp, bench.cpp, main.cpp) to confirm.

8. **LibTorch version**: does the build link against a LibTorch version
   that exposes `at::sdpa` and `torch::checkpoint::checkpointer`? Check
   `CMakeLists.txt` and the build logs.

9. **Python `models/zero/model.py`** forward (lines 1-1619): not read in
   full. The C++ `xorzen_model.cpp` forward was read, but the depth-mask
   application, the MoE reshape, and the CoT injection may differ in
   detail from Python.

10. **`config.cost_aware_routing` and `config.compute_budget`**: are these
    declared in the C++ `ModelConfig` struct? Need to read
    `include/xorzen/types.h` or wherever the struct is defined.

---

*End of audit. All claims are sourced from the files listed at the top of
each section. No path recommendation is made.*

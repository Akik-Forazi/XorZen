# HASS, SlicedFFN, Loss Investigation — Python vs C++

---

## 1. HASS Block

### 1.1 Pathway selection (top-k of 3 pathways per token)

**CONFIRMED MISMATCH — sparse dispatch vs dense compute.**

- Python `HASSBlock.forward` (`hass_block.py:1004-1157`) implements genuine **sparse pathway dispatch** when `routing_decision` is provided and `pathway_top_k < 3`:
  - Reads `top_k = getattr(self.config, 'pathway_top_k', 2)` (`hass_block.py:1084`).
  - Calls `sparse_pathway_dispatch(x_attn, path_probs, wrapped_fns, ['local','low_rank','ssm'], top_k, training=self.training, …)` (`hass_block.py:1110-1119`). Only the selected top-k pathways are invoked per token; pathways that no token selects are NOT called at all (`hass_block.py:1023-1024`). At training the straight-through estimator is used — forward uses the hard top-k mask, backward sees soft probs (`hass_block.py:1026-1030`).
- C++ `HASSBlockImpl::forward` (`hass_block.cpp:296-320`) **always computes all 3 pathways** unconditionally:
  ```cpp
  auto local_out = local->forward(xa, attention_mask);        // hass_block.cpp:301
  auto low_rank_out = low_rank->forward(xa);                  // hass_block.cpp:302
  auto ssm_out = ssm->forward(xa);                            // hass_block.cpp:303
  ```
  The `compute_all_pathways` flag (`hass_block.cpp:305`) only changes how the gate weights `w` are computed (router probs vs learned gate), never whether pathways are called. The top-k sparsity mechanism (`pathway_top_k`, `sparse_pathway_dispatch`) is **absent** from C++.

### 1.2 Local attention (SDPA vs matmul+softmax)

**Math equivalent; both apply ln_q/ln_k. Match (modulo fused vs unfused kernel).**

- Python `LocalAttentionPathway.forward` (`hass_block.py:91-207`):
  - Separate `q_proj/k_proj/v_proj` (`hass_block.py:52-54`), reshape to `[B,H,S,head_dim]` (`hass_block.py:116-118`).
  - Per-head LayerNorm `ln_q`/`ln_k` applied as `q = ln_q(q.transpose(1,2)).transpose(1,2)` (`hass_block.py:121-122`).
  - Builds additive `attn_bias` = causal mask + window mask + padding mask + position bias (`hass_block.py:141-186`).
  - Calls `F.scaled_dot_product_attention(q,k,v, attn_mask=attn_bias, dropout_p=…, is_causal=False)` (`hass_block.py:192-197`) — fused Flash/memory-efficient kernel, no `[B,H,S,S]` materialized.
- C++ `LocalAttentionPathwayImpl::forward` (`hass_block.cpp:32-66`):
  - Same q/k/v projections + reshape (`hass_block.cpp:38-40`).
  - Same `ln_q`/`ln_k` application with the same `.transpose(1,2).transpose(1,2)` pattern (`hass_block.cpp:42-43`).
  - When `XORZEN_ENABLE_FLASH_ATTN` is defined, calls `optimized::flash_attention_cpu(q,k,v,…)` (`hass_block.cpp:47`).
  - Otherwise falls back to **manual** `torch::matmul(q, k.transpose(-2,-1)) / sqrt(head_dim)` (`hass_block.cpp:50`), applies causal+window mask via `masked_fill(-inf)` (`hass_block.cpp:52-53`), `torch::softmax(scores,-1)` (`hass_block.cpp:60`), `torch::matmul(probs, v)` (`hass_block.cpp:62`).

**Parity harness fixture 06_attention**: PASS — `max_abs_error` within tolerance. The math is identical; only the kernel implementation differs.

### 1.3 Low-rank global pathway

**MAJOR ARCHITECTURAL DIVERGENCE.**

| Aspect | Python (`hass_block.py:309-368`) | C++ (`hass_block.cpp:81-93`) |
|---|---|---|
| Causal mask | lower-triangular `tril` | **none** |
| Long-seq fallback | chunked (T>512) | **none** |
| Second LayerNorm | none | `ln_low_rank` (fused with GELU) |
| Attention query | `low_rank` itself (Q=K=V) | learned `context_weights` parameter `[1,1,D]` |
| Softmax dim | -1 (over keys) | 1 |
| Extra parameter | — | `context_weights [1,1,D]` |

Python:
```python
low_rank = F.gelu(self.to_low_rank(x_norm))       # hass_block.py:325-326
causal_mask = torch.tril(torch.ones(S, S, dtype=bool))   # hass_block.py:331-333
scores = matmul(low_rank, low_rank.T) / sqrt(rk_dim)      # hass_block.py:337
scores = scores.masked_fill(~causal_mask, -inf)          # hass_block.py:339-341
attn_w = F.softmax(scores, dim=-1)                       # hass_block.py:342
global_context = matmul(attn_w, low_rank)                # hass_block.py:343
combined = low_rank + global_context                     # hass_block.py:362
```

C++:
```cpp
auto lr = fused_layernorm_gelu_simd(                     // hass_block.cpp:83-88
    to_low_rank->forward(xn), ln_low_rank->weight, ln_low_rank->bias, eps);
auto attn = softmax_simd(                                // hass_block.cpp:89-90
    matmul(lr, context_weights.transpose(-1,-2)) / sqrt(low_rank_dim), 1);
auto global_ctx = matmul(attn.transpose(-1,-2), lr).expand_as(lr);  // hass_block.cpp:91
return dropout(from_low_rank(lr + global_ctx));          // hass_block.cpp:92
```

These produce **different numerical outputs** — not a faithful port. The C++ adds an extra LayerNorm + a learned context query + non-causal attention. This will affect both training (gradient paths differ) and inference (output values differ).

### 1.4 SSM pathway

See `SSM_INVESTIGATION.md`. Four local fixes are required.

### 1.5 Pathway gates

**DIFFERENT MECHANISMS.**

- Python (`hass_block.py:897-903, 1064-1080`): `pathway_gate` module was **REMOVED in v0.5** (commented as dead code at `hass_block.py:897-903`). When `routing_decision is None`, uniform `1/3` weighting is used (`hass_block.py:1067-1070`). When `routing_decision` is provided, `path_probs` from the router are used directly (`hass_block.py:1076-1080`).
- C++ (`hass_block.cpp:287-289, 305-312`): `pathway_gate` is a `nn::Sequential(Linear(H,128) → LayerNorm(128) → GELU → Linear(128,3))` that **always runs**. When `routing_decision==nullptr || compute_all_pathways`, weights come from `softmax(pathway_gate(xa))` (`hass_block.cpp:306`). During training: `w = 0.9*path_probs + 0.1*gate_probs` (`hass_block.cpp:309`). At inference: `w = path_probs` directly (`hass_block.cpp:311`).

The C++ `pathway_gate` is **extra parameters not present in Python**; checkpoint loading will have orphan keys (`pathway_gate.*`).

### 1.6 Depth gating

**C++ has NO `forward_with_depth` equivalent.**

- Python: `HASSBlock.forward_with_depth` (`hass_block.py:1159-1215`) accepts a `depth_mask [B,T]` and applies `out = x + mask * (forward(x) - x)` so inactive tokens pass through unchanged (`hass_block.py:1213-1214`). `HASSBlockManager.forward` (`hass_block.py:1350-1409`) iterates blocks and skips those with `depth_mask.sum()==0` entirely (`hass_block.py:1385-1386`).
- C++: `HASSBlockImpl` has no `forward_with_depth` method. Depth masking is handled **at the model level** in `xorzen_model.cpp:102-113`:
  ```cpp
  for (int64_t i = 0; i < config.num_layers; ++i) {
      auto layer_mask = decision.depth_mask.select(-1, i);
      torch::Tensor block_out;
      if (!is_training() && !layer_mask.any().item<bool>()) {
          block_out = hidden;                                   // skip only at inference
      } else {
          block_out = block->forward(hidden, &decision, attention_mask);  // always computes block
      }
      hidden = block_out * layer_mask.unsqueeze(-1) + hidden * (1.0 - layer_mask.unsqueeze(-1));
  }
  ```
  - The skip optimization (`layer_mask.any()`) is **inference-only** — during training, every block is invoked regardless of `depth_mask`.
  - There is **no per-token residual gating inside the block**; the residual blend happens after the block returns.
  - Math is roughly equivalent for the residual blend (same `x + mask*(f(x)-x)` form), but the per-token pathway/FFN computation is **not** skipped inside the block. Compute savings are smaller than Python's `forward_with_depth`.

### 1.7 Aggregation of 3 pathway outputs

**Same algebraic formula, different `w` source.**

- Python (`hass_block.py:1071-1074, 1077-1080, 1093-1097`):
  ```python
  combined = sum(out * gate_weights[..., i:i+1] for i, out in enumerate(pathway_outputs))
  # or, with routing:
  combined = (
      local_out    * path_probs[..., 0:1] +
      low_rank_out * path_probs[..., 1:2] +
      ssm_out      * path_probs[..., 2:3]
  )
  ```
- C++ (`hass_block.cpp:313-315`):
  ```cpp
  auto combined = local_out    * w.slice(-1, 0, 1) +
                  low_rank_out * w.slice(-1, 1, 2) +
                  ssm_out      * w.slice(-1, 2, 3);
  ```
- Formula matches (`Σ wᵢ · outᵢ`), but `w` is computed differently (see §1.5).

### 1.8 FFN

- Python uses **`SlicedFFN`** by default (`hass_block.py:911-927`), with `max_width = int(config.hidden_size * 4.0)` (`hass_block.py:914`).
- C++ uses **`AdaptiveFFN`** with `ffn_multiplier=4.0` (`hass_block.cpp:290`), so `base_ffn_dim = hidden*4`.
- Detailed comparison in §2 below.

---

## 2. SlicedFFN (Python) vs AdaptiveFFN (C++)

### 2.1 Parameter shapes

- **Python `SlicedFFN`** (`sliced_ffn.py:93-97`):
  - `fc1 = nn.Linear(hidden_dim, max_width)` → weight `[max_width, hidden]`, bias `[max_width]`
  - `fc2 = nn.Linear(max_width, hidden_dim)` → weight `[hidden, max_width]`, bias `[hidden]`
  - `ln_input = nn.LayerNorm(hidden_dim)`
  - `ln_hidden = nn.LayerNorm(max_width)`
  - `max_width = int(config.hidden_size * 4.0)` (`hass_block.py:914`)
- **C++ `AdaptiveFFN`** (`hass_block.cpp:250-260`):
  - `base_ffn_dim = hidden * multiplier` with `multiplier=4.0` (`hass_block.cpp:251, 290`)
  - `fc1 = Linear(hidden, base_ffn_dim)` → weight `[base_ffn_dim, hidden]`, bias `[base_ffn_dim]`
  - `fc2 = Linear(base_ffn_dim, hidden)` → weight `[hidden, base_ffn_dim]`, bias `[hidden]`
  - `ln_input = LayerNorm(hidden)`
  - `ln_hidden = LayerNorm(base_ffn_dim)`

**`max_width` and `base_ffn_dim` are the same number** (`hidden * 4`). Parameter **shapes match exactly**.

### 2.2 Slicing behavior

- Python `SlicedFFN._forward_single_width` (`sliced_ffn.py:178-199`): **genuine weight slicing**:
  ```python
  fc1_w = self.fc1.weight[:w, :]     # [w, H]
  fc1_b = self.fc1.bias[:w]
  fc2_w = self.fc2.weight[:, :w]     # [H, w]
  fc2_b = self.fc2.bias              # [H]
  hidden = F.linear(x_norm, fc1_w, fc1_b)   # [B, T, w] — smaller matmul
  ```
  Per-token grouping via `_forward_per_token_width` (`sliced_ffn.py:201-246`) — tokens selecting width W_i are batched and processed with sliced matmuls; FLOPs scale with W_i.
- C++ `AdaptiveFFNImpl::forward` (`hass_block.cpp:262-278`): **no slicing whatsoever** — full matmuls always:
  ```cpp
  auto h = fc1->forward(xn);                  // full [B,T,base_ffn_dim]
  h = gelu_simd(h);
  h = ffn_dropout->forward(ln_hidden->forward(h));
  auto out = fc2->forward(h);                 // full [B,T,hidden]
  if (width_multiplier.defined()) out = out * width_multiplier;  // scale OUTPUT only
  ```
  `width_multiplier` is a scalar (or per-token) gain applied AFTER `fc2`, not a width selection.

### 2.3 Activation

- Python: configurable, default `F.gelu` (`sliced_ffn.py:101-108`).
- C++: configurable, default `gelu_simd` (`hass_block.cpp:266-272`).
- **Match** (both default to GELU; both support silu/relu). Note: C++ `gelu_simd` uses tanh-approximate GELU; Python `F.gelu` uses exact erf. Difference ~1e-3 in practice.

### 2.4 Width routing

- Python `SlicedFFN.forward` (`sliced_ffn.py:121-176`) accepts **per-token** width selection:
  - `width: int` (single width for whole batch, fast path),
  - `width_probs: [B,T,num_widths]` (training STE — hard argmax forward, soft backward; `sliced_ffn.py:150-169`),
  - `width_idx: [B,T]` (inference — per-token hard width grouping; `sliced_ffn.py:171-173`).
  - HASSBlock wires `width_probs` at training and `width_idx` at inference (`hass_block.py:1134-1143`).
- C++ `AdaptiveFFN::forward` accepts a **scalar/per-token `width_multiplier`** (`hass_block.cpp:262, 276`); HASSBlock passes `routing_decision->width_multiplier` (`hass_block.cpp:318`). The multiplier is `sum(probs * width_values) / hidden_size` (`routing.cpp:207`), a single scalar per token in `[0, ~4]`. **No width index, no per-token grouping.**

### 2.5 LayerNorm

- Python: `ln_input` always applied (`sliced_ffn.py:144`). `ln_hidden` applied **only when `w == max_width`** (`sliced_ffn.py:195-196`); skipped for partial widths (with a documented approximation note, `sliced_ffn.py:191-194, 257-261`).
- C++: `ln_input` always applied (`hass_block.cpp:263`). `ln_hidden` **always applied** (`hass_block.cpp:274`).
- **Mismatch** at partial widths (but at full width both apply `ln_hidden`).

### 2.6 Initialization

- Python `SlicedFFN._init_weights` (`sliced_ffn.py:112-116`): `xavier_uniform_(gain=1/sqrt(2))` for `fc1.weight, fc2.weight`; zeros for biases.
- C++ `AdaptiveFFNImpl` ctor (`hass_block.cpp:258-259`): `xavier_linear(fc1, 1/sqrt(2))`, `xavier_linear(fc2, 1/sqrt(2))` — same gain.
- **Match** (assuming `xavier_linear` uses `xavier_uniform_` underneath; same gain constant).

### 2.7 Parameter names (checkpoint compatibility)

| Component | Python `SlicedFFN` | C++ `AdaptiveFFN` |
|---|---|---|
| fc1 weight | `fc1.weight [max_width, H]` | `fc1.weight [base_ffn_dim, H]` ✓ |
| fc1 bias | `fc1.bias [max_width]` | `fc1.bias [base_ffn_dim]` ✓ |
| fc2 weight | `fc2.weight [H, max_width]` | `fc2.weight [H, base_ffn_dim]` ✓ |
| fc2 bias | `fc2.bias [H]` | `fc2.bias [H]` ✓ |
| ln_input | `ln_input.{weight,bias} [H]` | `ln_input.{weight,bias} [H]` ✓ |
| ln_hidden | `ln_hidden.{weight,bias} [max_width]` | `ln_hidden.{weight,bias} [base_ffn_dim]` ✓ |

**Names AND shapes match exactly.** A Python `SlicedFFN` state-dict can be loaded into a C++ `AdaptiveFFN` (and vice versa) without shape/naming conflicts at the FFN submodule level.

**Parity harness fixture 08_sliced_ffn**: PASS — at width=max, both produce identical output (within GELU tolerance).

### 2.8 Verdict

| Question | Answer |
|---|---|
| **Mathematically equivalent when width=max_width and multiplier=1.0?** | **YES** — at full width, both compute `dropout(fc2(ffn_dropout(ln_hidden(gelu(fc1(ln_input(x)))))))`. C++ multiplies by `width_multiplier=1.0` (no-op). The only sub-difference: C++ always applies `ln_hidden`; Python applies it at full width too. So the two are bit-equivalent up to fp noise. |
| **Parameter compatible?** | **YES** — names and shapes match exactly (§2.7). |
| **Checkpoint compatible?** | **YES, for the FFN submodule alone.** A Python `SlicedFFN` checkpoint loads cleanly into a C++ `AdaptiveFFN`. (Caveat: full-model checkpoint loading is blocked by other mismatches — extra `pathway_gate`, `context_weights`, `ln_low_rank` in C++; see §1.3, §1.5.) |
| **Adaptable (one can be made to behave like the other)?** | **UNCERTAIN** — at full width, behavior matches. At partial widths the two are architecturally different: Python slices weights (genuine FLOP reduction); C++ computes the full matmul and scales the output. The C++ `AdaptiveFFN` cannot replicate Python's per-token width sparsity without a rewrite (slicing + grouping). Going the other way (Python SlicedFFN emulating C++ AdaptiveFFN) is trivial: set `width=max_width` and ignore `width_multiplier`. |

---

## 3. Loss

### 3.1 Cross-entropy

- Python (`model.py:588-606`):
  ```python
  shift_logits = logits[..., :-1, :].contiguous()   # model.py:596
  shift_labels = labels[..., 1:].contiguous()        # model.py:597
  loss_fct = nn.CrossEntropyLoss(ignore_index=getattr(self.config, "pad_token_id", -100))  # model.py:600-602
  loss = loss_fct(shift_logits.view(-1, V), shift_labels.view(-1))                          # model.py:603-606
  ```
  - Shifted (predict token t+1 from token t). `ignore_index=pad_token_id`. Default reduction = `'mean'` (PyTorch default).
- C++ (`xorzen_model.cpp:127-135`):
  ```cpp
  if (labels.defined()) {
      auto shift_logits = logits.slice(1, 0, T - 1).contiguous();
      auto shift_labels = labels.slice(1, 1, T).contiguous();
      lm_loss = torch::nn::functional::cross_entropy(
          shift_logits.view({-1, config.vocab_size}),
          shift_labels.view({-1}),
          torch::nn::functional::CrossEntropyFuncOptions().ignore_index(config.pad_token_id));
  }
  ```
  - Same shift. `ignore_index=pad_token_id`. Default reduction = `'mean'` (LibTorch default).
- **Match** — same shift, same `ignore_index`, same reduction.

### 3.2 Routing loss

- Python `RoutingRegularizer.forward` (`routing.py:1691-1694`):
  ```python
  uncertainty_loss = decision.uncertainty.mean() * self.uncertainty_weight  # routing.py:1693
  return uncertainty_loss                                                    # routing.py:1694
  ```
  where `self.uncertainty_weight = getattr(config, 'routing_loss_weight', 0.01)` (`routing.py:1689`). Returns **only** the uncertainty term. The router's auxiliary losses (`load_balance_loss`, `router_z_loss`, `path_div_loss`, `width_div_loss`) are added **separately** in `model.py:617-626`, filtered by an explicit `_AUX_LOSS_KEYS` allowlist.
- C++ `RoutingRegularizerImpl::forward` (`routing.cpp:284-293`):
  ```cpp
  auto loss = decision.uncertainty.mean() * 0.0001;                       // routing.cpp:285
  loss = loss + path_diversity_loss(decision.path_probs) * 0.02;          // routing.cpp:286
  for (const auto& kv : decision.auxiliary) {
      if (kv.first.find("loss") != std::string::npos && kv.second.defined())
          loss = loss + kv.second.to(...);                                // routing.cpp:287-291
  }
  return loss;
  ```

**Three divergences**:
1. **Uncertainty weight**: Python `0.01` (config-driven) vs C++ `0.0001` (hardcoded). 100× difference.
2. **`path_div_loss` is added once in Python** (via the auxiliary dict, `routing.py:651`) and **twice in C++**: once explicitly at `routing.cpp:286`, and again via the auxiliary-dict loop at `routing.cpp:287-291` (because `decision.auxiliary["path_div_loss"]` was populated at `routing.cpp:170`). Double-counting bug.
3. **Auxiliary-loss filtering**: Python uses an explicit allowlist `_AUX_LOSS_KEYS = {'load_balance_loss','router_z_loss','path_div_loss','width_div_loss'}` (`model.py:617-620`). C++ uses a substring match `kv.first.find("loss") != npos` (`routing.cpp:288`) — looser, would catch any future key containing "loss".

### 3.3 Load-balance loss

There are **two different formulas** in play on each side.

**Python**:
- `load_balance_loss_switch` (`load_balance.py:151-177`) — the **Switch Transformer** formula `L = N · Σₑ fₑ·pₑ`, where `fₑ` = fraction of (token,slot) pairs dispatched to expert e (from one-hot indices, `load_balance.py:111-113`), `pₑ` = mean router prob (`load_balance.py:128`). Range `[1, N]`. Populated into `decision.auxiliary['load_balance_loss'] = self.lb_loss_weight * lb_loss` (`routing.py:644, 649`).
- `model.py:_compute_load_balance_loss` (`model.py:698-741`) — an **L2** variant: `load_balancing_weight * Σ(usage/(N·K) - 1/E)²` where `usage` is scatter-added expert weights. **This is zeroed by default** (`unify_load_balance=True`, `model.py:637-638`) to avoid double-counting with the Switch formula.

**C++**:
- `load_balance_loss` free function (`routing.cpp:11-17`): `(expert_probs.mean(0).mean(0) - 1/N)² · Σ` — an L2 loss using mean **router probs** (not load fractions). Used inside the router to populate `decision.auxiliary["load_balance_loss"] = 0.0001 * load_balance_loss(expert_probs, expert_indices, num_experts)` (`routing.cpp:168`). Note `expert_indices` is **unused** (signature accepts it but parameter is unnamed, `routing.cpp:12`).
- `XorzenModelImpl::compute_load_balance_loss` (`xorzen_model.cpp:211-219`): `load_balancing_weight * (usage/(N·K) - 1/E)² · Σ` where `usage` is scatter-added weights — matches the **Python L2 variant** (`_compute_load_balance_loss`).

**Net divergence**:
- The router-level `auxiliary['load_balance_loss']` in C++ uses router-probs L2; in Python it uses Switch `N·Σf·p`. **Different formula**, different gradient signal.
- The model-level `load_balance_loss` matches the Python L2 formula — but Python **disables it by default** (`unify_load_balance=True`), while C++ **always computes and adds it** (`xorzen_model.cpp:138, 140-141`). So in default config, C++ double-counts load balancing (router-side L2 + model-side L2), while Python uses only the Switch formula.

### 3.4 CoT consistency loss

- Python `_compute_cot_consistency_loss` (`model.py:743-766`):
  ```python
  cot_diff = cot_vector[:, 1:, :] - cot_vector[:, :-1, :]   # model.py:761
  consistency_loss = torch.mean(cot_diff ** 2)               # model.py:764
  return self.config.cot_consistency_weight * consistency_loss  # model.py:766
  ```
  Penalizes L2 of consecutive-token CoT differences. Computed when `_cot_enabled` (`model.py:646-647`).
- C++ (`xorzen_model.cpp:139, 143-148`):
  ```cpp
  torch::Tensor cot_consistency = torch::zeros({}, hidden.options());   // line 139
  if (cot->cot_enabled) {
      auto cot_losses = cot_loss_head->forward(cot_vector, labels, {}, attention_mask);
      cot_consistency = cot_losses["consistency"];                        // line 145
      auto total_cot_aux = cot_losses["total_auxiliary"];
      loss = loss + total_cot_aux;                                       // line 147
  }
  ```
  - `cot_consistency` is read out from `cot_loss_head->forward(...)` but **never added to `loss`** — only `total_cot_aux` is added. The Python `_compute_cot_consistency_loss` formula (consecutive-token L2) is **not implemented** in C++; instead a separate `CoTAuxiliaryLoss` head computes some other bundle of CoT losses and adds `total_auxiliary`. The "consistency" tensor reported in `ModelOutput` is just a tracking value.
  - **Divergence**: the explicit `mean((cot[t+1]-cot[t])²) · cot_consistency_weight` term in Python has no direct C++ counterpart.

### 3.5 Auxiliary-loss filtering / total summation

| Term | Python | C++ |
|---|---|---|
| LM CE | `+ loss` | `+ lm_loss` |
| uncertainty loss | `+ 0.01 · mean(uncertainty)` (in `routing_loss`) | `+ 0.0001 · mean(uncertainty)` (in `routing_loss`) |
| Switch LB | `+ lb_loss_weight · N·Σf·p` (in `routing_loss`) | **absent** — C++ uses L2 instead |
| Router L2 LB | (not present in router) | `+ 0.0001 · Σ(p̄ - 1/N)²` (in `routing_loss`, via aux) |
| Model-level L2 LB | `+ 0` (default, unify=True) | `+ load_balancing_weight · Σ(usage/NK - 1/N)²` (always) |
| router_z_loss | `+ z_loss_weight · logsumexp²` (in `routing_loss`) | `+ 0.0001 · logsumexp²` (in `routing_loss`, via aux) |
| path_div_loss | `+ path_div_weight · Σ(p̄_path - 1/3)²` (once) | `+ 0.02 · Σ(p̄_path - 1/3)²` **twice** (bug) |
| width_div_loss | `+ width_div_weight · Σ(p̄_width - 1/N)²` if `num_widths≥2` | **absent** — C++ router never populates `width_div_loss` |
| CoT consistency | `+ cot_consistency_weight · mean((Δcot)²)` | **not added** (reported only) |
| CoT aux head | (not present) | `+ cot_loss_head.total_auxiliary` (replaces CoT consistency) |

### 3.6 Configuration flags

- Python config fields used in loss computation:
  - `pad_token_id` (`model.py:601`) — CE ignore_index.
  - `routing_loss_weight` (default 0.01) — `RoutingRegularizer.uncertainty_weight` (`routing.py:1689`).
  - `lb_loss_weight`, `z_loss_weight`, `path_div_weight`, `width_div_weight` (`routing.py:649-651, 658`) — auxiliary-loss weights inside the router.
  - `unify_load_balance` (default True) (`model.py:637`) — toggles model-level L2 LB loss.
  - `load_balancing_weight` (`model.py:741`) — model-level L2 LB weight.
  - `cot_consistency_weight` (`model.py:766`) — CoT consistency weight.
  - `_cot_enabled` (`model.py:646`) — gates CoT consistency.
- C++ config fields used:
  - `pad_token_id` (`xorzen_model.cpp:134`) — CE ignore_index.
  - `load_balancing_weight` (`xorzen_model.cpp:218`) — model-level L2 LB weight.
  - `expert_count`, `top_k_experts` (`xorzen_model.cpp:213-217`) — for LB normalization.
  - Hardcoded constants for everything else: `0.0001` (uncertainty, z_loss, LB inside router), `0.02` (path_div) — see `routing.cpp:168-170, 285-286`.
  - No `unify_load_balance` flag — always computes both router-side and model-side LB losses.
  - No `cot_consistency_weight` — CoT consistency is not added.
- **Mismatch**: most loss weights are config-driven in Python but hardcoded in C++. Tuning the Python config will not affect C++ behavior.

### 3.7 Does C++ even implement a loss function?

**Yes**, but with significant deviations from the Python loss. C++ `XorzenModelImpl::forward` (`xorzen_model.cpp:63-171`) computes:
1. CE LM loss (matches Python) — `xorzen_model.cpp:127-135`.
2. `routing_loss` via `RoutingRegularizer` — `xorzen_model.cpp:137` (mismatches: wrong uncertainty weight, double-counted path_div, substring filter).
3. `load_balance` via `compute_load_balance_loss` — `xorzen_model.cpp:138, 211-219` (always-on L2, whereas Python disables it by default).
4. CoT consistency is **reported as 0** (`xorzen_model.cpp:139`) — not added to loss. Instead `total_cot_auxiliary` from a separate `CoTAuxiliaryLoss` head is added (`xorzen_model.cpp:143-148`).
5. Final: `loss = lm_loss + routing_loss + load_balance + total_cot_auxiliary` (when CoT enabled) — `xorzen_model.cpp:140-141, 147`.

The C++ loss is **not numerically equivalent** to the Python loss for any non-trivial config. Gradients will diverge.

---

## 4. Bonus finding — C++ has TWO `GatedMergerImpl` definitions

- `xorzen/hass.h:92-104` declares a `GatedMergerImpl` with members `gate`, `cot_proj`, `norm` — defined in `hass_block.cpp:322-343`. Uses **2 gates** (HASS, MoE) with a hardcoded `0.05 * cot` additive term, no dropout, no `gate_controller`.
- `xorzen/merger.h:40-65` declares **another** `GatedMergerImpl` with members `gate_controller`, `cot_proj`, `output_norm`, `dropout_` — defined in `merger.cpp:54-133`. Uses **3 gates** (HASS, MoE, CoT) via softmax, matches the Python `GatedMerger` (`merger.py:152-237`).
- `xorzen_model.cpp` includes `xorzen/model.h` which includes `xorzen/hass.h` (and **not** `xorzen/merger.h`). It instantiates `GatedMerger(config)` (`xorzen_model.cpp:24`), so it uses the **2-gate** version from `hass_block.cpp`.
- The proper 3-gate `merger.cpp` implementation is effectively **dead code** — not linked into the model.
- Python `model.py:57, 180` uses `xorzenMergerGate(config)` which wraps the 3-gate `GatedMerger` (`merger.py:271-297`).

**Parity harness fixture 11_merger**: ARCH_MISMATCH — confirmed.

**Impact**: C++ merger is architecturally inferior (no learned CoT gate, no dropout, hardcoded 0.05 weight) and structurally different from Python. Parameter names differ (`gate_controller.*` in Python vs `gate.*` in C++-hass.h), so checkpoint loading is broken at the merger submodule too.

---

## 5. Summary of action items (for a future fix — not part of this read-only investigation)

1. **C++ HASS**: implement top-k sparse pathway dispatch (`pathway_top_k`, `sparse_pathway_dispatch` equivalent) instead of always computing all 3 pathways.
2. **C++ LowRankGlobalPathway**: add causal lower-triangular mask, chunked fallback for T>512, remove `ln_low_rank` and `context_weights` (or port them to Python to match — pick one direction).
3. **C++ AdaptiveFFN**: implement genuine weight slicing + per-token width grouping, or rename to clarify it is not a SlicedFFN port. At minimum, add an `ln_hidden` skip for `w < max_width`.
4. **C++ HASSBlock**: remove the always-on `pathway_gate` Sequential to match Python v0.5 (or re-add it to Python if keeping the C++ behavior).
5. **C++ HASSBlock**: add a `forward_with_depth` method (or at least respect `depth_mask` inside the block during training, not just inference).
6. **C++ GatedMerger**: delete the `hass.h`/`hass_block.cpp` version; use the proper 3-gate version from `merger.cpp`. Update `xorzen_model.cpp` to use `XorzenMergerGate` (or `GatedMerger` from `merger.h`).
7. **C++ Loss**:
   - Fix the double-counted `path_div_loss` in `RoutingRegularizerImpl::forward` (`routing.cpp:286`).
   - Use config-driven weights instead of hardcoded `0.0001`/`0.02`.
   - Implement the Switch-formula `load_balance_loss` (the `load_balance_loss_switch` Python function) in the router.
   - Add `unify_load_balance` toggle; otherwise C++ always double-counts LB.
   - Implement `cot_consistency_loss = cot_consistency_weight · mean((cot[t+1]-cot[t])²)` and add it to `loss`.
   - Add `width_div_loss` to the C++ router.

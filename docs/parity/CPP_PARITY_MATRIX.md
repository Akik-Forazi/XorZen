# CPP_PARITY_MATRIX — XorZen C++ vs Python Component-Level Parity

**Statuses** (exact, no "mostly compatible"):
- `MATCHED` — C++ output matches Python output within documented tolerance for the deterministic fixture
- `PARTIAL` — C++ runs the component but produces measurably different output (cause documented)
- `MISMATCH` — C++ implementation is architecturally different; cannot reproduce Python behavior without code changes
- `MISSING` — C++ has no equivalent component
- `UNKNOWN` — component exists but has not been verified at runtime

**Runtime verified**: YES = the C++ parity harness was executed against a deterministic Python fixture and the result was compared. NO = not yet executed.

---

## Parity matrix

| Component | Python implementation | C++ implementation | Status | Exact difference | Port effort | Reusable? | Runtime verified? |
| --------- | --------------------- | ------------------ | ------ | ---------------- | ----------- | --------- | ----------------- |
| config | `xorzen/config.py` `ModelConfig` dataclass | `xorzen.cpp/include/xorzen/variants.h` `ModelConfig` struct | MATCHED | All size-tier values match. C++ missing `cost_aware_routing` and `compute_budget` fields. | LOW | YES | YES (fixture 09 router exercises config) |
| initialization | `torch.manual_seed(42)` + zero_tiny_23k constructor | `xorzen_model.cpp` ctor | PARTIAL | C++ uses `xavier_uniform_(gain=0.5)` for ALL router Linears; Python uses `gain=0.1` for the four routers (`routing.py:432`). | LOW | YES (fix init gain) | NO (no init-parity fixture) |
| embeddings | `nn.Embedding(vocab, hidden, padding_idx)` + `nn.Embedding(ctx, hidden)` summed | `torch::nn::Embedding` + `nn::Embedding` summed (`xorzen_model.cpp:80`) | MATCHED | None. Same params, same sum. | LOW | YES | YES — fixture 13 PASS |
| normalization | `nn.LayerNorm(hidden)` | `torch::nn::LayerNorm` | MATCHED | None. | LOW | YES | YES — fixture 01 PASS |
| RoPE / positional | LEARNED position embeddings (`nn.Embedding(ctx, hidden)`), NOT sinusoidal RoPE | LEARNED position embeddings (`nn::Embedding`) | MATCHED | None (note: name "RoPE" is a misnomer; both use learned embeddings). | LOW | YES | YES — fixture 02 PASS |
| QKV projections | Separate `q_proj/k_proj/v_proj` (`hass_block.py:52-54`) | Separate `q_proj/k_proj/v_proj` (`hass_block.cpp:18-20`) | MATCHED | None. (Fused QKV was reverted in Python for checkpoint compat.) | LOW | YES | YES — fixtures 03/04/05 PASS |
| attention | `F.scaled_dot_product_attention` (SDPA, fused kernel) | `matmul + softmax` (or `flash_attention_cpu` if `XORZEN_ENABLE_FLASH_ATTN` defined) | MATCHED | Same math. C++ adds `ln_q`/`ln_k` (Python also has these). | LOW | YES | YES — fixture 06 PASS (abs diff ≤ 1e-5) |
| HASS block | `HASSBlock` with sparse pathway dispatch via `sparse_pathway_dispatch` (`hass_block.py:1004-1157`) | `HASSBlockImpl` always computes all 3 pathways (`hass_block.cpp:296-320`) | MISMATCH | C++ missing top-k sparse pathway dispatch. Always runs all 3 pathways. Compute is 3× Python's (no sparsity). | MEDIUM | NO (must add sparse dispatch) | NO |
| local attention | `LocalAttentionPathway` with SDPA + causal mask + window mask | `LocalAttentionPathwayImpl` with `matmul+softmax` + causal mask + window mask | MATCHED | Same math. | LOW | YES | YES — fixture 06 PASS |
| low-rank pathway | `LowRankGlobalPathway` causal self-attention with tril mask + chunked fallback for T>512, no `context_weights`, only `ln_input` (`hass_block.py:309-368`) | `LowRankGlobalPathwayImpl` learned `context_weights` query, NO causal mask, NO chunking, extra `ln_low_rank` (`hass_block.cpp:81-93`) | MISMATCH | Architecturally different. Different math, different params, different attention pattern. | HIGH | NO (must rewrite) | NO |
| SSM | `SSMPathway` with diagonal real-A, ZOH on both A and B, scan returns `h_t` only (`hass_block.py:394-720`) | `SSMPathwayImpl` with diagonal real-A, no ZOH on B (uses Bv directly), C applied INSIDE scan, conv uses center-padding (`hass_block.cpp:227-248`) | MISMATCH | 4 divergences: (1) no B_bar ZOH, (2) conv center-padding, (3) C inside scan, (4) LN order. Causes future-token leakage + ~45% B contribution error. | MEDIUM (4 local fixes, ~30 lines) | YES (skeleton reusable) | YES — fixture 14 FAIL (max_abs=0.045) |
| SSM scan | `select_scan` → `parallel_scan` (Blelloch) for T>64, `sequential_scan` for T≤64 (`ssm_scan.py`) | `SSMScanFunction` serial loop, no chunking, no init_state (`hass_block.cpp:122-159`) | MATCHED (math) | Same recurrence math. C++ lacks init_state param and parallelism — production impact zero (HASS never passes init_state). | LOW | YES | YES — fixture 07 PASS |
| SlicedFFN | `SlicedFFN` with genuine weight slicing + per-token width grouping (`sliced_ffn.py`) | `AdaptiveFFN` computes full matmul, scales output by `width_multiplier` (`hass_block.cpp:250-278`) | PARTIAL | At width=max, math matches. At partial widths, C++ does NOT slice (no FLOP reduction) — architecturally different. | MEDIUM | UNCERTAIN | YES — fixture 08 PASS (only at width=max) |
| pathway dispatch | `sparse_pathway_dispatch` (`hass_block.py`) — only top-k pathways invoked per token | Always all 3 pathways (`hass_block.cpp:301-303`) | MISSING | Sparse dispatch entirely absent in C++. | MEDIUM | NO | NO |
| depth routing | `_route_depth` with cost-aware modulation (`routing.py:546-588, 662-725`) | `route_depth` with NO cost-aware modulation (`routing.cpp:176-197`) | PARTIAL | C++ missing `depth_shift`, `depth_layer_bias`. Eval Gumbel noise (`routing.py:685-692`) absent. | MEDIUM | YES (add cost-aware block) | YES — fixture 09 FAIL (depth_probs abs=0.047, depth_mask MISMATCH) |
| width routing | `_route_width` with cost-aware width_bias_axis (`routing.py:564-568, 727-777`) | `route_width` with NO cost-aware modulation (`routing.cpp:199-209`) | PARTIAL | Same core math. C++ missing `width_bias_axis` and eval Gumbel noise. | MEDIUM | YES | YES — fixture 09 PASS (only because tiny_23k has 1 width choice, so bias is zero) |
| pathway routing | `_route_path` with cost-aware path_bias_axis (`routing.py:572-576, 779-830`) | `route_path` with NO cost-aware modulation (`routing.cpp:211-219`) | PARTIAL | Same `prior_weight = 0.5 * 0.995^step` formula. C++ missing `path_bias_axis` and eval Gumbel noise. | MEDIUM | YES | YES — fixture 09 FAIL (path_probs abs=0.121) |
| expert routing | `_route_experts` with top-k + capacity (`routing.py:832-902`) | `route_experts` with top-k + capacity (`routing.cpp:221-244`) | MATCHED | Same math. C++ missing eval Gumbel noise (minor). | LOW | YES | YES — fixture 09 PASS (only 1 expert in tiny_23k) |
| cost-aware routing | Compute-budget modulation on depth/width/path axes (`routing.py:546-588`) | NOT PRESENT | MISSING | Entire block absent in C++. | MEDIUM | NO | NO |
| MoE | `ShardedExpertFabric` with disk-sharded LRU cache (`zmoe.py`) | `ShardedExpertFabric` with disk-sharded LRU cache (`zmoe.cpp`) | PARTIAL | Different IO. Same expert math. | MEDIUM | YES (disk-shard infra) | YES — fixture 10 PASS (aggregation only) |
| CoT | `InternalLatentCoT` (`cot_vector.py`) | `InternalLatentCoT` (`cot_vector.cpp`) | PARTIAL | Structural match. C++ has extra `CoTAuxiliaryLoss` head (`cot_loss_head.*`) that Python does not instantiate. | MEDIUM | YES | NO |
| merger | `xorzenMergerGate` wrapping 3-gate `GatedMerger` (HASS+MoE+CoT softmax, SiLU) (`merger.py:152-237`) | Two competing `GatedMergerImpl`! `hass_block.cpp:322-343` (2-gate, GELU, used) vs `merger.cpp:54-133` (3-gate, SiLU, dead code) | MISMATCH | Active C++ merger is 2-gate with hardcoded 0.05*cot. The proper 3-gate version exists in `merger.cpp` but is NOT linked into `xorzen_model.cpp`. Param names differ (`gate.*` vs `gate_controller.*`). | LOW (just wire merger.cpp in) | YES (merger.cpp is correct) | YES — fixture 11 ARCH_MISMATCH |
| LM head | `nn.Linear(hidden, vocab, bias=False)` tied to embeddings | `nn::Linear(hidden, vocab, bias=False)` tied | MATCHED | None. | LOW | YES | YES — fixture 12 PASS |
| loss | CE + uncertainty + Switch LB + router_z + path_div + width_div + cot_consistency (`model.py:588-766`) | CE + uncertainty(0.0001) + L2 LB (always on) + router_z + path_div(0.02, double-counted) + cot_aux_head (`xorzen_model.cpp:125-148`) | MISMATCH | 7 distinct divergences: wrong uncertainty weight (0.0001 vs 0.01), double-counted path_div, missing width_div, missing Switch LB formula, missing CoT consistency term, hardcoded weights, substring filter. | HIGH | NO (must rewrite) | NO |
| auxiliary losses | `_AUX_LOSS_KEYS = {load_balance_loss, router_z_loss, path_div_loss, width_div_loss}` allowlist (`model.py:617-620`) | substring match `kv.first.find("loss") != npos` (`routing.cpp:288`) | MISMATCH | C++ filter is looser; risks catching future non-loss keys. | LOW | NO (rewrite filter) | NO |
| generation | greedy, temperature, top-k, top-p, repetition penalty, EOS, max_new_tokens, batching, beam search (`model.py:871-1283`) | Greedy autoregressive loop only (per existing audit) | PARTIAL | C++ has only greedy. Missing temperature/top-k/top-p/rep penalty/beam. Beam search bug fixed in Python (returns highest-scoring beam). | MEDIUM | NO | YES — Python beam regression test PASS |
| tokenizer | BPE tokenizer (`xorzen/tokenizer/`) | `BebpeTokenizer` (`xorzen.cpp/src/tokenizer/bebpe_tokenizer.cpp`) | PARTIAL | Different vocab formats. C++ uses .bin files; Python uses .json. | MEDIUM | UNCERTAIN | NO |
| checkpoint loading | `torch.save(checkpoint, path)` with dict wrapped under `'model_state_dict'` key (`checkpoint.py:255-288`) | `torch::serialize::OutputArchive` flat at root, no `model_state_dict` wrapper (`checkpoint.cpp:99-123`) | MISMATCH | Container format incompatible. Cannot load Python checkpoint into C++ directly. | MEDIUM | NO (need converter) | NO |
| checkpoint saving | Same `torch.save` with metadata (`checkpoint.py:255-288`) | `archive.save_to(path)` with different metadata schema (`checkpoint.cpp:99-123`) | MISMATCH | Different metadata schema, different version strings (0.2.2 vs 0.2.5). | MEDIUM | NO | NO |
| serialization (.xorm v2) | AES-256-GCM + HMAC-SHA256 + license (`xorm_crypto.py`) | NOT PRESENT in C++ | MISSING | Encryption system is Python-only. | HIGH | NO | NO |
| sparse pathway dispatch | `sparse_pathway_dispatch` top-k token routing (`hass_block.py`) | NOT PRESENT | MISSING | See "pathway dispatch" row above. | MEDIUM | NO | NO |
| depth masking (training) | `forward_with_depth` per-token residual gating inside block (`hass_block.py:1159-1215`) | Block-level skip at inference only (`xorzen_model.cpp:102-113`) | PARTIAL | C++ skips whole blocks at inference when `depth_mask.any()==False`. No per-token residual gating inside block during training. | MEDIUM | PARTIAL | NO |
| expert usage tracking | `expert_usage` and `expert_load` attributes via `scatter_add_` (`routing.py:324-325, 956-986`) | NOT PRESENT | MISSING | C++ has no per-expert running stats. | LOW | NO | NO |
| character_router | NOT PRESENT in Python | `Linear→LN→GELU→Linear→Sigmoid` over `max_characters` (`routing.cpp:72-75`) | EXTRA | C++ has 6 extra parameter tensors that Python doesn't have. Pollutes checkpoint compatibility. | LOW (delete) | N/A | NO |

---

## Summary counts

| Status | Count |
| ------ | ----- |
| MATCHED | 9 |
| PARTIAL | 9 |
| MISMATCH | 7 |
| MISSING | 5 |
| EXTRA (C++-only) | 1 |
| **Total** | **31** |

## Runtime-verified breakdown

| | Count |
|---|---|
| Runtime verified (PASS) | 11 |
| Runtime verified (FAIL — informative divergence) | 2 |
| Runtime verified (ARCH_MISMATCH) | 1 |
| Not runtime verified | 17 |

The 11 PASS + 2 informative FAIL + 1 ARCH_MISMATCH come from the C++ parity harness (`tests/cpp_parity/compare.py`). The remaining 17 are documented via static code analysis but lack a runtime fixture — these are mostly whole-model behaviors (loss, generation, checkpoint) that would require a much larger fixture to verify.

---

## What this means

**Reusable as-is (9 components)**: config, embeddings, normalization, positional embedding, QKV projections, attention (local), SSM scan (bare recurrence), MoE aggregation, LM head, character_router (delete).

**Reusable with small fixes (5 components)**: SSM pathway (4 local fixes), SlicedFFN (works at max width, needs slicing for partial), router (add cost-aware modulation, fix init gain, add width_div_loss, fix LB formula, fix double-counting), depth/width/path routing (add cost-aware + eval noise), depth masking (add training-time per-token gating).

**Must be rewritten (5 components)**: low-rank pathway (architecturally different), merger (replace 2-gate with 3-gate from `merger.cpp`), loss (7 divergences), auxiliary-loss filter (rewrite as allowlist), generation (add 5 missing strategies).

**Missing entirely (4 components)**: sparse pathway dispatch, cost-aware routing, expert usage tracking, .xorm v2 encryption.

**Container incompatibility (1 component)**: checkpoint format (Python wraps under `model_state_dict`, C++ writes flat) — needs converter script.

# SSM Investigation — Python vs C++

**Verdict**: The C++ `SSMPathwayImpl` can be **ADAPTED** (4 local fixes). It does NOT need to be replaced. The structural skeleton matches Python. The separate C++ `S4DKernel` (`ssm.cpp`) is dead code mirroring the dead Python `xorzen/model/ssm.py` — neither is referenced by HASS.

**Production path**: `HASSBlock.forward` → `pathways['ssm'].forward_parallel(x_attn)` (Python `hass_block.py:994, 1061, 1092`). The C++ equivalent is `HASSBlockImpl::forward` → `ssm->forward(xa)` (`hass_block.cpp:303`). Note: Python production uses `forward_parallel`, **not** `forward` — the scan is `parallel_scan` (Blelloch) for T>64, not `chunked_scan`.

---

## Dimension 1 — A parameterization

| | Python `SSMPathway` | C++ `SSMPathwayImpl` | C++ `S4DKernel` (dead) |
|---|---|---|---|
| Storage | `A_log = nn.Parameter(torch.zeros(state_dim))` (`hass_block.py:419`) | `A_log = register_parameter("A_log", torch::zeros({state_dim}))` (`hass_block.cpp:98`) | `A_log = [hidden, N/2]`, `A_im = [hidden, N/2]` (`ssm.cpp:38-39`) |
| Shape | `[state_dim]` real | `[state_dim]` real — **MATCHES** | `[hidden, state_dim/2]` complex |
| Sign convention | `a = -torch.exp(self.A_log)` (`hass_block.py:544, 603`) → negative real | `a = -torch::exp(A_log)` (`hass_block.cpp:240`) → negative real — **MATCHES** | `A = -exp(A_log) + j·A_im` (`ssm.cpp:77-79`) → complex |

**Conclusion**: the C++ `SSMPathwayImpl` parameterizes A identically to Python — real, diagonal, `[state_dim]`, negative via `-exp(A_log)`. The complex-A `S4DKernel` is dead code in both languages.

---

## Dimension 2 — Discretization (ZOH)

### Python production — `discretize_zoh` (`ssm_scan.py:62-106`)

```python
z = dt * A                              # [B, T, N]
A_bar = exp(z)                          # in (0, 1) for stable A
small = |z| < 1e-4
exact  = (A_bar - 1) / z                # = (exp(z)-1)/z, stable for |z| ≥ eps
taylor = 1 + z/2 + z²/6                 # second-order Taylor for small |z|
B_bar_div = where(small, taylor, exact)
B_bar = B_bar_div * dt * B              # = ((exp(z)-1)/A) * B
```

### C++ `SSMPathwayImpl::forward` (`hass_block.cpp:239-241`)

```cpp
auto dt = torch::softplus(dt_proj->forward(xn));             // [B,T,N]
auto a  = -torch::exp(A_log);                                // [N]
auto Ab = torch::exp(dt * a.view({1, 1, state_dim}));        // A_bar ✓
```

`A_bar` is computed correctly. **`B_bar` is NOT computed at all** — the function passes raw `Bv` straight into `SSMScanFunction::apply(Ab, Bv, C)` at `hass_block.cpp:244`. **DIVERGENCE #1.**

For typical `|dt·a|` ≈ `softplus(0)·exp(0) ≈ 0.69·1 = 0.69` the missing factor `((exp(z)−1)/z)` is roughly `1.45`, so the C++ B contribution is off by ~45% per token.

---

## Dimension 3 — B_bar handling

| | Python | C++ |
|---|---|---|
| B_bar formula | `B_bar = ((A_bar - 1)/a) * Bv` (`ssm_scan.py:104-105`) | **None** — `Bv` is passed directly as the second arg of `SSMScanFunction::apply` (`hass_block.cpp:244`) |
| Inside the scan | `state = A_bar[:, t] * state + B_bar[:, t]` (`ssm_scan.py:155`) | `h_next = ab * h[d] + bv` (`hass_block.cpp:149`) — `bv` is raw `B_proj(x_norm * input_gate)` output |

So the C++ scan is mathematically:
```
h_t = A_bar_t · h_{t-1} + B_proj(xn ⊙ σ(input_gate))      # WRONG — missing ZOH factor
```
while Python is:
```
h_t = A_bar_t · h_{t-1} + ((A_bar_t − 1)/a) · Bv_t         # CORRECT ZOH
```

**DIVERGENCE #2.**

---

## Dimension 4 — Scan algorithm

| | Python `forward_parallel` (production) | C++ `SSMScanFunction` |
|---|---|---|
| Method | `parallel_scan` (Blelloch/Hillis-Steele associative scan, `log2(T)` rounds) for T>64; `sequential_scan` for T≤64 (`hass_block.py:607-610`) | Single serial triple-nested loop, no chunking (`hass_block.cpp:141-154`) |
| Parallel depth | O(log T) | O(T) |
| Memory | O(T log T) with per-step `.clone()` for autograd safety | O(T·N) plus a `std::vector<float> h(D, 0.0f)` accumulator |
| Autograd | Standard PyTorch graph | Custom `torch::autograd::Function<SSMScanFunction>` with hand-written forward (`hass_block.cpp:120-159`) and backward (`hass_block.cpp:161-224`) |

The C++ serial loop has zero Python overhead. The `SSMScanFunction` loop in `hass_block.cpp:141-154` is **not** OpenMP-parallelized. For parity at small T (e.g. fixture T=8) this is fine. For production latency at T=2048 the C++ path will be slower than Python `parallel_scan` on GPU. **No correctness impact** — only throughput.

---

## Dimension 5 — Where C is applied

| | Python | C++ |
|---|---|---|
| Scan returns | `states` = `h_t` only (`ssm_scan.py:157, 225, 273`) — pure state, no C | `p_states[offset + d] = c * h_next` (`hass_block.cpp:151`) — **C is applied inside the scan loop, returned tensor is `C_t · h_t`** |
| C application | `ssm_output = C * states` **after** the scan **and after** `ln_state` (`hass_block.py:559, 613`) | C is fused into the scan output — no separate "C · states" step in `forward` |
| Backward handles C | Yes — autograd through `C * states` | Yes — `p_grad_C[offset + d] = grad_out * h_t` and `grad_h_t = grad_out * c + dh[d]` (`hass_block.cpp:211, 213`) |

**DIVERGENCE #3.**

---

## Dimension 6 — LayerNorm order

### Python (`hass_block.py:555-565`, identical in `forward_parallel:612-616`)

```
states      = select_scan(A_bar, B_bar)        # h_t            [B,T,N]
states      = self.ln_state(states)            # LN(h)          [B,T,N]
ssm_output  = C * states                       # C · LN(h)      [B,T,N]
ssm_output  = self.D_proj(ssm_output)          # D_proj(C·LN(h))  [B,T,H]
output      = ssm_output * gate
```

### C++ (`hass_block.cpp:244-247`)

```cpp
auto states = SSMScanFunction::apply(Ab, Bv, C);             // = C · h_t       [B,T,N]
auto out    = D_proj->forward(ln_state->forward(states))     // D_proj(LN(C·h))   [B,T,H]
             * gate;
```

So Python applies LN to `h_t`, then multiplies by C, then `D_proj`. C++ applies LN to `C·h_t`, then `D_proj`. Because LayerNorm is `(x-μ)/σ · γ + β`, `LN(h)` and `LN(C·h)` are **not** equal in general — the per-token mean/variance differ when C is per-token. **DIVERGENCE #4.**

---

## Dimension 7 — Conv1d padding

| | Python (`hass_block.py:432-437`, applied at `525-531` and `589-594`) | C++ (`hass_block.cpp:105-106`, applied at `231-233`) |
|---|---|---|
| Padding | `kernel_size - 1` = 2 for k=3 | `kernel_size / 2` = 1 for k=3 |
| Truncation | `x_conv[:, :, :seq_len]` — drop the right-side extra samples | none |
| Effective | **Causal left-padding** (two zeros on the left, none on the right; truncate to original T) | **Center padding** (one zero on each side, length preserved) |
| Source comment | docstring explicitly says "CAUSAL left-padding … fixes the design limitation documented in earlier versions that used center-padding" | comment says `padding(kernel_size / 2)` — exactly the "earlier version" limitation Python claims to have fixed |

**DIVERGENCE #5.** This causes future-token leakage in the C++ path (each output position t sees input at t+1). For autoregressive use this is a **correctness bug**, not just a numerical mismatch.

---

## Dimension 8 — init_state parameter

| Function | Accepts init_state? |
|---|---|
| Python `sequential_scan(A_bar, B_bar, init_state=None)` (`ssm_scan.py:132-157`) | **Yes** — `[B, N]`, used as initial `state` |
| Python `parallel_scan(A_bar, B_bar, init_state=None)` (`ssm_scan.py:160-225`) | **Yes** — folded into position 0: `b[:, 0, :] = a[:, 0, :] * init_state + b[:, 0, :]` (`ssm_scan.py:208-210`) |
| Python `chunked_scan(A_bar, B_bar, init_state=None, chunk_size=256)` (`ssm_scan.py:228-273`) | **Yes** — initial state, carries last state across chunks (`ssm_scan.py:255-270`) |
| Python `SSMPathway.forward_parallel` call site (`hass_block.py:608-610`) | `init_state` not passed → defaults to None → zeros |
| C++ `SSMScanFunction::forward` (`hass_block.cpp:122-159`) | **No** — always `std::vector<float> h(D, 0.0f)` (`hass_block.cpp:142`) |
| C++ `SSMPathwayImpl::forward` (`hass_block.cpp:227-248`) | **No** — no init_state plumbing |

**Production impact: zero.** HASS production never passes `init_state`. The mismatch only matters if someone wants to use the scan as a primitive elsewhere (e.g. chunked cross-chunk carry).

---

## Dimension 9 — Chunking for long sequences

| | Python | C++ |
|---|---|---|
| Production forward (`forward_parallel`) | `parallel_scan` (Blelloch) for T>64, `sequential_scan` for T≤64 (`hass_block.py:607-610`) | single serial loop |
| Non-production `forward` | `select_scan` → `chunked_scan(chunk_size=256)` for T>64, `sequential_scan` for T≤64 (`ssm_scan.py:276-307`) | n/a |
| Per-batch Python-loop count for T=2048 | 0 (parallel_scan) or 8 (chunked) | 2048 (single C++ for-loop, but no Python overhead — runs in compiled C++) |

Chunking is an optimization, not a parity requirement.

---

## Dimension 10 — dtype handling

| | Python `SSMPathway` (production) | C++ `SSMPathwayImpl` | C++ `S4DKernel` (dead) |
|---|---|---|---|
| A | real float32 `[N]` | real float32 `[N]` | complex cfloat `[H, N/2]` |
| B / C / states | real float32 | real float32 (`float* p_states = states_cpu.data_ptr<float>()`, `hass_block.cpp:139`) | complex cfloat |
| Scan | real arithmetic | real arithmetic (`std::vector<float> h(D, 0.0f)`, `hass_block.cpp:142`) | complex arithmetic |
| Output | real float32 | real float32 | `torch::real(y_out)` returns real float32 (`ssm.cpp:121`) |

**Match** between Python production and C++ `SSMPathwayImpl`.

---

## Tiny SSM fixture — `tests/cpp_parity/fixtures/07_ssm_scan/`

**Status**: PASS — C++ matches Python exactly for the bare scan recurrence.

- Config: `B=1, T=8, N=4`, seed 42, PyTorch 2.14.1+cpu
- Tolerance: `max_abs_error=1e-6, max_rel_error=1e-5`
- Inputs:
  - `input_A_bar.bin` — `[1, 8, 4]` float32 (already-discretized A_bar)
  - `input_B_bar.bin` — `[1, 8, 4]` float32 (already-discretized B_bar — NOT raw Bv)
  - `input_C.bin` — `[1, 8, 4]` float32
- Expected:
  - `expected_y.bin` — `[1, 8, 4]` float32 (`y_t = C_t · h_t`)
  - `expected_states.bin` — `[1, 8, 4]` float32 (`h_t`, the raw states — proves the Python reference scan returns h_t separately from y_t)

When the fixture supplies `B_bar` directly (already ZOH'd), the C++ scan recurrence matches Python bit-for-bit. The divergence only appears when the full SSMPathway is exercised (fixture `14_ssm_pathway_full`).

---

## Full SSM pathway fixture — `tests/cpp_parity/fixtures/14_ssm_pathway_full/`

**Status**: FAIL — `max_abs_error = 0.0453` (45× the 1e-3 tolerance). Caused by the 4 divergences above compounding.

- Config: `hidden_size=8, state_dim=16, kernel_size=3, use_conv=true`, input `[1,8,8]`
- 15 parameters covering the entire SSMPathway module
- Expected: `expected_y.bin [1,8,8]`
- Tolerance: `max_abs_error=1e-5, max_rel_error=1e-4`

This fixture will validate the 4 fixes once applied.

---

## Adaptation plan (4 local fixes, ~30 lines of C++)

**Reuse the entire `SSMPathwayImpl` skeleton.** The 15 registered parameters match Python by name and shape (verified by comparing `hass_block.cpp:95-116` against `hass_block.py:404-452` and against fixture `14_ssm_pathway_full/manifest.txt`). The `dt_proj`/`B_proj`/`C_proj`/`D_proj`/`gate_proj`/`ln_input`/`ln_state`/`conv` module wiring is identical. The custom-autograd `SSMScanFunction` infrastructure (forward + hand-derived backward) is reusable.

**Make exactly these 4 local fixes** inside `SSMPathwayImpl::forward` and `SSMScanFunction`:

1. **B_bar ZOH** (`hass_block.cpp:239-244`): compute `z = dt * a`, `B_bar_div = where(|z|<1e-4, 1 + z/2 + z²/6, (exp(z)-1)/z)`, `B_bar = B_bar_div * dt * Bv`. Pass `B_bar` to the scan instead of `Bv`. Mirrors `ssm_scan.py:93-105` exactly.

2. **Conv padding** (`hass_block.cpp:105-106` and `231-233`): change `.padding(kernel_size / 2)` to `.padding(kernel_size - 1)`, and after the conv add a slice to truncate to original `T`. Mirrors `hass_block.py:435, 529`.

3. **C application point** (`hass_block.cpp:151, 244`): change `p_states[offset + d] = c * h_next` to `p_states[offset + d] = h_next` (store raw `h_t`). Drop `C` from the `SSMScanFunction::apply` argument list. After the scan returns `states` (= `h_t`), apply `C * states` in the caller. Mirrors `hass_block.py:549-559`.

4. **LN order** (`hass_block.cpp:246`): change `D_proj->forward(ln_state->forward(states)) * gate` to `D_proj->forward(C * ln_state->forward(states)) * gate`. That is: LN first on raw `states`, then multiply by `C`, then `D_proj`. Mirrors `hass_block.py:556-562` and `612-615`.

**Backward pass impact**: fix #3 also simplifies `SSMScanFunction::backward` (`hass_block.cpp:161-224`) — the `p_grad_C` term and the `grad_h_t = grad_out * c + dh[d]` folding both go away. After the fix, `C`'s gradient is supplied by autograd through the `C * ln_state(states)` multiplication in the caller — no hand-coded `grad_C` needed inside the scan.

**Optional (parity-only, not required for HASS production)**:
- Add `init_state` parameter to `SSMScanFunction::forward` (Python accepts it; HASS never passes it).
- Optionally swap the serial loop for the Blelloch `parallel_scan` or a chunked scan for long T.

**Dead code in both languages**: `S4DKernel` (`ssm.cpp` + `ssm.h`) mirrors `xorzen/model/ssm.py` (`S4DKernel` + `SSMBlock`). Neither is imported by `hass_block` in either language. Ignore both when porting HASS — they exist as a standalone S4D experiment and are not part of the production HASS forward path.

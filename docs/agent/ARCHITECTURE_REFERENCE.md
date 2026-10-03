# Architecture Reference — XORZEN v1.0.1

> Quick reference for navigating the codebase.
> For full implementation details, read the source files directly.

---

## Model Hierarchy

```
xorzen/
├── models/
│   ├── zero/
│   │   ├── model.py          → zeroModel (main LM, all sizes)
│   │   ├── variants.py       → size-specific variant helpers
│   │   ├── agentic_model.py  → ZeroAgenticModel (memory + actions)
│   │   └── agentic_config.py → AgenticConfig dataclass
│   └── greed/
│       └── model.py          → GreedModel(zeroModel)  [RESTORED in v1.0.1]
└── model/
    ├── base.py               → BaseModel
    ├── ssm.py                → SSM (state space model)
    ├── zmoe.py               → ShardedExpertFabric (MoE)
    └── components/
        ├── hass_block.py     → HASSBlock (core conditional-compute block)
        ├── sliced_ffn.py     → SlicedFFN (width-conditional FFN)
        ├── routing.py        → router, LowRankGlobalPathway
        ├── sparse_dispatch.py → sparse_pathway_dispatch
        ├── cot_vector.py     → CoT latent vector
        ├── merger.py         → output merger
        ├── load_balance.py   → load balancing losses
        └── ssm_scan.py       → SSM parallel scan kernel
```

---

## ConfigFactory Usage

```python
from xorzen.config import ConfigFactory, ModelSize

# Available sizes:
# '23K', '1M', '10M', '50M', '277M', '500M', '1B', '3B', '7B', '13B', '70B'

cfg = ConfigFactory.get_config(ModelSize('1M'))
```

**Key ModelConfig fields:**
| Field | Type | Default | Meaning |
|---|---|---|---|
| `hidden_size` | int | varies | Core hidden dimension |
| `num_layers` | int | varies | Number of HASS blocks |
| `vocab_size` | int | varies | Vocabulary size |
| `max_seq_len` | int | varies | Max sequence length |
| `use_sliced_ffn` | bool | `True` | Use SlicedFFN (width-conditional) |
| `use_moe` | bool | varies | Use MoE (ShardedExpertFabric) |
| `num_experts` | int | varies | Number of MoE experts |
| `top_k_experts` | int | varies | Top-k routing |
| `width_choices` | list | varies | Available FFN widths for SlicedFFN |
| `width_div_weight` | float | 0.01 | Width diversity loss weight |
| `path_div_weight` | float | 0.1 | Pathway diversity loss weight |
| `test_mode` | bool | `False` | Toy config for unit tests |

---

## Instantiate Models

### zeroModel (standard LM)
```python
from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.model import zeroModel

cfg = ConfigFactory.get_config(ModelSize('1M'))
model = zeroModel(cfg)

# Forward pass (returns dict)
import torch
input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
out = model(input_ids)
logits = out['logits']     # [B, L, vocab_size]
loss = out.get('loss')     # scalar if labels provided

# With labels
out = model(input_ids, labels=input_ids)
loss = out['loss']
```

### ZeroAgenticModel
```python
from xorzen.models.zero.agentic_model import ZeroAgenticModel
from xorzen.models.zero.agentic_config import AgenticConfig

cfg = ConfigFactory.get_config(ModelSize('23K'))
agentic_cfg = AgenticConfig(memory_slots=64, recurrence_depth=1)
model = ZeroAgenticModel(cfg, agentic_cfg)

input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
out = model(
    input_ids,
    action_targets=torch.randn(2, 16, 8),   # optional
    action_loss_weight=0.1                   # optional
)
```

### GreedModel (continuous input, classification)
```python
from xorzen.models.greed import GreedModel

cfg = ConfigFactory.get_config(ModelSize('23K'))
model = GreedModel(cfg, input_dim=32, num_classes=10)

x = torch.randn(2, 16, 32)   # [batch, seq, features]
logits = model(x)              # [batch, num_classes]
```

---

## HASSBlock — How It Works

`HASSBlock` is the core building block. Each block contains:
1. **Pathway router** — selects which of N pathways to use for each token
2. **SlicedFFN or AdaptiveFFN** — width-conditional feedforward
3. **SSM** — state space model (optional)
4. **Attention** — causal multi-head attention (optional)
5. **Depth gate** — binary mask deciding if this block runs at all (inference only)

**During training:**
- All pathways computed (but sparse_pathway_dispatch skips zero-token pathways)
- Full depth always (STE blend for depth gate — gradient flows through)
- Width selection uses STE (straight-through estimator)

**During inference (eval mode):**
- `forward_with_depth` — per-sample depth masking, processes each sample independently
- Pathway skipping — pathways with 0 tokens actually skipped
- Width argmax — select single narrowest sufficient width

---

## SlicedFFN — How Width Slicing Works

```
Input x: [B, L, H]

fc1.weight shape: [max_w, H]     (max_w = hidden_size * 4.0)
fc2.weight shape: [H, max_w]

For width w (from width_choices):
  intermediate = gelu(x @ fc1.weight[:w, :].T)   # [B, L, w]
  output = intermediate @ fc2.weight[:, :w].T      # [B, L, H]
```

Smaller `w` = fewer FLOPs, less compute. The router picks `w` per token.

SlicedFFN is wired in HASSBlock at `__init__` lines ~860–876:
```python
if config.use_sliced_ffn:
    self.ffn = SlicedFFN(hidden_size, max_width=hidden_size*4, ...)
else:
    self.ffn = AdaptiveFFN(...)
```

---

## MoE — ShardedExpertFabric

After BUG-CRITICAL-2 fix:
- `self.experts = nn.ModuleList([ExpertFFN(...) for _ in range(num_experts)])`
- All experts in RAM permanently (no lazy loading)
- `sync_to_disk()` — persist experts to disk shards
- `load_from_disk()` — reload from disk shards into registered modules
- Dispatch: `self.experts[expert_id](x_for_expert)`

Expert disk shards saved to: `experts/` directory in working dir.

---

## SSM — State Space Model

Located in `xorzen/model/ssm.py`.

Key properties:
- ZOH (zero-order hold) discretization
- Causal convolution for initial mixing
- Parallel scan for efficient sequence processing (see `ssm_scan.py`)
- Causal — no future token leakage (verified in audit)

---

## Training

```python
from xorzen.training.trainer import XorzenTrainer

trainer = XorzenTrainer(
    model=model,
    config=cfg,
    train_data_path="data/train.bin",
    val_data_path="data/val.bin",
    output_dir="experiments/run_name",
)
trainer.train(max_steps=10000)
```

Checkpoints saved to `output_dir/checkpoints/`.
Logs written to `logs/`.

---

## Inference / .xorm Format

```python
from xorzen.inference.xorm_format import XormWriter, XormReader

# Save
writer = XormWriter()
writer.save(model, "model.xorm")

# Load
reader = XormReader()
state = reader.load("model.xorm")
model.load_state_dict(state)

# Full session (if xorm_runtime.py is fully compatible)
from xorzen.inference.xorm_runtime import XormSession
session = XormSession.load("model.xorm")
output = session.generate("Hello", max_tokens=50)
```

> NOTE: `xorm_runtime.py` may need additional patching for full DevNet ConfigFactory compatibility. Verify before using `XormSession`.

---

## Tokenizers

Two pretrained vocabs included:
- `xorzen/tokenizer/pretrained/zero_bpe_10k.json` — 10k BPE vocab
- `xorzen/tokenizer/pretrained/xorzen_agi_tokenizer_65k.json` — 65k AGI vocab

```python
from xorzen.tokenizer.loader import load_tokenizer

tok = load_tokenizer("xorzen_agi_65k")  # or "zero_bpe_10k"
ids = tok.encode("Hello world")
text = tok.decode(ids)
```

---

## Known Issues (as of v1.0.1)

| Issue | Severity | Status |
|---|---|---|
| All N experts in RAM (no lazy loading) | Medium | Known limitation, documented |
| `forward_with_depth` is inference-only | Low | By design — training uses STE |
| `xorm_runtime.py` ConfigFactory compatibility | Low | May need additional patching |
| No GPU validation | Medium | CPU only tested |
| `tokenizers` library missing | Low | Non-fatal warning only |
| `pandas`/`pyarrow` missing | Low | Non-fatal warning only |
| GreedModel config presets missing | Medium | Task 2 in OUTSTANDING_TASKS.md |

---

## Reference: xorzen_0.2.4 Location

Original source for forensic reference:
```
c:\Users\user\akik\programing\ai\xorzen_0.2.4\
```

Key files that were ported to DevNet:
- `xorzen_0.2.4/xorzen/model/greed.py` → `DevNet/xorzen/models/greed/model.py`
- `xorzen_0.2.4/xorzen/inference/xorm_format.py` → `DevNet/xorzen/inference/xorm_format.py`
- `xorzen_0.2.4/xorzen/inference/xorm_runtime.py` → `DevNet/xorzen/inference/xorm_runtime.py`

**Do NOT modify `xorzen_0.2.4/`** — it is the reference baseline.

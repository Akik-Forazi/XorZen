"""
XORZENX GGUF Transfer — Load knowledge from local GGUF models (e.g. Nemotron, LLaMA, Mistral)
and transfer it into an xorzen-zero model without needing HuggingFace or internet access.

Supports any GGUF file using the `gguf` library for dequantization.

Usage:
    from xorzen.transfer.gguf_transfer import GGUFTransfer
    import xorzen

    model = xorzen.zero_1M()
    transfer = GGUFTransfer(r"C:\\path\\to\\model.gguf")
    model, tokenizer_vocab = transfer.apply(model)
"""

import os
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

try:
    import gguf
    from gguf import GGUFReader, GGMLQuantizationType, dequantize as gguf_dequantize
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_gguf():
    if not GGUF_AVAILABLE:
        raise ImportError(
            "The 'gguf' package is required. Install with:\n"
            "  pip install gguf --break-system-packages"
        )


def _dequantize_tensor(reader_tensor) -> np.ndarray:
    """Dequantize a GGUF ReaderTensor to float32 numpy array."""
    raw = reader_tensor.data          # raw quantized bytes as numpy uint8
    qtype = reader_tensor.tensor_type
    return gguf_dequantize(raw, qtype).astype(np.float32)


def _adapt_weight(src: torch.Tensor, target_shape: Tuple[int, int], noise: float = 0.0) -> torch.Tensor:
    """Crop / pad source weight to match target_shape."""
    tgt_h, tgt_w = target_shape
    src_h, src_w = src.shape

    result = torch.zeros(target_shape, dtype=torch.float32)
    min_h, min_w = min(src_h, tgt_h), min(src_w, tgt_w)
    result[:min_h, :min_w] = src[:min_h, :min_w]

    # Xavier fill for any unpopulated region
    if tgt_h > src_h or tgt_w > src_w:
        nn.init.xavier_uniform_(result, gain=0.1)
        result[:min_h, :min_w] = src[:min_h, :min_w]   # restore copied region

    if noise > 0:
        result += torch.randn_like(result) * noise

    return result


def _project_embedding(emb: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Project embedding matrix [V, src_dim] -> [V, target_dim]."""
    src_dim = emb.shape[1]
    if src_dim == target_dim:
        return emb
    proj = nn.Linear(src_dim, target_dim, bias=False)
    nn.init.orthogonal_(proj.weight, gain=0.5)
    with torch.no_grad():
        return proj(emb)


# ---------------------------------------------------------------------------
# GGUF metadata reader
# ---------------------------------------------------------------------------

class GGUFMeta:
    """Reads metadata and tensor index from a GGUF file without loading all tensors."""

    def __init__(self, path: str):
        _require_gguf()
        self.path = path
        self.reader = GGUFReader(path, mode='r')

        # Build tensor lookup: name -> ReaderTensor
        self._tensors: Dict[str, Any] = {t.name: t for t in self.reader.tensors}

        # Parse metadata fields into simple dict
        self.meta: Dict[str, Any] = {}
        for field in self.reader.fields.values():
            try:
                if field.types and len(field.parts) > field.data[0]:
                    raw = field.parts[field.data[0]]
                    self.meta[field.name] = bytes(raw).decode(errors='replace').rstrip('\x00')
                else:
                    self.meta[field.name] = None
            except Exception:
                self.meta[field.name] = None

    # Convenience getters ---------------------------------------------------

    @property
    def arch(self) -> str:
        return self.meta.get('general.architecture', 'llama')

    @property
    def n_layers(self) -> int:
        key = f'{self.arch}.block_count'
        v = self.meta.get(key) or self.meta.get('llama.block_count') or '32'
        try:
            return int(v)
        except Exception:
            return 32

    @property
    def hidden_size(self) -> int:
        key = f'{self.arch}.embedding_length'
        v = self.meta.get(key) or self.meta.get('llama.embedding_length') or '3072'
        try:
            return int(v)
        except Exception:
            return 3072

    @property
    def ffn_size(self) -> int:
        key = f'{self.arch}.feed_forward_length'
        v = self.meta.get(key) or self.meta.get('llama.feed_forward_length') or '8192'
        try:
            return int(v)
        except Exception:
            return 8192

    @property
    def vocab_size(self) -> int:
        v = self.meta.get('tokenizer.ggml.tokens')
        if v is None:
            key = f'{self.arch}.vocab_size'
            v = self.meta.get(key) or '32000'
        try:
            return int(v)
        except Exception:
            return 32000

    @property
    def tensor_names(self) -> List[str]:
        return list(self._tensors.keys())

    def get_tensor(self, name: str) -> Optional[torch.Tensor]:
        """Dequantize and return a named tensor as float32 CPU torch.Tensor."""
        t = self._tensors.get(name)
        if t is None:
            return None
        arr = _dequantize_tensor(t)
        return torch.from_numpy(arr)

    def get_tensor_shape(self, name: str) -> Optional[Tuple]:
        t = self._tensors.get(name)
        return tuple(t.shape) if t else None

    def has(self, name: str) -> bool:
        return name in self._tensors


# ---------------------------------------------------------------------------
# Layer name patterns for different architectures
# ---------------------------------------------------------------------------

LAYER_PATTERNS = {
    # (gate_proj, up_proj, down_proj)
    'llama':    ('blk.{i}.ffn_gate.weight', 'blk.{i}.ffn_up.weight', 'blk.{i}.ffn_down.weight'),
    'mistral':  ('blk.{i}.ffn_gate.weight', 'blk.{i}.ffn_up.weight', 'blk.{i}.ffn_down.weight'),
    'nemotron': ('blk.{i}.ffn_gate.weight', 'blk.{i}.ffn_up.weight', 'blk.{i}.ffn_down.weight'),
    'phi3':     ('blk.{i}.ffn_gate.weight', 'blk.{i}.ffn_up.weight', 'blk.{i}.ffn_down.weight'),
    'qwen2':    ('blk.{i}.ffn_gate.weight', 'blk.{i}.ffn_up.weight', 'blk.{i}.ffn_down.weight'),
    'default':  ('blk.{i}.ffn_gate.weight', 'blk.{i}.ffn_up.weight', 'blk.{i}.ffn_down.weight'),
}

EMBED_PATTERNS = [
    'token_embd.weight',
    'tok_embeddings.weight',
    'embeddings.weight',
]


# ---------------------------------------------------------------------------
# Main transfer class
# ---------------------------------------------------------------------------

class GGUFTransfer:
    """
    Transfer knowledge from a local GGUF model file into an xorzen-zero model.

    Steps performed:
      1. Read GGUF metadata (architecture, sizes, vocab)
      2. Extract & project token embeddings
      3. Seed xorzen MoE experts from teacher FFN layers (gate/up/down)
      4. Initialise router with balanced weights
      5. Freeze transferred embeddings for stable fine-tuning start
      6. Save a vocab JSON alongside the xorzen model for tokenizer reconstruction
    """

    def __init__(self, gguf_path: str, verbose: bool = True):
        _require_gguf()
        self.gguf_path = str(gguf_path)
        self.verbose = verbose
        self._log(f"Opening GGUF: {self.gguf_path}")
        self.meta = GGUFMeta(self.gguf_path)
        self._log(f"  Architecture : {self.meta.arch}")
        self._log(f"  Layers       : {self.meta.n_layers}")
        self._log(f"  Hidden size  : {self.meta.hidden_size}")
        self._log(f"  FFN size     : {self.meta.ffn_size}")
        self._log(f"  Tensors      : {len(self.meta.tensor_names)}")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[GGUFTransfer] {msg}")

    # ------------------------------------------------------------------
    # 1. Token embedding transfer
    # ------------------------------------------------------------------

    def _get_embedding_tensor(self) -> Optional[torch.Tensor]:
        for name in EMBED_PATTERNS:
            t = self.meta.get_tensor(name)
            if t is not None:
                self._log(f"  Found embedding tensor: '{name}' {tuple(t.shape)}")
                return t
        # Fallback: any tensor with 'embd' or 'embed' in name
        for name in self.meta.tensor_names:
            if 'embd' in name or 'embed' in name:
                t = self.meta.get_tensor(name)
                if t is not None and t.ndim == 2:
                    self._log(f"  Fallback embedding: '{name}' {tuple(t.shape)}")
                    return t
        return None

    def transfer_embeddings(self, xorzen_model) -> int:
        """
        Replace xorzen token_embedding + lm_head with projected teacher embeddings.
        Returns the new vocab size.
        """
        self._log("[1/4] Transferring token embeddings...")
        emb = self._get_embedding_tensor()

        if emb is None:
            self._log("  WARNING: No embedding tensor found in GGUF — skipping.")
            return xorzen_model.config.vocab_size

        # GGUF stores embeddings as [vocab, hidden] — same as PyTorch
        vocab_size, teacher_dim = emb.shape
        xorzen_dim = xorzen_model.config.hidden_size

        self._log(f"  Teacher: vocab={vocab_size:,} dim={teacher_dim}")
        self._log(f"  xorzen:  dim={xorzen_dim}")

        # Project embedding if dimensions differ
        if teacher_dim != xorzen_dim:
            self._log(f"  Projecting {teacher_dim} -> {xorzen_dim}")
            emb_projected = _project_embedding(emb, xorzen_dim).detach()
        else:
            emb_projected = emb

        # Rebuild embedding + lm_head with tied weights
        new_emb = nn.Embedding(vocab_size, xorzen_dim)
        new_emb.weight.data = emb_projected.cpu()

        new_lm_head = nn.Linear(xorzen_dim, vocab_size, bias=False)
        new_lm_head.weight = new_emb.weight   # tied

        xorzen_model.token_embedding = new_emb
        xorzen_model.lm_head = new_lm_head
        xorzen_model.config.vocab_size = vocab_size

        self._log(f"  Done: vocab={vocab_size:,}")
        return vocab_size

    # ------------------------------------------------------------------
    # 2. Expert seeding from FFN layers
    # ------------------------------------------------------------------

    def _ffn_layer_names(self, layer_idx: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (gate, up, down) tensor names for a given layer index."""
        arch = self.meta.arch
        pattern = LAYER_PATTERNS.get(arch, LAYER_PATTERNS['default'])
        g, u, d = [p.format(i=layer_idx) for p in pattern]

        # Verify existence, try fallbacks
        if not self.meta.has(g):
            g = f'blk.{layer_idx}.ffn_gate.weight' if self.meta.has(f'blk.{layer_idx}.ffn_gate.weight') else None
        if not self.meta.has(u):
            u = f'blk.{layer_idx}.ffn_up.weight' if self.meta.has(f'blk.{layer_idx}.ffn_up.weight') else None
        if not self.meta.has(d):
            d = f'blk.{layer_idx}.ffn_down.weight' if self.meta.has(f'blk.{layer_idx}.ffn_down.weight') else None

        return g, u, d

    def seed_experts(self, xorzen_model, noise_scale: float = 0.01):
        """
        Populate xorzen MoE experts from teacher FFN gate/up/down weights.
        Teachers have more layers than xorzen has experts — we tile round-robin.
        """
        self._log("[2/4] Seeding MoE experts from teacher FFN layers...")

        if xorzen_model.moe.test_mode:
            self._log("  Skipping — model is in test_mode")
            return

        num_experts = xorzen_model.moe.num_experts
        n_teacher_layers = self.meta.n_layers
        self._log(f"  Teacher layers: {n_teacher_layers}, xorzen experts: {num_experts}")

        seeded = 0
        for expert_id in range(num_experts):
            layer_idx = expert_id % n_teacher_layers
            gate_name, up_name, down_name = self._ffn_layer_names(layer_idx)

            if gate_name is None and up_name is None:
                self._log(f"  Layer {layer_idx}: no FFN tensors found, skipping")
                continue

            # Load expert from disk / cache
            expert = xorzen_model.moe.cache.get(expert_id)
            if expert is None:
                expert = xorzen_model.moe.disk_manager.load_expert(expert_id)

            noise = noise_scale * (1 + 0.1 * (expert_id // max(1, n_teacher_layers)))

            if gate_name and self.meta.has(gate_name):
                w = self.meta.get_tensor(gate_name)
                expert.gate_proj.weight.data = _adapt_weight(w, expert.gate_proj.weight.shape, noise)

            if up_name and self.meta.has(up_name):
                w = self.meta.get_tensor(up_name)
                expert.up_proj.weight.data = _adapt_weight(w, expert.up_proj.weight.shape, noise * 0.5)

            if down_name and self.meta.has(down_name):
                w = self.meta.get_tensor(down_name)
                expert.down_proj.weight.data = _adapt_weight(w, expert.down_proj.weight.shape, noise)

            # Persist back to disk
            xorzen_model.moe.disk_manager.save_expert(expert_id, expert)
            xorzen_model.moe.cache.put(expert_id, expert)
            seeded += 1

            if (expert_id + 1) % max(1, num_experts // 4) == 0:
                self._log(f"  Seeded {seeded}/{num_experts} experts...")

        self._log(f"  Done: {seeded}/{num_experts} experts seeded")

    # ------------------------------------------------------------------
    # 3. Router initialisation
    # ------------------------------------------------------------------

    def init_router(self, xorzen_model):
        """Balanced router init — prevents early routing collapse."""
        self._log("[3/4] Initialising router (balanced)...")
        router = xorzen_model.router
        if hasattr(router, 'depth_router'):
            router.depth_router.weight.data.fill_(0.0)
        if hasattr(router, 'expert_router'):
            nn.init.normal_(router.expert_router.weight, std=0.02)
        self._log("  Done")

    # ------------------------------------------------------------------
    # 4. Freeze embeddings
    # ------------------------------------------------------------------

    def freeze_embeddings(self, xorzen_model):
        """Freeze token + position embeddings for the first training phase."""
        self._log("[4/4] Freezing embeddings...")
        for param in xorzen_model.token_embedding.parameters():
            param.requires_grad = False
        if hasattr(xorzen_model, 'position_embedding'):
            for param in xorzen_model.position_embedding.parameters():
                param.requires_grad = False
        frozen = sum(1 for p in xorzen_model.parameters() if not p.requires_grad)
        self._log(f"  {frozen} params frozen")

    # ------------------------------------------------------------------
    # 5. Save vocab JSON for tokenizer reconstruction
    # ------------------------------------------------------------------

    def save_vocab(self, output_dir: str):
        """
        Dump the GGUF vocabulary to JSON so you can build a compatible tokenizer.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = out_dir / 'gguf_vocab.json'

        tokens_field = self.meta.reader.fields.get('tokenizer.ggml.tokens')
        if tokens_field is None:
            self._log("  WARNING: No tokenizer.ggml.tokens field in GGUF — vocab not saved")
            return

        tokens = []
        for idx in tokens_field.data:
            try:
                raw = tokens_field.parts[idx]
                tokens.append(bytes(raw).decode('utf-8', errors='replace').rstrip('\x00'))
            except Exception:
                tokens.append(f'<unk_{len(tokens)}>')

        vocab = {tok: i for i, tok in enumerate(tokens)}

        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        self._log(f"  Vocab saved: {vocab_path} ({len(tokens):,} tokens)")

        meta_path = out_dir / 'gguf_meta.json'
        summary = {
            'source_gguf': self.gguf_path,
            'architecture': self.meta.arch,
            'n_layers': self.meta.n_layers,
            'hidden_size': self.meta.hidden_size,
            'ffn_size': self.meta.ffn_size,
            'vocab_size': len(tokens),
        }
        with open(meta_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self._log(f"  Meta saved : {meta_path}")

    # ------------------------------------------------------------------
    # Master entry point
    # ------------------------------------------------------------------

    def apply(
        self,
        xorzen_model,
        output_dir: str = 'transfer_output',
        freeze_embeddings: bool = True,
        seed_experts: bool = True,
        noise_scale: float = 0.01,
    ):
        """
        Full transfer pipeline:
          1. Transfer embeddings
          2. Seed experts
          3. Init router
          4. Freeze embeddings (optional)
          5. Save vocab JSON

        Returns:
            (xorzen_model, vocab_dict)  where vocab_dict maps token_str -> id
        """
        print()
        print('=' * 70)
        print('XORZENX GGUF TRANSFER')
        print(f'  Source : {Path(self.gguf_path).name}')
        print(f'  Target : xorzen {xorzen_model.config.model_name}')
        print('=' * 70)
        t0 = time.time()

        self.transfer_embeddings(xorzen_model)

        if seed_experts:
            self.seed_experts(xorzen_model, noise_scale=noise_scale)
        else:
            self._log("[2/4] Expert seeding skipped")

        self.init_router(xorzen_model)

        if freeze_embeddings:
            self.freeze_embeddings(xorzen_model)
        else:
            self._log("[4/4] Embedding freezing skipped")

        self.save_vocab(output_dir)

        elapsed = time.time() - t0
        print()
        print(f'Transfer complete in {elapsed:.1f}s')
        print(f'Model ready for fine-tuning.')
        print('=' * 70)
        print()

        vocab_path = Path(output_dir) / 'gguf_vocab.json'
        vocab = {}
        if vocab_path.exists():
            with open(vocab_path, encoding='utf-8') as f:
                vocab = json.load(f)

        return xorzen_model, vocab


__all__ = ['GGUFTransfer', 'GGUFMeta']

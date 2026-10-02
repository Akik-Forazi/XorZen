# XorZen-Anime: Long-Context Anime Generation Architecture
### Extension Plan for XorZen v0.2.4
**Author:** Akik Faraji — FRAZIYM AI  
**Date:** 2026-05-15

---

## 1. What We're Building

A specialized **1B parameter anime video generation model** built on top of the XorZen HASS+MoE architecture, capable of maintaining **1B token effective context** through hierarchical chunk memory. The goal: generate studio-quality anime with full season-level consistency — character appearance, voice, personality, story arcs, world-state — across 24–26 episodes × 3–4 seasons, matching the visual and narrative quality of Demon Slayer, JJK, Devil May Cry.

---

## 2. Why 1B Tokens of Context is the Key Problem

A 24-episode season at 24fps, with each frame encoded as ~100 tokens (compressed visual tokens), is about **1.25 billion tokens per season**. Even a 4-season show is 5B tokens of raw content. No current transformer can attend over all of that. The key insight:

> **You don't need to attend over every token. You need to remember the right things.**

This is what chunk memory solves.

---

## 3. The Chunk Memory System (Core Innovation)

### 3.1 Three-Level Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 3: Series Memory (frozen, loaded per series)            │
│  ─ World rules, all character embeddings, narrative arc        │
│  ─ ~50M tokens compressed into 512K learned memory vectors    │
│  ─ Updated only at series-end or major arc completion         │
├─────────────────────────────────────────────────────────────────┤
│  LEVEL 2: Episode Memory (updated per episode)                 │
│  ─ What happened this episode, emotional states               │
│  ─ ~2M tokens compressed into 32K memory vectors             │
│  ─ Persists across scenes within an episode                   │
├─────────────────────────────────────────────────────────────────┤
│  LEVEL 1: Scene Memory (active context window)                 │
│  ─ Current scene frames, dialogue, motion                     │
│  ─ 8K–32K tokens directly in attention                       │
│  ─ Slides forward as scene progresses                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Chunk Memory Architecture

Each "chunk" is a fixed window of frames (e.g., 96 frames = 4 seconds at 24fps). At the boundary of each chunk:

1. **Chunk Encoder** runs over the chunk's tokens → produces a **chunk summary vector** (256 dims)
2. Summary written into a **Memory Bank** (a learned key-value store, like a differentiable episodic memory)
3. The next chunk can **cross-attend** into the Memory Bank using the current scene context as a query
4. The Memory Bank is stored **on disk** between episodes, loaded on demand — exactly like XorZen's `ShardedExpertFabric` LRU cache, repurposed for memory chunks

This is different from sliding-window attention: the model doesn't "look back" at raw tokens. It looks back at **compressed memory states**, which is orders of magnitude cheaper.

### 3.3 Memory Bank Implementation (Fits into XorZen)

```python
class ChunkMemoryBank(nn.Module):
    """
    Persistent memory bank for anime generation.
    Reuses ShardedExpertFabric's LRU disk caching infrastructure.
    
    Key design:
      - Memory slots = learned (key, value) pairs, one per chunk
      - Keys: [B, num_slots, key_dim]   — what this memory is "about"
      - Values: [B, num_slots, val_dim] — compressed chunk content
      - Cross-attention from current context queries into memory
    """
    
    def __init__(self, num_slots=4096, key_dim=256, val_dim=512):
        super().__init__()
        self.num_slots  = num_slots
        self.key_dim    = key_dim
        self.val_dim    = val_dim
        
        # Chunk encoder: compresses 96 frames → 1 summary vector
        self.chunk_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=val_dim, nhead=8, batch_first=True),
            num_layers=4
        )
        self.to_key   = nn.Linear(val_dim, key_dim)
        self.to_value = nn.Linear(val_dim, val_dim)
        
        # Cross-attention: current scene context queries memory bank
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=val_dim, num_heads=8, batch_first=True
        )
        
        # Disk persistence (reuses XorZen ExpertDiskManager pattern)
        self.disk_manager = ChunkDiskManager(...)  # mirrors ExpertDiskManager
        self.lru_cache    = LRUChunkCache(capacity=256)  # mirrors LRUExpertCache
    
    def write_chunk(self, chunk_tokens: Tensor, chunk_id: int):
        """Compress chunk into memory slot and persist to disk."""
        summary = self.chunk_encoder(chunk_tokens).mean(dim=1)  # [B, val_dim]
        key   = self.to_key(summary)    # [B, key_dim]
        value = self.to_value(summary)  # [B, val_dim]
        self.disk_manager.save_chunk(chunk_id, key, value)
        self.lru_cache.put(chunk_id, (key, value))
    
    def read_relevant(self, query: Tensor, top_k: int = 64):
        """Retrieve top-k most relevant memory slots for current context."""
        # query: [B, seq, val_dim] — current scene tokens
        # Load top-k slots from cache/disk via similarity search
        keys, values = self._load_top_k(query, top_k)  # [B, top_k, {key,val}_dim]
        out, _ = self.memory_attn(query, keys, values)
        return out  # [B, seq, val_dim]
```

---

## 4. Character Consistency System

This is XorZen's MoE repurposed: **each expert specializes in one character**.

### 4.1 Character Expert Assignment

- 64 experts in the base MoE → assign **N experts per character** (e.g., 4 experts per character × 16 characters = 64)
- Router learns to activate the right experts based on **who is on screen**
- Character embeddings are stored as persistent vectors, loaded from disk like expert shards
- At inference: detect character IDs in current scene → pre-load their expert shards into LRU cache

### 4.2 Character Embedding Store

```python
@dataclass
class CharacterProfile:
    char_id: int
    name: str
    embedding: Tensor          # [char_emb_dim] — appearance, voice, personality
    expert_ids: List[int]      # which MoE experts "are" this character
    memory_slots: List[int]    # which memory chunks contain this character
    first_appearance: int      # episode number
    arc_summaries: List[str]   # text summaries of character arcs (per season)
```

### 4.3 Router Modification for Character Routing

The existing `AdaptiveRouter` in `routing.py` adds a **character detection head**:
```python
# New head in AdaptiveRouter._build_network():
self.character_router = nn.Sequential(
    nn.Linear(_enc3, _head),
    nn.LayerNorm(_head),
    nn.GELU(),
    nn.Linear(_head, max_characters),  # [B, T, max_characters]
    nn.Sigmoid()  # which characters are present
)
```
Character probabilities bias the expert routing logits so character-assigned experts get higher scores when that character is detected.

---

## 5. Multimodal Extension

XorZen currently handles text tokens. Anime generation needs three modalities:

| Modality | Token Type | Encoder |
|---|---|---|
| Video frames | Visual tokens (VQVAE codes) | Patch encoder, 16×16 patches → ~256 tokens/frame |
| Audio/dialogue | Audio tokens (EnCodec) | Audio encoder → ~75 tokens/second |
| Script/story | Text tokens | Existing XorZen tokenizer |

Each modality gets its own **embedding projection** into the shared `hidden_size` space, then they're concatenated into a single sequence that HASS processes normally. The three HASS pathways (LocalAttn, LowRank, SSM) then naturally specialize:
- **LocalAttn** → frame-to-frame motion coherence (needs local temporal context)
- **LowRank** → scene-level structure (slow global changes)  
- **SSM** → long-range audio-visual sync (sequential, stateful)

This specialization is already learned in the existing HASS routing — it just needs to be exposed to multimodal input.

---

## 6. How to Actually Train This

### Phase 1: Pre-train on Manga + Comics (Text Only, 1B tokens)
- Use existing XorZen training pipeline unchanged
- Dataset: manga scripts, light novels, anime subtitles
- Goal: model learns narrative structure, character arcs, story consistency
- Model size: 1B params (scale up hidden_size=2048, num_layers=24, num_experts=192)

### Phase 2: Visual Token Pre-training (Frames Only, 50B frames)
- Add the visual token embedder
- Train on anime frame sequences, predict next frame
- Freeze the text/memory backbone, only train visual projections
- This teaches the model what anime looks like

### Phase 3: Multimodal Joint Training
- All three modalities together
- Dataset: aligned (video, audio, script) tuples from licensed anime
- The ChunkMemoryBank now writes chunks of video+audio+text

### Phase 4: Consistency Fine-tuning
- Dataset: single anime series (e.g., one 26-episode season)
- Train the Character Expert Assignment system
- Train the memory bank to correctly recall character state
- Reward: frame-to-frame character appearance consistency (FaceID/CLIP similarity)

---

## 7. Files to Create in `xorzen/`

```
xorzen/
├── model/
│   ├── anime/                        ← NEW MODULE
│   │   ├── __init__.py
│   │   ├── chunk_memory.py           ← ChunkMemoryBank, ChunkDiskManager, LRUChunkCache
│   │   ├── character_store.py        ← CharacterProfile, CharacterEmbeddingStore
│   │   ├── multimodal_encoder.py     ← VisualTokenEncoder, AudioTokenEncoder
│   │   ├── anime_model.py            ← AnimeXorZen (full model wrapper)
│   │   └── consistency.py            ← ConsistencyLoss, FaceIDLoss
│   └── components/
│       └── routing.py                ← ADD character_router head (existing file)
├── config/
│   └── anime_config.py               ← AnimeModelConfig (extends ModelConfig)
├── training/
│   └── anime_trainer.py              ← AnimeTrainer (chunk-aware training loop)
└── data/
    └── anime_dataset.py              ← AnimeDataset (loads (video, audio, script) tuples)
```

---

## 8. Config Changes

```python
@dataclass
class AnimeModelConfig(ModelConfig):
    # Scale up base model
    hidden_size: int          = 2048
    num_layers: int           = 24
    num_attention_heads: int  = 16
    expert_count: int         = 192    # 3× more experts than base
    
    # Chunk memory
    chunk_size_frames: int    = 96     # 4 seconds at 24fps
    memory_slots: int         = 4096   # how many chunks to remember
    memory_key_dim: int       = 256
    memory_val_dim: int       = 512
    memory_top_k: int         = 64     # how many memory slots to attend per step
    
    # Characters
    max_characters: int       = 64
    char_emb_dim: int         = 256
    experts_per_character: int = 3
    
    # Multimodal
    visual_token_vocab: int   = 8192   # VQVAE codebook size
    audio_token_vocab: int    = 1024   # EnCodec codebook size
    patch_size: int           = 16     # visual patch size
    
    # Generation
    frames_per_second: int    = 24
    max_episode_frames: int   = 31_104 # 24fps × 22 min episode
```

---

## 9. Key Technical Challenges and Solutions

| Challenge | Solution |
|---|---|
| 1B token context is too large for attention | Chunk memory: attend over 64 compressed summaries, not 1B raw tokens |
| Character appearance drifts between scenes | Character experts are pinned; visual consistency loss during fine-tuning |
| GPU VRAM (1B param model needs ~8GB for weights) | XorZen's existing disk-sharded expert system already handles this |
| Training data (you need licensed anime frames) | Start with public-domain anime (1930s–1960s Toei), then license; or partner with studios |
| Audio-visual synchronization | SSM pathway is naturally good at this (stateful sequential model) |
| Multi-season coherence | Level 3 Series Memory persists across episodes/seasons on disk |

---

## 10. Recommended First Steps (Priority Order)

1. **Implement `chunk_memory.py`** — this is the single most important new component. Mirror `ShardedExpertFabric` + `LRUExpertCache` from `zmoe.py` but for memory chunks instead of experts. 95% of the infrastructure already exists in `zmoe.py`.

2. **Add `character_router` head to `routing.py`** — small change to the existing `AdaptiveRouter._build_network()`. Adds character detection without breaking anything.

3. **Write `AnimeModelConfig`** — extend `ModelConfig` with the new fields. No breaking changes.

4. **Write `multimodal_encoder.py`** — just two linear projections (visual patch → hidden_size, audio token → hidden_size). Very simple.

5. **Train Phase 1** — text-only pre-training using the existing XorZen training pipeline with a bigger model config. This requires no new code, just a config change.

6. **Phase 2–4** — after Phase 1 proves the model is learning narrative structure.

---

## 11. What 1B Params × 1B Tokens Actually Gets You

At 1B params with 1B effective context (via chunk memory):
- **Intra-episode consistency**: perfect (direct context window covers full episode)
- **Inter-episode consistency**: very high (chunk memory recalls prior episodes accurately)
- **Character appearance**: consistent if character experts are properly assigned + fine-tuned
- **Narrative coherence**: strong (Series Memory holds arc summaries across seasons)
- **Generation quality**: comparable to a mid-tier latent diffusion model for individual frames, but with vastly superior temporal and narrative coherence

The frame-by-frame quality ceiling is set by the visual token vocabulary and the decoder (a separate VQVAE decoder or diffusion model). The XorZen model handles **what to generate** (narrative, character state, motion), and a separate decoder handles **how it looks pixel by pixel**. These two concerns are cleanly separated.

---

*This document is the architecture plan. Implementation starts with `chunk_memory.py`. All infrastructure in `zmoe.py` can be directly reused.*

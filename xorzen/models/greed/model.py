import torch
import torch.nn as nn
from typing import Optional

from xorzen.config import ModelConfig
from xorzen.models.zero.model import zeroModel


class GreedModel(zeroModel):
    """
    XorZen-GREED — the conversation quality evaluator.

    Inherits from zeroModel to reuse:
        - AdaptiveRouter (depth/width/path/expert routing)
        - HASSBlock x N  (Local Attention + LowRank Global + SSM pathways)
        - ShardedExpertFabric  (registered in-memory & disk-sharded MoE)
        - xorzenMergerGate  (HASS + MoE + CoT fusion)
        - RMSNorm

    Greed-specific traits:
        1. Continuous feature projection instead of discrete token embedding lookup
        2. Active InternalLatentCoT (unfrozen, trained end-to-end)
        3. Greedy Gated CoT Fusion — fuses reasoning trace back into hidden
           states before pooling, amplifying turns with high reasoning density
        4. Mean-pooled classification head instead of per-token LM head
    """

    def __init__(
        self,
        config: ModelConfig,
        input_dim: int = 24,
        num_classes: int = 5,
        test_mode: bool = False,
    ):
        # Set vocab_size to num_classes for consistency
        config.vocab_size = max(config.vocab_size, num_classes)
        super().__init__(config, test_mode=test_mode)

        self.input_dim = input_dim
        self.num_classes = num_classes

        # ── Feature projection (replaces token embedding for continuous vectors) ──
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

        # ── Unfreeze CoT (Greed trains it end-to-end) ──────────────────────
        self.enable_cot()

        # ── Greedy Gated CoT Fusion ─────────────────────────────────────────
        cot_total_dim = config.cot_dim * config.cot_components
        self.greedy_cot_gate = nn.Sequential(
            nn.Linear(config.hidden_size + cot_total_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.greedy_fusion_proj = nn.Linear(
            config.hidden_size + cot_total_dim, config.hidden_size
        )

        # ── Classification head ─────────────────────────────────────────────
        self.classification_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, num_classes),
        )

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(
        self,
        features: torch.Tensor,           # [B, T, input_dim]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Process a conversation as a sequence of per-turn feature vectors
        through the full XorZen stack, returning grade logits [B, num_classes].
        """
        batch_size, seq_length, _ = features.shape
        device = features.device

        if seq_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds context limit "
                f"{self.config.context_length}"
            )

        # ── STEP 1: Project features into hidden space ──────────────────────
        hidden_states = self.feature_proj(features)          # [B, T, H]

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = hidden_states + self.position_embedding(position_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=device
            )

        # ── STEP 2: Active CoT (Greed trait — unfrozen) ────────────────────
        cot_vector_seq, _ = self.cot(hidden_states)          # [B, T, cot_total]

        # ── STEP 3: Adaptive routing ────────────────────────────────────────
        routing_decision = self.router(
            x=hidden_states,
            cot_features=cot_vector_seq,
            training=self.training,
        )
        depth_mask      = routing_decision.depth_mask        # [B, T, n_layers]
        expert_indices  = routing_decision.expert_indices    # [B, T, top_k]
        expert_weights  = routing_decision.expert_weights    # [B, T, top_k]

        # ── STEP 4: HASS blocks ─────────────────────────────────────────────
        for layer_idx, block in enumerate(self.blocks):
            layer_mask = depth_mask[:, :, layer_idx]         # [B, T]

            if not self.training and not layer_mask.any():
                continue

            block_out = block(
                x=hidden_states,
                routing_decision=routing_decision,
                attention_mask=attention_mask,
            )
            layer_mask_3d = layer_mask.unsqueeze(-1)
            hidden_states = (
                block_out * layer_mask_3d
                + hidden_states * (1.0 - layer_mask_3d)
            )

        # ── STEP 5: MoE experts ─────────────────────────────────────────────
        batch_seq = batch_size * seq_length
        hidden_flat        = hidden_states.reshape(batch_seq, -1)
        expert_idx_flat    = expert_indices.reshape(batch_seq, -1)
        expert_weight_flat = expert_weights.reshape(batch_seq, -1)

        moe_flat, _ = self.moe(
            hidden_flat,
            expert_idx_flat,
            expert_weight_flat,
            attention_mask=(
                attention_mask.reshape(batch_seq)
                if attention_mask is not None
                else None
            ),
        )
        moe_output = moe_flat.reshape(batch_size, seq_length, -1)

        # ── STEP 6: Merger gate ─────────────────────────────────────────────
        merged_output = self.merger(
            hass_output=hidden_states,
            moe_output=moe_output,
            cot_vector=cot_vector_seq,
            attention_mask=attention_mask,
        )
        hidden_states = self.final_norm(merged_output)        # [B, T, H]

        # ── STEP 7: Greedy Gated CoT Fusion (Greed trait) ──────────────────
        concat = torch.cat([hidden_states, cot_vector_seq], dim=-1)  # [B, T, H+CoT]
        gate   = self.greedy_cot_gate(concat)                        # [B, T, 1]
        fused  = self.greedy_fusion_proj(concat)                     # [B, T, H]
        hidden_states = gate * fused + (1.0 - gate) * hidden_states

        # ── STEP 8: Pool + classify ─────────────────────────────────────────
        pooled = hidden_states.mean(dim=1)                   # [B, H]
        logits = self.classification_head(pooled)            # [B, num_classes]
        return logits

"""
XORZEN Jima (Agentic) Model — Built on the full XORZEN architecture.

Jima = zeroModel (HASS + SlicedFFN + AdaptiveRouter + MoE + MergerGate + CoT)
       + Agentic extensions (ActionHead, CritiqueModule, memory_vault, recurrence)

This replaces the previous ZeroAgenticModel which used a simplified
FlashSSM+GLA architecture. Jima now uses the SAME HASS blocks as zeroModel,
with agentic capabilities layered on top.

The agentic extensions are:
  1. ActionHead — predicts agentic actions (output/search/tool/reason)
  2. CritiqueModule — evaluates reasoning quality for recursion control
  3. RecursiveRouter — decides whether to re-process tokens (halt probability)
  4. MemoryVault — persistent associative memory across the sequence
  5. Recurrence — re-processes tokens through HASS blocks when quality is low

The base forward pass (embeddings → router → HASS blocks → MoE → merger → logits)
is identical to zeroModel. The agentic extensions add:
  - Memory vault retrieval injected after embeddings
  - Recursive re-processing of HASS blocks based on self-critique
  - Action head parallel to LM head
  - Ponder cost regularization (rewards efficient thinking)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from xorzen.model.base import BaseModel, ModelOutput
from xorzen.config import ModelConfig
from xorzen.utils.logger import get_logger
from xorzen.utils.math_utils import RMSNorm
from xorzen.model.components.hass_block import HASSBlock
from xorzen.model.components.routing import AdaptiveRouter
from xorzen.model.components.merger import xorzenMergerGate
from xorzen.model.components.cot_vector import InternalLatentCoT
from xorzen.model.zmoe import ShardedExpertFabric

from .agentic_config import ZeroAgenticConfig

logger = get_logger()


# ═══════════════════════════════════════════════════════════════════
# Agentic Extensions (the "Bit Upgrades" on top of XORZEN core)
# ═══════════════════════════════════════════════════════════════════

class ActionHead(nn.Module):
    """
    Predicts agentic actions per token.
    Actions: 0=Output Token, 1=Search Memory, 2=Use Tool, 3=Internal Reasoning
    """
    def __init__(self, d_model, num_actions):
        super().__init__()
        self.proj = nn.Linear(d_model, num_actions)

    def forward(self, x):
        return self.proj(x)


class CritiqueModule(nn.Module):
    """
    Evaluates reasoning quality per token.
    Outputs a sigmoid score [0,1] — low quality triggers more recursion.
    """
    def __init__(self, d_model):
        super().__init__()
        self.quality_eval = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.quality_eval(x)


class RecursiveRouter(nn.Module):
    """
    Decides whether to halt (stop thinking) or continue recursing.
    Outputs a per-token halt probability [0,1].
    """
    def __init__(self, d_model):
        super().__init__()
        self.halt_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        return torch.sigmoid(self.halt_proj(x))


# ═══════════════════════════════════════════════════════════════════
# Jima Model — XORZEN core + agentic extensions
# ═══════════════════════════════════════════════════════════════════

class JimaModel(BaseModel):
    """
    Jima — Agentic XORZEN Model.

    Uses the FULL XORZEN architecture (HASS blocks with Local/LowRank/SSM
    pathways, SlicedFFN with per-token width, AdaptiveRouter with 4-axis
    routing, ShardedExpertFabric MoE, GatedMerger, InternalLatentCoT) PLUS
    agentic extensions (ActionHead, CritiqueModule, RecursiveRouter,
    memory_vault, recurrence).

    This is NOT a separate architecture — it IS zeroModel with agentic
    capabilities added on top. The forward pass through HASS blocks, MoE,
    and merger is identical to zeroModel. The agentic extensions add:

    1. Memory vault retrieval (after embeddings, before routing)
    2. Recursive HASS re-processing (based on self-critique quality)
    3. Action head (parallel to LM head)
    4. Ponder cost regularization (rewards efficient thinking)
    """

    def __init__(self, config: ZeroAgenticConfig, test_mode: bool = True):
        super().__init__(config)
        self.config = config
        self.test_mode = test_mode

        logger.info("jima", f"Initializing Jima ({config.hidden_size} dim, {config.num_layers} layers, "
                            f"recurrence_depth={config.recurrence_depth})")

        # ═══════════════════════════════════════════════════════════════
        # 1. XORZEN Core (identical to zeroModel)
        # ═══════════════════════════════════════════════════════════════

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # CoT (frozen during pre-training, unfrozen for fine-tuning)
        self.cot = InternalLatentCoT(config)
        self._cot_enabled = False
        self._freeze_cot()

        # Adaptive Router (4-axis: depth, width, path, expert)
        self.router = AdaptiveRouter(config)

        # HASS Blocks (Local Attention + Low-Rank Global + SSM + SlicedFFN)
        self.blocks = nn.ModuleList([
            HASSBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])

        # MoE (Sharded Expert Fabric with LRU cache)
        self.moe = ShardedExpertFabric(config, test_mode=test_mode)

        # Merger Gate (fuses HASS + MoE + CoT)
        self.merger = xorzenMergerGate(config)

        # Output
        self.final_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tied
        self.token_embedding.weight.data[config.pad_token_id].zero_()

        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing

        # Routing regularizer
        from xorzen.model.components.routing import RoutingRegularizer
        self.routing_regularizer = RoutingRegularizer(config)

        # ═══════════════════════════════════════════════════════════════
        # 2. Agentic Extensions (Jima-specific)
        # ═══════════════════════════════════════════════════════════════

        # Persistent Memory Vault
        self.memory_vault = nn.Parameter(
            torch.randn(1, config.memory_slots, config.hidden_size) * 0.02
        )

        # Agentic components
        self.action_head = ActionHead(config.hidden_size, config.num_action_slots)
        self.critique = CritiqueModule(config.hidden_size)
        self.recursive_router = RecursiveRouter(config.hidden_size)

        # Initialize weights
        self.apply(self._init_weights)
        # Re-tie weights after apply
        self.lm_head.weight = self.token_embedding.weight
        self.token_embedding.weight.data[config.pad_token_id].zero_()
        # Scale lm_head
        self.lm_head.weight.data.mul_(1.0 / math.sqrt(config.hidden_size))

    def _freeze_cot(self):
        """Freeze CoT parameters (same as zeroModel)."""
        for param in self.cot.parameters():
            param.requires_grad = False
        self._cot_enabled = False

    def enable_cot(self):
        """Enable CoT for fine-tuning (same as zeroModel)."""
        for param in self.cot.parameters():
            param.requires_grad = True
        self._cot_enabled = True
        logger.info("jima", "CoT enabled for fine-tuning")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        action_targets: Optional[torch.Tensor] = None,
        action_loss_weight: float = 0.1,
        output_hidden_states: bool = False,
        output_routing_info: bool = False,
        return_dict: bool = True,
    ):
        """
        Forward pass — XORZEN core + agentic extensions.

        The base forward (embeddings → router → HASS → MoE → merger → logits)
        is identical to zeroModel. Agentic extensions add:
        - Memory vault retrieval after embeddings
        - Recursive HASS re-processing based on self-critique
        - Action head parallel to LM head
        - Ponder cost regularization
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        if seq_length > self.config.context_length:
            raise ValueError(f"Sequence length {seq_length} exceeds maximum {self.config.context_length}")

        # ═══ STEP 1: EMBEDDINGS ═══
        hidden_states = self.token_embedding(input_ids)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        # ═══ STEP 1.5: MEMORY VAULT RETRIEVAL (agentic extension) ═══
        vault = self.memory_vault.expand(batch_size, -1, -1)
        mem_scores = torch.bmm(hidden_states, vault.transpose(1, 2)) / math.sqrt(self.config.hidden_size)
        mem_weights = F.softmax(mem_scores, dim=-1)
        mem_retrieved = torch.bmm(mem_weights, vault)
        hidden_states = hidden_states + mem_retrieved

        # ═══ STEP 2: CoT ═══
        cot_total_dim = self.config.cot_dim * self.config.cot_components
        if self._cot_enabled:
            cot_vector_seq, _ = self.cot(hidden_states)
        else:
            cot_vector_seq = torch.zeros(batch_size, seq_length, cot_total_dim, device=device, dtype=hidden_states.dtype)

        # ═══ STEP 3: ADAPTIVE ROUTING ═══
        routing_decision = self.router(
            x=hidden_states,
            cot_features=cot_vector_seq,
            training=self.training,
            position_ids=position_ids,
        )

        depth_mask = routing_decision.depth_mask
        expert_indices = routing_decision.expert_indices
        expert_weights = routing_decision.expert_weights

        # ═══ STEP 4: HASS BLOCKS (with agentic recurrence) ═══
        layer_outputs = [] if output_hidden_states else None
        total_recurrence_steps = 0
        ponder_cost = 0.0

        for layer_idx, block in enumerate(self.blocks):
            layer_mask = depth_mask[:, :, layer_idx]

            # Standard HASS block forward (same as zeroModel)
            if not self.training:
                active = (layer_mask > 0.5)
                n_active = int(active.sum().item())
                if n_active == 0:
                    if output_hidden_states:
                        layer_outputs.append(hidden_states)
                    continue
                if n_active < batch_size * seq_length:
                    hidden_states = block.forward_with_depth(
                        x=hidden_states, depth_mask=active.float(),
                        routing_decision=routing_decision, attention_mask=attention_mask,
                    )
                    if output_hidden_states:
                        layer_outputs.append(hidden_states)
                    continue

            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                def create_fwd(blk, rd, am):
                    def fwd(hs):
                        out, _ = blk(x=hs, routing_decision=rd, attention_mask=am)
                        return out
                    return fwd
                block_out = checkpoint(create_fwd(block, routing_decision, attention_mask), hidden_states, use_reentrant=False)
            else:
                block_out, _ = block(
                    x=hidden_states, routing_decision=routing_decision,
                    attention_mask=attention_mask,
                )

            layer_mask_3d = layer_mask.unsqueeze(-1)
            hidden_states = block_out * layer_mask_3d + hidden_states * (1.0 - layer_mask_3d)

            # ═══ AGENTIC RECURRENCE (Jima extension) ═══
            if self.config.recurrence_depth > 1 and not self.training:
                for rec_step in range(self.config.recurrence_depth - 1):
                    halt_prob = self.recursive_router(hidden_states)
                    quality = self.critique(hidden_states)
                    agentic_halt = halt_prob * quality
                    ponder_cost = ponder_cost + agentic_halt.mean()

                    # Re-process through the same HASS block
                    new_hidden, _ = block(
                        x=hidden_states, routing_decision=routing_decision,
                        attention_mask=attention_mask,
                    )
                    hidden_states = agentic_halt * hidden_states + (1 - agentic_halt) * new_hidden

                    if agentic_halt.mean() > 0.8:
                        break
                    total_recurrence_steps += 1

            if output_hidden_states:
                layer_outputs.append(hidden_states)

        # ═══ STEP 5: MoE EXPERTS ═══
        batch_seq = batch_size * seq_length
        hidden_flat = hidden_states.reshape(batch_seq, -1)
        expert_indices_flat = expert_indices.reshape(batch_seq, -1)
        expert_weights_flat = expert_weights.reshape(batch_seq, -1)
        moe_output_flat, expert_stats = self.moe(
            hidden_flat, expert_indices_flat, expert_weights_flat,
            attention_mask=attention_mask.reshape(batch_seq) if attention_mask is not None else None,
        )
        moe_output = moe_output_flat.reshape(batch_size, seq_length, -1)

        # ═══ STEP 6: MERGER GATE ═══
        merged_output = self.merger(
            hass_output=hidden_states, moe_output=moe_output,
            cot_vector=cot_vector_seq, attention_mask=attention_mask,
        )

        # ═══ STEP 7: OUTPUT ═══
        hidden_states = self.final_norm(merged_output)
        logits = self.lm_head(hidden_states)

        # Action head (agentic extension)
        actions = self.action_head(hidden_states)

        # ═══ STEP 8: LOSS ═══
        loss = None
        lm_loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(f"labels shape {tuple(labels.shape)} must match input_ids shape {tuple(input_ids.shape)}")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=getattr(self.config, "pad_token_id", -100))
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = lm_loss

            # Agentic: reward efficient thinking (subtract ponder cost)
            loss = loss - 0.01 * ponder_cost

            # Action head regularization
            if action_targets is not None:
                loss = loss + action_loss_weight * F.mse_loss(actions, action_targets)
            else:
                action_probs = F.softmax(actions, dim=-1)
                action_log_probs = F.log_softmax(actions, dim=-1)
                entropy_reg = -(action_probs * action_log_probs).sum(dim=-1).mean()
                loss = loss - 0.01 * entropy_reg

        # Routing losses
        routing_loss = self.routing_regularizer(routing_decision)
        load_balance_loss = torch.tensor(0.0, device=device)
        if hasattr(routing_decision, 'auxiliary') and routing_decision.auxiliary:
            _AUX_LOSS_KEYS = frozenset({'load_balance_loss', 'router_z_loss', 'path_div_loss', 'width_div_loss'})
            for aux_key, aux_val in routing_decision.auxiliary.items():
                if aux_key in _AUX_LOSS_KEYS and isinstance(aux_val, torch.Tensor) and aux_val.requires_grad:
                    routing_loss = routing_loss + aux_val

        if loss is not None:
            loss = loss + routing_loss

        if not return_dict:
            outputs = (logits, loss) if loss is not None else (logits,)
            if output_routing_info:
                outputs += (routing_decision,)
            return outputs

        return ModelOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm_loss,
            cot_vector=cot_vector_seq,
            routing_info=routing_decision if output_routing_info else None,
            layer_outputs=layer_outputs,
            expert_stats=expert_stats,
            routing_loss=routing_loss,
            load_balance_loss=load_balance_loss,
            active_params=None,
            compute_cost=None,
        )


# ═══════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════

def jima_nano(vocab_size=10000, **kwargs):
    """Jima Nano — compact agentic XORZEN (~5M params)."""
    config = ZeroAgenticConfig(
        hidden_size=256, num_layers=6, num_attention_heads=8,
        vocab_size=vocab_size, recurrence_depth=3, memory_slots=32,
        max_depth=6, min_depth=2,
        use_sliced_ffn=True, width_choices=(128, 256),
        expert_count=4, top_k_experts=2, pathway_top_k=2,
        **kwargs
    )
    return JimaModel(config, test_mode=True)


def jima_micro(vocab_size=10000, **kwargs):
    """Jima Micro — agentic XORZEN (~50M params)."""
    config = ZeroAgenticConfig(
        hidden_size=512, num_layers=12, num_attention_heads=8,
        vocab_size=vocab_size, recurrence_depth=4, memory_slots=64,
        max_depth=12, min_depth=3,
        use_sliced_ffn=True, width_choices=(256, 512),
        expert_count=8, top_k_experts=2, pathway_top_k=2,
        **kwargs
    )
    return JimaModel(config, test_mode=True)

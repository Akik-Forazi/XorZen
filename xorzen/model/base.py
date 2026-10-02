"""
Base Model for xorzen
Version: 1.0
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Iterator
from dataclasses import dataclass, fields as _dc_fields

@dataclass(init=False)
class ModelOutput:
    """Complete model output with all intermediate states.
    
    Compatible with ``torch.nn.DataParallel.gather()`` — supports both
    normal construction ``ModelOutput(logits=..., loss=...)`` and the
    ``ModelOutput(map_object)`` call pattern used by PyTorch's
    ``gather_map`` fallthrough branch when reconstructing gathered
    outputs from multiple GPU replicas.
    """
    # Primary outputs
    logits: torch.Tensor  # [batch, seq, vocab] - next token predictions
    loss: Optional[torch.Tensor] = None  # scalar - training loss
    
    # Internal states (for analysis/debugging)
    cot_vector: Optional[torch.Tensor] = None  # [batch, seq, cot_dim] - reasoning trace
    routing_info: Optional[Any] = None  # routing decisions
    layer_outputs: Optional[List[torch.Tensor]] = None  # intermediate layer outputs
    expert_stats: Optional[Dict[str, Any]] = None  # expert usage statistics
    
    # Auxiliary losses
    routing_loss: Optional[torch.Tensor] = None  # routing regularization
    load_balance_loss: Optional[torch.Tensor] = None  # expert load balancing
    cot_consistency_loss: Optional[torch.Tensor] = None  # CoT stability
    lm_loss: Optional[torch.Tensor] = None  # pure cross-entropy (before aux losses added)
    
    # Performance metrics
    active_params: Optional[int] = None  # number of active parameters
    compute_cost: Optional[float] = None  # FLOPs estimate
    
    def __init__(self, *args, **kwargs):
        """Initialize ModelOutput.
        
        Supports three call patterns:
        1. ``ModelOutput(logits=t, loss=l, ...)`` — keyword args (preferred).
        2. ``ModelOutput(logits_tensor, loss=...)`` — positional + keyword args.
        3. ``ModelOutput(iterable_of_field_values)`` — single iterable positional
           arg with field values in declaration order. This pattern is used
           by ``torch.nn.DataParallel.gather()`` when reconstructing gathered
           outputs from multiple GPU replicas via
           ``type(out)(map(gather_map, zip(*outputs)))``.
        """
        field_names = [f.name for f in _dc_fields(self)]
        
        # Pattern 3: single iterable positional arg (PyTorch's gather_map call)
        if (len(args) == 1 and not kwargs
                and not isinstance(args[0], torch.Tensor)
                and not isinstance(args[0], (str, bytes))
                and hasattr(args[0], '__iter__')):
            values = list(args[0])
            if len(values) == len(field_names):
                for name, value in zip(field_names, values):
                    setattr(self, name, value)
                return
            # Fall through to default handling if length doesn't match
        
        # Pattern 1 + 2: positional args followed by keyword args
        for name, value in zip(field_names, args):
            setattr(self, name, value)
        for name, value in kwargs.items():
            if name not in field_names:
                raise TypeError(f"__init__() got an unexpected keyword argument '{name}'")
            setattr(self, name, value)
    
    def total_loss(self) -> Optional[torch.Tensor]:
        """Compute total weighted loss."""
        if self.loss is None:
            return None
        
        total = self.loss
        if self.routing_loss is not None:
            total = total + self.routing_loss
        if self.load_balance_loss is not None:
            total = total + self.load_balance_loss
        if self.cot_consistency_loss is not None:
            total = total + self.cot_consistency_loss
        
        return total
    
    def __iter__(self) -> Iterator:
        """Iterate over field values in declaration order, with non-tensor
        fields replaced by None.
        
        This makes ModelOutput compatible with ``torch.nn.DataParallel.gather()``.
        Without ``__iter__``, PyTorch's ``gather_map`` falls through to
        ``type(out)(map(gather_map, zip(*outputs)))`` which crashes with
        ``TypeError: 'ModelOutput' object is not iterable``.
        
        Non-tensor fields (``active_params``, ``compute_cost``, ``routing_info``,
        ``expert_stats``, ``layer_outputs``) are yielded as ``None`` because
        PyTorch's ``gather_map`` cannot gather plain ints/floats/dicts/lists
        (it tries to ``zip(*outputs)`` on them, which fails for non-iterable
        types). The gathered ModelOutput will have ``None`` for these fields.
        This is acceptable because they are diagnostic metadata with no impact
        on loss or gradients.
        
        Tensor fields (``logits``, ``loss``, ``cot_vector``, ``routing_loss``,
        ``load_balance_loss``, ``cot_consistency_loss``, ``lm_loss``) are
        yielded as-is. PyTorch's ``Gather.apply`` concatenates them along
        dim=0. For batched tensors (``logits``, ``cot_vector``) this is correct.
        For scalar tensors (``loss``, aux losses) the result is a 1-D tensor
        with one element per replica — call ``.mean()`` to recover a scalar.
        """
        for f in _dc_fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                yield v
            else:
                yield None

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    
    # Special tokens
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    # Early stopping
    early_stopping: bool = True
    min_length: int = 10
    
    # Advanced
    no_repeat_ngram_size: int = 0
    length_penalty: float = 1.0

class BaseModel(nn.Module):
    """
    Base class for all xorzen models.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, path: str, **kwargs):
        raise NotImplementedError

    def load_checkpoint(self, path: str, **kwargs):
        raise NotImplementedError

    def count_parameters(self, only_trainable: bool = False) -> int:
        """Count total or trainable parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_memory_footprint(self) -> Dict[str, Any]:
        """Get model memory footprint."""
        param_size = 0
        param_count = 0
        buffer_size = 0
        buffer_count = 0
        
        for param in self.parameters():
            param_count += param.numel()
            param_size += param.numel() * param.element_size()
        
        for buffer in self.buffers():
            buffer_count += buffer.numel()
            buffer_size += buffer.numel() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'param_count': param_count,
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_count': buffer_count,
            'buffer_size_mb': buffer_size / (1024 ** 2),
            'total_size_mb': total_size / (1024 ** 2),
            'total_size_gb': total_size / (1024 ** 3)
        }


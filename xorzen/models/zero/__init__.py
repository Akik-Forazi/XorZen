"""
zero Model Package
Provides the core zero architecture and all variants, including agentic sub-variants.

Usage:
    # Method 1: Direct variant import (RECOMMENDED)
    >>> from xorzen.models.zero import zero_277M
    >>> model = zero_277M()

    # Method 2: Base model with config
    >>> from xorzen.models.zero import zeroModel
    >>> from xorzen.config import ConfigFactory, ModelSize
    >>> config = ConfigFactory.get_config(ModelSize.MINI_277M)
    >>> model = zeroModel(config)

    # Method 3: Legacy compatibility
    >>> from xorzen.models.zero import zero277M  # Alias for zero_277M
    >>> model = zero277M()

    # Method 4: Agentic variants (Jima — recursive inference with self-critique)
    >>> from xorzen.models.zero import jima_nano
    >>> model = jima_nano()

    # Method 5: Legacy agentic aliases (now use Jima internally)
    >>> from xorzen.models.zero import zero_agentic_nano
    >>> model = zero_agentic_nano()  # Same as jima_nano
"""

# Core architecture
from .model import zeroModel

# All variants (explicit parameter counts)
from .variants import (
    zeroBase,
    zero_tiny_23k,
    zero_1M,
    zero_10M,
    zero_50M,
    zero_277M,
    zero_500M,
    zero_1_3B,
    zero_7B,
    # Legacy aliases
    zeroModel277M,
    zero277M,
)

# Jima (Agentic XORZEN) — uses full HASS/SlicedFFN/AdaptiveRouter/MoE/Merger
from .jima_model import JimaModel, jima_nano, jima_micro

# Agentic config (shared between old and new)
from .agentic_config import ZeroAgenticConfig, ZeroAgenticSize

# Backward compatibility: old ZeroAgenticModel now aliases JimaModel
from .jima_model import JimaModel as ZeroAgenticModel

# Backward compatibility: old agentic factory functions now use Jima
def zero_agentic_nano(vocab_size=10000, **kwargs):
    """Legacy alias for jima_nano."""
    return jima_nano(vocab_size, **kwargs)

def zero_agentic_micro(vocab_size=10000, **kwargs):
    """Legacy alias for jima_micro."""
    return jima_micro(vocab_size, **kwargs)

__all__ = [
    # Core
    'zeroModel',
    'zeroBase',

    # Variants (by size)
    'zero_tiny_23k',
    'zero_1M',
    'zero_10M',
    'zero_50M',
    'zero_277M',
    'zero_500M',
    'zero_1_3B',
    'zero_7B',

    # Legacy
    'zeroModel277M',
    'zero277M',

    # Jima (Agentic XORZEN)
    'JimaModel',
    'jima_nano',
    'jima_micro',

    # Agentic (backward compat)
    'ZeroAgenticConfig',
    'ZeroAgenticSize',
    'ZeroAgenticModel',  # alias for JimaModel
    'zero_agentic_nano',  # alias for jima_nano
    'zero_agentic_micro',  # alias for jima_micro
]

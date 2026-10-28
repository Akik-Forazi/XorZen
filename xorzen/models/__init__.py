
"""
xorzen Models Package
Central hub for all model architectures and variants.

Usage:
    # List available models
    >>> from xorzen.models import list_models
    >>> print(list_models())
    ['zero_1m', 'zero_10m', 'zero_50m', 'zero_277m', ...]

    # Get model by name
    >>> from xorzen.models import get_model
    >>> ModelClass = get_model('zero_277m')
    >>> model = ModelClass()

    # Direct import (recommended)
    >>> from xorzen.models.zero import zero_277M
    >>> model = zero_277M()
"""

# Registry system
from .registry import (
    ModelRegistry,
    list_models,
    get_model,
    create_model,
)

# zero models
from .zero import (
    zeroModel,
    zeroBase,
    zero_1M,
    zero_10M,
    zero_50M,
    zero_277M,
    zero_500M,
    zero_1_3B,
    zero_7B,
    zero277M,  # Legacy alias
)

# Jima (Agentic XORZEN) — uses full HASS/SlicedFFN/AdaptiveRouter/MoE/Merger
from .zero.jima_model import JimaModel, jima_nano, jima_micro
from .zero.agentic_config import ZeroAgenticConfig

# Backward compatibility: ZeroAgenticModel now aliases JimaModel
from .zero.jima_model import JimaModel as ZeroAgenticModel

# Backward compatibility: old agentic factory functions
def zero_agentic_nano(vocab_size=10000, **kwargs):
    return jima_nano(vocab_size, **kwargs)

def zero_agentic_micro(vocab_size=10000, **kwargs):
    return jima_micro(vocab_size, **kwargs)

# Greed model (conversation quality evaluator)
from .greed import GreedModel

__all__ = [
    # Registry
    'ModelRegistry',
    'list_models',
    'get_model',
    'create_model',

    # zero
    'zeroModel',
    'zeroBase',
    'zero_1M',
    'zero_10M',
    'zero_50M',
    'zero_277M',
    'zero_500M',
    'zero_1_3B',
    'zero_7B',
    'zero277M',

    # Jima (Agentic XORZEN)
    'JimaModel',
    'jima_nano',
    'jima_micro',

    # Agentic (backward compat — aliases for Jima)
    'ZeroAgenticModel',  # alias for JimaModel
    'ZeroAgenticConfig',
    'zero_agentic_nano',  # alias for jima_nano
    'zero_agentic_micro',  # alias for jima_micro

    # Greed
    'GreedModel',
]

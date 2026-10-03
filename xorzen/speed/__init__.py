"""
xorzen Speed Booster — Master __init__
"""
from .booster import SpeedBooster, boost_model, SpeedProfile
from .fast_attention import FlashLocalAttention
from .fast_ssm import FastSSMPathway
from .fast_router import CachedRouter
from .fast_moe import PreloadedExpertFabric
from .fast_trainer import FastTrainer
from .xorzen_compile import compile as xorzen_compile, warmup_compiled
from .native_kernels import (
    native_available,
    fused_rmsnorm,
    fused_gelu,
    fused_swiglu,
    fused_layernorm_gelu,
    expert_dispatch,
    diagonal_ssm_scan,
    get_native_status,
    auto_patch_model,
)

__all__ = [
    "SpeedBooster", "boost_model", "SpeedProfile",
    "FlashLocalAttention", "FastSSMPathway",
    "CachedRouter", "PreloadedExpertFabric", "FastTrainer",
    "xorzen_compile", "warmup_compiled",
    # Native C++ kernels
    "native_available", "fused_rmsnorm", "fused_gelu", "fused_swiglu",
    "fused_layernorm_gelu", "expert_dispatch", "diagonal_ssm_scan",
    "get_native_status", "auto_patch_model",
]

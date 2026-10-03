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

__all__ = [
    "SpeedBooster", "boost_model", "SpeedProfile",
    "FlashLocalAttention", "FastSSMPathway",
    "CachedRouter", "PreloadedExpertFabric", "FastTrainer",
    "xorzen_compile", "warmup_compiled",
]

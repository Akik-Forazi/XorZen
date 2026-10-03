"""
KIDŌ MOTION — Kinetic Coherence Architecture Module for XORZEN
機動モーション — Anime Video Generation Component

This module provides anime-specific video generation capabilities to XORZEN,
including:
  - Coherence Field Theory (CFT) for character consistency
  - Spectral Temporal Decomposition (STD) for motion generation
  - Directed Consistency Manifolds (DCM) for real-time director control
  - Harmonic Cross-Modal Embedding (HCME) for phase-aligned multimodal fusion
  
KIDŌ is NOT a standalone model - it's a component that enhances XORZEN
with anime generation capabilities.
"""

from .coherence_field import CoherenceFieldTheory, CoherenceFieldConfig
from .spectral_motion import SpectralTemporalDecomposition, SpectralMotionConfig
from .manifold import DirectedConsistencyManifold, ManifoldConfig
from .harmonic_embedding import HarmonicCrossModalEmbedding, HarmonicConfig
from .kido_block import KidoBlock, KidoConfig

__all__ = [
    'CoherenceFieldTheory',
    'CoherenceFieldConfig',
    'SpectralTemporalDecomposition',
    'SpectralMotionConfig',
    'DirectedConsistencyManifold',
    'ManifoldConfig',
    'HarmonicCrossModalEmbedding',
    'HarmonicConfig',
    'KidoBlock',
    'KidoConfig',
]

__version__ = '1.0.0'
__author__ = 'Akik Faraji — FRAZIYM AI'

"""
Multi-Fidelity System for MRMFMBD.

Theory-correct coarse-to-fine fidelity ladder:
   - Early steps: coarse sim (low cost, fast)
   - Late steps: fine sim (high fidelity, accurate)
   - Optional: adaptive upgrade based on score gap

Cost model: total_cost ∝ Σ_k cost(ℓ_k), prefer more steps at coarse.
"""

from __future__ import annotations

from .specs import FidelityLevelSpec, FidelitySystemConfig, default_fidelity_system_config
from .ladder import (
    BlockFidelityLadder,
    FidelityLadderType,
    create_fidelity_ladder,
)
from .upgrade import UpgradeRule, ScoreGapUpgradeRule, ESSUpgradeRule, CompositeUpgradeRule

__all__ = [
    "FidelityLevelSpec",
    "FidelitySystemConfig",
    "default_fidelity_system_config",
    "BlockFidelityLadder",
    "FidelityLadderType",
    "create_fidelity_ladder",
    "UpgradeRule",
    "ScoreGapUpgradeRule",
    "ESSUpgradeRule",
    "CompositeUpgradeRule",
]

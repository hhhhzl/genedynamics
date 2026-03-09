"""
Inference primitives for model-based diffusion and posterior sampling.

This module provides task-agnostic abstractions for:

- annealed_bridge: Annealed posterior bridge π_k ∝ p0 * p(y|θ)^β_k
- mcsa: Monte Carlo Score Ascent (proposal, weighting, score estimation)
- fidelity: Multi-fidelity ladder and upgrade rules
- diagnostics: ESS, weight entropy, degeneracy flags

Used by:
- mbd3d: 3DGS robust mapping (observation-likelihood bridge)
- mrmfmbd: Soft-robot mode-marginal + multi-fidelity
"""

from .annealed_bridge import (
    BridgeSchedule,
    BridgeScheduleConfig,
    LinearBridgeSchedule,
    GeometricBridgeSchedule,
    log_pi,
    create_linear_bridge_schedule,
    create_geometric_bridge_schedule,
)
from .mcsa import (
    ProposalSampler,
    ImportanceWeighter,
    MCSAScoreEstimator,
    MCSADiagnostics,
)
from .fidelity import (
    FidelityLadder,
    FidelityConfig,
    UpgradeRule,
    ScoreGapUpgradeRule,
)
from .diagnostics import (
    effective_sample_size,
    weight_entropy,
    degeneracy_flags,
    InferenceDiagnostics,
)

__all__ = [
    # Annealed bridge
    "BridgeSchedule",
    "BridgeScheduleConfig",
    "LinearBridgeSchedule",
    "GeometricBridgeSchedule",
    "log_pi",
    "create_linear_bridge_schedule",
    "create_geometric_bridge_schedule",
    # MCSA
    "ProposalSampler",
    "ImportanceWeighter",
    "MCSAScoreEstimator",
    "MCSADiagnostics",
    # Fidelity
    "FidelityLadder",
    "UpgradeRule",
    "ScoreGapUpgradeRule",
    "FidelityConfig",
    # Diagnostics
    "effective_sample_size",
    "weight_entropy",
    "degeneracy_flags",
    "InferenceDiagnostics",
]

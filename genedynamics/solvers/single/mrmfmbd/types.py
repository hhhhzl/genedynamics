"""
Type definitions for MRMFMBD (soft-robot co-design (mode marginalization + fidelity ladder)) solver.

Multi-Resolution, Multi-Fidelity Model-Based Diffusion for deformable/soft robots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

Array = Any


@dataclass
class FidelityLevel:
    """
    Fidelity level descriptor for multi-fidelity simulation.

    Attributes:
        level: integer level (0=coarse, 1=medium, 2=fine)
        mesh_resolution: optional mesh refinement factor
        sub_steps: optional sub-steps per dt
        physics_accuracy: optional accuracy flag
        state_dim: state dimension at this level
        act_dim: action dimension
    """

    level: int
    mesh_resolution: Optional[float] = None
    sub_steps: Optional[int] = None
    physics_accuracy: Optional[str] = None
    state_dim: Optional[int] = None
    act_dim: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "mesh_resolution": self.mesh_resolution,
            "sub_steps": self.sub_steps,
            "physics_accuracy": self.physics_accuracy,
            "state_dim": self.state_dim,
            "act_dim": self.act_dim,
        }


@dataclass
class ModeRegime:
    """
    Contact/friction regime for mode-marginalization.

    Attributes:
        regime_id: integer regime index
        name: human-readable name (e.g. "sticking", "sliding")
        log_prior: log p(regime)
    """

    regime_id: int
    name: str
    log_prior: float = 0.0


@dataclass
class MRMFMBDResult:
    """
    Result from MRMFMBD solve.

    Attributes:
        states: trajectory states (H+1, state_dim)
        actions: trajectory actions (H, act_dim)
        rewards: per-step rewards (H,)
        fidelity_history: fidelity level used per diffusion step
        mode_responsibilities: optional (K,) responsibilities if mode-marginal
        diagnostics: MCSA/fidelity diagnostics
        diffusion_history: optional diffusion trajectory
    """

    states: List[Array]
    actions: List[Array]
    rewards: Array
    fidelity_history: Optional[List[int]] = None
    mode_responsibilities: Optional[Array] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    diffusion_history: Optional[Dict[str, Any]] = None

    def to_trajectory_info(self) -> Dict[str, Any]:
        return {
            "rewards": self.rewards,
            "fidelity_history": self.fidelity_history,
            "mode_responsibilities": self.mode_responsibilities,
            "diagnostics": self.diagnostics,
            "diffusion_history": self.diffusion_history,
        }

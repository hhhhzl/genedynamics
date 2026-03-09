"""
Multi-fidelity ladder and upgrade rules.

Provides:
- FidelityLadder: maps step index k to fidelity level ℓ
- UpgradeRule: protocol for deciding when to upgrade fidelity
- ScoreGapUpgradeRule: upgrade when ‖s_ℓ - s_{ℓ-1}‖ > δ
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


@dataclass
class FidelityConfig:
    """
    Configuration for multi-fidelity ladder.

    Attributes:
        levels: list of fidelity levels (e.g. [0, 0, 1, 1, 2])
        ladder: explicit mapping k -> ℓ (overrides levels if len matches K)
        step_ratio: for geometric ladder, ratio of steps per level
    """

    levels: Optional[List[int]] = None
    ladder: Optional[Dict[int, int]] = None
    step_ratio: float = 1.5
    num_levels: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "levels": self.levels,
            "ladder": self.ladder,
            "step_ratio": self.step_ratio,
            "num_levels": self.num_levels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FidelityConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class FidelityLadder:
    """
    Maps diffusion step index k to simulation fidelity level ℓ.

    Supports:
    - Explicit ladder: list or dict
    - Geometric: early steps at coarse, late steps at fine
    """

    def __init__(self, config: FidelityConfig, K: int = 100):
        self.config = config
        self.K = K
        self._ladder = self._build_ladder()

    def _build_ladder(self) -> List[int]:
        if self.config.levels is not None and len(self.config.levels) >= self.K:
            return self.config.levels[: self.K]
        if self.config.ladder is not None:
            return [self.config.ladder.get(k, 0) for k in range(self.K)]

        # Geometric: divide K into segments, each level gets more steps
        L = self.config.num_levels
        r = self.config.step_ratio
        # Geometric progression: n_0 + n_0*r + n_0*r^2 + ... = K
        # n_0 * (1 + r + r^2 + ... + r^{L-1}) = n_0 * (r^L - 1)/(r - 1) = K
        if abs(r - 1.0) < 1e-6:
            n_per = self.K // L
            remainder = self.K % L
            ladder = []
            for ell in range(L):
                n = n_per + (1 if ell < remainder else 0)
                ladder.extend([ell] * n)
            return ladder[: self.K]

        total = (r ** L - 1) / (r - 1)
        n0 = self.K / total
        n_per_level = [max(1, int(n0 * (r ** i))) for i in range(L)]
        remainder = self.K - sum(n_per_level)
        n_per_level[-1] += remainder
        ladder = []
        for ell, n in enumerate(n_per_level):
            ladder.extend([ell] * n)
        return ladder[: self.K]

    def __call__(self, k: int) -> int:
        """Return fidelity level at step k."""
        return self._ladder[min(k, self.K - 1)]

    def level(self, k: int) -> int:
        """Alias for __call__."""
        return self(k)


class UpgradeRule(ABC):
    """Protocol for deciding when to upgrade fidelity during inference."""

    @abstractmethod
    def should_upgrade(
        self,
        k: int,
        score_curr: Any,
        score_prev: Any,
        fidelity_curr: int,
        fidelity_prev: int,
        **kwargs: Any,
    ) -> bool:
        """Return True if fidelity should be upgraded."""
        pass


@dataclass
class ScoreGapUpgradeRule(UpgradeRule):
    """
    Upgrade when ‖s_ℓ - s_{ℓ-1}‖ > δ.

    Used when coarse and fine scores disagree significantly.
    """

    delta: float = 0.1
    norm: str = "l2"
    backend: str = "jax"

    def should_upgrade(
        self,
        k: int,
        score_curr: Any,
        score_prev: Any,
        fidelity_curr: int,
        fidelity_prev: int,
        **kwargs: Any,
    ) -> bool:
        if fidelity_curr <= fidelity_prev:
            return False
        diff = score_curr - score_prev
        if self.backend == "jax" and JAX_AVAILABLE:
            gap = float(jnp.linalg.norm(jnp.ravel(diff)))
        else:
            gap = float(np.linalg.norm(np.ravel(diff)))
        return gap > self.delta

"""
S3 Fidelity Upgrade Rules: when to switch from coarse to fine.

Used for adaptive fidelity: upgrade when coarse and fine disagree (score gap)
or when ESS is low (need more accurate evaluation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


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

    Coarse and fine scores disagree significantly -> need fine for accuracy.
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
        if fidelity_curr <= fidelity_prev or score_prev is None:
            return False
        diff = score_curr - score_prev
        if self.backend == "jax" and JAX_AVAILABLE and hasattr(diff, "block_until_ready"):
            gap = float(jnp.linalg.norm(jnp.ravel(diff)))
        else:
            gap = float(np.linalg.norm(np.ravel(np.asarray(diff))))
        return gap > self.delta


@dataclass
class ESSUpgradeRule(UpgradeRule):
    """
    Upgrade when ESS < ess_min.

    Low ESS indicates degeneracy -> finer evaluation may help.
    """

    ess_min: float = 2.0

    def should_upgrade(
        self,
        k: int,
        score_curr: Any,
        score_prev: Any,
        fidelity_curr: int,
        fidelity_prev: int,
        *,
        ess: Optional[float] = None,
        **kwargs: Any,
    ) -> bool:
        if ess is None:
            return False
        return float(ess) < self.ess_min and fidelity_curr < kwargs.get("max_fidelity", 2)


@dataclass
class CompositeUpgradeRule(UpgradeRule):
    """
    Combine multiple rules: upgrade if ANY rule says upgrade.
    """

    rules: tuple = field(default_factory=tuple)
    require_all: bool = False

    def should_upgrade(
        self,
        k: int,
        score_curr: Any,
        score_prev: Any,
        fidelity_curr: int,
        fidelity_prev: int,
        **kwargs: Any,
    ) -> bool:
        if not self.rules:
            return False
        results = [
            r.should_upgrade(
                k, score_curr, score_prev, fidelity_curr, fidelity_prev, **kwargs
            )
            for r in self.rules
        ]
        return all(results) if self.require_all else any(results)

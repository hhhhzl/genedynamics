"""
S3 Fidelity Ladder: maps bridge step k to fidelity level ℓ.

Theory: Early steps use coarse (low cost), late steps use fine (high fidelity).
Cost-optimal: geometric allocation n_ℓ ∝ r^ℓ gives more steps at coarse.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .specs import FidelityLevelSpec, FidelitySystemConfig


class FidelityLadderType(str, Enum):
    """Ladder schedule type."""

    FIXED = "fixed"
    GEOMETRIC = "geometric"
    LINEAR = "linear"
    COSINE = "cosine"


def _build_geometric_ladder(K: int, L: int, r: float) -> List[int]:
    """
    Geometric: n_0 + n_0*r + n_0*r^2 + ... + n_0*r^{L-1} = K.

    More steps at coarse (level 0). n_ℓ = n_0 * r^ℓ.
    """
    if L <= 0:
        return [0] * K
    if abs(r - 1.0) < 1e-6:
        n_per = max(1, K // L)
        remainder = K - n_per * L
        ladder = []
        for ell in range(L):
            n = n_per + (1 if ell < remainder else 0)
            ladder.extend([ell] * n)
        return ladder[:K]

    total = (r ** L - 1) / (r - 1)
    n0 = K / total
    n_per_level = [max(1, int(n0 * (r ** i))) for i in range(L)]
    remainder = K - sum(n_per_level)
    n_per_level[-1] = max(1, n_per_level[-1] + remainder)
    ladder = []
    for ell, n in enumerate(n_per_level):
        ladder.extend([ell] * n)
    return ladder[:K]


def _build_linear_ladder(K: int, L: int) -> List[int]:
    """
    Linear: level increases linearly with k.
    ℓ(k) = round((L-1) * k / (K-1))
    """
    if K <= 1 or L <= 1:
        return [0] * K
    ladder = []
    for k in range(K):
        alpha = k / max(K - 1, 1)
        ell = min(int(round(alpha * (L - 1))), L - 1)
        ladder.append(ell)
    return ladder


def _build_cosine_ladder(K: int, L: int) -> List[int]:
    """
    Cosine: smooth transition via cosine schedule.
    ℓ(k) = round((L-1) * 0.5 * (1 - cos(π * k/(K-1))))
    """
    if K <= 1 or L <= 1:
        return [0] * K
    ladder = []
    for k in range(K):
        alpha = 0.5 * (1.0 - np.cos(np.pi * k / max(K - 1, 1)))
        ell = min(int(round(alpha * (L - 1))), L - 1)
        ladder.append(ell)
    return ladder


class FidelityLadderS3:
    """
    S3 Fidelity Ladder: maps bridge step k to fidelity level ℓ.

    Supports fixed, geometric, linear, cosine schedules.
    Cost-aware: geometric allocates more steps at coarse.
    """

    def __init__(
        self,
        config: FidelitySystemConfig,
        K: int = 100,
        *,
        explicit_ladder: Optional[List[int]] = None,
        ladder_type_override: Optional[str] = None,
    ):
        self.config = config
        self.K = K
        self._ladder = self._build_ladder(explicit_ladder, ladder_type_override)

    def _build_ladder(
        self,
        explicit: Optional[List[int]],
        type_override: Optional[str],
    ) -> List[int]:
        if explicit is not None and len(explicit) >= self.K:
            return explicit[: self.K]

        L = self.config.num_levels
        if L <= 0:
            return [0] * self.K

        ladder_type = type_override or self.config.ladder_type

        if ladder_type == FidelityLadderType.FIXED.value or ladder_type == "fixed":
            n_per = max(1, self.K // L)
            remainder = self.K - n_per * L
            ladder = []
            for ell in range(L):
                n = n_per + (1 if ell < remainder else 0)
                ladder.extend([ell] * n)
            return ladder[: self.K]

        if ladder_type == FidelityLadderType.GEOMETRIC.value or ladder_type == "geometric":
            return _build_geometric_ladder(
                self.K, L, self.config.step_ratio
            )

        if ladder_type == FidelityLadderType.LINEAR.value or ladder_type == "linear":
            return _build_linear_ladder(self.K, L)

        if ladder_type == FidelityLadderType.COSINE.value or ladder_type == "cosine":
            return _build_cosine_ladder(self.K, L)

        return _build_geometric_ladder(self.K, L, self.config.step_ratio)

    def __call__(self, k: int) -> int:
        """Return fidelity level at step k."""
        return self._ladder[min(max(0, k), self.K - 1)]

    def level(self, k: int) -> int:
        """Alias for __call__."""
        return self(k)

    def get_ladder(self) -> List[int]:
        """Return full ladder for inspection."""
        return list(self._ladder)

    def cumulative_cost(self) -> float:
        """Total cost of ladder: Σ_k cost(ℓ_k)."""
        return sum(self.config.get_cost(ell) for ell in self._ladder)

    def steps_per_level(self) -> Dict[int, int]:
        """Count steps per level."""
        counts: Dict[int, int] = {}
        for ell in self._ladder:
            counts[ell] = counts.get(ell, 0) + 1
        return counts


def create_fidelity_ladder(
    K: int = 100,
    num_levels: int = 3,
    ladder_type: str = "geometric",
    step_ratio: float = 1.5,
    *,
    config: Optional[FidelitySystemConfig] = None,
    explicit_levels: Optional[List[int]] = None,
) -> FidelityLadderS3:
    """
    Factory for FidelityLadderS3.

    Args:
        K: Number of bridge steps
        num_levels: Number of fidelity levels
        ladder_type: "fixed" | "geometric" | "linear" | "cosine"
        step_ratio: For geometric
        config: Override with full config
        explicit_levels: Override with explicit [ℓ_0, ℓ_1, ..., ℓ_{K-1}]

    Returns:
        FidelityLadderS3 instance
    """
    if config is None:
        from .specs import default_fidelity_system_config
        config = default_fidelity_system_config(num_levels)
    # Forward the step_ratio arg into the config (previously dead-dropped,
    # which meant all yaml-level `fidelity_step_ratio` overrides were ignored
    # and the ladder always defaulted to r=1.5 → fine-heavy).
    try:
        config.step_ratio = float(step_ratio)
    except Exception:
        pass
    # Also forward ladder_type so config is self-consistent.
    try:
        config.ladder_type = ladder_type
    except Exception:
        pass
    return FidelityLadderS3(
        config=config,
        K=K,
        explicit_ladder=explicit_levels,
        ladder_type_override=ladder_type,
    )

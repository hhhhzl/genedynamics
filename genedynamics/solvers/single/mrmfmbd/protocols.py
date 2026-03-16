"""
Protocols for MRMFMBD (Soft-robot S1+S3) solver.

Defines abstract interfaces for multi-fidelity simulation and
optional mode-marginalization (contact/friction regimes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from .types import FidelityLevel


# -----------------------------------------------------------------------------
# Multi-fidelity simulation
# -----------------------------------------------------------------------------


@runtime_checkable
class FidelitySimulator(Protocol):
    """
    Protocol for multi-fidelity rollout.

    At each fidelity level ℓ, rollout may use different mesh resolution,
    sub-steps, or physics accuracy.
    """

    def rollout_rewards(
        self,
        x0: Any,
        actions: Any,
        fidelity_level: int,
        **kwargs: Any,
    ) -> Any:
        """
        Rollout actions from x0 at given fidelity level, return rewards.

        Args:
            x0: initial state
            actions: (H, act_dim) action sequence
            fidelity_level: 0=coarse, 1=medium, 2=fine

        Returns:
            rewards: (H,) per-step rewards (negative cost)
        """
        ...

    def rollout_states(
        self,
        x0: Any,
        actions: Any,
        fidelity_level: int,
        **kwargs: Any,
    ) -> Any:
        """Rollout and return states (H+1, state_dim)."""
        ...

    def get_fidelity_levels(self) -> List[FidelityLevel]:
        """Return available fidelity level descriptors."""
        ...

    def state_dim(self, fidelity_level: int) -> int:
        """State dimension at fidelity level."""
        ...

    def act_dim(self) -> int:
        """Action dimension."""
        ...


# -----------------------------------------------------------------------------
# Mode-marginalization (contact/friction regimes)
# -----------------------------------------------------------------------------


@runtime_checkable
class ModeMarginalizer(Protocol):
    """
    Protocol for mode-marginalization over contact/friction regimes.

    Used when dynamics depend on discrete regime (sticking/sliding, etc.).
    """

    def log_prob_per_mode(
        self,
        x0: Any,
        actions: Any,
        rewards_per_mode: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Compute unnormalized log-prob per regime.

        Args:
            x0: initial state
            actions: action sequence
            rewards_per_mode: (K, H) rewards for each regime

        Returns:
            log_terms: (K,) unnormalized log p(data | regime)
        """
        ...

    def num_modes(self) -> int:
        """Number of regimes."""
        ...


# -----------------------------------------------------------------------------
# Abstract base implementations
# -----------------------------------------------------------------------------


class BaseFidelitySimulator(ABC):
    """Abstract base for FidelitySimulator implementations."""

    @abstractmethod
    def rollout_rewards(
        self,
        x0: Any,
        actions: Any,
        fidelity_level: int,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def rollout_states(
        self,
        x0: Any,
        actions: Any,
        fidelity_level: int,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def get_fidelity_levels(self) -> List[FidelityLevel]:
        pass

    def state_dim(self, fidelity_level: int) -> int:
        levels = self.get_fidelity_levels()
        for fl in levels:
            if fl.level == fidelity_level and fl.state_dim is not None:
                return fl.state_dim
        return levels[0].state_dim if levels else 0

    @abstractmethod
    def act_dim(self) -> int:
        pass

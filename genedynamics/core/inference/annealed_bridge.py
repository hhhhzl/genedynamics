"""
Annealed posterior bridge for model-based diffusion.

Defines π_k(θ) ∝ p0(θ) * p(y|θ)^β_k with 0 = β_0 < β_1 < ... < β_K = 1,
providing a smooth path from prior to posterior for stable inference.

Useful when a planner needs to introduce a likelihood or energy term gradually
instead of applying its full weight in one step.
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

from genedynamics.core.constraints.core.types import ScheduleState


def log_pi(
    log_prior: Union[float, Any],
    log_likelihood: Union[float, Any],
    beta: float,
) -> Union[float, Any]:
    """
    Compute log π_k(θ) = log p0(θ) + β_k * log p(y|θ) up to additive constant.

    Args:
        log_prior: log p0(θ)
        log_likelihood: log p(y|θ)
        beta: annealing coefficient in [0, 1]

    Returns:
        log π_k(θ)
    """
    if JAX_AVAILABLE and (hasattr(log_prior, "block_until_ready") or hasattr(log_likelihood, "block_until_ready")):
        return log_prior + beta * log_likelihood
    return float(log_prior) + beta * float(log_likelihood)


@dataclass
class BridgeScheduleConfig:
    """
    Configuration for annealed bridge schedule.

    Attributes:
        K: number of annealing steps
        beta0: initial β (typically 0)
        betaK: final β (typically 1)
        schedule_type: "linear" | "geometric" | "cosine"
        sigma_schedule: optional list of smoothing σ_k per step
        eta_schedule: optional list of step sizes η_k
        tau_schedule: optional list of exploration noise τ_k
    """

    K: int = 100
    beta0: float = 0.0
    betaK: float = 1.0
    schedule_type: str = "linear"
    sigma_schedule: Optional[List[float]] = None
    eta_schedule: Optional[List[float]] = None
    tau_schedule: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "K": self.K,
            "beta0": self.beta0,
            "betaK": self.betaK,
            "schedule_type": self.schedule_type,
            "sigma_schedule": self.sigma_schedule,
            "eta_schedule": self.eta_schedule,
            "tau_schedule": self.tau_schedule,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeScheduleConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BridgeSchedule(ABC):
    """
    Abstract interface for annealed bridge schedules.

    Provides β_k, σ_k, η_k, τ_k as functions of step index k.
    """

    @abstractmethod
    def beta(self, k: int) -> float:
        """Annealing coefficient at step k."""
        pass

    @abstractmethod
    def sigma(self, k: int) -> float:
        """Smoothing/perturbation scale at step k."""
        pass

    @abstractmethod
    def eta(self, k: int) -> float:
        """Score ascent step size at step k."""
        pass

    @abstractmethod
    def tau(self, k: int) -> float:
        """Exploration noise scale at step k."""
        pass

    def params(self, state: ScheduleState) -> Dict[str, Any]:
        """Return all params for given schedule state (for scheduler compatibility)."""
        k = state.k
        return {
            "beta_k": self.beta(k),
            "sigma_k": self.sigma(k),
            "eta_k": self.eta(k),
            "tau_k": self.tau(k),
        }


class LinearBridgeSchedule(BridgeSchedule):
    """Linear interpolation: β_k = β0 + (βK - β0) * k / K."""

    def __init__(self, config: BridgeScheduleConfig):
        self.config = config
        self._betas = np.linspace(config.beta0, config.betaK, config.K, dtype=np.float32)
        self._sigmas = self._build_sigmas()
        self._etas = self._build_etas()
        self._taus = self._build_taus()

    def _build_sigmas(self) -> np.ndarray:
        if self.config.sigma_schedule is not None:
            return np.asarray(self.config.sigma_schedule, dtype=np.float32)
        # Default: decay from 0.1 to 0.01
        return np.linspace(0.1, 0.01, self.config.K, dtype=np.float32)

    def _build_etas(self) -> np.ndarray:
        if self.config.eta_schedule is not None:
            return np.asarray(self.config.eta_schedule, dtype=np.float32)
        return np.full(self.config.K, 0.01, dtype=np.float32)

    def _build_taus(self) -> np.ndarray:
        if self.config.tau_schedule is not None:
            return np.asarray(self.config.tau_schedule, dtype=np.float32)
        return np.linspace(0.01, 0.001, self.config.K, dtype=np.float32)

    def beta(self, k: int) -> float:
        return float(self._betas[min(k, self.config.K - 1)])

    def sigma(self, k: int) -> float:
        return float(self._sigmas[min(k, self.config.K - 1)])

    def eta(self, k: int) -> float:
        return float(self._etas[min(k, self.config.K - 1)])

    def tau(self, k: int) -> float:
        return float(self._taus[min(k, self.config.K - 1)])


class GeometricBridgeSchedule(BridgeSchedule):
    """Geometric schedule: β_k = β0 * (βK/β0)^(k/K)."""

    def __init__(self, config: BridgeScheduleConfig):
        self.config = config
        if config.beta0 <= 0:
            config.beta0 = 1e-6
        self._betas = np.geomspace(config.beta0, config.betaK, config.K, dtype=np.float32)
        self._sigmas = self._build_sigmas()
        self._etas = self._build_etas()
        self._taus = self._build_taus()

    def _build_sigmas(self) -> np.ndarray:
        if self.config.sigma_schedule is not None:
            return np.asarray(self.config.sigma_schedule, dtype=np.float32)
        return np.geomspace(0.1, 0.01, self.config.K, dtype=np.float32)

    def _build_etas(self) -> np.ndarray:
        if self.config.eta_schedule is not None:
            return np.asarray(self.config.eta_schedule, dtype=np.float32)
        return np.full(self.config.K, 0.01, dtype=np.float32)

    def _build_taus(self) -> np.ndarray:
        if self.config.tau_schedule is not None:
            return np.asarray(self.config.tau_schedule, dtype=np.float32)
        return np.geomspace(0.01, 0.001, self.config.K, dtype=np.float32)

    def beta(self, k: int) -> float:
        return float(self._betas[min(k, self.config.K - 1)])

    def sigma(self, k: int) -> float:
        return float(self._sigmas[min(k, self.config.K - 1)])

    def eta(self, k: int) -> float:
        return float(self._etas[min(k, self.config.K - 1)])

    def tau(self, k: int) -> float:
        return float(self._taus[min(k, self.config.K - 1)])


def create_linear_bridge_schedule(
    K: int = 100,
    beta0: float = 0.0,
    betaK: float = 1.0,
    **kwargs: Any,
) -> BridgeSchedule:
    """Factory for linear bridge schedule."""
    config = BridgeScheduleConfig(K=K, beta0=beta0, betaK=betaK, schedule_type="linear", **kwargs)
    return LinearBridgeSchedule(config)


def create_geometric_bridge_schedule(
    K: int = 100,
    beta0: float = 1e-6,
    betaK: float = 1.0,
    **kwargs: Any,
) -> BridgeSchedule:
    """Factory for geometric bridge schedule."""
    config = BridgeScheduleConfig(K=K, beta0=beta0, betaK=betaK, schedule_type="geometric", **kwargs)
    return GeometricBridgeSchedule(config)

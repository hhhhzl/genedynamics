"""
Monte Carlo Score Ascent (MCSA) primitives.

Provides:
- ProposalSampler: protocol for generating perturbed proposals
- ImportanceWeighter: stable importance weight computation
- MCSAScoreEstimator: score = (1/σ) E[w * δ] from weighted perturbations
- MCSADiagnostics: ESS, weight stats, degeneracy flags

Used by:
- 3DGS robust mapping: observation-likelihood MCSA
- Soft-robot MR-MF-MBD: reward-based MCSA (ES gradient)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from genedynamics.core.prob.mixtures import stable_log_softmax
from genedynamics.core.inference.diagnostics import (
    effective_sample_size,
    degeneracy_flags,
    weight_entropy,
)


class ProposalSampler(Protocol):
    """
    Protocol for proposal sampling in MCSA.

    Given current state θ and scale σ, produce proposals z_m = θ + σ * δ_m.
    Subspace variants restrict δ to a low-dimensional basis.
    """

    def sample(
        self,
        theta: Any,
        sigma: float,
        M: int,
        rng: Any,
    ) -> Tuple[Any, Any]:
        """
        Sample M proposals and their perturbation vectors.

        Args:
            theta: current state (flattened or structured)
            sigma: perturbation scale
            M: number of proposals
            rng: random key (JAX) or Generator (numpy)

        Returns:
            proposals: (M, ...) proposed states
            deltas: (M, ...) perturbation vectors such that proposals = theta + sigma * deltas
        """
        ...


class ImportanceWeighter:
    """
    Numerically stable importance weight computation for MCSA.

    Converts unnormalized log-weights to normalized weights with
    log-sum-exp trick and optional temperature scaling.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        backend: str = "jax",
    ):
        self.temperature = temperature
        self.backend = backend

    def __call__(
        self,
        log_weights: Any,
        *,
        axis: int = -1,
    ) -> Any:
        """
        Compute normalized importance weights from log-weights.

        w_m = exp((log_w_m - max) / T) / sum(exp(...))

        Args:
            log_weights: unnormalized log π(z_m)
            axis: axis of particles

        Returns:
            weights: normalized, sum to 1 along axis
        """
        scaled = log_weights / max(self.temperature, 1e-8)
        if self.backend == "jax" and JAX_AVAILABLE:
            log_w = jax.nn.log_softmax(scaled, axis=axis)
            return jnp.exp(log_w)
        scaled = np.asarray(scaled, dtype=np.float64)
        x_max = np.max(scaled, axis=axis, keepdims=True)
        exp_shifted = np.exp(scaled - x_max)
        w = exp_shifted / (np.sum(exp_shifted, axis=axis, keepdims=True) + 1e-12)
        return w.astype(np.float32)


@dataclass
class MCSADiagnostics:
    """
    Diagnostics from a single MCSA step.

    Attributes:
        ess: effective sample size
        weight_entropy: -sum w log w
        max_weight: largest weight
        min_weight: smallest weight
        degeneracy: whether ESS < threshold
    """

    ess: float = 0.0
    weight_entropy: float = 0.0
    max_weight: float = 0.0
    min_weight: float = 0.0
    degeneracy: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ess": self.ess,
            "weight_entropy": self.weight_entropy,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "degeneracy": self.degeneracy,
            **self.extra,
        }


class MCSAScoreEstimator:
    """
    Monte Carlo Score Ascent score estimator.

    ∇ log π_σ(θ) ≈ (1/σ) Σ_m w_m δ_m
    where w_m ∝ π(z_m), z_m = θ + σ δ_m, δ_m ~ N(0,I).
    """

    def __init__(
        self,
        sigma: float,
        weighter: Optional[ImportanceWeighter] = None,
        ess_min: float = 1.0,
        backend: str = "jax",
    ):
        self.sigma = sigma
        self.weighter = weighter or ImportanceWeighter(backend=backend)
        self.ess_min = ess_min
        self.backend = backend

    def estimate(
        self,
        proposals: Any,
        deltas: Any,
        log_probs: Any,
        *,
        axis: int = 0,
    ) -> Tuple[Any, MCSADiagnostics]:
        """
        Estimate score from weighted perturbations.

        Args:
            proposals: (M, ...) proposed states (unused, for logging)
            deltas: (M, ...) perturbation vectors
            log_probs: (M,) log π(z_m)
            axis: particle axis

        Returns:
            score: estimated ∇ log π_σ(θ)
            diagnostics: ESS, weight stats, etc.
        """
        weights = self.weighter(log_probs, axis=axis)
        ess = effective_sample_size(weights, axis=axis, backend=self.backend)
        flags = degeneracy_flags(weights, ess_min=self.ess_min, backend=self.backend)

        if self.backend == "jax" and JAX_AVAILABLE:
            score = jnp.sum(weights[:, None] * deltas, axis=axis) / max(self.sigma, 1e-8)
        else:
            weights = np.asarray(weights, dtype=np.float32)
            deltas = np.asarray(deltas, dtype=np.float32)
            score = np.sum(weights[:, None] * deltas, axis=axis) / max(self.sigma, 1e-8)

        ent = weight_entropy(weights, axis=axis, backend=self.backend)
        diag = MCSADiagnostics(
            ess=float(np.asarray(ess)),
            weight_entropy=float(np.asarray(ent)),
            max_weight=float(np.max(weights)),
            min_weight=float(np.min(weights)),
            degeneracy=flags.get("degenerate", False),
        )
        return score, diag

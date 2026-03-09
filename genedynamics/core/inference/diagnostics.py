"""
Inference diagnostics for importance sampling and MCSA.

Provides:
- effective_sample_size: ESS = (sum w)^2 / sum(w^2)
- weight_entropy: -sum w log w
- degeneracy_flags: ESS threshold, max weight, etc.
- InferenceDiagnostics: aggregated diagnostic bundle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


def effective_sample_size(
    weights: Any,
    *,
    axis: int = -1,
    normalized: bool = True,
    backend: Optional[str] = None,
) -> Union[float, Any]:
    """
    Compute effective sample size: ESS = (sum w)^2 / sum(w^2).

    Args:
        weights: importance weights
        axis: axis of particles
        normalized: whether weights already sum to 1
        backend: "jax" | "numpy" | None

    Returns:
        ESS as scalar
    """
    if backend is None:
        backend = "jax" if (JAX_AVAILABLE and hasattr(weights, "block_until_ready")) else "numpy"

    if backend == "jax" and JAX_AVAILABLE:
        w = jnp.asarray(weights, dtype=jnp.float32)
        if not normalized:
            w = w / (jnp.sum(w, axis=axis, keepdims=True) + 1e-10)
        sum_w = jnp.sum(w, axis=axis)
        sum_w2 = jnp.sum(w ** 2, axis=axis)
        ess = (sum_w ** 2) / (sum_w2 + 1e-10)
        return jnp.squeeze(ess)

    w = np.asarray(weights, dtype=np.float32)
    if not normalized:
        w = w / (np.sum(w, axis=axis, keepdims=True) + 1e-10)
    sum_w = np.sum(w, axis=axis)
    sum_w2 = np.sum(w ** 2, axis=axis)
    ess = (sum_w ** 2) / (sum_w2 + 1e-10)
    return float(np.squeeze(ess))


def weight_entropy(
    weights: Any,
    *,
    axis: int = -1,
    backend: Optional[str] = None,
) -> Union[float, Any]:
    """
    Compute weight entropy: H = -sum w log w.

    Higher entropy = more uniform weights (less degeneracy).

    Args:
        weights: normalized importance weights
        axis: axis of particles
        backend: "jax" | "numpy" | None

    Returns:
        entropy as scalar
    """
    if backend is None:
        backend = "jax" if (JAX_AVAILABLE and hasattr(weights, "block_until_ready")) else "numpy"

    if backend == "jax" and JAX_AVAILABLE:
        w = jnp.asarray(weights, dtype=jnp.float32)
        log_w = jnp.where(w > 1e-12, jnp.log(w + 1e-12), 0.0)
        H = -jnp.sum(w * log_w, axis=axis)
        return jnp.squeeze(H)

    w = np.asarray(weights, dtype=np.float32)
    log_w = np.where(w > 1e-12, np.log(w + 1e-12), 0.0)
    H = -np.sum(w * log_w, axis=axis)
    return float(np.squeeze(H))


def degeneracy_flags(
    weights: Any,
    *,
    ess_min: float = 1.0,
    max_weight_threshold: float = 0.99,
    axis: int = -1,
    backend: Optional[str] = None,
) -> Dict[str, bool]:
    """
    Compute degeneracy flags for importance sampling.

    Args:
        weights: importance weights
        ess_min: ESS below this indicates degeneracy
        max_weight_threshold: max weight above this indicates collapse
        axis: particle axis
        backend: "jax" | "numpy" | None

    Returns:
        dict with keys: degenerate, collapsed, low_ess
    """
    if backend is None:
        backend = "jax" if (JAX_AVAILABLE and hasattr(weights, "block_until_ready")) else "numpy"

    ess = effective_sample_size(weights, axis=axis, backend=backend)
    if backend == "jax" and JAX_AVAILABLE:
        ess_val = float(np.asarray(ess))
        max_w = float(jnp.max(weights))
    else:
        ess_val = float(ess)
        max_w = float(np.max(weights))

    degenerate = ess_val < ess_min
    collapsed = max_w > max_weight_threshold
    low_ess = ess_val < max(ess_min * 2, 10.0)

    return {
        "degenerate": degenerate,
        "collapsed": collapsed,
        "low_ess": low_ess,
    }


@dataclass
class InferenceDiagnostics:
    """
    Aggregated diagnostics from an inference step.

    Attributes:
        ess: effective sample size
        weight_entropy: -sum w log w
        max_weight: largest weight
        min_weight: smallest weight
        degenerate: ESS < threshold
        step_index: diffusion/inference step
        wall_time_ms: optional wall-clock time
        extra: extensible dict for method-specific stats
    """

    ess: float = 0.0
    weight_entropy: float = 0.0
    max_weight: float = 0.0
    min_weight: float = 0.0
    degenerate: bool = False
    step_index: int = 0
    wall_time_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "ess": self.ess,
            "weight_entropy": self.weight_entropy,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "degenerate": self.degenerate,
            "step_index": self.step_index,
            "wall_time_ms": self.wall_time_ms,
            **self.extra,
        }
        return {k: v for k, v in out.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceDiagnostics":
        known = {"ess", "weight_entropy", "max_weight", "min_weight", "degenerate", "step_index", "wall_time_ms"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs["extra"] = extra
        return cls(**kwargs)

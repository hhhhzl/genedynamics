"""
S1 Contact Mode System for MRMFMBD.

Theory-correct mode marginalization over contact/friction regimes.

Theory:
  p(R|θ) = Σ_c p(c) p(R|θ,c)     [marginal likelihood]
  p(R|θ,c) ∝ exp(R_c/T)          [Boltzmann-style reward-to-likelihood]
  log p(R|θ) = logsumexp(log p(c) + R_c/T)
  w_c = p(c|θ,R) = softmax(log p(c) + R_c/T)  [responsibilities]
"""

from __future__ import annotations

from .specs import ModeSpec, ModeSystemConfig, default_mode_system_config
from .marginalizer import ModeMarginalizerS1, compute_marginal_log_likelihood, compute_responsibilities

__all__ = [
    "ModeSpec",
    "ModeSystemConfig",
    "ModeMarginalizerS1",
    "compute_marginal_log_likelihood",
    "compute_responsibilities",
    "default_mode_system_config",
]

"""
Probability and statistical primitives for inference.

This module provides backend-agnostic numerical primitives used across
model-based diffusion, MCSA, and posterior inference:

- gaussian_lowrank: Woodbury identity, log-determinant lemma, quadratic forms
  for low-rank correlated Gaussian likelihoods (Σ = AA' + σ²I).

- mixtures: Responsibility weights, mixture log-probability, weighted score
  combination for mode-marginalization and importance sampling.

These primitives are task-agnostic and suitable for:
- 3DGS robust mapping (correlated observation noise)
- Soft-robot mode-marginalization (contact/friction regimes)
- Any inference requiring low-rank covariance or mixture models.
"""

from .gaussian_lowrank import (
    woodbury_solve,
    lowrank_logdet,
    quad_form_lowrank,
    LowRankCovariance,
)
from .mixtures import (
    responsibilities,
    mixture_logprob,
    weighted_score,
    stable_log_softmax,
)
from .noise_sampler import NoiseSampler, IsotropicGaussian
from .structured_noise import StructuredNoise

__all__ = [
    # Gaussian low-rank
    "woodbury_solve",
    "lowrank_logdet",
    "quad_form_lowrank",
    "LowRankCovariance",
    # Mixtures
    "responsibilities",
    "mixture_logprob",
    "weighted_score",
    "stable_log_softmax",
    # Noise samplers (pluggable, backend-agnostic)
    "NoiseSampler",
    "IsotropicGaussian",
    "StructuredNoise",
]

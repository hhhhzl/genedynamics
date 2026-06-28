"""Shared prior infrastructure (genedynamics/learning/priors).

Reusable by ANY solver via the additive `prior=` seam (None => unchanged). NOT
MDAC-private. Multi-backend, mirroring the solver architecture:
  rl/        - RLPrior: model-free policy prior; jax backend = brax.training
  diffusion/ - LearnedDiffusionPrior: s_theta score model (transport score_g)
"""
from genedynamics.learning.priors.base import Prior, DiffusionPrior
from genedynamics.learning.priors.elite_buffer import EliteBuffer
from genedynamics.learning.priors.registry import register_prior, make_prior, list_priors

__all__ = [
    "Prior", "DiffusionPrior", "EliteBuffer",
    "register_prior", "make_prior", "list_priors",
]

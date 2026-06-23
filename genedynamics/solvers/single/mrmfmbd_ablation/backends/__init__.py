"""Frozen legacy MBD backend (Stage 1 of the theory-faithful refactor).

Holds the pre-refactor engine verbatim for paper Q2/Q3 ablations. The method
path lives in ``genedynamics.solvers.single.mrmfmbd``.
"""

from .mrmfmbd_legacy_jax import MRMFMBDLegacyBackend, LegacyMBDConfig
from .mrmfmbd_posterior_jax import MRMFMBDPosteriorBackendJax, PosteriorBridgeConfig

__all__ = ["MRMFMBDLegacyBackend", "LegacyMBDConfig",
           "MRMFMBDPosteriorBackendJax", "PosteriorBridgeConfig"]

"""SHAC — Short-Horizon Actor-Critic with truncated BPTT through jax_mpm.

Public surface:
- SHACConfig — hyperparameter dataclass
- SHACSolver — high-level solver
- SHACBackendJax — concrete JAX-MPM backend (used by the solver shell)

Algorithm reference: Xu et al., "Accelerated Policy Learning with Parallel
Differentiable Simulation", ICLR 2022. Source code at NVlabs/DiffRL — see
third_party/diffrl/README.md for the full citation.
"""

from .protocols import SHACConfig, SHACInfo
from .shac import SHACSolver
from .backends import SHACBackendJax

__all__ = [
    "SHACConfig",
    "SHACInfo",
    "SHACSolver",
    "SHACBackendJax",
]

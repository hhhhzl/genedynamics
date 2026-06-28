"""DIAL-MPC solver (multi-backend, genedynamics MBD-family pattern).

DIAL-MPC == RecedingHorizonController(<DIAL backend>): a control-space MBD
weighted-mean reverse update (spline-node parametrised) driven by the
backend-agnostic receding-horizon bridge.
"""

from genedynamics.solvers.single.dial.spline import NodeSpline
from genedynamics.solvers.single.dial.backend_impl import DialBackend, to_unified_backend
from genedynamics.solvers.single.dial.dial import DIALMPCSolver, _get_dial_backend

# Backend kernel symbols (lazy: require jax; safe to import here since jax is a
# core dep, but keep behind the package for discoverability).
from genedynamics.solvers.single.dial.backends.dial_jax import (
    DialBackendJax,
    reverse_once,
    make_sigma_control,
    make_traj_diffuse_factors,
)

__all__ = [
    "NodeSpline",
    "DialBackend",
    "to_unified_backend",
    "DIALMPCSolver",
    "DialBackendJax",
    "reverse_once",
    "make_sigma_control",
    "make_traj_diffuse_factors",
]

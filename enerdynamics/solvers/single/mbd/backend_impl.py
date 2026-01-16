"""
Backend protocol/base for MBD solver.

This mirrors the lightweight structure used by MBD backends (JAX/NumPy),
without enforcing heavy inheritance. It is kept minimal for clarity and
future extension (e.g., optional JIT/Numpy toggles).
"""

from __future__ import annotations
from typing import Protocol, Any, Dict, List
import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None

from enerdynamics.core.types import Trajectory


class MBDBackend(Protocol):
    """
    Protocol for MBD backend implementations.
    """

    def plan(self, x0: Any, rng_key: Any | None = None) -> Dict[str, Any]:
        """
        Run diffusion planning and return result dict.
        Expected keys (by convention):
          - states: (H+1, state_dim)
          - actions: (H, act_dim)
          - rewards: (H,)
          - energies: (H,)
          - reward_history: (Ndiffuse-1,) optional
          - diffusion_actions_traj: (Ndiffuse-1, H, act_dim) optional
          - diffusion_sampled_actions: (Ndiffuse-1, M, H, act_dim) optional
        """
        ...

    def sample_trajectories(
        self, x0: Any, n_samples: int, rng_key: Any | None = None
    ) -> List[Trajectory]:
        """
        Generate trajectories for SamplingSolver API compatibility.
        """
        ...


class MBDBackendBase:
    """
    Optional base class to share config between backends.
    """

    def __init__(self, solver: Any, **config: Any):
        self.solver = solver
        self.env = solver._env_adapter
        self.energy = solver._legacy_energy
        self.horizon = solver.horizon
        self.Nsample = solver.Nsample
        self.Ndiffuse = solver.Ndiffuse
        self.temp_sample = solver.temp_sample
        self.beta0 = solver.beta0
        self.betaT = solver.betaT
        self.action_limit = solver.action_limit
        self.seed = solver.seed
        self.scheduler = getattr(solver, "scheduler", None)
        self.show_tqdm = bool(getattr(solver, "show_tqdm", False))

    # Convenience converters
    @staticmethod
    def to_np(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if jnp is not None and isinstance(x, jnp.ndarray):
            return np.asarray(x)
        return np.asarray(x)


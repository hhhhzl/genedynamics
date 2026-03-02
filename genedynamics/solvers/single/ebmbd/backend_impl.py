"""
Backend protocol/base for EB-MBD solver.

Provides lightweight interface definitions similar to EDOC, but tailored
to EB-MBD (reverse diffusion returning result dict).
"""

from __future__ import annotations
from typing import Protocol, Any, Dict, List
import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None

from genedynamics.core.types import Trajectory


class EBMBDBackend(Protocol):
    """
    Protocol for EB-MBD backend implementations.
    """

    def reverse_diffuse(self, rng_key: Any, state_init: np.ndarray) -> Dict[str, Any]:
        """
        Run reverse diffusion and return dict with (at minimum):
          - states: (H+1, state_dim)
          - actions: (H, act_dim)
          - rewards: (H,)
          - energies: (H,)
        Optional (for viz/debug):
          - reward_history: (Ndiffuse-1,)
          - diffusion_actions_traj: (Ndiffuse-1, H, act_dim)
          - diffusion_sampled_actions: (Ndiffuse-1, M, H, act_dim)
        """
        ...

    def sample_trajectories(
        self, x0: Any, n_samples: int, rng_key: Any | None = None
    ) -> List[Trajectory]:
        """
        Optional helper to satisfy SamplingSolver API.
        """
        ...


class EBMBDBackendBase:
    """
    Optional base to hold shared solver config.
    """

    def __init__(self, solver: Any, **config: Any):
        self.solver = solver
        self.env = solver._env_adapter
        self.energy = solver._legacy_energy
        self.horizon = solver.horizon
        self.Nsample = solver.Nsample
        self.Ndiffuse = solver.Ndiffuse
        self.temp = solver.temp_sample
        self.beta0 = solver.beta0
        self.betaT = solver.betaT
        self.action_limit = solver.action_limit
        self.action_extra_sigma = getattr(solver, "action_extra_sigma", 0.0)
        self.mu = solver.mu
        self.alpha = solver.alpha
        self.bound = solver.bound
        self.use_min_over_time = solver.use_min_over_time
        self.terminal_energy_weight = float(getattr(solver, "terminal_energy_weight", 0.0))
        self.obstacles = getattr(solver, "_obstacles", None)
        self.obstacle_config = getattr(solver, "_obstacle_config", {}) or {}
        self.scheduler = getattr(solver, "scheduler", None)
        self.show_tqdm = bool(getattr(solver, "show_tqdm", False))

    @staticmethod
    def to_np(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if jnp is not None and isinstance(x, jnp.ndarray):
            return np.asarray(x)
        return np.asarray(x)


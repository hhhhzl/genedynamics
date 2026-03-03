"""
Backend protocol/base for CEM.
"""

from __future__ import annotations
from typing import Protocol, Any, Dict, List
import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None

from genedynamics.core.types import Trajectory


class CEMBackend(Protocol):
    def plan(self, x0: Any, rng_key: Any | None = None) -> Dict[str, Any]:
        ...

    def sample_trajectories(
        self, x0: Any, n_samples: int, rng_key: Any | None = None
    ) -> List[Trajectory]:
        ...


class CEMBackendBase:
    def __init__(self, solver: Any, **config: Any):
        self.solver = solver
        self.env = solver._env_adapter
        self.energy = solver._legacy_energy
        self.horizon = solver.horizon
        self.num_samples = solver.num_samples
        self.num_iterations = solver.num_iterations
        self.elite_frac = solver.elite_frac
        self.init_std = solver.init_std
        self.min_std = solver.min_std
        self.action_limit = solver.action_limit
        self.seed = solver.seed
        self.scheduler = getattr(solver, "scheduler", None)
        self.show_tqdm = bool(getattr(solver, "show_tqdm", False))

    @staticmethod
    def to_np(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if jnp is not None and isinstance(x, jnp.ndarray):
            return np.asarray(x)
        return np.asarray(x)


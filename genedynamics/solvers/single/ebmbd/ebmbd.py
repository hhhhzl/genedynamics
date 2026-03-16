"""
EB-MBD solver implementation (Emerging-Barrier Model-Based Diffusion).
"""

from __future__ import annotations
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.common.model_based_diffusion import BaseModelBasedDiffusionSolver
from genedynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter
from genedynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from genedynamics.core.backends import Backend
from genedynamics.core.types import State, Trajectory
from genedynamics.core.energy import EnergyToLegacyAdapter
from genedynamics.core.task_spec import legacy_extract_position
from genedynamics.solvers.single.ebmbd.backend_impl import to_unified_backend

# Register backend implementations
try:
    from .backends import ebmbd_jax  
except ImportError:
    pass

# Register solver to registry (optional, matches EDOC pattern)
try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


class EBMBDSolver(BaseModelBasedDiffusionSolver):
    """
    EB-MBD solver using emerging barriers in reverse diffusion (action space).

    This class mirrors MBDSolver's simplicity while keeping the EDOC-like
    packaging (backend registry + solver wrapper). Only JAX backend is
    supported.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        horizon: int = 80,
        dt: float = 0.1,
        Nsample: int = 2048,
        Ndiffuse: int = 100,
        temp_sample: float = 0.1,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        action_limit: float = 1.0,
        action_extra_sigma: float = 0.0,
        mu: float = 10.0,
        alpha: float = 1.0,
        bound: float = 0.8,
        use_min_over_time: bool = True,
        terminal_energy_weight: float = 0.0,
        obstacles: Optional[Any] = None,
        obstacle_config: Optional[dict] = None,
        seed: int = 0,
        scheduler: Any = None,
        show_tqdm: bool = False,
        num_modes: int = 1,
        mode_strategy: str = "multirun",
        diversity_eta: float = 1.0,
        diversity_topK_cand: int = None,
        diversity_use_state: bool = True,
        use_target_line: bool = False,
        num_targets: int = 4,
        position_extractor=None,
        position_dim: int = 2,
        **kwargs,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name not in ("jax", "numpy"):
            raise ValueError(
                f"EB-MBD solver supports only JAX or NumPy backend, got {backend.name}."
            )

        self.horizon = horizon
        self.dt = dt
        self.Nsample = Nsample
        self.Ndiffuse = Ndiffuse
        self.temp_sample = temp_sample
        self.beta0 = beta0
        self.betaT = betaT
        self.action_limit = action_limit
        self.action_extra_sigma = action_extra_sigma
        self.mu = mu
        self.alpha = alpha
        self.bound = bound
        self.use_min_over_time = use_min_over_time
        self.terminal_energy_weight = float(terminal_energy_weight)
        self.seed = seed
        self.scheduler = scheduler
        self.show_tqdm = bool(show_tqdm)
        self.num_modes = int(num_modes)  # Number of candidate trajectories to return
        # How to produce multiple modes:
        # - "multirun": run the solver C times with different RNG keys, take the best of each run.
        # - "diverse_topk": pick C diverse candidates from samples inside a single diffusion run (backend-provided).
        self.mode_strategy = str(mode_strategy)
        self.diversity_eta = float(diversity_eta)  # Diversity weight for diverse top-K
        self.diversity_topK_cand = int(diversity_topK_cand) if diversity_topK_cand is not None else None
        self.diversity_use_state = bool(diversity_use_state)  # Use state features (True) or action features (False)
        self.use_target_line = bool(use_target_line)
        self.num_targets = int(num_targets)
        self.config.update(
            dict(
                num_modes=self.num_modes,
                mode_strategy=self.mode_strategy,
                diversity_eta=self.diversity_eta,
                diversity_topK_cand=self.diversity_topK_cand,
                diversity_use_state=self.diversity_use_state,
            )
        )
        self.position_extractor = position_extractor or legacy_extract_position
        self.position_dim = int(position_dim)
        # Optional obstacle manager + config (for fast JAX SDF via texture)
        self._obstacles = obstacles
        self._obstacle_config = obstacle_config or {}

        # Env adapter (JAX-friendly interfaces)
        self._env_adapter = DynamicsToEnvAdapter(dynamics, dt)

        # Legacy energy adapter
        if isinstance(energy, LegacyEnergyFunctional):
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy

        self.act_dim = self._env_adapter.act_dim

        # Backend instantiation (JAX preferred; NumPy for debugging)
        from genedynamics.core.registry.edoc_backends import get_edoc_backend_registry

        registry = get_edoc_backend_registry()
        if backend.name == "jax":
            backend_class = registry.get("ebmbd_jax") or registry.get("jax")
        else:
            backend_class = registry.get("ebmbd_numpy") or registry.get("numpy")
        if backend_class is None:
            raise ValueError("EB-MBD backend implementation not found in registry.")

        self._backend_impl = to_unified_backend(backend_class(self))

    def _get_backend_impl(self):
        return self._backend_impl

    def sample_trajectories(
        self,
        x0: State,
        horizon: int,
        n_samples: int,
        **kwargs,
    ):
        """
        Generate candidate trajectories for interface completeness.

        For EB-MBD we optimize directly; here we return one optimized trajectory
        replicated n_samples times to satisfy SamplingSolver API.
        """
        traj = self.solve(x0, horizon, rng_key=kwargs.get("rng_key", jax.random.PRNGKey(self.seed)))
        return [traj for _ in range(max(1, n_samples))]


# Fix placeholder registration to point to the solver class
if register_solver is not None:
    try:
        register_solver("ebmbd", EBMBDSolver)
    except Exception:
        pass


"""
EB-MBD solver implementation (Emerging-Barrier Model-Based Diffusion).
"""

from __future__ import annotations
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.core.solvers import SamplingSolver
from enerdynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter
from enerdynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from enerdynamics.core.backends import Backend
from enerdynamics.core.types import State, Trajectory
from enerdynamics.solvers.single.edoc import EnergyToLegacyAdapter

# Register backend implementations
try:
    from .backends import ebmbd_jax  
except ImportError:
    pass

# Register solver to registry (optional, matches EDOC pattern)
try:
    from enerdynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


class EBMBDSolver(SamplingSolver):
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
        from enerdynamics.core.registry.edoc_backends import get_edoc_backend_registry

        registry = get_edoc_backend_registry()
        if backend.name == "jax":
            backend_class = registry.get("ebmbd_jax") or registry.get("jax")
        else:
            backend_class = registry.get("ebmbd_numpy") or registry.get("numpy")
        if backend_class is None:
            raise ValueError("EB-MBD backend implementation not found in registry.")

        self._backend_impl = backend_class(self)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def solve(self, x0: State, horizon: int, **kwargs) -> Trajectory:
        """Solve for optimal trajectory using EB-MBD diffusion."""
        if horizon != self.horizon:
            self.horizon = horizon

        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))

        result = self._backend_impl.reverse_diffuse(rng, x0_data)

        # Convert to Trajectory
        states = result["states"]
        actions = result["actions"]
        states_list = [np.asarray(s, dtype=np.float32) for s in states]
        actions_list = [np.asarray(a, dtype=np.float32) for a in actions]

        return Trajectory(states=states_list, actions=actions_list, info=result)

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


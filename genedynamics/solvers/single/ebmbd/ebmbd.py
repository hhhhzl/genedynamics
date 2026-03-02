"""
EB-MBD solver implementation (Emerging-Barrier Model-Based Diffusion).
"""

from __future__ import annotations
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core.solvers import SamplingSolver
from genedynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter
from genedynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from genedynamics.core.backends import Backend
from genedynamics.core.types import State, Trajectory
from genedynamics.solvers.single.edoc import EnergyToLegacyAdapter

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
        num_modes: int = 1,
        mode_strategy: str = "multirun",
        diversity_eta: float = 1.0,
        diversity_topK_cand: int = None,
        diversity_use_state: bool = True,
        use_target_line: bool = False,
        num_targets: int = 4,
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

        # Multi-mode strategy: run C independent solves (like emerging_barrier_mbd's vmap over seeds)
        if self.num_modes > 1 and self.mode_strategy.lower() == "multirun":
            # Check backend type to determine how to handle rng
            backend_name = getattr(self.backend, 'name', 'jax') if hasattr(self, 'backend') else 'jax'
            if backend_name == "jax":
                # Ensure rng is a JAX PRNG key (convert if needed)
                if not isinstance(rng, jnp.ndarray) or rng.shape != (2,):
                    # If not a JAX key, create one from seed or convert
                    if isinstance(rng, (int, np.integer)):
                        rng = jax.random.PRNGKey(int(rng))
                    else:
                        # Try to extract seed or use default
                        rng = jax.random.PRNGKey(self.seed)
                keys = jax.random.split(rng, self.num_modes)
            else:
                # NumPy backend: convert to integer seeds
                if isinstance(rng, jnp.ndarray) and rng.shape == (2,):
                    # Convert JAX key to integer seed (use hash of key values)
                    base_seed = int(rng[0]) ^ int(rng[1])
                elif isinstance(rng, (int, np.integer)):
                    base_seed = int(rng)
                else:
                    base_seed = self.seed
                # Generate C different seeds
                keys = [base_seed + i for i in range(self.num_modes)]
            backend_num_modes_orig = int(getattr(self._backend_impl, "num_modes", 1))
            # Disable backend's within-run multi-mode selection; we aggregate runs here.
            self._backend_impl.num_modes = 1
            try:
                # Use batch version for parallel execution
                if hasattr(self._backend_impl, "reverse_diffuse_batch"):
                    results = self._backend_impl.reverse_diffuse_batch(keys, x0_data)
                else:
                    # Fallback to sequential if batch not available
                    results = [self._backend_impl.reverse_diffuse(k, x0_data) for k in keys]
            finally:
                self._backend_impl.num_modes = backend_num_modes_orig

            candidate_states_list = [np.asarray(r["states"], dtype=np.float32) for r in results]
            candidate_actions_list = [np.asarray(r["actions"], dtype=np.float32) for r in results]
            candidate_costs = np.asarray(
                [float(np.asarray(r.get("candidate_costs", [np.nan]))[int(r.get("best_idx", 0))]) for r in results],
                dtype=np.float32,
            )
            best_idx = int(np.nanargmin(candidate_costs))
            best_result = dict(results[best_idx])
            best_result["candidate_states"] = candidate_states_list
            best_result["candidate_actions"] = candidate_actions_list
            best_result["candidate_costs"] = candidate_costs
            best_result["best_idx"] = best_idx
            best_result["mode_strategy"] = "multirun"
            best_result["multirun_keys"] = keys
            best_result["multirun_diffusion_data"] = [
                {"diffusion_actions_traj": r.get("diffusion_actions_traj"), "diffusion_sampled_actions": r.get("diffusion_sampled_actions")}
                for r in results
            ]
            result = best_result
        else:
            # Single-run (backend handles either 1 mode or diverse_topk selection)
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


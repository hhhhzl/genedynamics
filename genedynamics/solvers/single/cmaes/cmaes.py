"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) solver.

A general black-box optimizer over the action sequence, packaged like the CEM
solver with the same unified Solver interface and backend split.

Mirrors `solvers/single/cem/cem.py` structure: a Solver wrapper + a JAX backend
planner that owns the CMA-ES update (separable / diagonal covariance with
rank-mu mean update and step-size adaptation).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core.solvers import SamplingSolver
from genedynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter
from genedynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from genedynamics.core.backends import Backend
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.types import State, Trajectory
from genedynamics.core.energy import EnergyToLegacyAdapter

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


try:
    from genedynamics.solvers.single.cmaes.backends import cmaes_jax  # noqa: F401
except ImportError:
    pass


def _get_cmaes_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.cmaes.backends.cmaes_jax import CMAESBackendJax

        return CMAESBackendJax
    return None


class CMAESSolver(SamplingSolver):
    """General CMA-ES solver matching the unified Solver interface (cf. CEMSolver)."""

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        horizon: int = 80,
        dt: float = 0.1,
        num_samples: int = 64,        # CMA-ES population size (lambda)
        num_iterations: int = 8,      # generations
        elite_frac: float = 0.5,      # mu = elite_frac * lambda
        sigma0: float = 0.5,          # initial step size
        init_std: float = 0.5,        # alias for sigma0 (kept for kw-compat with CEM)
        min_std: float = 1e-3,
        action_limit: float = 1.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name not in {"jax"}:
            raise ValueError(f"CMA-ES solver supports backend 'jax', got {backend.name}.")

        self.horizon = horizon
        self.dt = dt
        self.seed = seed
        # sigma0 takes precedence; otherwise fall back to init_std (CEM-style kw).
        sigma0 = float(sigma0 if sigma0 is not None else init_std)
        self.config.update(
            dict(
                num_samples=num_samples,
                num_iterations=num_iterations,
                elite_frac=elite_frac,
                sigma0=sigma0,
                min_std=min_std,
                action_limit=action_limit,
            )
        )

        self._env_adapter = DynamicsToEnvAdapter(dynamics, dt)
        if isinstance(energy, LegacyEnergyFunctional):
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy

        self._backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_cmaes_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"CMA-ES backend '{backend.name}' not found")
            self._backend_impl = backend_cls(
                env_adapter=self._env_adapter,
                legacy_energy=self._legacy_energy,
                horizon=self.horizon,
                dt=self.dt,
                num_samples=self.config["num_samples"],
                num_iterations=self.config["num_iterations"],
                elite_frac=self.config["elite_frac"],
                sigma0=self.config["sigma0"],
                min_std=self.config["min_std"],
                action_limit=self.config["action_limit"],
                seed=self.seed,
            )
        return self._backend_impl

    # ------------------------------------------------------------------ #
    # SamplingSolver API
    # ------------------------------------------------------------------ #
    def sample_trajectories(self, x0: State, horizon: int, n_samples: int, **kwargs) -> List[Trajectory]:
        if horizon != self.horizon:
            self.horizon = horizon
            self._backend_impl = None
        planner = self._get_backend_impl()
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        return planner.sample_trajectories(x0_data, n_samples, rng_key=rng_key)

    def solve(self, x0: State, horizon: int, **kwargs) -> Trajectory:
        if horizon != self.horizon:
            self.horizon = horizon
            self._backend_impl = None
        planner = self._get_backend_impl()
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        result = planner.plan(x0_data, rng_key)
        actions_list = [np.asarray(a, dtype=np.float32) for a in result.get("actions", [])]
        states_list = [np.asarray(s, dtype=np.float32) for s in result.get("states", [])]
        return Trajectory(states=states_list, actions=actions_list, info=result)


if register_solver is not None:
    try:
        register_solver("cmaes", CMAESSolver)
    except Exception:
        pass


def run_cmaes(args, initial_state: Optional[np.ndarray] = None):
    """Backward-compatible entry point (mirrors run_cem)."""
    from genedynamics.envs.factories import make_env, make_energy
    from genedynamics.core import get_backend

    env = make_env(args.env_name)
    energy = make_energy(args.env_name)
    if hasattr(env, "dt"):
        env.dt = args.dt
    if hasattr(env, "horizon"):
        env.horizon = args.horizon

    dynamics = DynamicsToEnvAdapter(env, dt=args.dt)
    backend = get_backend("jax")
    solver = CMAESSolver(
        dynamics=dynamics, energy=energy, backend=backend,
        horizon=args.horizon, dt=args.dt,
        num_samples=getattr(args, "num_samples", 64),
        num_iterations=getattr(args, "num_iterations", 8),
        sigma0=getattr(args, "sigma0", 0.5),
        action_limit=getattr(args, "action_limit", 1.0),
        seed=args.seed,
    )
    if initial_state is None:
        rng = jax.random.PRNGKey(args.seed)
        try:
            x0, _ = env.reset(rng)
        except TypeError:
            x0, _ = env.reset()
    else:
        x0 = np.asarray(initial_state, dtype=np.float32)
    traj = solver.solve(x0, horizon=args.horizon, rng_key=jax.random.PRNGKey(args.seed))
    return traj.info

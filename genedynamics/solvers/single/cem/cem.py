"""
CEM (Cross-Entropy Method) solver with EDOC-style architecture.

This refactor aligns CEM with the EDOC packaging pattern:
- Planner class that owns backend-specific logic (JAX-only for now)
- Solver wrapper that matches the unified Solver interface
- Legacy energy adapter for new EnergyFunctional
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
from genedynamics.solvers.single.edoc import EnergyToLegacyAdapter

# Optional solver registry (mirrors EDOC style)
try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


# ============================================================================
# Backend registry helpers
# ============================================================================
try:
    from genedynamics.solvers.single.cem.backends import cem_jax, cem_numpy  # noqa: F401
except ImportError:
    pass


def _get_cem_backend(backend_name: str):
    """Return backend implementation for CEM."""
    if backend_name == "jax":
        from genedynamics.solvers.single.cem.backends.cem_jax import CEMBackendJax

        return CEMBackendJax
    if backend_name == "numpy":
        from genedynamics.solvers.single.cem.backends.cem_numpy import CEMBackendNumpy

        return CEMBackendNumpy
    return None


# ============================================================================
# CEM Solver (Unified interface wrapper)
# ============================================================================

class CEMSolver(SamplingSolver):
    """Solver wrapper that mirrors EDOC's structure but runs CEM."""

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        horizon: int = 80,
        dt: float = 0.1,
        num_samples: int = 512,
        num_iterations: int = 6,
        elite_frac: float = 0.1,
        init_std: float = 0.5,
        min_std: float = 0.05,
        action_limit: float = 1.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name not in {"jax", "numpy"}:
            raise ValueError(
                f"CEM solver supports backends {{'jax','numpy'}}, got {backend.name}."
            )

        self.horizon = horizon
        self.dt = dt
        self.seed = seed
        self.config.update(
            dict(
                num_samples=num_samples,
                num_iterations=num_iterations,
                elite_frac=elite_frac,
                init_std=init_std,
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
            backend_cls = _get_cem_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"CEM backend '{backend.name}' not found")
            self._backend_impl = backend_cls(
                env_adapter=self._env_adapter,
                legacy_energy=self._legacy_energy,
                horizon=self.horizon,
                dt=self.dt,
                num_samples=self.config["num_samples"],
                num_iterations=self.config["num_iterations"],
                elite_frac=self.config["elite_frac"],
                init_std=self.config["init_std"],
                min_std=self.config["min_std"],
                action_limit=self.config["action_limit"],
                seed=self.seed,
            )
        return self._backend_impl

    # ------------------------------------------------------------------ #
    # SamplingSolver API
    # ------------------------------------------------------------------ #
    def sample_trajectories(
        self,
        x0: State,
        horizon: int,
        n_samples: int,
        **kwargs,
    ) -> List[Trajectory]:
        if horizon != self.horizon:
            self.horizon = horizon
            self._planner = None
        planner = self._get_backend_impl()

        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        return planner.sample_trajectories(x0_data, n_samples, rng_key=rng_key)

    def solve(
        self,
        x0: State,
        horizon: int,
        **kwargs,
    ) -> Trajectory:
        if horizon != self.horizon:
            self.horizon = horizon
            self._planner = None
        planner = self._get_backend_impl()

        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        result = planner.plan(x0_data, rng_key)

        actions = result.get("actions", [])
        states = result.get("states", [])

        actions_list = [np.asarray(a, dtype=np.float32) for a in actions]
        states_list = [np.asarray(s, dtype=np.float32) for s in states]

        return Trajectory(states=states_list, actions=actions_list, info=result)


# Register to solver registry (optional)
if register_solver is not None:
    try:
        register_solver("cem", CEMSolver)
    except Exception:
        pass


# ============================================================================
# Main Entry Point (for backward compatibility)
# ============================================================================

def run_cem(args, initial_state: Optional[np.ndarray] = None):
    """
    Run CEM planner (backward-compatible entry point).
    """
    from genedynamics.envs.factories import make_env, make_energy
    from genedynamics.core import get_backend

    env = make_env(args.env_name)
    energy = make_energy(args.env_name)

    if hasattr(env, "dt"):
        env.dt = args.dt
    if hasattr(env, "horizon"):
        env.horizon = args.horizon
    if hasattr(env, "control_limit"):
        env.control_limit = args.action_limit

    dynamics = DynamicsToEnvAdapter(env, dt=args.dt)
    backend = get_backend("jax")

    solver = CEMSolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=args.horizon,
        dt=args.dt,
        num_samples=args.num_samples,
        num_iterations=args.num_iterations,
        elite_frac=args.elite_frac,
        init_std=args.init_std,
        min_std=args.min_std,
        action_limit=args.action_limit,
        seed=args.seed,
    )

    # Get initial state
    if initial_state is None:
        rng = jax.random.PRNGKey(args.seed)
        try:
            x0, _ = env.reset(rng)
        except TypeError:
            x0, _ = env.reset()
    else:
        x0 = np.asarray(initial_state, dtype=np.float32)

    traj = solver.solve(x0, horizon=args.horizon, rng_key=jax.random.PRNGKey(args.seed))

    return traj.info if hasattr(traj, "info") else {
        "actions": np.stack(traj.actions, axis=0),
        "states": np.stack(traj.states, axis=0),
    }


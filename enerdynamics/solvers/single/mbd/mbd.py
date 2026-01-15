"""
MBD (Multi-scale Barrier Diffusion) solver with EDOC-style architecture.

This refactor separates planner/backend logic (JAX-only for now) from the
solver wrapper to match the EDOC packaging pattern.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.core.solvers import SamplingSolver
from enerdynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter
from enerdynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from enerdynamics.core.backends import Backend
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.core.types import State, Trajectory
from enerdynamics.solvers.single.edoc import EnergyToLegacyAdapter

try:
    from enerdynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


# ============================================================================
# Backend registry helpers
# ============================================================================
try:
    from enerdynamics.solvers.single.mbd.backends import mbd_jax  # noqa: F401
except ImportError:
    pass


def _get_mbd_backend(backend_name: str):
    if backend_name == "jax":
        from enerdynamics.solvers.single.mbd.backends.mbd_jax import MBDBackendJax

        return MBDBackendJax
    return None


# ============================================================================
# MBD Solver (wrapper)
# ============================================================================

class MBDSolver(SamplingSolver):
    """Solver wrapper that mirrors EDOC structure but runs MBD diffusion."""

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
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name != "jax":
            raise ValueError(
                f"MBD solver requires JAX backend, got {backend.name}. "
                "Use backend=get_backend('jax') when creating the solver."
            )

        self.horizon = horizon
        self.dt = dt
        self.seed = seed
        self.config.update(
            dict(
                Nsample=Nsample,
                Ndiffuse=Ndiffuse,
                temp_sample=temp_sample,
                beta0=beta0,
                betaT=betaT,
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
            backend_cls = _get_mbd_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"MBD backend '{backend.name}' not found")
            self._backend_impl = backend_cls(
                env_adapter=self._env_adapter,
                legacy_energy=self._legacy_energy,
                horizon=self.horizon,
                dt=self.dt,
                Nsample=self.config["Nsample"],
                Ndiffuse=self.config["Ndiffuse"],
                temp_sample=self.config["temp_sample"],
                beta0=self.config["beta0"],
                betaT=self.config["betaT"],
                action_limit=self.config["action_limit"],
                seed=self.seed,
            )
        return self._backend_impl

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
        states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]
        actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
        return Trajectory(states=states_list, actions=actions_list, info=result)


if register_solver is not None:
    try:
        register_solver("mbd", MBDSolver)
    except Exception:
        pass


# ============================================================================
# Main Entry Point (backward compatibility)
# ============================================================================

def run_mbd(args):
    """Run MBD diffusion planner (backward-compatible entry point)."""
    from enerdynamics.envs.factories import make_env, make_energy
    from enerdynamics.core import get_backend

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

    solver = MBDSolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=args.horizon,
        dt=args.dt,
        Nsample=args.Nsample,
        Ndiffuse=args.Ndiffuse,
        temp_sample=args.temp_sample,
        beta0=args.beta0,
        betaT=args.betaT,
        action_limit=args.action_limit,
        seed=args.seed,
    )

    rng = jax.random.PRNGKey(args.seed)
    try:
        x0, _ = env.reset(rng)
    except TypeError:
        x0, _ = env.reset()

    traj = solver.solve(x0, horizon=args.horizon, rng_key=jax.random.PRNGKey(args.seed))
    return traj.info if hasattr(traj, "info") else {
        "actions": np.stack(traj.actions, axis=0),
        "states": np.stack(traj.states, axis=0),
    }

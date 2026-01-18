"""
MDOC (enerdynamics version).

Definition:
  MDOC = MBD-style diffusion driver + ConstraintFilter (action-space filtering per diffusion step)
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
from enerdynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter

try:
    from enerdynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


def _get_mdoc_backend(backend_name: str):
    if backend_name == "jax":
        from enerdynamics.solvers.single.mdoc.backends.mdoc_jax import MDOCBackendJax

        return MDOCBackendJax
    return None


class MDOCSolver(SamplingSolver):
    """
    Solver wrapper: delegates to MDOC backend.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        horizon: int = 64,
        dt: float = 0.05,
        Nsample: int = 256,
        Ndiffuse: int = 100,
        temp_sample: float = 0.3,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        action_limit: float = 1.0,
        seed: int = 0,
        scheduler: Any = None,
        constraint_filter: Optional[ConstraintFilter] = None,
        obstacles: Any = None,
        show_tqdm: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name not in {"jax"}:
            raise ValueError(f"MDOC solver currently supports backend {{'jax'}}, got {backend.name}.")

        self.horizon = int(horizon)
        self.dt = float(dt)
        self.seed = int(seed)
        self.config.update(
            dict(
                Nsample=int(Nsample),
                Ndiffuse=int(Ndiffuse),
                temp_sample=float(temp_sample),
                beta0=float(beta0),
                betaT=float(betaT),
                action_limit=float(action_limit),
                scheduler=scheduler,
                show_tqdm=bool(show_tqdm),
            )
        )

        self._env_adapter = DynamicsToEnvAdapter(dynamics, dt)
        if isinstance(energy, LegacyEnergyFunctional):
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy

        self._backend_impl = None
        self.constraint_filter = constraint_filter or NoOpConstraintFilter()
        self.obstacles = obstacles
        # Store additional CBF params
        self.cbf_params = {
            "cbf_tau": kwargs.get("cbf_tau", 0.005),
            "cbf_eta": kwargs.get("cbf_eta", 1.5),
            "cbf_margin": kwargs.get("cbf_margin", 0.1),
            "base_beta": kwargs.get("base_beta", 0.05),
            "terminal_energy_weight": kwargs.get("terminal_energy_weight", 50.0),
            "guide_weight": kwargs.get("guide_weight", 10.0),
        }

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_mdoc_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"MDOC backend '{backend.name}' not found")
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
                scheduler=self.config.get("scheduler"),
                show_tqdm=self.config.get("show_tqdm", False),
                constraint_filter=self.constraint_filter,
                obstacles=self.obstacles,
                **self.cbf_params,
            )
        return self._backend_impl

    def sample_trajectories(self, x0: State, horizon: int, n_samples: int, **kwargs) -> List[Trajectory]:
        if horizon != self.horizon:
            self.horizon = int(horizon)
            self._backend_impl = None
        planner = self._get_backend_impl()
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        return planner.sample_trajectories(x0_data, n_samples, rng_key=rng_key)

    def solve(self, x0: State, horizon: int, **kwargs) -> Trajectory:
        if horizon != self.horizon:
            self.horizon = int(horizon)
            self._backend_impl = None
        planner = self._get_backend_impl()
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        rng_key = kwargs.get("rng_key", jax.random.PRNGKey(self.seed))
        result = planner.plan(x0_data, rng_key)
        states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]
        actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
        return Trajectory(states=states_list, actions=actions_list, info=result)


if register_solver is not None:
    try:
        register_solver("mdoc", MDOCSolver)
    except Exception:
        pass


def run_mdoc(args):
    """Backward-compatible entry point (similar to run_mbd)."""
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

    solver = MDOCSolver(
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
        scheduler=getattr(args, "scheduler", None),
        obstacles=getattr(args, "obstacles", None),
    )

    traj = solver.solve(x0=env.reset()[0], horizon=args.horizon, rng_key=jax.random.PRNGKey(args.seed))
    return traj.info if hasattr(traj, "info") else {"actions": np.stack(traj.actions), "states": np.stack(traj.states)}



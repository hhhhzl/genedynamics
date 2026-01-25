"""
CFS-MBD (enerdynamics version).

Definition:
  CFS-MBD = MBD-style diffusion driver + Augmented Lagrangian objective + CFS-based per-step QP projection
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


def _get_cfsmbd_backend(backend_name: str):
    if backend_name == "jax":
        from enerdynamics.solvers.single.cfsmbd.backends.cfsmbd_jax import CFSMBDBackendJax
        return CFSMBDBackendJax
    if backend_name == "numpy":
        from enerdynamics.solvers.single.cfsmbd.backends.cfsmbd_numpy import CFSMBDBackendNumpy
        return CFSMBDBackendNumpy
    return None


class CFSMBDSolver(SamplingSolver):
    """
    Solver wrapper: delegates to CFS-MBD backend.
    
    CFS-MBD combines:
    - MBD-style reverse diffusion
    - Augmented Lagrangian objective (J + λ^T [g]_+ + (ρ/2) ||[g]_+||^2)
    - CFS-based per-step QP projection for constraint satisfaction
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        horizon: int = 64,
        dt: float = 0.05,
        Nsample: int = 4096,
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
        aug_lambda: float = 0.0,
        aug_rho: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)

        if backend.name not in {"jax", "numpy"}:
            raise ValueError(
                f"CFS-MBD solver currently supports backends {{'jax','numpy'}}, got {backend.name}."
            )

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
                aug_lambda=float(aug_lambda),
                aug_rho=float(aug_rho),
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

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_cfsmbd_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"CFS-MBD backend '{backend.name}' not found")
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
                aug_lambda=self.config.get("aug_lambda", 0.0),
                aug_rho=self.config.get("aug_rho", 1.0),
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
        register_solver("cfsmbd", CFSMBDSolver)
    except Exception:
        pass

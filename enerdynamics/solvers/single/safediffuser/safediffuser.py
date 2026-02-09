"""
SafeDiffuser solver wrapper for enerdynamics.

This integrates a SafeDiffuser-style safety correction into the same denoising
architecture used by DPCC (vendored `third_party/diffuser`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from enerdynamics.core.backends import Backend
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.core.dynamics import DynamicsModel
from enerdynamics.core.energy import EnergyFunctional
from enerdynamics.core.solvers import SamplingSolver
from enerdynamics.core.types import State, Trajectory
from enerdynamics.solvers.single.safediffuser.backends.safediffuser_plan_torch import (
    SafeDiffuserBackendTorch,
)

try:
    from enerdynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


class SafeDiffuserSolver(SamplingSolver):
    """
    Solver wrapper: delegates SafeDiffuser planning to a torch backend.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        env: Any,
        diffusion: Any,
        normalizer: Any,
        plan_config: Dict[str, Any],
        goal_xy: Optional[np.ndarray] = None,
        device: str = "cuda",
        seed: int = 0,
        **kwargs: Any,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)
        self.env = env
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.plan_config = plan_config
        self.goal_xy = goal_xy
        self.device = device
        self.seed = int(seed)

        self._backend_impl: SafeDiffuserBackendTorch | None = None

    def _get_backend_impl(self) -> SafeDiffuserBackendTorch:
        if self._backend_impl is None:
            runtime_backend = RuntimeBackendManager.get_backend()
            if runtime_backend.name != "torch":
                raise ValueError(
                    f"SafeDiffuser solver requires torch backend, got '{runtime_backend.name}'."
                )
            self._backend_impl = SafeDiffuserBackendTorch(
                env=self.env,
                diffusion=self.diffusion,
                normalizer=self.normalizer,
                plan_config=self.plan_config,
                device=str(self.device),
                seed=self.seed,
                goal_xy=self.goal_xy,
            )
        return self._backend_impl

    def sample_trajectories(self, x0: State, horizon: int, n_samples: int, **kwargs: Any) -> List[Trajectory]:
        # SafeDiffuser policy does its own batching; for now, just return n_samples repeats of solve().
        _ = (horizon, n_samples)
        traj = self.solve(x0, horizon=horizon, **kwargs)
        return [traj]

    def solve(self, x0: State, horizon: int, **kwargs: Any) -> Trajectory:
        # NOTE: `Solver.solve(x0, horizon, ...)` requires `horizon` by interface.
        # We use `plan_config.horizon` (or model horizon) as the authoritative one.
        _ = horizon
        planner = self._get_backend_impl()
        result = planner.plan(np.asarray(x0, dtype=np.float32), rng_key=kwargs.get("rng_key"))
        states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]
        actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
        print(states_list)
        return Trajectory(states=states_list, actions=actions_list[:-1], info=result.get("info"))


if register_solver is not None:
    try:
        register_solver("safediffuser", SafeDiffuserSolver)
    except Exception:
        pass


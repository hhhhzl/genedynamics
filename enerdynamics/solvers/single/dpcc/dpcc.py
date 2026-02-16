"""
DPCC solver wrapper for enerdynamics.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from enerdynamics.core.backends import Backend
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.core.dynamics import DynamicsModel
from enerdynamics.core.energy import EnergyFunctional
from enerdynamics.core.solvers import SamplingSolver
from enerdynamics.core.types import State, Trajectory
from enerdynamics.solvers.single.dpcc.backends.dpcc_plan_torch import DPCCBackendTorch

try:
    from enerdynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


class DPCCSolver(SamplingSolver):
    """
    Solver wrapper: delegates DPCC planning to a torch backend.
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
        constraint_config: Dict[str, Any],
        indices: Dict[str, Dict[str, int]],
        device: str = "cuda",
        seed: int = 0,
        **kwargs: Any,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)
        self.env = env
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.plan_config = plan_config
        self.constraint_config = constraint_config
        self.indices = indices
        self.device = device
        self.seed = seed

        self._backend_impl = None

    def _get_backend_impl(self):
        if self._backend_impl is None:
            runtime_backend = RuntimeBackendManager.get_backend()
            if runtime_backend.name != "torch":
                raise ValueError(
                    f"DPCC solver requires torch backend, got '{runtime_backend.name}'."
                )
            self._backend_impl = DPCCBackendTorch(
                env=self.env,
                diffusion=self.diffusion,
                normalizer=self.normalizer,
                plan_config=self.plan_config,
                constraint_config=self.constraint_config,
                indices=self.indices,
                device=self.device,
                seed=self.seed,
            )
        return self._backend_impl

    def sample_trajectories(self, x0: State, horizon: int, n_samples: int, **kwargs) -> List[Trajectory]:
        planner = self._get_backend_impl()
        return planner.sample_trajectories(x0, n_samples, rng_key=kwargs.get("rng_key"))

    def solve(self, x0: State, horizon: int, **kwargs) -> Trajectory:
        planner = self._get_backend_impl()
        result = planner.plan(x0, rng_key=kwargs.get("rng_key"))
        states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]
        actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
        info = dict(result.get("info") or {})
        for k in ("candidate_states", "candidate_actions", "candidate_costs", "best_idx"):
            if k in result:
                info[k] = result[k]
        return Trajectory(states=states_list, actions=actions_list, info=info)


if register_solver is not None:
    try:
        register_solver("dpcc", DPCCSolver)
    except Exception:
        pass

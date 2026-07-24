"""MPPI method plugin for the unified experiment runner."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.dynamics.adapters import EnvDynamicsAdapter
from genedynamics.solvers.single.mppi import MPPISolver

from ...framework.base import MethodPlugin
from ._result_utils import normalize_result_from_trajectory


class MPPIMethodPlugin(MethodPlugin):
    """Expose the existing :class:`MPPISolver` to YAML experiments."""

    @property
    def name(self) -> str:
        return "mppi"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> MPPISolver:
        backend = RuntimeBackendManager.get_backend()
        horizon = int(config.get("horizon", getattr(env, "horizon", 80)))
        dt = float(config.get("dt", getattr(env, "dt", 0.1)))

        return MPPISolver(
            dynamics=EnvDynamicsAdapter(env),
            energy=energy,
            backend=backend,
            horizon=horizon,
            dt=dt,
            num_samples=int(config.get("num_samples", config.get("Nsample", 512))),
            num_iterations=int(config.get("num_iterations", config.get("Ndiffuse", 6))),
            noise_sigma=float(config.get("noise_sigma", 0.3)),
            lambda_=float(config.get("lambda_", 1.0)),
            action_limit=float(
                config.get("action_limit", getattr(env, "control_limit", 1.0))
            ),
            seed=int(config.get("np_random_seed", config.get("seed", 0))),
        )

    def plan(self, planner: MPPISolver, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        traj = planner.solve(initial_state, horizon=planner.horizon, rng_key=rng)
        result = normalize_result_from_trajectory(
            traj,
            initial_state=initial_state,
            preserve_info=True,
        )

        # The stepping-stones deploy entrypoint consumes the experiment runner's
        # multi-candidate schema. MPPI returns one best trajectory, so expose it
        # as a one-element candidate set without changing the solver itself.
        states = np.asarray(result["states"], dtype=np.float32)
        actions = np.asarray(result["actions"], dtype=np.float32)
        total_reward = result.get("total_reward")
        if total_reward is not None:
            candidate_cost = -float(total_reward)
        else:
            candidate_cost = float(
                np.sum(np.asarray(result.get("energies", []), dtype=np.float64))
            )

        result["candidate_states"] = [states]
        result["candidate_actions"] = [actions]
        result["candidate_costs"] = np.asarray([candidate_cost], dtype=np.float32)
        result["best_idx"] = 0
        return result

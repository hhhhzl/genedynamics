"""
SafeDiffuser method plugin implementation.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from enerdynamics.core.dynamics import DynamicsToEnvAdapter
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from ...framework.base import MethodPlugin

try:
    from enerdynamics.solvers.single.safediffuser import SafeDiffuserSolver
except Exception:
    SafeDiffuserSolver = None


class SafeDiffuserMethodPlugin(MethodPlugin):
    @property
    def name(self) -> str:
        return "safediffuser"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> "SafeDiffuserSolver":
        if SafeDiffuserSolver is None:
            raise ImportError(
                "SafeDiffuser solver not available."
            )

        plan_config = dict(config.get("plan_config", {}))
        plan_config.setdefault("batch_size", int(config.get("batch_size", 8)))

        ckpt_dir = config.get("safediffuser_checkpoint_dir") or plan_config.get("checkpoint_dir")
        if not ckpt_dir:
            raise ValueError(
                "Missing `safediffuser_checkpoint_dir` in config. "
                "It should point to a converted checkpoint containing "
                "`safediffuser_planning.yaml` and `state_*.pt`."
            )
        plan_config.setdefault("checkpoint_dir", ckpt_dir)
        plan_config.setdefault("device", str(config.get("device", "cuda:0")))
        plan_config.setdefault("diffusion_epoch", config.get("diffusion_epoch", "latest"))

        # Goal comes from d3il env wrapper by default.
        goal_xy = np.asarray(config.get("goal_xy", getattr(env, "target", None)), dtype=np.float32) if getattr(env, "target", None) is not None or config.get("goal_xy") is not None else None

        dynamics = DynamicsToEnvAdapter(env, dt=float(getattr(env, "dt", 0.1)))
        backend = RuntimeBackendManager.get_backend()

        return SafeDiffuserSolver(
            dynamics=dynamics,
            energy=energy,
            backend=backend,
            env=env,
            plan_config=plan_config,
            goal_xy=goal_xy,
        )

    def plan(self, planner: "SafeDiffuserSolver", initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        # NOTE: `Solver.solve(x0, horizon, ...)` requires a horizon argument by interface.
        # For SafeDiffuser, the *actual* horizon is fixed by the converted checkpoint's
        # `safediffuser_planning.yaml`, so this value is a placeholder and is ignored.
        traj = planner.solve(initial_state, horizon=1, rng_key=rng)
        return {
            "states": traj.states,
            "actions": traj.actions,
            "initial_state": initial_state,
            "info": traj.info,
        }

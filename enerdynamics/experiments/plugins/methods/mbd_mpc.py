"""
MBD MPC method plugin for D3IL Avoiding.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

from enerdynamics.solvers.single.mbd import MBDSolver
from enerdynamics.core.dynamics.adapters import EnvDynamicsAdapter
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.envs.external.d3il.avoiding_plan_env import AvoidingPlanEnv, AvoidingPlanSpec
from enerdynamics.experiments.common.d3il_mpc import run_mpc_episode
from ...framework.base import MethodPlugin


class MBDMPCMethodPlugin(MethodPlugin):
    @property
    def name(self) -> str:
        return "mbd_mpc"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        backend = RuntimeBackendManager.get_backend()

        horizon = int(config.get("horizon", 20))
        dt = float(config.get("dt", getattr(env, "dt", 0.035)))
        action_limit = float(config.get("action_limit", getattr(env, "control_limit", 0.05)))

        plan_env = AvoidingPlanEnv(
            AvoidingPlanSpec(dt=dt, horizon=horizon, control_limit=action_limit)
        )
        dynamics = EnvDynamicsAdapter(plan_env)

        solver = MBDSolver(
            dynamics=dynamics,
            energy=energy,
            backend=backend,
            horizon=horizon,
            dt=dt,
            Nsample=int(config.get("Nsample", 1024)),
            Ndiffuse=int(config.get("Ndiffuse", 50)),
            temp_sample=float(config.get("temp_sample", 0.1)),
            beta0=float(config.get("beta0", 1e-4)),
            betaT=float(config.get("betaT", 1e-2)),
            action_limit=action_limit,
            seed=int(config.get("np_random_seed", 0) or 0),
            scheduler=config.get("scheduler"),
            show_tqdm=bool(config.get("show_tqdm", False)),
        )

        return {"exec_env": env, "plan_env": plan_env, "solver": solver, "config": config}

    def plan(self, planner: Any, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        _ = initial_state
        exec_env = planner["exec_env"]
        plan_env: AvoidingPlanEnv = planner["plan_env"]
        solver: MBDSolver = planner["solver"]
        cfg: Dict[str, Any] = planner["config"]

        max_steps = int(cfg.get("max_episode_length", getattr(exec_env, "horizon", 150)))

        def plan_step_fn(mbd_solver: MBDSolver, x0: np.ndarray, rng_key: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
            plan_env.set_initial_state(x0)
            traj = mbd_solver.solve(x0, horizon=mbd_solver.horizon, rng_key=rng_key)
            if not traj.actions:
                raise RuntimeError("MBD returned no actions")
            u0 = np.asarray(traj.actions[0], dtype=np.float32)
            return u0, {"mbd_info_keys": list(getattr(traj, "info", {}).keys())}

        return run_mpc_episode(
            exec_env=exec_env,
            planner=solver,
            plan_step_fn=plan_step_fn,
            rng=rng,
            max_steps=max_steps,
        )



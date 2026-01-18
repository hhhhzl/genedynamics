"""
EDOC MPC method plugin for D3IL Avoiding.

This runs EDOC on a simple JAX-friendly planning model and executes the first
action on the real (stateful) D3IL MuJoCo environment.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

from enerdynamics.solvers.single.edoc import EDOCPlanner
from enerdynamics.envs.external.d3il.avoiding_plan_env import AvoidingPlanEnv, AvoidingPlanSpec
from enerdynamics.experiments.common.d3il_mpc import run_mpc_episode
from ...framework.base import MethodPlugin


class EDOCMPCMethodPlugin(MethodPlugin):
    @property
    def name(self) -> str:
        return "edoc_mpc"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        """
        Create an EDOC planner that plans on AvoidingPlanEnv.

        `env` here is the execution env (D3ILAvoidingEnv), but we do not pass it
        into EDOC. Instead we build a planning env with matching state/action dims.
        """
        horizon = int(config.get("horizon", 20))
        dt = float(config.get("dt", getattr(env, "dt", 0.035)))
        control_limit = float(config.get("action_limit", getattr(env, "control_limit", 0.05)))

        plan_env = AvoidingPlanEnv(
            AvoidingPlanSpec(
                dt=dt,
                horizon=horizon,
                control_limit=control_limit,
            )
        )

        planner = EDOCPlanner(
            env=plan_env,
            energy=energy,
            horizon=horizon,
            dt=dt,
            action_space=True,
            diffusion_mode=config.get("diffusion_mode", "reverse"),
            action_diffuse_steps=int(config.get("action_diffuse_steps", 50)),
            action_nsample=int(config.get("action_nsample", 64)),
            use_antithetic=bool(config.get("use_antithetic", True)),
            action_score_mode=config.get("action_score_mode", "energy"),
            constraint_manager=config.get("constraint_manager"),
            constraint_pipeline=config.get("constraint_pipeline"),
            scheduler=config.get("scheduler"),
            use_constraint_in_scoring=bool(config.get("use_constraint_in_scoring", True)),
            lambda_energy=float(config.get("lambda_energy", 1.0)),
            terminal_energy_weight=float(config.get("terminal_energy_weight", 0.0)),
            np_random_seed=config.get("np_random_seed", None),
            show_tqdm=bool(config.get("show_tqdm", False)),
        )

        # bundle exec env and plan env for plan()
        return {"exec_env": env, "plan_env": plan_env, "planner": planner, "config": config}

    def plan(self, planner: Any, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        """
        Run MPC episode and return executed trajectory.

        Note: `initial_state` is ignored for execution env; D3IL env controls its own reset.
        """
        _ = initial_state
        exec_env = planner["exec_env"]
        plan_env: AvoidingPlanEnv = planner["plan_env"]
        edoc: EDOCPlanner = planner["planner"]
        cfg: Dict[str, Any] = planner["config"]

        max_steps = int(cfg.get("max_episode_length", getattr(exec_env, "horizon", 150)))

        def plan_step_fn(edoc_planner: EDOCPlanner, x0: np.ndarray, rng_key: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
            # Set initial state for planning env, then plan
            plan_env.set_initial_state(x0)
            result = edoc_planner.plan(rng_key)
            actions = result.get("actions", None)
            if actions is None or len(actions) == 0:
                raise RuntimeError("EDOC returned no actions")
            u0 = np.asarray(actions[0], dtype=np.float32)
            return u0, {"edoc_result": {k: v for k, v in result.items() if k in {"terminal_energy", "scheduler_params_history"}}}

        return run_mpc_episode(
            exec_env=exec_env,
            planner=edoc,
            plan_step_fn=plan_step_fn,
            rng=rng,
            max_steps=max_steps,
        )



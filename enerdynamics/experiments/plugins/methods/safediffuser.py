"""
SafeDiffuser method plugin implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys
import numpy as np

from enerdynamics.core.dynamics import DynamicsToEnvAdapter
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from ...framework.base import MethodPlugin
from enerdynamics.solvers.single.safediffuser import SafeDiffuserSolver

# Ensure vendored diffuser is importable.
project_root = Path(__file__).resolve().parents[4]
third_party = (project_root / "third_party").resolve()
if str(third_party) not in sys.path:
    sys.path.insert(0, str(third_party))
import diffuser.utils as d_utils  # type: ignore



class SafeDiffuserMethodPlugin(MethodPlugin):
    @property
    def name(self) -> str:
        return "safediffuser"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> "SafeDiffuserSolver":
        exp = config.get("exp", "avoiding-d3il")
        # Match DPCC convention: prefer explicit seed from method_params.
        seed = int(config.get("seed"))
        device = str(config.get("device", "cuda"))

        # Mirror DPCC's loading convention.
        loadbase = config.get("loadbase", "logs")
        dataset = config.get("dataset", exp)
        diffusion_loadpath = config.get("diffusion_loadpath", "diffusion")
        diffusion_epoch = config.get("diffusion_epoch", "best")

        diffusion_experiment = d_utils.load_diffusion(
            loadbase,
            dataset,
            diffusion_loadpath,
            str(seed),
            epoch=diffusion_epoch,
            device=device,
        )
        diffusion = diffusion_experiment.diffusion
        normalizer = diffusion_experiment.dataset.normalizer
        if hasattr(diffusion, "clip_denoised"):
            diffusion.clip_denoised = True

        plan_config = dict(config.get("plan_config", {}))
        plan_config.setdefault("exp", exp)
        plan_config.setdefault("horizon", getattr(diffusion, "horizon", 8))
        plan_config.setdefault("batch_size", int(config.get("batch_size", 4)))
        plan_config.setdefault(
            "safediffuser_config_path", 
            config.get("safediffuser_config_path", "enerdynamics/solvers/single/safediffuser/config/avoiding_d3il.yaml")
        )
        plan_config.setdefault("enable_safety", bool(config.get("enable_safety", True)))
        plan_config.setdefault("correct_all_steps", bool(config.get("correct_all_steps", False)))
        plan_config.setdefault("which_trajectory", int(config.get("which_trajectory", 0)))
        plan_config.setdefault("receding_horizon", bool(config.get("receding_horizon", True)))
        plan_config.setdefault("num_modes", int(config.get("num_modes", 1)))
        plan_config.setdefault("plan_once_chunks", int(config.get("plan_once_chunks", 1)))
        plan_config.setdefault("plan_once_steps_per_chunk", int(config.get("plan_once_steps_per_chunk", 0)))
        plan_config.setdefault("max_episode_length", int(config.get("max_episode_length", 200)))
        plan_config.setdefault("safediffuser_pos_idx", tuple(config.get("safediffuser_pos_idx", (0,1))))
        plan_config.setdefault("derive_action_from_states", bool(config.get("derive_action_from_states", True)))
        plan_config.setdefault("native_9d", bool(config.get("native_9d", False)))
        # Align with enerdynamics unified constraint stack when available.
        plan_config.setdefault("use_framework_constraints", bool(config.get("use_framework_constraints", True)))
        # Align CBF obstacle set with framework runtime obstacle generation.
        plan_config.setdefault(
            "align_constraints_with_framework",
            bool(config.get("align_constraints_with_framework", True)),
        )
        plan_config.setdefault(
            "halfspace_variants",
            config.get("halfspace_variants", None),
        )
        plan_config.setdefault("obstacles", config.get("obstacles"))
        plan_config.setdefault("obstacle_config", config.get("obstacle_config", {}))

        # inner default config
        plan_config.setdefault("return_diffusion", bool(config.get("return_diffusion", True)))
        plan_config.setdefault("test_ret", float(config.get("test_ret", 0.0)))
        plan_config.setdefault("preprocess_fns", config.get("preprocess_fns", []))
        plan_config.setdefault("use_target_line", config.get("use_target_line", False))
        plan_config.setdefault("num_targets", config.get("num_targets", 4))
        # 9D extra action-space safety projection (optional)
        plan_config.setdefault("action_cbf_alpha", float(config.get("action_cbf_alpha", 0.5)))
        plan_config.setdefault("action_cbf_margin", float(config.get("action_cbf_margin", 0.0)))

        # Goal comes from d3il env wrapper by default.
        goal_xy = np.asarray(config.get("goal_xy", getattr(env, "target", None)), dtype=np.float32) if getattr(env, "target", None) is not None or config.get("goal_xy") is not None else None

        dynamics = DynamicsToEnvAdapter(env, dt=float(getattr(env, "dt", 0.1)))
        backend = RuntimeBackendManager.get_backend()

        return SafeDiffuserSolver(
            dynamics=dynamics,
            energy=energy,
            backend=backend,
            env=env,
            diffusion=diffusion,
            normalizer=normalizer,
            plan_config=plan_config,
            goal_xy=goal_xy,
            device=device,
            seed=seed,
            constraint_manager=config.get("constraint_manager"),
            constraint_pipeline=config.get("constraint_pipeline"),
        )

    def plan(self, planner: "SafeDiffuserSolver", initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        try:
            seed = int(getattr(rng, "integers", lambda low, high: 0)(0, 2**31 - 1))
            reset_out = planner.env.reset(seed=seed)  # 有些 env 支持 seed
        except TypeError:
            reset_out = planner.env.reset()
        except Exception:
            reset_out = planner.env.reset()

        if reset_out is not None:
            if isinstance(reset_out, (tuple, list)) and len(reset_out) > 0:
                initial_state = np.asarray(reset_out[0], dtype=np.float32)
            else:
                initial_state = np.asarray(reset_out, dtype=np.float32)
        horizon = int(planner.plan_config.get("horizon", getattr(planner.diffusion, "horizon", 8)))
        traj = planner.solve(initial_state, horizon=horizon, rng_key=rng)
        out = {
            "states": traj.states,
            "actions": traj.actions,
            "initial_state": initial_state,
            "info": traj.info,
        }
        info = traj.info if isinstance(traj.info, dict) else {}
        if "states_9d" in info and "actions_9d" in info:
            out["states_9d"] = info["states_9d"]
            out["actions_9d"] = info["actions_9d"]
        if isinstance(traj.info, dict):
            for k in (
                "candidate_states",
                "candidate_actions",
                "candidate_costs",
                "best_idx",
                # Optional per-mode goal assignment metadata (multi-target debugging)
                "candidate_goals_xy",
                "candidate_goals_idx",
                "use_target_line",
                "num_targets",
            ):
                if k in traj.info:
                    out[k] = traj.info[k]
        return out

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
        # inner default config
        plan_config.setdefault("return_diffusion", bool(config.get("return_diffusion", True)))
        plan_config.setdefault("test_ret", float(config.get("test_ret", 0.0)))
        plan_config.setdefault("preprocess_fns", config.get("preprocess_fns", []))

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
        )

    def plan(self, planner: "SafeDiffuserSolver", initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        horizon = int(planner.plan_config.get("horizon", getattr(planner.diffusion, "horizon", 8)))
        traj = planner.solve(initial_state, horizon=horizon, rng_key=rng)
        return {
            "states": traj.states,
            "actions": traj.actions,
            "initial_state": initial_state,
            "info": traj.info,
        }

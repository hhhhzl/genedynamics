"""
MDOC method plugin implementation.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from genedynamics.solvers.single.mdoc import MDOCSolver
from genedynamics.core.dynamics import DynamicsToEnvAdapter
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.task_spec import get_default_task_spec
from genedynamics.core.constraints.action_filters import (
    NoOpConstraintFilter,
    ClosedFormCBFFilter,
    QPBasedCBFFilter,
    ClosedFormCBFFilterJointLift,
    QPBasedCBFFilterJointLift,
)
from ...framework.base import MethodPlugin
from ._result_utils import normalize_result_from_trajectory


class MDOCMethodPlugin(MethodPlugin):
    @property
    def name(self) -> str:
        return "mdoc"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        horizon = int(config.get("horizon", getattr(env, "horizon", 64)))
        dt = float(config.get("dt", getattr(env, "dt", 0.05)))

        backend = RuntimeBackendManager.get_backend()
        dynamics = DynamicsToEnvAdapter(env, dt=dt)

        # Select constraint filter based on YAML mode
        mode = config.get("mdoc_constraint_mode", "noop")
        if mode == "cbf_closed_form_perstep":
            constraint_filter = ClosedFormCBFFilter()
        elif mode == "cbf_qp_perstep":
            constraint_filter = QPBasedCBFFilter()
        elif mode == "cbf_closed_form_joint_lift_perstep":
            constraint_filter = ClosedFormCBFFilterJointLift()
        elif mode == "cbf_qp_joint_lift_perstep":
            constraint_filter = QPBasedCBFFilterJointLift()
        else:
            constraint_filter = NoOpConstraintFilter()

        # Extract parameters
        cbf_params = {
            "cbf_tau": float(config.get("cbf_tau", 0.005)),
            "cbf_eta": float(config.get("cbf_eta", 1.5)),
            "cbf_margin": float(config.get("cbf_margin", 0.1)),
            "base_beta": float(config.get("base_beta", 0.05)),
            "terminal_energy_weight": float(config.get("terminal_energy_weight", 100.0)),
            "guide_weight": float(config.get("guide_weight", 20.0)),
        }

        task_spec = get_default_task_spec(config.get("env_plugin"), config.get("env_name"))
        solver = MDOCSolver(
            dynamics=dynamics,
            energy=energy,
            backend=backend,
            horizon=horizon,
            dt=dt,
            Nsample=int(config.get("action_nsample", config.get("Nsample", 256))),
            Ndiffuse=int(config.get("action_diffuse_steps", config.get("Ndiffuse", 100))),
            temp_sample=float(config.get("temp_sample", 0.3)),
            beta0=float(config.get("beta0", 1e-4)),
            betaT=float(config.get("betaT", 1e-2)),
            action_limit=float(config.get("action_limit", getattr(env, "control_limit", 1.0))),
            seed=int(config.get("np_random_seed", 0)),
            scheduler=config.get("scheduler"),
            constraint_filter=constraint_filter,
            obstacles=config.get("obstacles"),
            show_tqdm=bool(config.get("show_tqdm", False)),
            num_modes=int(config.get("num_modes", 1)),  # Number of candidate trajectories to return
            mode_strategy=config.get("mode_strategy", "multirun"),
            diversity_eta=config.get("diversity_eta", 1.0),  # Diversity weight
            diversity_topK_cand=config.get("diversity_topK_cand", None),  # Pre-filter candidates
            diversity_use_state=config.get("diversity_use_state", True),  # Use state or action features
            position_extractor=task_spec.extract_position,
            position_dim=task_spec.position_dim,
            **cbf_params,
        )
        solver.env = env
        return solver

    def plan(self, planner: Any, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        rng_key = rng
        result_traj = planner.solve(initial_state, horizon=getattr(planner, "horizon", 64), rng_key=rng_key)
        return normalize_result_from_trajectory(result_traj, initial_state=initial_state, preserve_info=True)

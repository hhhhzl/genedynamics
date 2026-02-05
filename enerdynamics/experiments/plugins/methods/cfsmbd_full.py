"""
CFS-MBD method plugin implementation with full trajectory QP.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from enerdynamics.solvers.single.cfsmbd import CFSMBDSolver
from enerdynamics.core.dynamics import DynamicsToEnvAdapter
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.core.constraints.action_filters.cfs_qp_full import CFSQPFullFilter
from ...framework.base import MethodPlugin


class CFSMBDFullMethodPlugin(MethodPlugin):
    @property
    def name(self) -> str:
        return "cfsmbd_full"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        horizon = int(config.get("horizon", getattr(env, "horizon", 64)))
        dt = float(config.get("dt", getattr(env, "dt", 0.05)))

        backend = RuntimeBackendManager.get_backend()
        dynamics = DynamicsToEnvAdapter(env, dt=dt)

        # Create CFS-based full trajectory QP filter
        constraint_filter = CFSQPFullFilter(
            max_constraints_per_point=int(config.get("max_constraints_per_point", 8)),
            constraint_margin=float(config.get("constraint_margin", 0.25)),
            use_slack=False,
        )

        solver = CFSMBDSolver(
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
            aug_lambda=float(config.get("aug_lambda", 0.0)),  # Fixed lambda
            aug_rho=float(config.get("aug_rho", 1.0)),  # Fixed rho
            action_extra_sigma=float(config.get("action_extra_sigma", 0.0)),  # Extra noise for diversity
            num_modes=int(config.get("num_modes", 1)),  # Number of candidate trajectories to return
            mode_strategy=str(config.get("mode_strategy", "multirun")),  # How to generate C paths
            multirun_use_plan_batch_minimal=bool(config.get("multirun_use_plan_batch_minimal", False)),
            diversity_eta=float(config.get("diversity_eta", 1.0)),  # Diversity weight
            diversity_topK_cand=config.get("diversity_topK_cand", None),  # Pre-filter candidates
            diversity_use_state=bool(config.get("diversity_use_state", True)),  # Use state or action features
        )
        solver.env = env
        return solver

    def plan(self, planner: Any, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        rng_key = rng
        result_traj = planner.solve(initial_state, horizon=getattr(planner, "horizon", 64), rng_key=rng_key)
        
        # CRITICAL: result_traj.info MUST be returned directly to preserve all keys (including diffusion_*)
        if hasattr(result_traj, "info") and isinstance(result_traj.info, dict):
            # Also ensure states/actions exist for common metric extraction
            out = result_traj.info
            out["states"] = np.stack(result_traj.states, axis=0)
            out["actions"] = np.stack(result_traj.actions, axis=0)
            return out
            
        return {
            "actions": np.stack(result_traj.actions, axis=0),
            "states": np.stack(result_traj.states, axis=0),
        }

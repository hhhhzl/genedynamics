"""
2GO method plugin (Phase-1 minimal runnable).
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.constraints.action_filters.cfs_qp_full import CFSQPFullFilter
from genedynamics.core.dynamics import DynamicsToEnvAdapter
from genedynamics.core.task_spec import get_default_task_spec
from genedynamics.solvers.single.twogo import TwoGOSolver
from ...framework.base import MethodPlugin
from ._result_utils import normalize_result_from_trajectory


class TwoGOMethodPlugin(MethodPlugin):
    @property
    def name(self) -> str:
        return "2go"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        horizon = int(config.get("horizon", getattr(env, "horizon", 64)))
        dt = float(config.get("dt", getattr(env, "dt", 0.05)))

        backend = RuntimeBackendManager.get_backend()
        dynamics = DynamicsToEnvAdapter(env, dt=dt)

        task_spec = get_default_task_spec(config.get("env_plugin"), config.get("env_name"))
        # Phase-1: use full-trajectory CFS filter to preserve high-quality feasibility path.
        constraint_filter = CFSQPFullFilter(
            max_constraints_per_point=int(config.get("max_constraints_per_point", 8)),
            constraint_margin=float(config.get("constraint_margin", 0.25)),
            use_slack=False,
            convexifier_name=str(config.get("cfs_action_convexifier", "cfs_action")),
            position_extractor=task_spec.extract_position,
        )

        solver = TwoGOSolver(
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
            aug_lambda=float(config.get("aug_lambda", 0.0)),
            aug_rho=float(config.get("aug_rho", 1.0)),
            action_extra_sigma=float(config.get("action_extra_sigma", 0.0)),
            num_modes=int(config.get("num_modes", 1)),
            mode_strategy=str(config.get("mode_strategy", "multirun")),
            diversity_eta=float(config.get("diversity_eta", 1.0)),
            diversity_topK_cand=config.get("diversity_topK_cand", None),
            diversity_use_state=bool(config.get("diversity_use_state", True)),
            position_extractor=task_spec.extract_position,
            position_dim=task_spec.position_dim,
            # 2GO knobs
            twogo_sigma_max=float(config.get("twogo_sigma_max", 0.25)),
            twogo_gate_vrate_threshold=float(config.get("twogo_gate_vrate_threshold", 0.01)),
            twogo_sigma_q=float(config.get("twogo_sigma_q", 1.0)),
            twogo_sigma_q_lambda=float(config.get("twogo_sigma_q_lambda", 1.0)),
            twogo_delta0=float(config.get("twogo_delta0", 0.02)),
            twogo_delta_r=float(config.get("twogo_delta_r", 1.0)),
            twogo_delta_r_lambda=float(config.get("twogo_delta_r_lambda", 1.0)),
            twogo_lambda0=float(config.get("twogo_lambda0", 1.0)),
            twogo_theta_start=float(config.get("twogo_theta_start", 0.6)),
            twogo_theta_end=float(config.get("twogo_theta_end", 0.2)),
            twogo_cvar_alpha=float(config.get("twogo_cvar_alpha", 0.9)),
            twogo_window_size=int(config.get("twogo_window_size", 16)),
            twogo_window_stride=int(config.get("twogo_window_stride", 8)),
            twogo_tail_ratio=float(config.get("twogo_tail_ratio", 0.3)),
            twogo_active_topk=int(config.get("twogo_active_topk", 8)),
            twogo_agp_eta=float(config.get("twogo_agp_eta", 0.08)),
            twogo_cfs_gain=float(config.get("twogo_cfs_gain", 0.35)),
            twogo_multi_scale=float(config.get("twogo_multi_scale", 0.25)),
            twogo_enable_agp_refine=bool(config.get("twogo_enable_agp_refine", False)),
            twogo_enable_local_gating=bool(config.get("twogo_enable_local_gating", True)),
            twogo_enable_sample_tail=bool(config.get("twogo_enable_sample_tail", True)),
            twogo_enable_agp_batch=bool(config.get("twogo_enable_agp_batch", False)),
            twogo_use_jax_scan_core=bool(config.get("twogo_use_jax_scan_core", True)),
            twogo_retract_qp_boost=bool(config.get("twogo_retract_qp_boost", True)),
            twogo_stability_eps=float(config.get("twogo_stability_eps", 1e-6)),
            twogo_gamma_init=float(config.get("twogo_gamma_init", 1.0)),
            twogo_tail_mix_ratio=float(config.get("twogo_tail_mix_ratio", 0.5)),
            twogo_tail_pool_ratio=float(config.get("twogo_tail_pool_ratio", 0.25)),
            twogo_m_eff_min=int(config.get("twogo_m_eff_min", 16)),
            twogo_force_retract_from=float(config.get("twogo_force_retract_from", 0.8)),
        )
        solver.env = env
        return solver

    def plan(self, planner: Any, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        result_traj = planner.solve(initial_state, horizon=getattr(planner, "horizon", 64), rng_key=rng)
        return normalize_result_from_trajectory(result_traj, initial_state=initial_state, preserve_info=True)


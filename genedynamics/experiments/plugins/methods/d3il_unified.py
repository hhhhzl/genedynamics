"""
Unified D3IL method plugin: one entry point for all solvers (MBD, EBMBD, MDOC, CFS-MBD)
with execution mode mpc or plan_once. All execution goes through run_d3il_unified.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from genedynamics.core.dynamics.adapters import EnvDynamicsAdapter


class _ScaledStageEnergy:
    """Wraps a legacy energy to scale stage (intermediate) cost by a constant."""

    def __init__(self, energy: Any, scale: float):
        self._energy = energy
        self._scale = float(scale)

    def compute(self, x: Any, u: Any, ctx: Any = None) -> Any:
        return self._scale * self._energy.compute(x, u, ctx)
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.experiments.common.d3il_mpc import run_d3il_unified
from ...framework.base import MethodPlugin

# 4D (tcp_xy + x_des,y_des style)
from genedynamics.envs.external.d3il.avoiding_plan_env import AvoidingPlanEnv, AvoidingPlanSpec
# 9D (tcp_xy + q)
from genedynamics.envs.external.d3il.avoiding_plan_env_9d import AvoidingPlanEnv9D, AvoidingPlanSpec9D


def _make_plan_env_4d(config: Dict[str, Any], env: Any) -> AvoidingPlanEnv:
    horizon = int(config.get("horizon", 20))
    dt = float(config.get("dt", getattr(env, "dt", 0.035)))
    action_limit = float(config.get("action_limit", getattr(env, "control_limit", 0.05)))
    return AvoidingPlanEnv(
        AvoidingPlanSpec(dt=dt, horizon=horizon, control_limit=action_limit)
    )


def _make_plan_env_9d(config: Dict[str, Any], env: Any) -> AvoidingPlanEnv9D:
    horizon = int(config.get("horizon", 20))
    dt = float(config.get("dt", getattr(env, "dt", 0.035)))
    qdot_limit = float(config.get("action_limit", config.get("qdot_limit", 1.5)))
    return AvoidingPlanEnv9D(
        AvoidingPlanSpec9D(dt=dt, horizon=horizon, qdot_limit=qdot_limit)
    )


class D3ILUnifiedMethodPlugin(MethodPlugin):
    """
    Single method plugin for D3IL: config.solver (mbd | ebmbd | mdoc | cfsmbd | cfsmbd_full)
    and config.execution (mpc | plan_once). All solvers use the same run_d3il_unified execution.
    """

    @property
    def name(self) -> str:
        return "d3il_unified"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        state_dim = getattr(env, "state_dim", 4)
        if state_dim >= 9:
            plan_env = _make_plan_env_9d(config, env)
        else:
            plan_env = _make_plan_env_4d(config, env)
        # Sync plan env target to exec env (e.g. center 0.5, 0.35) so planning and cost use same target
        if hasattr(env, "target") and hasattr(plan_env, "target"):
            plan_env.target = np.asarray(env.target, dtype=np.float32).reshape(-1)[:2]
        # Sync robot_radius so CFS filter gets correct clearance (plan_env is passed via adapter to filter)
        obstacle_config = config.get("obstacle_config") or {}
        plan_env.robot_radius = float(
            obstacle_config.get("robot_radius", getattr(env, "robot_radius", 0.01)))

        backend = RuntimeBackendManager.get_backend()
        dynamics = EnvDynamicsAdapter(plan_env)
        solver_name = config.get("solver", "mbd").lower()

        stage_weight = float(config.get("stage_cost_weight", 1.0))
        energy_to_use = _ScaledStageEnergy(energy, stage_weight) if stage_weight != 1.0 else energy

        horizon = int(config.get("horizon", getattr(plan_env, "horizon", 20)))
        dt = float(config.get("dt", getattr(plan_env, "dt", 0.035)))
        action_limit = float(config.get("action_limit", getattr(plan_env, "control_limit", 0.05)))

        if solver_name == "mbd":
            from genedynamics.solvers.single.mbd import MBDSolver
            solver = MBDSolver(
                dynamics=dynamics,
                energy=energy_to_use,
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
                num_modes=int(config.get("num_modes", 1)),
                mode_strategy=config.get("mode_strategy", "multirun"),
                diversity_eta=float(config.get("diversity_eta", 1.0)),
                diversity_topK_cand=config.get("diversity_topK_cand"),
                diversity_use_state=bool(config.get("diversity_use_state", True)),
                terminal_energy_weight=float(config.get("terminal_energy_weight", 0.0)),
                use_target_line=bool(config.get("use_target_line", False)),
                num_targets=int(config.get("num_targets", 4)),
            )
        elif solver_name == "ebmbd":
            from genedynamics.solvers.single.ebmbd import EBMBDSolver
            solver = EBMBDSolver(
                dynamics=dynamics,
                energy=energy_to_use,
                backend=backend,
                horizon=horizon,
                dt=dt,
                Nsample=int(config.get("Nsample", 2048)),
                Ndiffuse=int(config.get("Ndiffuse", 100)),
                temp_sample=float(config.get("temp_sample", 0.1)),
                beta0=float(config.get("beta0", 1e-4)),
                betaT=float(config.get("betaT", 1e-2)),
                action_limit=action_limit,
                action_extra_sigma=float(config.get("action_extra_sigma", 0.0)),
                mu=float(config.get("mu", 10.0)),
                alpha=float(config.get("alpha", 1.0)),
                bound=float(config.get("bound", 0.8)),
                use_min_over_time=bool(config.get("use_min_over_time", True)),
                terminal_energy_weight=float(config.get("terminal_energy_weight", 0.0)),
                obstacles=config.get("obstacles"),
                obstacle_config=config.get("obstacle_config"),
                seed=int(config.get("np_random_seed", 0) or 0),
                scheduler=config.get("scheduler"),
                show_tqdm=bool(config.get("show_tqdm", False)),
                num_modes=int(config.get("num_modes", 1)),
                mode_strategy=config.get("mode_strategy", "multirun"),
                diversity_eta=float(config.get("diversity_eta", 1.0)),
                diversity_topK_cand=config.get("diversity_topK_cand"),
                diversity_use_state=bool(config.get("diversity_use_state", True)),
                use_target_line=bool(config.get("use_target_line", False)),
                num_targets=int(config.get("num_targets", 4)),
            )
        elif solver_name == "mdoc":
            from genedynamics.solvers.single.mdoc import MDOCSolver
            from genedynamics.core.constraints.action_filters import (
                NoOpConstraintFilter,
                ClosedFormCBFFilter,
                QPBasedCBFFilter,
                ClosedFormCBFFilterJointLift,
                QPBasedCBFFilterJointLift,
            )
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
            cbf_params = {
                "cbf_tau": float(config.get("cbf_tau", 0.005)),
                "cbf_eta": float(config.get("cbf_eta", 1.5)),
                "cbf_margin": float(config.get("cbf_margin", 0.1)),
                "base_beta": float(config.get("base_beta", 0.05)),
                "terminal_energy_weight": float(config.get("terminal_energy_weight", 100.0)),
                "guide_weight": float(config.get("guide_weight", 20.0)),
            }
            solver = MDOCSolver(
                dynamics=dynamics,
                energy=energy_to_use,
                backend=backend,
                horizon=horizon,
                dt=dt,
                Nsample=int(config.get("action_nsample", config.get("Nsample", 256))),
                Ndiffuse=int(config.get("action_diffuse_steps", config.get("Ndiffuse", 100))),
                temp_sample=float(config.get("temp_sample", 0.3)),
                beta0=float(config.get("beta0", 1e-4)),
                betaT=float(config.get("betaT", 1e-2)),
                action_limit=action_limit,
                seed=int(config.get("np_random_seed", 0)),
                scheduler=config.get("scheduler"),
                constraint_filter=constraint_filter,
                obstacles=config.get("obstacles"),
                show_tqdm=bool(config.get("show_tqdm", False)),
                num_modes=int(config.get("num_modes", 1)),
                mode_strategy=config.get("mode_strategy", "multirun"),
                diversity_eta=float(config.get("diversity_eta", 1.0)),
                diversity_topK_cand=config.get("diversity_topK_cand"),
                diversity_use_state=bool(config.get("diversity_use_state", True)),
                use_target_line=bool(config.get("use_target_line", False)),
                num_targets=int(config.get("num_targets", 4)),
                **cbf_params,
            )
        elif solver_name in ("cfsmbd", "cfsmbd_full"):
            from genedynamics.solvers.single.cfsmbd import CFSMBDSolver
            from genedynamics.core.constraints.action_filters.cfs_qp_perstep import CFSQPPerStepFilter
            from genedynamics.core.constraints.action_filters.cfs_qp_full import CFSQPFullFilter
            cfs_convexifier_name = str(config.get("cfs_action_convexifier", "cfs_action"))
            if solver_name == "cfsmbd_full":
                constraint_filter = CFSQPFullFilter(
                    max_constraints_per_point=int(config.get("max_constraints_per_point", 8)),
                    constraint_margin=float(config.get("constraint_margin", 0.25)),
                    use_slack=False,
                    convexifier_name=cfs_convexifier_name,
                )
            else:
                constraint_filter = CFSQPPerStepFilter(
                    max_constraints_per_point=int(config.get("max_constraints_per_point", 8)),
                    constraint_margin=float(config.get("constraint_margin", 0.25)),
                    use_slack=True,
                    convexifier_name=cfs_convexifier_name,
                )
            solver = CFSMBDSolver(
                dynamics=dynamics,
                energy=energy_to_use,
                backend=backend,
                horizon=horizon,
                dt=dt,
                Nsample=int(config.get("action_nsample", config.get("Nsample", 256))),
                Ndiffuse=int(config.get("action_diffuse_steps", config.get("Ndiffuse", 100))),
                temp_sample=float(config.get("temp_sample", 0.3)),
                beta0=float(config.get("beta0", 1e-4)),
                betaT=float(config.get("betaT", 1e-2)),
                action_limit=action_limit,
                seed=int(config.get("np_random_seed", 0)),
                scheduler=config.get("scheduler"),
                constraint_filter=constraint_filter,
                obstacles=config.get("obstacles"),
                show_tqdm=bool(config.get("show_tqdm", False)),
                aug_lambda=float(config.get("aug_lambda", 0.0)),
                aug_rho=float(config.get("aug_rho", 1.0)),
                action_extra_sigma=float(config.get("action_extra_sigma", 0.0)),
                num_modes=int(config.get("num_modes", 1)),
                mode_strategy=config.get("mode_strategy", "multirun"),
                diversity_eta=float(config.get("diversity_eta", 1.0)),
                diversity_topK_cand=config.get("diversity_topK_cand"),
                diversity_use_state=bool(config.get("diversity_use_state", True)),
                use_target_line=bool(config.get("use_target_line", False)),
                num_targets=int(config.get("num_targets", 4)),
            )
            if solver_name == "cfsmbd_full":
                solver.multirun_use_plan_batch_minimal = bool(config.get("multirun_use_plan_batch_minimal", False))
        else:
            raise ValueError(f"d3il_unified: unknown solver={solver_name}. Use mbd, ebmbd, mdoc, cfsmbd, or cfsmbd_full.")

        return {"exec_env": env, "plan_env": plan_env, "solver": solver, "config": config}

    def plan(self, planner: Any, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        _ = initial_state
        return run_d3il_unified(planner, initial_state, rng)

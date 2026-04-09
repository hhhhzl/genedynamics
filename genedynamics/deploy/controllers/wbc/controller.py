"""Humanoid whole-body controller.

Implements the :class:`~genedynamics.deploy.interfaces.controller.Controller`
protocol from Phase 2 by composing the WBC modules in this package:

* :mod:`config`            — grouped tuning surface
* :mod:`task_stack`        — per-task acceleration commands
* :mod:`contact_blocks`    — support-foot equality blocks
* :mod:`friction_cone`     — inequality constraints
* :mod:`qp_builder`        — assemble the QP and recover torques
* :mod:`qp_solver_adapter` — multi-method fallback solver

The controller's :meth:`act` is a thin orchestration: pull a
:class:`HumanoidTaskSpec` from ``intent.extras["humanoid_tasks"]``, query
the IO for ``M`` / ``bias``, assemble the QP, solve, recover torques,
clip, and emit a :class:`ControlCommand` (kind=``"joint_pos"``) with the
solver telemetry attached in ``cmd.extras``.

Two design notes:

* The controller is **stateless** with respect to physics — it queries the
  IO every ``act`` call. The only stored state is the QP adapter (warm
  start) and the startup-blend step counter.
* The "joint reference" emitted by the controller is computed from a
  one-step Euler integration of the QP's ``ddq`` solution, exactly as the
  legacy controller did. ``q_ref = q_act + qd_ref · dt``, with ``qd_ref``
  itself clipped, then ``q_ref`` clipped to joint limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.controllers.wbc.config import WBCConfig
from genedynamics.deploy.controllers.wbc.qp_builder import (
    QPProblem,
    assemble_qp,
    recover_torques,
)
from genedynamics.deploy.controllers.wbc.qp_solver_adapter import (
    QPSolverAdapter,
    SolveResult,
)
from genedynamics.deploy.controllers.wbc.task_stack import lower_body_target
from genedynamics.deploy.followers.humanoid.task_spec import HumanoidTaskSpec
from genedynamics.deploy.interfaces.messages import ControlCommand, Intent, RobotState
from genedynamics.deploy.io.mujoco_io import MujocoRobotIO

__all__ = ["HumanoidWBCController", "WBCResult"]


# ---------------------------------------------------------------------------
# Internal result type — what the controller computes per step
# ---------------------------------------------------------------------------


@dataclass
class WBCResult:
    """Diagnostic bundle exposed via ``ControlCommand.extras["wbc"]``."""

    q_ref: np.ndarray
    qd_ref: np.ndarray
    ddq_full: np.ndarray
    tau_ff: np.ndarray
    lambda_ref: np.ndarray
    method: str
    eq_residual: float
    ineq_violation: float
    task_errors: Dict[str, float]
    contact_feet: Tuple[str, ...]
    torque_saturation_max: float
    torque_bound_violation_max: float
    startup_blend: float = 1.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class HumanoidWBCController:
    """Whole-body controller for the Unitree G1 (and compatible humanoids).

    Args:
        io: A :class:`MujocoRobotIO` (or :class:`MjxRobotIO` with host shadow
            sync enabled). The controller uses the IO for physics queries
            (mass matrix, jacobians, contacts) and never touches MuJoCo
            directly.
        cfg: Optional :class:`WBCConfig`. Defaults reproduce the legacy
            ``G1WBCTaskStackConfig`` exactly.
        startup_blend_second_step_scale: Lower-body actuation blend at the
            second step (matches the legacy 0.70 default).
    """

    runtime: str = "numpy"
    produces: Tuple[str, ...] = ("joint_pos",)

    def __init__(
        self,
        io: MujocoRobotIO,
        cfg: Optional[WBCConfig] = None,
        *,
        startup_blend_second_step_scale: float = 0.70,
    ) -> None:
        self.io = io
        self.spec = io.spec
        self.cfg = cfg or WBCConfig()
        self.startup_blend_second_step_scale = float(startup_blend_second_step_scale)
        self._qp = QPSolverAdapter(self.cfg.solver)

    # ------------------------------------------------------------------
    # Controller protocol
    # ------------------------------------------------------------------

    def reset(self, io: Optional[MujocoRobotIO] = None) -> None:
        """Re-bind the controller to ``io`` (if provided) and reset warm start."""
        if io is not None:
            self.io = io
            self.spec = io.spec
        self._qp = QPSolverAdapter(self.cfg.solver)

    def act(self, state: RobotState, intent: Intent) -> ControlCommand:
        """Solve the WBC QP for the current step and emit a joint-pos command.

        ``intent.extras`` must contain a ``"humanoid_tasks"`` key holding a
        :class:`HumanoidTaskSpec`. The follower (corridor follower in this
        codebase) is responsible for assembling that struct from the
        14-D plan frame and contact phase information.

        ``state.extras`` may contain ``"dt"``; if missing the controller
        falls back to ``intent.extras.get("dt", 0.02)``.
        """
        tasks = intent.extras.get("humanoid_tasks") if intent.extras else None
        if not isinstance(tasks, HumanoidTaskSpec):
            raise ValueError(
                "HumanoidWBCController requires intent.extras['humanoid_tasks'] "
                "to be a HumanoidTaskSpec instance produced by the follower."
            )
        dt = float(
            (state.extras or {}).get(
                "dt", (intent.extras or {}).get("dt", 0.02)
            )
        )
        result = self._compute(tasks, state, dt=dt)

        # Apply startup blend if the follower exposed a phase via intent.extras
        phase = (intent.extras or {}).get("phase")
        if phase is not None:
            result = self._apply_startup_blend(result, phase)

        return ControlCommand(
            kind="joint_pos",
            joint_pos=result.q_ref,
            joint_vel=result.qd_ref,
            joint_torque=result.tau_ff,
            extras={"wbc": result},
        )

    # ------------------------------------------------------------------
    # Inner: one full WBC step (no I/O on the IO side after queries)
    # ------------------------------------------------------------------

    def _compute(
        self,
        tasks: HumanoidTaskSpec,
        state: RobotState,
        *,
        dt: float,
    ) -> WBCResult:
        spec = self.spec
        nv = self.io.model.nv
        dt = max(dt, 1e-6)

        # 1. Push state into the IO's data so physics queries are consistent.
        q_full = np.asarray(state.qpos, dtype=np.float64).reshape(-1)
        qvel_full = np.asarray(state.qvel, dtype=np.float64).reshape(-1)
        if qvel_full.size != nv:
            qvel_full = np.zeros(nv, dtype=np.float64)
        self.io.data.qpos[:] = q_full
        self.io.data.qvel[:] = qvel_full
        self.io._mujoco.mj_forward(self.io.model, self.io.data)

        q_act = spec.actuated_qpos_from_full(q_full)
        qd_act = self._actuated_qvel_from_full(qvel_full)

        # 2. Build joint hint vector and lower-body target
        q_hint = spec.joint_dict_to_vector(tasks.joint_hints, base=spec.stand_ctrl)
        lower_q, _ = lower_body_target(
            spec, tasks, q_act, gains=self.cfg.gains, limits=self.cfg.limits
        )

        # 3. Mass + bias from MuJoCo
        M = self.io.mass_matrix()
        bias = self.io.bias()

        # 4. Assemble + solve QP
        problem: QPProblem = assemble_qp(
            self.io,
            tasks,
            M=M,
            bias=bias,
            qvel_full=qvel_full,
            q_act=q_act,
            qd_act=qd_act,
            q_hint=q_hint,
            lower_body_target_q=lower_q,
            cfg=self.cfg,
        )

        prefer_trust_constr = len(problem.blocks) == 1
        solve: SolveResult = self._qp.solve(
            problem.H,
            problem.f,
            problem.C_eq,
            problem.d_eq,
            problem.G_ineq,
            problem.h_ineq,
            prefer_trust_constr=prefer_trust_constr,
        )
        x = solve.x
        ddq_full = np.asarray(x[: problem.nv], dtype=np.float64).copy()
        lambda_ref = (
            np.asarray(x[problem.nv:], dtype=np.float64).copy()
            if problem.n_lambda > 0
            else np.zeros((0,), dtype=np.float64)
        )

        # 5. Recover actuator torques + clip to limits
        tau_ff = recover_torques(
            spec, M, bias, ddq_full, problem.support_jacobian, lambda_ref
        )
        torque_limit = self.cfg.limits.torque_limit_scale * spec.torque_limit_vector()
        bound_violation = float(np.max(np.maximum(np.abs(tau_ff) - torque_limit, 0.0))) if tau_ff.size else 0.0
        if bound_violation > 1e-6:
            tau_clipped = np.clip(tau_ff, -torque_limit, torque_limit)
            saturation = float(np.max(np.abs(tau_ff - tau_clipped))) if tau_ff.size else 0.0
            tau_ff = tau_clipped
        else:
            saturation = 0.0

        # 6. Joint reference via Euler integration of ddq
        ddq_act = ddq_full[spec.actuated_dof_indices]
        qd_ref = np.clip(
            qd_act + ddq_act * dt,
            -self.cfg.limits.max_qd_ref,
            self.cfg.limits.max_qd_ref,
        )
        q_ref = q_act + qd_ref * dt
        q_ref = np.clip(
            q_ref - q_act, -self.cfg.limits.max_q_step, self.cfg.limits.max_q_step
        ) + q_act
        q_ref = spec.clip_to_joint_limits(q_ref)

        return WBCResult(
            q_ref=q_ref,
            qd_ref=qd_ref,
            ddq_full=ddq_full,
            tau_ff=tau_ff,
            lambda_ref=lambda_ref,
            method=solve.method,
            eq_residual=solve.eq_residual,
            ineq_violation=solve.ineq_violation,
            task_errors=problem.task_errors,
            contact_feet=tuple(b.name for b in problem.blocks),
            torque_saturation_max=saturation,
            torque_bound_violation_max=bound_violation,
        )

    # ------------------------------------------------------------------
    # Startup blend (lifted from legacy pipeline.py)
    # ------------------------------------------------------------------

    def _apply_startup_blend(self, result: WBCResult, phase: Any) -> WBCResult:
        beta = self._startup_blend_factor(phase)
        if beta >= 0.999:
            result.startup_blend = 1.0
            return result

        spec = self.spec
        stand = np.asarray(spec.stand_ctrl, dtype=np.float64)
        upper_mask = np.zeros((spec.num_actuated,), dtype=np.float64)
        upper_joint_names = (
            tuple(spec.waist_joints)
            + tuple(spec.left_arm_joints)
            + tuple(spec.right_arm_joints)
        )
        for name in upper_joint_names:
            if name in spec.actuated_joints:
                upper_mask[spec.actuated_joints.index(name)] = 1.0
        leg_mask = 1.0 - upper_mask
        leg_beta = min(1.0, 0.70 + 0.30 * beta)

        q_ref = result.q_ref
        blended = stand + upper_mask * beta * (q_ref - stand) + leg_mask * leg_beta * (q_ref - stand)
        result.q_ref = blended
        if result.qd_ref is not None:
            result.qd_ref = upper_mask * beta * result.qd_ref + leg_mask * leg_beta * result.qd_ref
        if result.tau_ff.size:
            result.tau_ff = upper_mask * beta * result.tau_ff + leg_mask * leg_beta * result.tau_ff
        if result.ddq_full.size:
            ddq = result.ddq_full.copy()
            act_idx = np.asarray(spec.actuated_dof_indices, dtype=np.int32)
            ddq[act_idx] = upper_mask * beta * ddq[act_idx] + leg_mask * leg_beta * ddq[act_idx]
            result.ddq_full = ddq
        result.startup_blend = float(beta)
        return result

    def _startup_blend_factor(self, phase: Any) -> float:
        step_index = int(getattr(phase, "step_index", 0))
        if step_index <= 0:
            phase_value = getattr(getattr(phase, "phase", None), "value", phase.phase if hasattr(phase, "phase") else None)
            if phase_value == "double_support":
                alpha = float(np.clip(getattr(phase, "alpha", 0.0), 0.0, 1.0))
                return float(0.10 + 0.50 * alpha)
            liftoff = bool(getattr(phase, "metadata", {}).get("liftoff_confirmed", False))
            if not liftoff:
                return 0.35
            release = float(np.clip((float(getattr(phase, "alpha", 0.0)) - 0.18) / 0.82, 0.0, 1.0))
            return float(0.80 + 0.20 * release)
        if step_index == 1:
            alpha = float(np.clip(getattr(phase, "alpha", 0.0), 0.0, 1.0))
            return float(min(1.0, self.startup_blend_second_step_scale + 0.45 * alpha))
        return 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _actuated_qvel_from_full(self, qvel_full: np.ndarray) -> np.ndarray:
        idx = self.spec.actuated_dof_indices
        if idx.size == 0 or int(idx.max()) >= qvel_full.size:
            return np.zeros((self.spec.num_actuated,), dtype=np.float64)
        return np.asarray(qvel_full[idx], dtype=np.float64).copy()

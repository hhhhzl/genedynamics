"""
MuJoCo-backed G1 IK solver for stage-1 corridor following.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from genedynamics.deploy.sim_plan.humanoid_corridor.g1_model import G1ModelSpec
from genedynamics.deploy.sim_plan.humanoid_corridor.schema import (
    FollowerTasks,
    JointTargets,
)
from genedynamics.deploy.sim_plan.humanoid_corridor.solver_base import CorridorTaskSolverBase


@dataclass
class G1IKSolverConfig:
    max_iterations: int = 24
    damping: float = 1e-3
    step_size: float = 0.65
    position_tolerance: float = 5e-3
    use_leg_ik: bool = True
    body_height_nominal: float = 0.75
    body_height_min: float = 0.55
    crouch_hip_pitch_gain: float = 0.35
    crouch_knee_gain: float = 0.95
    crouch_ankle_pitch_gain: float = -0.45


class G1CorridorIKSolver(CorridorTaskSolverBase):
    """
    Stage-1 task solver:
    - maps torso/arms directly into joint hints
    - seeds symmetric crouch posture
    - solves left/right foot position IK in MuJoCo

    This is intentionally kept solver-only so the same task interface can be
    reused by a future WBC/QP backend.
    """

    def __init__(
        self,
        model: object,
        data: object,
        model_spec: G1ModelSpec,
        cfg: Optional[G1IKSolverConfig] = None,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.model_spec = model_spec
        self.cfg = cfg or G1IKSolverConfig()
        self._root_height_offset = self._infer_root_height_offset()

    def solve(
        self,
        tasks: FollowerTasks,
        *,
        qpos: Sequence[float],
        qvel: Optional[Sequence[float]] = None,
        dt: float = 0.0,
    ) -> JointTargets:
        _ = qvel, dt
        current_full = np.asarray(qpos, dtype=np.float64).reshape(-1)
        base_full = self._apply_pelvis_pose(current_full, tasks)
        q_ref = self._seed_joint_vector(base_full, tasks)
        q_ref = self._apply_joint_hints(q_ref, tasks.joint_hints)

        metadata: Dict[str, object] = {
            "solver": "ik",
            "use_leg_ik": bool(self.cfg.use_leg_ik),
        }
        full_q = self.model_spec.apply_actuated_qpos(base_full, q_ref)
        if self.cfg.use_leg_ik:
            full_q, leg_meta = self._solve_feet(full_q, tasks)
            q_ref = self.model_spec.actuated_qpos_from_full(full_q)
            metadata.update(leg_meta)
        q_ref = self.model_spec.clip_to_joint_limits(q_ref)
        qd_ref = np.zeros_like(q_ref)
        return JointTargets(q_ref=q_ref, qd_ref=qd_ref, metadata=metadata)

    def _seed_joint_vector(self, current_full_qpos: np.ndarray, tasks: FollowerTasks) -> np.ndarray:
        try:
            q_ref = self.model_spec.actuated_qpos_from_full(current_full_qpos)
        except ValueError:
            q_ref = self.model_spec.stand_ctrl.copy()
        crouch = self._crouch_ratio(float(tasks.pelvis.position_world[2]))
        for prefix in ("left", "right"):
            q_ref = self._set_joint_value(
                q_ref,
                f"{prefix}_hip_pitch_joint",
                self._get_joint_value(q_ref, f"{prefix}_hip_pitch_joint") + self.cfg.crouch_hip_pitch_gain * crouch,
            )
            q_ref = self._set_joint_value(
                q_ref,
                f"{prefix}_knee_joint",
                self._get_joint_value(q_ref, f"{prefix}_knee_joint") + self.cfg.crouch_knee_gain * crouch,
            )
            q_ref = self._set_joint_value(
                q_ref,
                f"{prefix}_ankle_pitch_joint",
                self._get_joint_value(q_ref, f"{prefix}_ankle_pitch_joint") + self.cfg.crouch_ankle_pitch_gain * crouch,
            )
        return q_ref

    def _apply_joint_hints(self, q_ref: np.ndarray, joint_hints: Dict[str, float]) -> np.ndarray:
        out = np.asarray(q_ref, dtype=np.float64).copy()
        for name, value in joint_hints.items():
            if name not in self.model_spec.actuated_joints:
                continue
            out = self._set_joint_value(out, name, float(value))
        return out

    def _solve_feet(self, qpos_full: np.ndarray, tasks: FollowerTasks) -> Tuple[np.ndarray, Dict[str, object]]:
        out = np.asarray(qpos_full, dtype=np.float64).copy()
        errors: Dict[str, float] = {}
        iters: Dict[str, int] = {}

        out, err_l, it_l = self._solve_leg_site_ik(
            out,
            self.model_spec.left_leg_joints,
            "left_foot",
            tasks.left_foot.position_world,
        )
        out, err_r, it_r = self._solve_leg_site_ik(
            out,
            self.model_spec.right_leg_joints,
            "right_foot",
            tasks.right_foot.position_world,
        )
        errors["left_foot"] = err_l
        errors["right_foot"] = err_r
        iters["left_foot"] = it_l
        iters["right_foot"] = it_r
        return out, {"foot_position_error": errors, "ik_iterations": iters}

    def _solve_leg_site_ik(
        self,
        qpos_full: np.ndarray,
        joint_names: Sequence[str],
        site_name: str,
        target_world: Sequence[float],
    ) -> Tuple[np.ndarray, float, int]:
        sid = self.model_spec.site_id.get(site_name)
        if sid is None:
            return np.asarray(qpos_full, dtype=np.float64).copy(), float("nan"), 0

        target = np.asarray(target_world, dtype=np.float64).reshape(3)
        q_work = np.asarray(qpos_full, dtype=np.float64).copy()
        qpos_idx = np.asarray([self.model_spec.joint_qpos_index[n] for n in joint_names], dtype=np.int32)
        dof_idx = np.asarray([self.model_spec.joint_dof_index[n] for n in joint_names], dtype=np.int32)
        ranges = np.asarray([self.model_spec.joint_range[n] for n in joint_names], dtype=np.float64)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        for it in range(int(self.cfg.max_iterations)):
            self.data.qpos[:] = q_work
            self.mujoco.mj_forward(self.model, self.data)
            current = np.asarray(self.data.site_xpos[sid], dtype=np.float64).copy()
            err = target - current
            err_norm = float(np.linalg.norm(err))
            if err_norm <= float(self.cfg.position_tolerance):
                return q_work, err_norm, it
            jacp.fill(0.0)
            self.mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
            J = np.asarray(jacp[:, dof_idx], dtype=np.float64)
            gram = J @ J.T + float(self.cfg.damping) * np.eye(3, dtype=np.float64)
            dq = J.T @ np.linalg.solve(gram, err)
            q_work[qpos_idx] += float(self.cfg.step_size) * dq
            q_work[qpos_idx] = np.clip(q_work[qpos_idx], ranges[:, 0], ranges[:, 1])

        self.data.qpos[:] = q_work
        self.mujoco.mj_forward(self.model, self.data)
        final_err = float(np.linalg.norm(target - np.asarray(self.data.site_xpos[sid], dtype=np.float64)))
        return q_work, final_err, int(self.cfg.max_iterations)

    def _get_joint_value(self, q_ref: np.ndarray, joint_name: str) -> float:
        idx = self.model_spec.actuated_joints.index(joint_name)
        return float(q_ref[idx])

    def _set_joint_value(self, q_ref: np.ndarray, joint_name: str, value: float) -> np.ndarray:
        out = np.asarray(q_ref, dtype=np.float64).copy()
        idx = self.model_spec.actuated_joints.index(joint_name)
        out[idx] = float(value)
        return out

    def _crouch_ratio(self, body_height: float) -> float:
        denom = max(self.cfg.body_height_nominal - self.cfg.body_height_min, 1e-6)
        return float(np.clip((self.cfg.body_height_nominal - body_height) / denom, 0.0, 1.0))

    def _infer_root_height_offset(self) -> float:
        try:
            kid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "stand")
            if kid >= 0 and self.model.key_qpos.shape[1] >= 3:
                stand_root_z = float(self.model.key_qpos[kid, 2])
                return stand_root_z - float(self.cfg.body_height_nominal)
        except Exception:
            pass
        if getattr(self.data, "qpos", None) is not None and len(self.data.qpos) >= 3:
            return float(self.data.qpos[2]) - float(self.cfg.body_height_nominal)
        return 0.0

    def _apply_pelvis_pose(self, qpos_full: np.ndarray, tasks: FollowerTasks) -> np.ndarray:
        out = np.asarray(qpos_full, dtype=np.float64).copy()
        if out.size < 7:
            return out
        pelvis_pos = np.asarray(tasks.pelvis.position_world, dtype=np.float64).reshape(3)
        out[0] = float(pelvis_pos[0])
        out[1] = float(pelvis_pos[1])
        out[2] = float(pelvis_pos[2]) + float(self._root_height_offset)
        out[3:7] = self._quat_wxyz_from_rpy(
            float(tasks.pelvis.roll_world),
            float(tasks.pelvis.pitch_world),
            float(tasks.pelvis.yaw_world),
        )
        return out

    @staticmethod
    def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = np.cos(0.5 * roll)
        sr = np.sin(0.5 * roll)
        cp = np.cos(0.5 * pitch)
        sp = np.sin(0.5 * pitch)
        cy = np.cos(0.5 * yaw)
        sy = np.sin(0.5 * yaw)
        return np.asarray(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )

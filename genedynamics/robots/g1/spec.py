"""Unitree G1 robot specification.

This is the **single source of truth** for G1 metadata used by every layer of
the deploy pipeline (controllers, IO adapters, follower trajectory builders,
real-hardware backends).

The module exports two things:

* **Joint topology constants** (``G1_LEFT_LEG_JOINTS`` and friends, plus
  ``G1_ACTUATED_JOINTS``). These are the canonical actuator names from the
  Unitree G1 29-DoF MJCF, ordered legs → waist → arms.
* **``G1RobotSpec``** — a dataclass populated by introspecting a loaded
  ``mujoco.MjModel``. All MuJoCo-side indices, joint ranges, actuator gains
  and torque bounds are read from the MJCF; nothing is hardcoded except the
  joint *names* themselves.

To construct a spec::

    import mujoco
    from genedynamics.robots.g1 import g1_scene_path, G1RobotSpec

    path = g1_scene_path()
    model = mujoco.MjModel.from_xml_path(path)
    spec  = G1RobotSpec.from_mujoco_model(model, model_xml_path=path)

The spec is **read-only after construction** by convention; controllers
should consume it without mutating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "G1_LEFT_LEG_JOINTS",
    "G1_RIGHT_LEG_JOINTS",
    "G1_WAIST_JOINTS",
    "G1_LEFT_ARM_JOINTS",
    "G1_RIGHT_ARM_JOINTS",
    "G1_ACTUATED_JOINTS",
    "G1_FOOT_SITE_NAMES",
    "G1_FOOT_BODY_NAMES",
    "G1_IMU_SITE_NAMES",
    "G1_BODY_NAMES",
    "G1RobotSpec",
]


# ---------------------------------------------------------------------------
# Joint topology — the only G1-specific strings in the codebase.
# ---------------------------------------------------------------------------

G1_LEFT_LEG_JOINTS: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
)

G1_RIGHT_LEG_JOINTS: Tuple[str, ...] = (
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

G1_WAIST_JOINTS: Tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

G1_LEFT_ARM_JOINTS: Tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

G1_RIGHT_ARM_JOINTS: Tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

#: Canonical actuator order: legs → waist → arms (29 names total).
G1_ACTUATED_JOINTS: Tuple[str, ...] = (
    G1_LEFT_LEG_JOINTS
    + G1_RIGHT_LEG_JOINTS
    + G1_WAIST_JOINTS
    + G1_LEFT_ARM_JOINTS
    + G1_RIGHT_ARM_JOINTS
)

G1_FOOT_SITE_NAMES: Tuple[str, str] = ("left_foot", "right_foot")
G1_FOOT_BODY_NAMES: Tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_IMU_SITE_NAMES: Tuple[str, str] = ("imu_in_pelvis", "imu_in_torso")
G1_BODY_NAMES: Tuple[str, str] = ("pelvis", "torso_link")


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=False)
class G1RobotSpec:
    """Introspected metadata for a loaded G1 MuJoCo model.

    All ``Dict`` fields are keyed by joint / site / body / actuator name.
    Index arrays follow the canonical actuator order in ``actuated_joints``.
    """

    # Provenance ------------------------------------------------------------
    model_xml_path: str

    # Joint topology (defaults match the canonical G1 layout) --------------
    actuated_joints: Tuple[str, ...] = G1_ACTUATED_JOINTS
    left_leg_joints: Tuple[str, ...] = G1_LEFT_LEG_JOINTS
    right_leg_joints: Tuple[str, ...] = G1_RIGHT_LEG_JOINTS
    waist_joints: Tuple[str, ...] = G1_WAIST_JOINTS
    left_arm_joints: Tuple[str, ...] = G1_LEFT_ARM_JOINTS
    right_arm_joints: Tuple[str, ...] = G1_RIGHT_ARM_JOINTS

    foot_site_names: Tuple[str, str] = G1_FOOT_SITE_NAMES
    foot_body_names: Tuple[str, str] = G1_FOOT_BODY_NAMES
    imu_site_names: Tuple[str, str] = G1_IMU_SITE_NAMES

    # MuJoCo indices (filled by ``from_mujoco_model``) ---------------------
    joint_qpos_index: Dict[str, int] = field(default_factory=dict)
    joint_dof_index: Dict[str, int] = field(default_factory=dict)
    joint_range: Dict[str, np.ndarray] = field(default_factory=dict)
    site_id: Dict[str, int] = field(default_factory=dict)
    body_id: Dict[str, int] = field(default_factory=dict)
    actuator_id: Dict[str, int] = field(default_factory=dict)

    # Actuator parameters --------------------------------------------------
    actuator_kp: Dict[str, float] = field(default_factory=dict)
    actuator_ctrlrange: Dict[str, np.ndarray] = field(default_factory=dict)
    actuator_forcerange: Dict[str, np.ndarray] = field(default_factory=dict)
    torque_limit: Dict[str, float] = field(default_factory=dict)

    # Reference postures ---------------------------------------------------
    stand_ctrl: np.ndarray = field(
        default_factory=lambda: np.zeros(len(G1_ACTUATED_JOINTS), dtype=np.float64)
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_mujoco_model(
        cls,
        model: object,
        *,
        model_xml_path: str,
        actuated_joints: Optional[Sequence[str]] = None,
    ) -> "G1RobotSpec":
        """Build a spec by introspecting a loaded ``mujoco.MjModel``.

        Args:
            model: A ``mujoco.MjModel`` instance.
            model_xml_path: Path the model was loaded from (recorded for provenance).
            actuated_joints: Optional override of the actuator name order. Must
                be a subset of the joints declared in the MJCF.

        Raises:
            KeyError: if any expected joint is missing from the MJCF.
        """
        import mujoco  # local import keeps the spec module importable without mujoco

        joint_names = tuple(actuated_joints) if actuated_joints is not None else G1_ACTUATED_JOINTS

        joint_qpos_index: Dict[str, int] = {}
        joint_dof_index: Dict[str, int] = {}
        joint_range: Dict[str, np.ndarray] = {}
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"G1 joint not found in MJCF: {name}")
            joint_qpos_index[name] = int(model.jnt_qposadr[jid])
            joint_dof_index[name] = int(model.jnt_dofadr[jid])
            joint_range[name] = np.asarray(model.jnt_range[jid], dtype=np.float64).copy()

        site_id: Dict[str, int] = {}
        for name in (*G1_FOOT_SITE_NAMES, *G1_IMU_SITE_NAMES):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                site_id[name] = int(sid)

        body_id: Dict[str, int] = {}
        for name in (*G1_BODY_NAMES, *G1_FOOT_BODY_NAMES):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                body_id[name] = int(bid)

        actuator_id: Dict[str, int] = {}
        actuator_kp: Dict[str, float] = {}
        actuator_ctrlrange: Dict[str, np.ndarray] = {}
        actuator_forcerange: Dict[str, np.ndarray] = {}
        torque_limit: Dict[str, float] = {}
        for name in joint_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                torque_limit[name] = float("inf")
                continue
            actuator_id[name] = int(aid)
            gain = np.asarray(model.actuator_gainprm[aid], dtype=np.float64).reshape(-1)
            actuator_kp[name] = float(gain[0]) if gain.size > 0 else 0.0
            actuator_ctrlrange[name] = np.asarray(model.actuator_ctrlrange[aid], dtype=np.float64).copy()
            actuator_forcerange[name] = np.asarray(model.actuator_forcerange[aid], dtype=np.float64).copy()
            torque_limit[name] = float(np.max(np.abs(actuator_forcerange[name])))

        stand_ctrl = np.zeros(len(joint_names), dtype=np.float64)
        try:
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            if kid >= 0:
                key_ctrl = np.asarray(model.key_ctrl[kid], dtype=np.float64)
                if key_ctrl.size >= len(joint_names):
                    stand_ctrl = key_ctrl[: len(joint_names)].copy()
        except Exception:
            pass

        return cls(
            model_xml_path=str(model_xml_path),
            actuated_joints=joint_names,
            joint_qpos_index=joint_qpos_index,
            joint_dof_index=joint_dof_index,
            joint_range=joint_range,
            site_id=site_id,
            body_id=body_id,
            actuator_id=actuator_id,
            actuator_kp=actuator_kp,
            actuator_ctrlrange=actuator_ctrlrange,
            actuator_forcerange=actuator_forcerange,
            torque_limit=torque_limit,
            stand_ctrl=stand_ctrl,
        )

    # ------------------------------------------------------------------
    # Convenient views
    # ------------------------------------------------------------------

    @property
    def num_actuated(self) -> int:
        return len(self.actuated_joints)

    @property
    def actuated_qpos_indices(self) -> np.ndarray:
        """qpos indices for actuated joints (np.int32, length ``num_actuated``)."""
        return np.fromiter(
            (self.joint_qpos_index[name] for name in self.actuated_joints),
            dtype=np.int32,
            count=self.num_actuated,
        )

    @property
    def actuated_dof_indices(self) -> np.ndarray:
        """qvel/dof indices for actuated joints (np.int32, length ``num_actuated``)."""
        return np.fromiter(
            (self.joint_dof_index[name] for name in self.actuated_joints),
            dtype=np.int32,
            count=self.num_actuated,
        )

    def actuator_kp_vector(self) -> np.ndarray:
        """Per-actuator position gains, ordered like ``actuated_joints``."""
        return np.fromiter(
            (self.actuator_kp.get(name, 0.0) for name in self.actuated_joints),
            dtype=np.float64,
            count=self.num_actuated,
        )

    def torque_limit_vector(self) -> np.ndarray:
        """Per-actuator torque magnitude bounds, ordered like ``actuated_joints``."""
        return np.fromiter(
            (self.torque_limit.get(name, np.inf) for name in self.actuated_joints),
            dtype=np.float64,
            count=self.num_actuated,
        )

    # ------------------------------------------------------------------
    # qpos / joint-vector helpers
    # ------------------------------------------------------------------

    def actuated_qpos_from_full(self, qpos: Sequence[float]) -> np.ndarray:
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        idx = self.actuated_qpos_indices
        if idx.size == 0 or int(idx.max()) >= arr.size:
            raise ValueError(
                f"Full qpos shape {arr.shape} incompatible with G1 actuated indices "
                f"(max={int(idx.max()) if idx.size else -1})"
            )
        return arr[idx].copy()

    def apply_actuated_qpos(
        self,
        qpos_full: Sequence[float],
        q_actuated: Sequence[float],
    ) -> np.ndarray:
        out = np.asarray(qpos_full, dtype=np.float64).reshape(-1).copy()
        vec = np.asarray(q_actuated, dtype=np.float64).reshape(-1)
        if vec.size != self.num_actuated:
            raise ValueError(
                f"Expected {self.num_actuated} actuated joints, got {vec.size}"
            )
        out[self.actuated_qpos_indices] = vec
        return out

    def joint_dict_to_vector(
        self,
        joint_targets: Mapping[str, float],
        *,
        base: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        if base is None:
            vec = self.stand_ctrl.copy()
        else:
            vec = np.asarray(base, dtype=np.float64).reshape(-1).copy()
        if vec.size != self.num_actuated:
            raise ValueError(
                f"Expected base vector with {self.num_actuated} entries, got {vec.size}"
            )
        for i, name in enumerate(self.actuated_joints):
            if name in joint_targets:
                vec[i] = float(joint_targets[name])
        return vec

    def joint_vector_to_dict(self, joint_vector: Sequence[float]) -> Dict[str, float]:
        vec = np.asarray(joint_vector, dtype=np.float64).reshape(-1)
        if vec.size != self.num_actuated:
            raise ValueError(f"Expected {self.num_actuated} entries, got {vec.size}")
        return {name: float(vec[i]) for i, name in enumerate(self.actuated_joints)}

    def clip_to_joint_limits(
        self,
        joint_vector: Sequence[float],
        *,
        margin: float = 0.0,
    ) -> np.ndarray:
        vec = np.asarray(joint_vector, dtype=np.float64).reshape(-1).copy()
        if vec.size != self.num_actuated:
            raise ValueError(f"Expected {self.num_actuated} entries, got {vec.size}")
        for i, name in enumerate(self.actuated_joints):
            lo, hi = self.joint_range[name]
            vec[i] = float(np.clip(vec[i], lo + margin, hi - margin))
        return vec

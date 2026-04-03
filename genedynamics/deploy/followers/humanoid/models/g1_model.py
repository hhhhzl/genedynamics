"""
G1 model metadata and MuJoCo index helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


LEFT_LEG_JOINTS: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
)
RIGHT_LEG_JOINTS: Tuple[str, ...] = (
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
WAIST_JOINTS: Tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
LEFT_ARM_JOINTS: Tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS: Tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
ACTUATED_JOINTS: Tuple[str, ...] = LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS + WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
FOOT_SITE_NAMES: Tuple[str, str] = ("left_foot", "right_foot")
IMU_SITE_NAMES: Tuple[str, str] = ("imu_in_pelvis", "imu_in_torso")
BODY_NAMES: Tuple[str, str] = ("pelvis", "torso_link")


def resolve_g1_model_path(model_xml_path: Optional[str] = None) -> str:
    def _prefer_scene(path: Path) -> Path:
        if path.name in ("g1.xml", "g1_mjx.xml"):
            for scene_name in ("scene.xml", "scene_mjx.xml"):
                scene_path = path.parent / scene_name
                if scene_path.exists():
                    return scene_path
        return path

    if model_xml_path:
        path = Path(model_xml_path).expanduser().resolve()
        if path.exists():
            return str(_prefer_scene(path))
        raise FileNotFoundError(path)

    try:
        from genedynamics.robots.registry import _get_g1_path

        path = _get_g1_path()
        if path:
            return str(_prefer_scene(Path(path).resolve()))
    except Exception:
        pass

    root = Path(__file__).resolve().parents[5]
    candidates = (
        root / "third_party" / "mujoco_menagerie" / "unitree_g1" / "scene.xml",
        root / "third_party" / "mujoco_menagerie" / "unitree_g1" / "scene_mjx.xml",
        root / "third_party" / "mujoco_menagerie" / "unitree_g1" / "g1.xml",
        root / "third_party" / "mujoco_menagerie" / "unitree_g1" / "g1_mjx.xml",
    )
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    raise FileNotFoundError("Unable to resolve Unitree G1 MuJoCo model path")


@dataclass
class G1ModelSpec:
    model_xml_path: str
    actuated_joints: Tuple[str, ...] = ACTUATED_JOINTS
    left_leg_joints: Tuple[str, ...] = LEFT_LEG_JOINTS
    right_leg_joints: Tuple[str, ...] = RIGHT_LEG_JOINTS
    waist_joints: Tuple[str, ...] = WAIST_JOINTS
    left_arm_joints: Tuple[str, ...] = LEFT_ARM_JOINTS
    right_arm_joints: Tuple[str, ...] = RIGHT_ARM_JOINTS
    foot_site_names: Tuple[str, str] = FOOT_SITE_NAMES
    imu_site_names: Tuple[str, str] = IMU_SITE_NAMES
    joint_qpos_index: Dict[str, int] = field(default_factory=dict)
    joint_dof_index: Dict[str, int] = field(default_factory=dict)
    joint_range: Dict[str, np.ndarray] = field(default_factory=dict)
    site_id: Dict[str, int] = field(default_factory=dict)
    body_id: Dict[str, int] = field(default_factory=dict)
    stand_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(len(ACTUATED_JOINTS), dtype=np.float64))

    @classmethod
    def from_mujoco_model(cls, model: object, *, model_xml_path: str) -> "G1ModelSpec":
        import mujoco

        joint_qpos_index: Dict[str, int] = {}
        joint_dof_index: Dict[str, int] = {}
        joint_range: Dict[str, np.ndarray] = {}
        for name in ACTUATED_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"Joint not found in G1 model: {name}")
            joint_qpos_index[name] = int(model.jnt_qposadr[jid])
            joint_dof_index[name] = int(model.jnt_dofadr[jid])
            joint_range[name] = np.asarray(model.jnt_range[jid], dtype=np.float64).copy()

        site_id: Dict[str, int] = {}
        for name in FOOT_SITE_NAMES + IMU_SITE_NAMES:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                site_id[name] = int(sid)

        body_id: Dict[str, int] = {}
        for name in BODY_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                body_id[name] = int(bid)

        stand_ctrl = np.zeros(len(ACTUATED_JOINTS), dtype=np.float64)
        try:
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            if kid >= 0:
                stand_ctrl = np.asarray(model.key_ctrl[kid], dtype=np.float64).copy()
        except Exception:
            pass

        return cls(
            model_xml_path=str(model_xml_path),
            joint_qpos_index=joint_qpos_index,
            joint_dof_index=joint_dof_index,
            joint_range=joint_range,
            site_id=site_id,
            body_id=body_id,
            stand_ctrl=stand_ctrl,
        )

    @property
    def num_actuated(self) -> int:
        return len(self.actuated_joints)

    @property
    def actuated_qpos_indices(self) -> np.ndarray:
        return np.asarray([self.joint_qpos_index[n] for n in self.actuated_joints], dtype=np.int32)

    @property
    def actuated_dof_indices(self) -> np.ndarray:
        return np.asarray([self.joint_dof_index[n] for n in self.actuated_joints], dtype=np.int32)

    def actuated_qpos_from_full(self, qpos: Sequence[float]) -> np.ndarray:
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        idx = self.actuated_qpos_indices
        if idx.size == 0 or int(np.max(idx)) >= arr.size:
            raise ValueError(f"Full qpos shape {arr.shape} incompatible with G1 actuated indices")
        return arr[idx].copy()

    def apply_actuated_qpos(self, qpos_full: Sequence[float], q_actuated: Sequence[float]) -> np.ndarray:
        out = np.asarray(qpos_full, dtype=np.float64).reshape(-1).copy()
        vec = np.asarray(q_actuated, dtype=np.float64).reshape(-1)
        if vec.size != self.num_actuated:
            raise ValueError(f"Expected {self.num_actuated} actuated joints, got {vec.size}")
        out[self.actuated_qpos_indices] = vec
        return out

    def joint_dict_to_vector(
        self,
        joint_targets: Dict[str, float],
        *,
        base: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        vec = self.stand_ctrl.copy() if base is None else np.asarray(base, dtype=np.float64).reshape(-1).copy()
        if vec.size != self.num_actuated:
            raise ValueError(f"Expected base vector with {self.num_actuated} entries, got {vec.size}")
        for i, name in enumerate(self.actuated_joints):
            if name in joint_targets:
                vec[i] = float(joint_targets[name])
        return vec

    def joint_vector_to_dict(self, joint_vector: Sequence[float]) -> Dict[str, float]:
        vec = np.asarray(joint_vector, dtype=np.float64).reshape(-1)
        if vec.size != self.num_actuated:
            raise ValueError(f"Expected {self.num_actuated} entries, got {vec.size}")
        return {name: float(vec[i]) for i, name in enumerate(self.actuated_joints)}

    def clip_to_joint_limits(self, joint_vector: Sequence[float], *, margin: float = 0.0) -> np.ndarray:
        vec = np.asarray(joint_vector, dtype=np.float64).reshape(-1).copy()
        if vec.size != self.num_actuated:
            raise ValueError(f"Expected {self.num_actuated} entries, got {vec.size}")
        for i, name in enumerate(self.actuated_joints):
            lo, hi = np.asarray(self.joint_range[name], dtype=np.float64)
            vec[i] = float(np.clip(vec[i], lo + margin, hi - margin))
        return vec

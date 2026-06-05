"""Scene-frame localization wrapper (deploy glue).

Wraps ANY ``BaseLocalizationPlugin`` (Vicon / OptiTrack / SLAM / mock — the
tracker-agnostic source, in the WORLD frame) and returns the base pose/twist in
the corridor **SCENE frame**, using the ``T_world_scene`` SE(2) calibration
(``Se2Transform``). This is what makes the robot's reported pose match the plan
(authored in the scene frame): drop it in as ``UnitreeG1RobotIO(localization=
SceneFrameLocalization(vicon, T_world_scene))`` and ``_read_state`` puts the
scene-frame base straight into ``qpos[0:7]``.

Pure software; verify with a known pose (conventions §2).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.ar.scene_source import Se2Transform
from genedynamics.deploy.localization.base_plugin import BaseLocalizationPlugin


def _yaw_quat(a: float) -> np.ndarray:
    return np.array([math.cos(0.5 * a), 0.0, 0.0, math.sin(0.5 * a)], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w,x,y,z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


class SceneFrameLocalization(BaseLocalizationPlugin):
    """WORLD-frame localization → SCENE-frame pose/twist via ``T_world_scene``."""

    def __init__(self, inner: BaseLocalizationPlugin, T_world_scene: Any,
                 config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config or {})
        self.inner = inner
        self.T = Se2Transform.from_any(T_world_scene)

    def get_state(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        st = self.inner.get_state()
        if st is None:
            return None
        pose, twist = st
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        twist = np.asarray(twist, dtype=np.float64).reshape(6)

        x, y = self.T.point_to_scene(float(pose[0]), float(pose[1]))
        # scene orientation = R_z(-yaw0) ∘ world orientation (frames differ by yaw only)
        q_scene = _quat_mul(_yaw_quat(-self.T.yaw), pose[3:7])
        out_pose = np.array([x, y, float(pose[2]), *q_scene], dtype=np.float64)

        vx, vy = self.T.vel_to_scene(float(twist[0]), float(twist[1]))
        wx, wy = self.T.vel_to_scene(float(twist[3]), float(twist[4]))  # angular vel rotates too
        out_twist = np.array([vx, vy, float(twist[2]), wx, wy, float(twist[5])], dtype=np.float64)
        return out_pose, out_twist

    def get_last_update_time(self) -> Optional[float]:
        return self.inner.get_last_update_time()

    def health(self) -> str:
        return self.inner.health()

"""Single source of truth for corridor obstacle geometry.

``SceneSource`` holds **one** :class:`CorridorScene` (the same
``CorridorObstacle`` primitives the planner and the body-SDF governor use)
plus the ``T_world_scene`` SE(2) transform that places the scene frame into
the Vicon world frame. Its :meth:`to_contract` emits the JSON contract that
is shared, unchanged, by:

* the AR clients (Vision Pro / phone / tablet) — to render holographic
  obstacles in the Vicon world frame, and
* the robot side — re-planning (``replan_from_scene``) and the body-SDF
  governor (``diagnose(body_sdf_scene=...)``), which already consumes the
  ``corridor_scene_to_dict`` schema via ``BodySdfAdmissibleSet.from_scene_dict``.

There is deliberately **no second obstacle definition** anywhere: the
contract's ``obstacles`` come verbatim from
:func:`genedynamics.envs.domains.humanoid.corridor.corridor_scene_to_dict`.

Every mutation bumps :attr:`revision` so a server can tell when to push.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from genedynamics.envs.domains.humanoid.corridor import (
    CorridorObstacle,
    CorridorScene,
    corridor_scene_to_dict,
    resolve_corridor_scene_preset,
)

__all__ = ["SCHEMA_VERSION", "Se2Transform", "SceneSource"]

SCHEMA_VERSION = 1

# Field names accepted by CorridorObstacle(**o) — used to filter user-authored
# blobs so stray keys don't blow up construction.
_OBS_FIELDS = {f.name for f in dataclasses.fields(CorridorObstacle)}


@dataclass
class Se2Transform:
    """Planar (SE(2)) transform placing the scene frame in the world frame.

    ``world_xy = R(yaw) @ scene_xy + (x, y)``. Measured once on-site (e.g.
    tape the corridor start + +x direction, probe two points with a Vicon
    wand). Identity until calibrated.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "yaw": float(self.yaw)}

    @classmethod
    def from_any(cls, value: Any) -> "Se2Transform":
        if value is None:
            return cls()
        if isinstance(value, Se2Transform):
            return value
        if isinstance(value, dict):
            return cls(float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("yaw", 0.0)))
        seq = list(value)
        return cls(*(float(v) for v in (seq + [0.0, 0.0, 0.0])[:3]))

    # ------------------------------------------------------------------
    # Frame transforms (M4): Vicon WORLD frame <-> corridor SCENE frame.
    # The plan / governor / obstacles live in the SCENE frame; the robot's
    # Vicon pose is in the WORLD frame. world = R(yaw)·scene + (x, y).
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
        """Yaw (rotation about world +z) from a quaternion [w, x, y, z]."""
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def point_to_world(self, x: float, y: float) -> Tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.x + c * x - s * y, self.y + s * x + c * y)

    def point_to_scene(self, X: float, Y: float) -> Tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = X - self.x, Y - self.y
        return (c * dx + s * dy, -s * dx + c * dy)

    def yaw_to_world(self, psi: float) -> float:
        return self._wrap(psi + self.yaw)

    def yaw_to_scene(self, Psi: float) -> float:
        return self._wrap(Psi - self.yaw)

    def vel_to_scene(self, VX: float, VY: float) -> Tuple[float, float]:
        """Rotate a world-frame planar velocity into the scene frame (no translation)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (c * VX + s * VY, -s * VX + c * VY)

    def vel_to_world(self, vx: float, vy: float) -> Tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (c * vx - s * vy, s * vx + c * vy)

    def pose_to_scene(self, X: float, Y: float, Psi: float) -> Tuple[float, float, float]:
        x, y = self.point_to_scene(X, Y)
        return (x, y, self.yaw_to_scene(Psi))

    def pose_to_world(self, x: float, y: float, psi: float) -> Tuple[float, float, float]:
        X, Y = self.point_to_world(x, y)
        return (X, Y, self.yaw_to_world(psi))

    def scene_base_from_world(
        self, qpos: Sequence[float], qvel: Optional[Sequence[float]] = None
    ) -> Dict[str, float]:
        """Convert a Vicon WORLD-frame base estimate into the SCENE-frame planar
        base state the follower / governor consume.

        Args:
            qpos: ``[x, y, z, qw, qx, qy, qz]`` (the localization-plugin contract).
            qvel: optional ``[vx, vy, vz, wx, wy, wz]`` (world frame).

        Returns dict ``{x, y, psi}`` (+ ``vx, vy, omega`` when qvel given), where
        ``vx, vy`` are the SCENE-frame planar velocity (the world linear velocity
        rotated by ``-yaw``) and ``omega`` is the yaw rate (frame-invariant).
        """
        q = list(qpos)
        Psi = self.yaw_from_quat(q[3], q[4], q[5], q[6])
        x, y, psi = self.pose_to_scene(q[0], q[1], Psi)
        out = {"x": x, "y": y, "psi": psi}
        if qvel is not None:
            v = list(qvel)
            vx, vy = self.vel_to_scene(v[0], v[1])
            out.update({"vx": vx, "vy": vy, "omega": float(v[5])})
        return out


@dataclass
class SceneSource:
    """Mutable holder of the authoritative corridor scene + world transform."""

    scene: CorridorScene
    scene_preset: Optional[str] = None
    T_world_scene: Se2Transform = field(default_factory=Se2Transform)
    robot_pose_world: Optional[Dict[str, float]] = None  # live Vicon base pose (world frame)
    plan_xy_scene: Optional[List[List[float]]] = None     # planned path (scene frame), for AR ghost line
    revision: int = 0

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_preset(cls, preset: str, *, T_world_scene: Any = None) -> "SceneSource":
        return cls(
            scene=resolve_corridor_scene_preset(preset),
            scene_preset=str(preset),
            T_world_scene=Se2Transform.from_any(T_world_scene),
        )

    @classmethod
    def from_scene(cls, scene: CorridorScene, *, scene_preset: Optional[str] = None,
                   T_world_scene: Any = None) -> "SceneSource":
        return cls(scene=scene, scene_preset=scene_preset,
                   T_world_scene=Se2Transform.from_any(T_world_scene))

    @classmethod
    def from_file(cls, path: str | Path, *, T_world_scene: Any = None) -> "SceneSource":
        scene, preset, t = cls._parse_file(path)
        return cls(scene=scene, scene_preset=preset,
                   T_world_scene=Se2Transform.from_any(T_world_scene if T_world_scene is not None else t))

    @classmethod
    def from_contract(cls, contract: Dict[str, Any], *, T_world_scene: Any = None) -> "SceneSource":
        """Rebuild a SceneSource from a wire contract (the inverse of
        :meth:`to_contract`). Used by the robot side (replanning / governor)."""
        merged = {**(contract.get("corridor") or {}),
                  "obstacles": contract.get("obstacles") or [],
                  "scene_preset": contract.get("scene_preset")}
        scene, preset = cls._scene_from_blob(merged)
        t = T_world_scene if T_world_scene is not None else contract.get("T_world_scene")
        src = cls(scene=scene, scene_preset=preset, T_world_scene=Se2Transform.from_any(t))
        src.robot_pose_world = contract.get("robot_pose_world")
        src.plan_xy_scene = contract.get("plan_xy_scene")
        return src

    # ------------------------------------------------------------------
    # Mutation (each bumps revision)
    # ------------------------------------------------------------------
    def set_scene(self, scene: CorridorScene, *, scene_preset: Optional[str] = None) -> None:
        self.scene = scene
        self.scene_preset = scene_preset
        self.revision += 1

    def set_preset(self, preset: str) -> None:
        self.set_scene(resolve_corridor_scene_preset(preset), scene_preset=str(preset))

    def set_obstacles(self, obstacles: Sequence[CorridorObstacle | Dict[str, Any]]) -> None:
        self.scene.obstacles = [self._coerce_obstacle(o) for o in obstacles]
        self.scene_preset = None  # no longer a named preset once edited
        self.revision += 1

    def set_robot_pose_world(self, pose: Optional[Dict[str, float]]) -> None:
        self.robot_pose_world = pose
        self.revision += 1

    def set_plan_xy_scene(self, plan: Optional[Sequence[Sequence[float]]]) -> None:
        self.plan_xy_scene = [[float(p[0]), float(p[1])] for p in plan] if plan is not None else None
        self.revision += 1

    def set_T_world_scene(self, value: Any) -> None:
        self.T_world_scene = Se2Transform.from_any(value)
        self.revision += 1

    def load_file(self, path: str | Path) -> None:
        """Reload scene from a JSON spec (used by the server's file-watch)."""
        scene, preset, t = self._parse_file(path)
        self.scene = scene
        self.scene_preset = preset
        if t is not None:
            self.T_world_scene = Se2Transform.from_any(t)
        self.revision += 1

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------
    def to_contract(self) -> Dict[str, Any]:
        """Serialize to the shared JSON contract (pure — no wall-clock)."""
        blob = corridor_scene_to_dict(self.scene, scene_preset=self.scene_preset)
        w = float(blob["corridor_width"])
        return {
            "schema_version": SCHEMA_VERSION,
            "frame": "vicon_world",
            "units": "m_rad",
            "revision": int(self.revision),
            "scene_preset": self.scene_preset,
            "T_world_scene": self.T_world_scene.to_dict(),
            "corridor": {
                "corridor_width": w,
                "corridor_length": float(blob["corridor_length"]),
                "start_pos": [float(v) for v in blob["start_pos"]],
                "goal_pos": [float(v) for v in blob["goal_pos"]],
                "wall_y_min": -0.5 * w,
                "wall_y_max": 0.5 * w,
            },
            "obstacles": blob["obstacles"],
            "robot_pose_world": self.robot_pose_world,
            "plan_xy_scene": self.plan_xy_scene,
        }

    # ------------------------------------------------------------------
    # Governor bridge (M3): scene → body-SDF scene the runtime governor eats
    # ------------------------------------------------------------------
    def to_body_sdf_scene(self) -> Dict[str, Any]:
        """Return the flat scene dict the reference governor consumes via
        ``BodySdfAdmissibleSet.from_scene_dict`` (== ``corridor_scene_to_dict``).

        This is the runtime bridge that makes the robot "perceive" the AR
        obstacles: the same single obstacle source served to the AR clients is
        handed to ``diagnose(body_sdf_scene=...)`` / the governor, which then
        tightens the command envelope to keep the body clear of them. The
        governor works in the SCENE frame; convert the robot's Vicon world pose
        with ``inv(T_world_scene)`` before feeding it (handled at deploy time).
        """
        return corridor_scene_to_dict(self.scene, scene_preset=self.scene_preset)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_obstacle(o: CorridorObstacle | Dict[str, Any]) -> CorridorObstacle:
        if isinstance(o, CorridorObstacle):
            return o
        return CorridorObstacle(**{k: v for k, v in o.items() if k in _OBS_FIELDS})

    @classmethod
    def _scene_from_blob(cls, blob: Dict[str, Any]) -> Tuple[CorridorScene, Optional[str]]:
        preset = blob.get("scene_preset") or blob.get("preset")
        if preset:
            return resolve_corridor_scene_preset(str(preset)), str(preset)
        kwargs: Dict[str, Any] = {}
        for key in ("corridor_width", "corridor_length"):
            if blob.get(key) is not None:
                kwargs[key] = float(blob[key])
        if blob.get("start_pos") is not None:
            kwargs["start_pos"] = tuple(float(v) for v in blob["start_pos"])
        if blob.get("goal_pos") is not None:
            kwargs["goal_pos"] = tuple(float(v) for v in blob["goal_pos"])
        obstacles = [cls._coerce_obstacle(o) for o in (blob.get("obstacles") or [])]
        return CorridorScene(obstacles=obstacles, **kwargs), None

    @classmethod
    def _parse_file(cls, path: str | Path) -> Tuple[CorridorScene, Optional[str], Optional[Any]]:
        blob = json.loads(Path(path).read_text())
        # Accept either a bare scene spec or a full contract (with "corridor").
        if "corridor" in blob and "obstacles" in blob:
            merged = {**blob.get("corridor", {}), "obstacles": blob["obstacles"],
                      "scene_preset": blob.get("scene_preset")}
            scene, preset = cls._scene_from_blob(merged)
        else:
            scene, preset = cls._scene_from_blob(blob)
        return scene, preset, blob.get("T_world_scene")

"""
Scene intermediate representation (IR) for trajectory visualizations.

This module decouples *what* to draw (corridor walls, obstacles, stones,
robot poses) from *how* to draw it (matplotlib top-down, isometric,
pyrender mesh, ...).

Two env types currently feed into this IR:
  * humanoid corridor 2D  -> kind="corridor"
  * stepping stones plan  -> kind="stepping"

Renderer backends consume only this IR plus a list of `RobotPoseIR`
frames; they should not import env classes directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------
@dataclass
class BoxIR:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float = 0.0
    z_max: float = 2.0
    name: str = ""
    # Source shape of the underlying obstacle. "box" for true boxes;
    # "sphere"/"qc" if this BoxIR is just an AABB proxy added so legacy
    # backends can place a label. Iso/3D backends should skip drawing
    # the cuboid for non-"box" shapes.
    shape: str = "box"


@dataclass
class SphereIR:
    cx: float
    cy: float
    radius: float
    z_min: float = 0.0
    z_max: float = 2.0
    name: str = ""


@dataclass
class QuarterCircleIR:
    cx: float
    cy: float
    radius: float
    clip_sign: float
    z_min: float = 0.0
    z_max: float = 2.0
    name: str = ""


@dataclass
class DiskIR:
    cx: float
    cy: float
    radius: float
    z: float = 0.0


# ---------------------------------------------------------------------------
# Scene IR
# ---------------------------------------------------------------------------
@dataclass
class SceneIR:
    kind: str  # "corridor" | "stepping"
    bounds: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
    walls: List[BoxIR] = field(default_factory=list)
    boxes: List[BoxIR] = field(default_factory=list)
    spheres: List[SphereIR] = field(default_factory=list)
    quarter_circles: List[QuarterCircleIR] = field(default_factory=list)
    stones: List[DiskIR] = field(default_factory=list)
    platforms: List[BoxIR] = field(default_factory=list)
    river: Optional[Tuple[float, float, float, float]] = None
    start: Optional[Tuple[float, float]] = None
    goal: Optional[Tuple[float, float]] = None


# ---------------------------------------------------------------------------
# Robot pose IR
# ---------------------------------------------------------------------------
@dataclass
class RobotPoseIR:
    """One robot pose frame. Fields not relevant to a given env stay default."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0          # heading psi (corridor)
    torso_yaw: float = 0.0    # psi_torso relative to yaw
    body_height: float = 0.78
    arm_tuck_L: float = 0.0
    arm_tuck_R: float = 0.0
    arm_posture_L: float = 0.0   # p_L: shoulder-pitch posture in [-1, 1]
    arm_posture_R: float = 0.0   # p_R
    feet: Optional[Dict[str, Tuple[float, float]]] = None
    swing_legs: Tuple[str, ...] = ()
    mode: int = 0
    in_collision: bool = False


# ---------------------------------------------------------------------------
# Adapters: corridor
# ---------------------------------------------------------------------------
def corridor_scene_to_ir(scene: Any) -> SceneIR:
    """Convert a CorridorScene into a SceneIR (kind='corridor')."""
    hw = scene.corridor_width / 2.0
    length = scene.corridor_length

    boxes: List[BoxIR] = []
    spheres: List[SphereIR] = []
    qcs: List[QuarterCircleIR] = []
    for obs in scene.obstacles:
        shape = getattr(obs, "shape", "box")
        if shape == "sphere":
            spheres.append(SphereIR(
                cx=float(obs.cx), cy=float(obs.cy), radius=float(obs.radius),
                z_min=float(obs.z_min), z_max=float(obs.z_max), name=str(obs.name),
            ))
        elif shape == "qc":
            qcs.append(QuarterCircleIR(
                cx=float(obs.cx), cy=float(obs.cy), radius=float(obs.radius),
                clip_sign=float(getattr(obs, "qc_clip_sign", 1.0)),
                z_min=float(obs.z_min), z_max=float(obs.z_max), name=str(obs.name),
            ))
        # Always include the AABB form too so legacy box-style draw can find
        # x_min/x_max etc. For pure boxes this is the only entry.
        boxes.append(BoxIR(
            x_min=float(obs.x_min), x_max=float(obs.x_max),
            y_min=float(obs.y_min), y_max=float(obs.y_max),
            z_min=float(obs.z_min), z_max=float(obs.z_max),
            name=str(obs.name),
            shape=shape,
        ))

    return SceneIR(
        kind="corridor",
        bounds=(0.0, length, -hw, hw),
        boxes=boxes,
        spheres=spheres,
        quarter_circles=qcs,
        start=(float(scene.start_pos[0]), float(scene.start_pos[1])),
        goal=(float(scene.goal_pos[0]), float(scene.goal_pos[1])),
    )


def corridor_states_to_poses(states: np.ndarray) -> List[RobotPoseIR]:
    """Convert a (T, >=7) state array into a list of RobotPoseIR.

    Layout assumed: [x, y, psi, h, psi_torso, a_L, a_R, p_L, p_R, ...].
    Posture p_L/p_R (indices 7, 8) are optional — used by the articulated
    arm drawing to mirror the follower's shoulder-pitch behaviour.
    """
    states = np.asarray(states, dtype=np.float32)
    poses: List[RobotPoseIR] = []
    for s in states:
        n = s.shape[0]
        poses.append(RobotPoseIR(
            x=float(s[0]), y=float(s[1]),
            yaw=float(s[2]),
            body_height=float(s[3]),
            torso_yaw=float(s[4]),
            arm_tuck_L=float(s[5]),
            arm_tuck_R=float(s[6]),
            arm_posture_L=float(s[7]) if n > 7 else 0.0,
            arm_posture_R=float(s[8]) if n > 8 else 0.0,
        ))
    return poses


# ---------------------------------------------------------------------------
# Adapters: stepping stones
# ---------------------------------------------------------------------------
def stepping_scene_to_ir(scene: Any) -> SceneIR:
    """Convert a SteppingStonesScene into a SceneIR (kind='stepping')."""
    xmin, xmax = float(scene.map_x[0]), float(scene.map_x[1])
    ymin, ymax = float(scene.map_y[0]), float(scene.map_y[1])

    river = None
    if bool(getattr(scene, "has_river", True)):
        rx0, rx1 = scene.river_x
        river = (float(rx0), float(rx1), ymin, ymax)

    platforms: List[BoxIR] = []
    raw_plat = np.asarray(
        getattr(scene, "support_platforms", np.zeros((0, 4))),
        dtype=np.float32,
    ).reshape(-1, 4)
    for row in raw_plat:
        platforms.append(BoxIR(
            x_min=float(row[0]), x_max=float(row[1]),
            y_min=float(row[2]), y_max=float(row[3]),
        ))

    stones: List[DiskIR] = []
    centers = np.asarray(scene.stones_centers, dtype=np.float32).reshape(-1, 2)
    radii = np.asarray(scene.stones_radii, dtype=np.float32).reshape(-1)
    for c, r in zip(centers, radii):
        stones.append(DiskIR(cx=float(c[0]), cy=float(c[1]), radius=float(r)))

    start = None
    goal = None
    if hasattr(scene, "start_pos"):
        start = (float(scene.start_pos[0]), float(scene.start_pos[1]))
    if hasattr(scene, "goal_pos"):
        goal = (float(scene.goal_pos[0]), float(scene.goal_pos[1]))

    return SceneIR(
        kind="stepping",
        bounds=(xmin, xmax, ymin, ymax),
        platforms=platforms,
        stones=stones,
        river=river,
        start=start,
        goal=goal,
    )


def stepping_decoded_to_poses(
    body: np.ndarray,
    feet: Dict[str, np.ndarray],
    mode: np.ndarray,
) -> List[RobotPoseIR]:
    """Convert decoded stepping plan arrays into RobotPoseIR frames."""
    body = np.asarray(body, dtype=np.float32)
    mode = np.asarray(mode).reshape(-1)
    poses: List[RobotPoseIR] = []
    T = int(body.shape[0])
    for t in range(T):
        m = int(mode[t]) if t < mode.shape[0] else 0
        swing = ("FR", "RL") if (m % 2 == 0) else ("FL", "RR")
        feet_t = {leg: (float(feet[leg][t, 0]), float(feet[leg][t, 1]))
                  for leg in ("FL", "FR", "RL", "RR") if leg in feet}
        poses.append(RobotPoseIR(
            x=float(body[t, 0]), y=float(body[t, 1]),
            feet=feet_t, swing_legs=swing, mode=m,
        ))
    return poses

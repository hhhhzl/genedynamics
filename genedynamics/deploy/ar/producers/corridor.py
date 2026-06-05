"""Corridor producer: ``SceneSource`` (corridor scene) → generic ``Entity`` list.

Makes the existing corridor scene *one producer* of the generic ``WorldState``.
Obstacles/walls/goals are baked from the scene frame into the world frame via
``T_world_scene`` (so the snapshot is self-contained in `frame="world"`); the
quarter-circle taper is approximated as a cylinder (same as the renderers).
"""

from __future__ import annotations

from typing import List

from genedynamics.deploy.ar.scene_source import SceneSource
from genedynamics.deploy.ar.world_state import Entity, Geometry, Pose, WorldState, yaw_quat

_WALL_RGBA = 0x8899AACC
_OBS_RGBA = 0xCC5544DD
_GOAL_RGBA = 0x33CC66DD


def corridor_entities(src: SceneSource, *, prefix: str = "") -> List[Entity]:
    """Convert *src* into world-frame entities (walls, obstacles, start/goal)."""
    T = src.T_world_scene
    scene = src.scene
    yaw_w = T.yaw_to_world(0.0)            # scene axes' orientation in world
    q_axis = yaw_quat(yaw_w)              # for scene-axis-aligned boxes
    ents: List[Entity] = []

    def world_pose(cx: float, cy: float, cz: float, q=(1.0, 0.0, 0.0, 0.0)) -> Pose:
        X, Y = T.point_to_world(cx, cy)
        return Pose(p=(X, Y, cz), q=q)

    # Corridor walls: two long thin boxes at y = ±width/2, full length, ~2 m tall.
    w = float(scene.corridor_width)
    L = float(scene.corridor_length)
    for name, y in ((f"{prefix}wall_lo", -0.5 * w), (f"{prefix}wall_hi", 0.5 * w)):
        ents.append(Entity(
            id=name, type="wall",
            pose=world_pose(0.5 * L, y, 1.0, q_axis),
            geom=Geometry(kind="box", half_extents=(0.5 * L, 0.01, 1.0)),
            color_rgba=_WALL_RGBA,
        ))

    # Obstacles.
    for o in scene.obstacles:
        oid = f"{prefix}{o.name}" if o.name else f"{prefix}obs_{len(ents)}"
        zc = 0.5 * (float(o.z_min) + float(o.z_max))
        zh = max(0.02, float(o.z_max) - float(o.z_min))
        shape = getattr(o, "shape", "box")
        if shape == "sphere":
            ents.append(Entity(
                id=oid, type="obstacle",
                pose=world_pose(float(o.cx), float(o.cy), zc),
                geom=Geometry(kind="sphere", radius=float(o.radius)),
                color_rgba=_OBS_RGBA,
            ))
        elif shape == "qc":  # quarter-circle taper → cylinder approximation
            ents.append(Entity(
                id=oid, type="obstacle",
                pose=world_pose(float(o.cx), float(o.cy), zc),
                geom=Geometry(kind="cylinder", radius=float(o.radius), height=zh),
                color_rgba=_OBS_RGBA,
            ))
        else:  # box
            cx = 0.5 * (float(o.x_min) + float(o.x_max))
            cy = 0.5 * (float(o.y_min) + float(o.y_max))
            ents.append(Entity(
                id=oid, type="obstacle",
                pose=world_pose(cx, cy, zc, q_axis),
                geom=Geometry(kind="box", half_extents=(
                    max(0.01, 0.5 * (float(o.x_max) - float(o.x_min))),
                    max(0.01, 0.5 * (float(o.y_max) - float(o.y_min))),
                    0.5 * zh)),
                color_rgba=_OBS_RGBA,
            ))

    # Start / goal markers.
    for name, (gx, gy) in ((f"{prefix}start", tuple(scene.start_pos)),
                           (f"{prefix}goal", tuple(scene.goal_pos))):
        ents.append(Entity(
            id=name, type="goal",
            pose=world_pose(float(gx), float(gy), 0.05),
            geom=Geometry(kind="sphere", radius=0.08),
            color_rgba=_GOAL_RGBA,
        ))

    return ents


def populate_corridor(world: WorldState, src: SceneSource, *, prefix: str = "") -> None:
    """Upsert the corridor's entities into *world* (one of possibly many producers)."""
    for ent in corridor_entities(src, prefix=prefix):
        world.upsert(ent)

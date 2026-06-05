"""Tracker-pose producer: robot base + occluder proxy → ``WorldState`` entities.

Tracker-agnostic (the second producer, alongside ``corridor``). It consumes a
``BaseLocalizationPlugin`` (`get_state() -> (qpos[7] world, qvel[6])`, any of
vicon / optitrack / slam / vio / fiducial / mock) — or a raw `qpos` — and emits,
in `frame="world"` (the tracker already provides world poses):

  * a ``robot`` entity (the base; a cylinder proxy or a URDF asset), and
  * an ``occluder`` entity at the same pose: a slightly inflated, *invisible*
    holdout (``color_rgba`` alpha = 0) so each renderer can draw it depth-only —
    making the **real** robot correctly occlude virtual obstacles behind it
    (ARCHITECTURE §4 occlusion; precise because the pose comes from the tracker).

This is how multi-robot / multi-arm scenes compose: one ``populate_tracker`` per
tracked body (distinct ``robot_id``) writes into the same ``WorldState``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional, Sequence

from genedynamics.deploy.ar.world_state import Entity, Geometry, Pose, WorldState

# G1-ish standing proxy (radius ~ shoulder half-width, height ~ stature).
DEFAULT_ROBOT_GEOM = Geometry(kind="cylinder", radius=0.18, height=1.3)
_ROBOT_RGBA = 0x4488CCBB    # blue, semi
_OCCLUDER_RGBA = 0x00000000  # alpha 0 → renderer uses a depth-only holdout material


def _inflate(geom: Geometry, pad: float) -> Geometry:
    """Return a slightly larger copy of *geom* (conservative occluder)."""
    return replace(
        geom,
        radius=geom.radius + pad if geom.radius else geom.radius,
        height=geom.height + 2.0 * pad if geom.height else geom.height,
        half_extents=tuple(float(e) + pad for e in geom.half_extents) if any(geom.half_extents) else geom.half_extents,
    )


def _qpos(source: Any) -> Optional[Sequence[float]]:
    """Accept a BaseLocalizationPlugin (call get_state) or a raw 7-vec qpos."""
    if hasattr(source, "get_state"):
        st = source.get_state()
        if st is None:
            return None
        qpos, _qvel = st
        return list(qpos)
    return list(source)


def robot_entities(
    qpos: Sequence[float],
    *,
    robot_id: str = "g1",
    geom: Optional[Geometry] = None,
    urdf: Optional[str] = None,
    occluder: bool = True,
    occluder_pad: float = 0.05,
) -> List[Entity]:
    """Build the robot-base (+ optional occluder) entities from a world-frame qpos.

    Args:
        qpos: ``[x, y, z, qw, qx, qy, qz]`` in the world frame.
        geom: robot proxy geometry; defaults to a G1-ish cylinder (or a URDF
            asset when ``urdf`` is given).
        occluder: also emit the invisible holdout proxy (default True).
    """
    q = list(qpos)
    pose = Pose(p=(float(q[0]), float(q[1]), float(q[2])),
                q=(float(q[3]), float(q[4]), float(q[5]), float(q[6])))
    base_geom = geom or (Geometry(kind="urdf", asset_uri=urdf) if urdf else DEFAULT_ROBOT_GEOM)
    ents = [Entity(id=f"{robot_id}/base", type="robot", frame="world",
                   pose=pose, geom=base_geom, color_rgba=_ROBOT_RGBA)]
    if occluder:
        ents.append(Entity(id=f"{robot_id}/occluder", type="occluder", frame="world",
                           pose=pose, geom=_inflate(base_geom, occluder_pad),
                           color_rgba=_OCCLUDER_RGBA,
                           meta='{"render":"depth_only"}'))
    return ents


def populate_tracker(world: WorldState, source: Any, *, robot_id: str = "g1",
                     **kwargs) -> List[Entity]:
    """Upsert one tracked body's robot+occluder entities into *world*.

    *source* is a localization plugin or a raw qpos. Returns the upserted
    entities, or ``[]`` if the plugin had no update yet.
    """
    qpos = _qpos(source)
    if qpos is None:
        return []
    return [world.upsert(e) for e in robot_entities(qpos, robot_id=robot_id, **kwargs)]

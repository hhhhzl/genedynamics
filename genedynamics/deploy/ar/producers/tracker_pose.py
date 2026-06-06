"""Tracker-pose producer: robot base + occluder proxy + viewer headset.

Tracker-agnostic (the second producer, alongside ``corridor``). It consumes a
``BaseLocalizationPlugin`` (`get_state() -> (qpos[7] world, qvel[6])`, any of
vicon / optitrack / slam / vio / fiducial / mock) — or a raw `qpos` — and emits,
in `frame="world"` (the tracker already provides world poses):

  * a ``robot`` entity (the base; a cylinder proxy or a URDF asset),
  * an ``occluder`` entity at the same pose: a slightly inflated, *invisible*
    holdout (``color_rgba`` alpha = 0) so each renderer can draw it depth-only —
    making the **real** robot correctly occlude virtual obstacles behind it
    (ARCHITECTURE §4 occlusion; precise because the pose comes from the tracker), and
  * a ``viewer`` entity (:func:`populate_headset`) for an AR device tracked in the
    same world frame (e.g. a Vicon marker cluster on a Quest). Its pose drives that
    device's registration — ``OpenXRAnchorProvider`` reads it off the existing
    ``WorldSnapshot`` stream and solves ``anchor = headset_unity ∘ L⁻¹`` (ARCHITECTURE
    §4a). The wearer's own client skips drawing it (it IS the camera); every *other*
    client shows it as a collaborator marker. The marker→head mount offset is applied
    HERE/upstream (just like the robot base), so the Unity side stays calibration-free.

This is how multi-robot / multi-arm / multi-viewer scenes compose: one
``populate_tracker`` (or ``populate_headset``) per tracked body — distinct
``robot_id`` / ``headset_id`` — writes into the same ``WorldState``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional, Sequence

from genedynamics.deploy.ar.world_state import Entity, Geometry, Pose, WorldState

# G1-ish standing proxy (radius ~ shoulder half-width, height ~ stature).
DEFAULT_ROBOT_GEOM = Geometry(kind="cylinder", radius=0.18, height=1.3)
_ROBOT_RGBA = 0x4488CCBB    # blue, semi
_OCCLUDER_RGBA = 0x00000000  # alpha 0 → renderer uses a depth-only holdout material

# Headset proxy (~Quest form factor) so OTHER devices see a collaborator's head;
# the wearer's own client skips it. World axes: x=fwd (depth), y=left (width), z=up.
DEFAULT_HEADSET_GEOM = Geometry(kind="box", half_extents=(0.06, 0.09, 0.05))
_HEADSET_RGBA = 0xDDDD22AA   # yellow, semi


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


def _pose_from_qpos(qpos: Sequence[float]) -> Pose:
    """World-frame ``Pose`` from ``[x, y, z, qw, qx, qy, qz]``."""
    q = list(qpos)
    return Pose(p=(float(q[0]), float(q[1]), float(q[2])),
                q=(float(q[3]), float(q[4]), float(q[5]), float(q[6])))


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
    pose = _pose_from_qpos(qpos)
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


def headset_entities(
    qpos: Sequence[float],
    *,
    headset_id: str = "headset",
    geom: Optional[Geometry] = None,
) -> List[Entity]:
    """Build the ``viewer`` entity for one AR headset from a world-frame qpos.

    The headset is just another tracked body (mount offset applied upstream, like
    the robot base). Id is ``"{headset_id}/base"`` — mirror of the robot — and the
    wearer's ``OpenXRAnchorProvider`` (`WorldRenderer.LocalHeadsetId`) matches on it.

    Args:
        qpos: ``[x, y, z, qw, qx, qy, qz]`` in the world frame (the Quest's head
            frame, i.e. the marker pose already shifted by the mount offset).
        headset_id: stable per-device id.
        geom: viewer proxy geometry; defaults to a Quest-ish box.
    """
    return [Entity(id=f"{headset_id}/base", type="viewer", frame="world",
                   pose=_pose_from_qpos(qpos), geom=geom or DEFAULT_HEADSET_GEOM,
                   color_rgba=_HEADSET_RGBA)]


def populate_headset(world: WorldState, source: Any, *, headset_id: str = "headset",
                     **kwargs) -> List[Entity]:
    """Upsert one headset's ``viewer`` entity into *world* (alongside robots).

    *source* is a localization plugin or a raw world-frame qpos. Returns the
    upserted entities, or ``[]`` if the plugin had no update yet.
    """
    qpos = _qpos(source)
    if qpos is None:
        return []
    return [world.upsert(e) for e in headset_entities(qpos, headset_id=headset_id, **kwargs)]

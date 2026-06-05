"""Generic world model for the multi-robot AR digital twin (P1 compat core).

``WorldState`` is the engine-neutral generalization of ``SceneSource``: a set of
**entities** (`{id, type, frame, pose, geom, color, rev, meta}`) that obstacles,
walls, robot bases, arm links, goals, planned paths and occluder proxies all map
onto. Producers (corridor scene, tracker poses, ROS2 topics) write into one
``WorldState``; encoders serialize it (JSON now, FlatBuffers in P2) as a
``WorldSnapshot`` — either a full **keyframe** or a **delta**.

Mirrors `schema/world.fbs`; see `schema/conventions.md` for the normative rules.
Pure Python — no NATS, no rendering, no hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 2

# Entity / geometry kind vocabularies (mirror the .fbs enums).
ENTITY_TYPES = ("obstacle", "wall", "robot", "arm_link", "goal", "path", "occluder")
GEOM_KINDS = ("box", "sphere", "cylinder", "capsule", "usd", "urdf", "path")


@dataclass
class Pose:
    p: Tuple[float, float, float] = (0.0, 0.0, 0.0)        # position in Entity.frame (m)
    q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # unit quat (w, x, y, z)

    def to_dict(self) -> dict:
        return {"p": [float(v) for v in self.p], "q": [float(v) for v in self.q]}


@dataclass
class Geometry:
    kind: str = "box"
    half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    height: float = 0.0
    asset_uri: str = ""
    joints: List[float] = field(default_factory=list)
    polyline: List[Tuple[float, float, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "half_extents": [float(v) for v in self.half_extents],
            "radius": float(self.radius),
            "height": float(self.height),
            "asset_uri": self.asset_uri,
            "joints": [float(v) for v in self.joints],
            "polyline": [[float(c) for c in pt] for pt in self.polyline],
        }


@dataclass
class Entity:
    id: str
    type: str = "obstacle"
    frame: str = "world"
    pose: Pose = field(default_factory=Pose)
    geom: Optional[Geometry] = None
    color_rgba: int = 0xFFFFFFFF
    rev: int = 0
    meta: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "frame": self.frame,
            "pose": self.pose.to_dict(),
            "geom": self.geom.to_dict() if self.geom is not None else None,
            "color_rgba": int(self.color_rgba),
            "rev": int(self.rev),
            "meta": self.meta,
        }


@dataclass
class WorldState:
    """Mutable set of entities with monotonic revisions (keyframe + delta)."""

    site: str = "default"
    frame: str = "world"
    units: str = "m_rad"
    tracker: str = ""

    _entities: Dict[str, Entity] = field(default_factory=dict)
    _removed: Dict[str, int] = field(default_factory=dict)  # id -> rev when removed
    _rev: int = 0  # global monotonic revision

    # ------------------------------------------------------------------
    # Mutation (producers call these)
    # ------------------------------------------------------------------
    @property
    def rev(self) -> int:
        return self._rev

    def upsert(self, entity: Entity) -> Entity:
        """Insert or update an entity; stamps it with the new global rev."""
        if entity.type not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type {entity.type!r}")
        if entity.geom is not None and entity.geom.kind not in GEOM_KINDS:
            raise ValueError(f"unknown geom kind {entity.geom.kind!r}")
        self._rev += 1
        entity.rev = self._rev
        self._entities[entity.id] = entity
        self._removed.pop(entity.id, None)
        return entity

    def remove(self, entity_id: str) -> bool:
        if entity_id not in self._entities:
            return False
        self._rev += 1
        del self._entities[entity_id]
        self._removed[entity_id] = self._rev
        return True

    def get(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def __len__(self) -> int:
        return len(self._entities)

    # ------------------------------------------------------------------
    # Snapshots (encoders consume these dicts)
    # ------------------------------------------------------------------
    def snapshot(self, *, stamp_ns: int = 0) -> dict:
        """Full self-contained keyframe (client replaces its whole world)."""
        return self._message(
            entities=list(self._entities.values()),
            removed_ids=[],
            is_keyframe=True,
            stamp_ns=stamp_ns,
        )

    def delta(self, since_rev: int, *, stamp_ns: int = 0) -> Tuple[dict, int]:
        """Entities changed since *since_rev* + ids removed since then.

        Returns ``(message, cursor_rev)``; pass ``cursor_rev`` as the next
        ``since_rev``. If the client has no keyframe, send :meth:`snapshot` first.
        """
        changed = [e for e in self._entities.values() if e.rev > since_rev]
        removed = [eid for eid, r in self._removed.items() if r > since_rev]
        return (
            self._message(entities=changed, removed_ids=removed,
                          is_keyframe=False, stamp_ns=stamp_ns),
            self._rev,
        )

    # ------------------------------------------------------------------
    def _message(self, *, entities: List[Entity], removed_ids: List[str],
                 is_keyframe: bool, stamp_ns: int) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "site": self.site,
            "stamp_ns": int(stamp_ns),
            "frame": self.frame,
            "tracker": self.tracker,
            "units": self.units,
            "is_keyframe": bool(is_keyframe),
            "entities": [e.to_dict() for e in entities],
            "removed_ids": list(removed_ids),
        }


# ---------------------------------------------------------------------------
# Small helpers shared by producers
# ---------------------------------------------------------------------------
def yaw_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) for a rotation of *yaw* about world +z."""
    import math
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))

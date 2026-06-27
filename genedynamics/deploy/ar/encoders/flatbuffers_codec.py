"""FlatBuffers codec for ``WorldSnapshot`` — the zero-copy P2 hot path.

Same message shape as ``json_codec`` / ``binary_codec`` (and the C#/Swift/TS
clients), encoded with the ``flatc``-generated ``twin`` package under
``schema/generated/python``. Clients read fields straight out of the received
buffer with no parse/allocate (the perf win); this module is the Python
(server-side) encode + a decode for tests/tools.

Regenerate the cross-language code after editing ``world.fbs``::

    flatc --python --csharp --swift --ts -o schema/generated/<lang> schema/world.fbs

Requires the ``flatbuffers`` runtime (``pip install flatbuffers``); raises a
clear error if it or the generated code is missing (FlatBuffers is the P2
upgrade — JSON/binary codecs work without it).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Make the flatc-generated ``twin`` package importable.
_GEN = Path(__file__).resolve().parent.parent / "schema" / "generated" / "python"
if _GEN.is_dir() and str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

try:
    import flatbuffers  # type: ignore
    from twin import Entity as _E  # type: ignore
    from twin import Geometry as _G  # type: ignore
    from twin import Pose as _P  # type: ignore
    from twin import Quat as _Q  # type: ignore   (struct used via CreatePose)
    from twin import Vec3 as _V3  # type: ignore
    from twin import WorldSnapshot as _WS  # type: ignore
    _AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    _AVAILABLE = False
    _IMPORT_ERROR = _exc

BINARY = True

_ETYPE = {"obstacle": 0, "wall": 1, "robot": 2, "arm_link": 3, "goal": 4, "path": 5, "occluder": 6, "viewer": 7}
_GKIND = {"box": 0, "sphere": 1, "cylinder": 2, "capsule": 3, "usd": 4, "urdf": 5, "path": 6}
_ETYPE_R = {v: k for k, v in _ETYPE.items()}
_GKIND_R = {v: k for k, v in _GKIND.items()}


def _require():
    if not _AVAILABLE:
        raise RuntimeError(
            "flatbuffers_codec unavailable: install the runtime (`pip install flatbuffers`) "
            "and generate code (`flatc --python -o schema/generated/python schema/world.fbs`). "
            f"Original import error: {_IMPORT_ERROR!r}"
        )


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------
def _build_geom(b, g: Dict[str, Any]) -> int:
    asset_off = b.CreateString(g.get("asset_uri", "") or "")
    joints = [float(v) for v in (g.get("joints") or [])]
    _G.GeometryStartJointsVector(b, len(joints))
    for v in reversed(joints):
        b.PrependFloat32(v)
    joints_off = b.EndVector()
    poly = g.get("polyline") or []
    _G.GeometryStartPolylineVector(b, len(poly))
    for pt in reversed(poly):
        _V3.CreateVec3(b, float(pt[0]), float(pt[1]), float(pt[2]))
    poly_off = b.EndVector()

    _G.GeometryStart(b)
    _G.GeometryAddKind(b, _GKIND.get(g.get("kind", "box"), 0))
    he = g.get("half_extents") or (0.0, 0.0, 0.0)
    _G.GeometryAddHalfExtents(b, _V3.CreateVec3(b, float(he[0]), float(he[1]), float(he[2])))
    _G.GeometryAddRadius(b, float(g.get("radius", 0.0)))
    _G.GeometryAddHeight(b, float(g.get("height", 0.0)))
    _G.GeometryAddAssetUri(b, asset_off)
    _G.GeometryAddJoints(b, joints_off)
    _G.GeometryAddPolyline(b, poly_off)
    return _G.GeometryEnd(b)


def _build_entity(b, e: Dict[str, Any]) -> int:
    id_off = b.CreateString(e["id"])
    frame_off = b.CreateString(e.get("frame", "world") or "world")
    meta_off = b.CreateString(e.get("meta", "") or "")
    geom = e.get("geom")
    geom_off = _build_geom(b, geom) if geom else None

    _E.EntityStart(b)
    _E.EntityAddId(b, id_off)
    _E.EntityAddType(b, _ETYPE.get(e.get("type", "obstacle"), 0))
    _E.EntityAddFrame(b, frame_off)
    p = e["pose"]
    pp, pq = p["p"], p["q"]
    _E.EntityAddPose(b, _P.CreatePose(b, float(pp[0]), float(pp[1]), float(pp[2]),
                                      float(pq[0]), float(pq[1]), float(pq[2]), float(pq[3])))
    if geom_off is not None:
        _E.EntityAddGeom(b, geom_off)
    _E.EntityAddColorRgba(b, int(e.get("color_rgba", 0xFFFFFFFF)) & 0xFFFFFFFF)
    _E.EntityAddRev(b, int(e.get("rev", 0)))
    _E.EntityAddMeta(b, meta_off)
    return _E.EntityEnd(b)


def encode(snapshot: Dict[str, Any]) -> bytes:
    """Serialize a snapshot dict to a FlatBuffers buffer (binary frame)."""
    _require()
    b = flatbuffers.Builder(4096)
    site = b.CreateString(snapshot.get("site", "") or "")
    frame = b.CreateString(snapshot.get("frame", "world") or "world")
    tracker = b.CreateString(snapshot.get("tracker", "") or "")
    units = b.CreateString(snapshot.get("units", "m_rad") or "m_rad")

    ent_offs = [_build_entity(b, e) for e in snapshot.get("entities", [])]
    _WS.WorldSnapshotStartEntitiesVector(b, len(ent_offs))
    for off in reversed(ent_offs):
        b.PrependUOffsetTRelative(off)
    ents_vec = b.EndVector()

    rem_offs = [b.CreateString(s) for s in snapshot.get("removed_ids", [])]
    _WS.WorldSnapshotStartRemovedIdsVector(b, len(rem_offs))
    for off in reversed(rem_offs):
        b.PrependUOffsetTRelative(off)
    rem_vec = b.EndVector()

    _WS.WorldSnapshotStart(b)
    _WS.WorldSnapshotAddSchemaVersion(b, int(snapshot.get("schema_version", 2)))
    _WS.WorldSnapshotAddSite(b, site)
    _WS.WorldSnapshotAddStampNs(b, int(snapshot.get("stamp_ns", 0)))
    _WS.WorldSnapshotAddFrame(b, frame)
    _WS.WorldSnapshotAddTracker(b, tracker)
    _WS.WorldSnapshotAddUnits(b, units)
    _WS.WorldSnapshotAddIsKeyframe(b, bool(snapshot.get("is_keyframe", True)))
    _WS.WorldSnapshotAddEntities(b, ents_vec)
    _WS.WorldSnapshotAddRemovedIds(b, rem_vec)
    b.Finish(_WS.WorldSnapshotEnd(b))
    return bytes(b.Output())


# ---------------------------------------------------------------------------
# decode (round-trip / tooling; clients read zero-copy via generated accessors)
# ---------------------------------------------------------------------------
def _s(x) -> str:
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else (x or "")


def decode(buf: bytes) -> Dict[str, Any]:
    """Parse a FlatBuffers buffer back into a snapshot dict (floats are f32)."""
    _require()
    ws = _WS.WorldSnapshot.GetRootAs(buf, 0)
    entities = []
    for i in range(ws.EntitiesLength()):
        e = ws.Entities(i)
        pose = e.Pose()
        pp, pq = pose.P(_V3.Vec3()), pose.Q(_Q.Quat())  # struct sub-structs need a target obj
        g = e.Geom()
        geom = None
        if g is not None:
            he = g.HalfExtents()
            geom = {
                "kind": _GKIND_R.get(g.Kind(), "box"),
                "half_extents": [he.X(), he.Y(), he.Z()] if he is not None else [0.0, 0.0, 0.0],
                "radius": g.Radius(),
                "height": g.Height(),
                "asset_uri": _s(g.AssetUri()),
                "joints": [g.Joints(j) for j in range(g.JointsLength())],
                "polyline": [[g.Polyline(k).X(), g.Polyline(k).Y(), g.Polyline(k).Z()]
                             for k in range(g.PolylineLength())],
            }
        entities.append({
            "id": _s(e.Id()),
            "type": _ETYPE_R.get(e.Type(), "obstacle"),
            "frame": _s(e.Frame()),
            "pose": {"p": [pp.X(), pp.Y(), pp.Z()], "q": [pq.W(), pq.X(), pq.Y(), pq.Z()]},
            "geom": geom,
            "color_rgba": int(e.ColorRgba()),
            "rev": int(e.Rev()),
            "meta": _s(e.Meta()),
        })
    return {
        "schema_version": int(ws.SchemaVersion()),
        "site": _s(ws.Site()),
        "stamp_ns": int(ws.StampNs()),
        "frame": _s(ws.Frame()),
        "tracker": _s(ws.Tracker()),
        "units": _s(ws.Units()),
        "is_keyframe": bool(ws.IsKeyframe()),
        "entities": entities,
        "removed_ids": [_s(ws.RemovedIds(i)) for i in range(ws.RemovedIdsLength())],
    }

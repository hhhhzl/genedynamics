"""ROS 2 producer: bridge ROS2 topics → generic ``WorldState`` entities.

The interop point for the whole ROS2 / RViz ecosystem (ARCHITECTURE §6). Any
ROS2 robot/planner that publishes **visualization_msgs/MarkerArray** (obstacles,
walls, paths — the same messages RViz draws) or **geometry_msgs/PoseStamped**
(robot/headset poses) joins the AR twin with no changes: this mirrors them into a
``WorldState`` that the existing encoders/transport stream to every renderer.

Two halves, deliberately split:
  * **Pure converters** (``marker_to_entity`` / ``markerarray_to_entities`` /
    ``ros_pose_to_pose`` …) — no ``rclpy`` import, duck-typed on the message
    fields, so they unit-test with plain stubs and never drag ROS2 into a non-ROS
    process.
  * **``Ros2Bridge``** — owns an ``rclpy`` node (imported lazily, only in
    :meth:`start`), subscribes, and upserts on each callback. **Pump-driven** so it
    slots into ``world_ws``'s single-threaded ``on_tick`` (no ``WorldState``
    locking needed):

        bridge = Ros2Bridge(world, marker_topics=["/twin/markers"],
                            pose_entities=[("/vicon/g1", "g1/base", None)])
        bridge.start()
        WorldStreamServer(world, on_tick=lambda w: bridge.pump()).run()
        bridge.stop()

Frames: a marker's ``header.frame_id`` becomes ``Entity.frame``; publish in the
shared world frame (conventions §1) or resolve via TF upstream. Quaternions are
reordered ROS ``(x,y,z,w)`` → contract ``(w,x,y,z)``; colors pack to ``0xRRGGBBAA``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from genedynamics.deploy.ar.world_state import Entity, Geometry, Pose, WorldState

# visualization_msgs/Marker.type (kept as constants — no rclpy needed).
_M_ARROW, _M_CUBE, _M_SPHERE, _M_CYLINDER = 0, 1, 2, 3
_M_LINE_STRIP, _M_LINE_LIST, _M_CUBE_LIST, _M_SPHERE_LIST = 4, 5, 6, 7
_M_POINTS, _M_TEXT, _M_MESH, _M_TRIANGLE_LIST = 8, 9, 10, 11
# visualization_msgs/Marker.action (ADD == MODIFY == 0).
_A_ADD, _A_DELETE, _A_DELETEALL = 0, 2, 3

_DEFAULT_RGBA = 0xCC5544DD       # obstacle red, for markers that leave color unset
_DEFAULT_POSE_RGBA = 0x4488CCBB  # blue, for pose-only entities


# ---------------------------------------------------------------------------
# Pure converters (duck-typed; no rclpy)
# ---------------------------------------------------------------------------
def _quat_ros_to_wxyz(o: Any) -> Tuple[float, float, float, float]:
    """ROS ``(x,y,z,w)`` orientation → contract ``(w,x,y,z)``; unset → identity."""
    w, x, y, z = float(o.w), float(o.x), float(o.y), float(o.z)
    if w == 0.0 and x == 0.0 and y == 0.0 and z == 0.0:
        return (1.0, 0.0, 0.0, 0.0)   # markers often leave orientation (0,0,0,0)
    return (w, x, y, z)


def _pack_rgba(c: Any, *, fallback: int = _DEFAULT_RGBA) -> int:
    """std_msgs/ColorRGBA (floats 0..1) → packed ``0xRRGGBBAA``; all-zero → fallback."""
    r, g, b, a = float(c.r), float(c.g), float(c.b), float(c.a)
    if r == 0.0 and g == 0.0 and b == 0.0 and a == 0.0:
        return fallback               # unset color → a visible default
    to8 = lambda v: max(0, min(255, int(round(v * 255.0))))
    return (to8(r) << 24) | (to8(g) << 16) | (to8(b) << 8) | to8(a)


def ros_pose_to_pose(p: Any) -> Pose:
    """geometry_msgs/Pose → contract ``Pose``."""
    return Pose(p=(float(p.position.x), float(p.position.y), float(p.position.z)),
                q=_quat_ros_to_wxyz(p.orientation))


def ros_transform_to_pose(t: Any) -> Pose:
    """geometry_msgs/Transform → contract ``Pose``."""
    return Pose(p=(float(t.translation.x), float(t.translation.y), float(t.translation.z)),
                q=_quat_ros_to_wxyz(t.rotation))


def _marker_id(m: Any) -> str:
    ns = getattr(m, "ns", "") or ""
    return f"{ns}/{m.id}" if ns else f"marker/{m.id}"


def _marker_geom(m: Any) -> Optional[Geometry]:
    """Marker.type + Marker.scale → ``Geometry`` (RViz scale = full size / diameter)."""
    t = int(m.type)
    s = m.scale
    if t == _M_CUBE:
        return Geometry(kind="box", half_extents=(0.5 * float(s.x), 0.5 * float(s.y), 0.5 * float(s.z)))
    if t == _M_SPHERE:
        return Geometry(kind="sphere", radius=0.5 * float(s.x))
    if t == _M_CYLINDER:
        return Geometry(kind="cylinder", radius=0.5 * float(s.x), height=float(s.z))
    if t in (_M_LINE_STRIP, _M_LINE_LIST, _M_POINTS):
        pts = [(float(p.x), float(p.y), float(p.z)) for p in getattr(m, "points", [])]
        return Geometry(kind="path", polyline=pts) if pts else None
    if t == _M_MESH:
        uri = getattr(m, "mesh_resource", "") or ""
        return Geometry(kind="usd", asset_uri=uri) if uri else None
    return None  # ARROW / TEXT / *_LIST / TRIANGLE_LIST — no single-entity twin geom


# A namespace→entity-type map: a dict ({"walls": "wall"}) or a callable (ns -> type).
NsTypeMap = Union[Mapping[str, str], Callable[[str], Optional[str]]]


def _entity_type(ns: str, type_for_ns: Optional[NsTypeMap], default_type: str) -> str:
    if type_for_ns is None:
        return default_type
    t = type_for_ns(ns) if callable(type_for_ns) else type_for_ns.get(ns, default_type)
    return t or default_type


def marker_to_entity(m: Any, *, type_for_ns: Optional[NsTypeMap] = None,
                     default_type: str = "obstacle") -> Optional[Entity]:
    """Convert one visualization_msgs/Marker to an ``Entity`` (RViz-compatible).

    Returns ``None`` for non-geometric / unsupported marker types. DELETE /
    DELETEALL are handled by the caller via ``Marker.action`` (see
    :func:`markerarray_to_entities`). Entity type defaults to ``"obstacle"``;
    pass *type_for_ns* to map a marker namespace to ``wall`` / ``goal`` / etc.
    """
    geom = _marker_geom(m)
    if geom is None:
        return None
    frame = getattr(getattr(m, "header", None), "frame_id", "") or "world"
    return Entity(
        id=_marker_id(m),
        type=_entity_type(getattr(m, "ns", "") or "", type_for_ns, default_type),
        frame=frame,
        pose=ros_pose_to_pose(m.pose),
        geom=geom,
        color_rgba=_pack_rgba(m.color),
    )


def markerarray_to_entities(
    msg: Any, *, type_for_ns: Optional[NsTypeMap] = None, default_type: str = "obstacle",
) -> Tuple[List[Entity], List[str], bool]:
    """Split a visualization_msgs/MarkerArray into ``(upserts, removed_ids, clear_all)``."""
    upserts: List[Entity] = []
    removed: List[str] = []
    clear_all = False
    for m in msg.markers:
        action = int(getattr(m, "action", _A_ADD))
        if action == _A_DELETEALL:
            clear_all = True
            continue
        if action == _A_DELETE:
            removed.append(_marker_id(m))
            continue
        e = marker_to_entity(m, type_for_ns=type_for_ns, default_type=default_type)
        if e is not None:
            upserts.append(e)
    return upserts, removed, clear_all


# ---------------------------------------------------------------------------
# Live bridge (lazy rclpy)
# ---------------------------------------------------------------------------
# (topic, entity_id, optional Geometry proxy if the entity doesn't pre-exist)
PoseBinding = Tuple[str, str, Optional[Geometry]]


class Ros2Bridge:
    """Mirror ROS2 topics into a ``WorldState`` (pump-driven; one node)."""

    def __init__(
        self,
        world: WorldState,
        *,
        node_name: str = "ar_twin_bridge",
        marker_topics: Sequence[str] = (),
        pose_entities: Sequence[PoseBinding] = (),
        type_for_ns: Optional[NsTypeMap] = None,
        default_type: str = "obstacle",
        qos_depth: int = 10,
    ) -> None:
        self.world = world
        self.node_name = node_name
        self.marker_topics = list(marker_topics)
        self.pose_entities = list(pose_entities)
        self.type_for_ns = type_for_ns
        self.default_type = default_type
        self.qos_depth = int(qos_depth)

        self._node = None
        self._owns_rclpy = False
        self._marker_ids: set[str] = set()  # marker-sourced ids (scopes DELETEALL)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Init the rclpy node and subscriptions (imports rclpy lazily)."""
        import rclpy  # noqa: PLC0415 — lazy: module stays importable without ROS2
        from geometry_msgs.msg import PoseStamped
        from visualization_msgs.msg import MarkerArray

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._node = rclpy.create_node(self.node_name)

        for topic in self.marker_topics:
            self._node.create_subscription(
                MarkerArray, topic, self._on_markers, self.qos_depth)
        for topic, eid, geom in self.pose_entities:
            # bind eid/geom per subscription without a late-binding closure bug
            self._node.create_subscription(
                PoseStamped, topic,
                lambda msg, _eid=eid, _geom=geom: self._on_pose(_eid, _geom, msg),
                self.qos_depth)

    def pump(self, timeout_sec: float = 0.0) -> None:
        """Process pending callbacks once (call from ``world_ws`` ``on_tick``)."""
        if self._node is None:
            return
        import rclpy  # noqa: PLC0415
        rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def spin(self) -> None:
        """Blocking spin (standalone use instead of :meth:`pump`)."""
        if self._node is None:
            self.start()
        import rclpy  # noqa: PLC0415
        rclpy.spin(self._node)

    def stop(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy:
            import rclpy  # noqa: PLC0415
            if rclpy.ok():
                rclpy.shutdown()
            self._owns_rclpy = False

    # ------------------------------------------------------------------
    def _on_markers(self, msg: Any) -> None:
        upserts, removed, clear_all = markerarray_to_entities(
            msg, type_for_ns=self.type_for_ns, default_type=self.default_type)
        if clear_all:
            for eid in list(self._marker_ids):
                self.world.remove(eid)
            self._marker_ids.clear()
        for e in upserts:
            self.world.upsert(e)
            self._marker_ids.add(e.id)
        for eid in removed:
            if self.world.remove(eid):
                self._marker_ids.discard(eid)

    def _on_pose(self, eid: str, default_geom: Optional[Geometry], msg: Any) -> None:
        # Pose-only update: keep the existing entity's geom/type/color if it exists
        # (so a tracker stream just moves it), else create a small proxy.
        existing = self.world.get(eid)
        frame = getattr(getattr(msg, "header", None), "frame_id", "") or "world"
        self.world.upsert(Entity(
            id=eid,
            type=existing.type if existing is not None else "robot",
            frame=frame,
            pose=ros_pose_to_pose(msg.pose),
            geom=(existing.geom if existing is not None
                  else (default_geom or Geometry(kind="sphere", radius=0.05))),
            color_rgba=existing.color_rgba if existing is not None else _DEFAULT_POSE_RGBA,
        ))

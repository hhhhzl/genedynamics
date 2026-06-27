"""NatNet (OptiTrack/Motive) → shared-memory bridge for the corridor deploy pipeline.

Runs in a **ROS2 environment** (sourced ``rclpy``). Subscribes to the
`natnet_ros2 <https://github.com/L2S-lab/natnet_ros2>`_ driver's per-rigid-body
``geometry_msgs/PoseStamped`` topic (``/<body>/pose``) and writes the robot base
6-DoF into the ``mocap_state_shm`` shared block in the exact ``q13d`` layout that
:class:`genedynamics.deploy.localization.vicon_shm_plugin.ViconShmPlugin` reads.

So the rest of the pipeline never sees ROS2: the fedguide twin server and the
robot-control process keep using ``--localization vicon`` (no ``rclpy`` there).
This is the NatNet/ROS2 drop-in **replacement for the pyvicon-based**
``ViconDemoWriter`` — it plays the same "Vicon writer" role from
``ar/instruction.md`` §A7/§C1, just sourced from natnet_ros2 instead of the
Vicon DataStream SDK.

Pipeline::

    Motive  (Streaming: Up Axis = Z, units = meters)
      → natnet_ros2 driver      (ROS2 env)   publishes  /g1_base/pose  [PoseStamped]
      → natnet_shm_writer.py    (ROS2 env)   writes     mocap_state_shm [q13d]
      → ViconShmPlugin          (fedguide)   --localization vicon
      → { run_twin_server.py (AR) , run_real_g1.py (governor) }

Two source paths, same ``mocap_state_shm`` output:
  * **direct NatNet (no ROS)** — read Motive's NatNet stream straight off the
    wire. Use on hosts without ROS (e.g. the Mac control host). No ``rclpy``.
  * **via ROS** — subscribe to natnet_ros2 / natnet_ros_cpp ``PoseStamped``.

Run — **direct NatNet, no ROS** (validated against Motive / NatNet 4.5, multicast)::

    python -m genedynamics.deploy.localization.natnet_shm_writer --natnet \
        --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5
    # --local-ip = this host's NIC on Motive's subnet — REQUIRED so the multicast
    #   group is joined on the right interface (on macOS multicast otherwise
    #   defaults to the wrong NIC). --rigid-body-id = the base body's Motive
    #   streaming ID (omit → first tracked body). Add --unicast if Motive's
    #   Transmission Type is Unicast.

Run — **via ROS** (in the ROS2 env, after ``source /opt/ros/<distro>/setup.bash``)::

    python -m genedynamics.deploy.localization.natnet_shm_writer --rigid-body g1_base
    # explicit topic instead of <rigid-body>/pose:
    python -m genedynamics.deploy.localization.natnet_shm_writer --topic /g1_base/pose

Conventions — get these right or everything downstream is rotated/scaled:
  * Motive Streaming → **Up Axis = Z** and **units = meters**. NatNet/ROS publish
    meters, so we do NOT scale (unlike the Vicon DataStream, which is mm → ÷1000).
  * world frame: right-handed, z up, x forward, y left (``ar/schema/conventions.md`` §1).
  * Set the rigid body's **pivot at the pelvis center, +x = robot forward** in Motive
    (instruction §A5); the published pose then needs no extra mount offset.
  * shm ``q13d`` = int64 utime(µs) + 13 float64: pos[3], quat_xyzw[4], vel[3], omega[3];
    the quaternion is stored **xyzw** (``ViconShmPlugin`` rolls it to wxyz on read).
"""

from __future__ import annotations

import argparse
import struct
import time
from typing import Optional, Tuple

import numpy as np

from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin


class _TwistEstimator:
    """Finite-difference (vel, omega) in the world frame from a pose stream.

    ``PoseStamped`` carries no twist, so we differentiate position and orientation
    (same scheme as ``ViconDemoWriter``). Optional first-order low-pass (``alpha``
    in [0, 1); 0 = raw) tames mocap finite-difference noise the lateral governor
    backstop would otherwise see.
    """

    def __init__(self, alpha: float = 0.0) -> None:
        from scipy.spatial.transform import Rotation as _R  # noqa: PLC0415 — lazy
        self._R = _R
        self._alpha = float(max(0.0, min(1.0, alpha)))
        self._t: Optional[float] = None
        self._pos: Optional[np.ndarray] = None
        self._quat: Optional[np.ndarray] = None
        self._vel = np.zeros(3)
        self._omega = np.zeros(3)

    def update(self, t: float, pos: np.ndarray, quat_xyzw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vel = np.zeros(3)
        omega = np.zeros(3)
        if self._t is not None and self._pos is not None:
            dt = t - self._t
            if dt > 1e-6:
                vel = (pos - self._pos) / dt
                dq = self._R.from_quat(quat_xyzw) * self._R.from_quat(self._quat).inv()
                omega = dq.as_rotvec() / dt
        self._t, self._pos, self._quat = t, pos.copy(), quat_xyzw.copy()
        if self._alpha > 0.0:
            a = self._alpha
            self._vel = a * self._vel + (1.0 - a) * vel
            self._omega = a * self._omega + (1.0 - a) * omega
            return self._vel.copy(), self._omega.copy()
        return vel, omega


def _stamp_to_sec(stamp) -> float:
    """Header stamp → seconds, tolerant of ROS2 (.sec/.nanosec) and ROS1 (.secs/.nsecs)."""
    if hasattr(stamp, "sec"):            # ROS2 builtin_interfaces/Time
        return float(stamp.sec) + float(getattr(stamp, "nanosec", 0)) * 1e-9
    if hasattr(stamp, "secs"):           # ROS1 rospy Time fields
        return float(stamp.secs) + float(getattr(stamp, "nsecs", 0)) * 1e-9
    try:
        return float(stamp.to_sec())     # ROS1 rospy.Time
    except Exception:
        return 0.0


def pack_q13d(
    utime_us: int,
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
    vel: np.ndarray,
    omega: np.ndarray,
) -> bytes:
    """Pack one mocap sample into the ``q13d`` shm record (pure; rclpy-free, testable)."""
    return struct.pack(
        ViconShmPlugin.STRUCT_FMT,
        int(utime_us),
        float(pos[0]), float(pos[1]), float(pos[2]),
        float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3]),
        float(vel[0]), float(vel[1]), float(vel[2]),
        float(omega[0]), float(omega[1]), float(omega[2]),
    )


# ============================================================================
# Direct NatNet path (ROS-free). Read Motive's NatNet stream straight off the
# wire and mirror the base rigid body to shm — no ROS, no natnet_ros2 driver.
# Conventions (Motive → Streaming): Up Axis = Z, units = meters, Rigid Bodies on.
# Validated against Motive / NatNet 4.5 (Transmission Type = Multicast).
# ============================================================================

NAT_CONNECT = 0
NAT_FRAME_OF_DATA = 7
DEFAULT_MULTICAST = "239.255.42.99"
DEFAULT_DATA_PORT = 1511
DEFAULT_CMD_PORT = 1510


def parse_frame_of_data(data: bytes) -> Optional[Tuple[int, list]]:
    """Parse a NatNet 4.x ``FrameOfData`` packet → ``(frame_number, [rigid bodies])``.

    NatNet 3.0+ lays each top-level section out as ``[count i32][byteSize i32]
    [payload]``; the ``byteSize`` lets a client skip section types it does not
    consume. We stream only Rigid Bodies (Motive → Streaming: Rigid Bodies on,
    markers/skeletons off), so we skip the marker-set and legacy-marker sections
    by their byte size and parse the rigid bodies. Each rigid body is 38 bytes:
    ``id(i32) x,y,z(3f) qx,qy,qz,qw(4f) meanMarkerError(f) trackingFlags(i16)``
    (bit 0 of the flags = tracking valid). Returns ``None`` for other messages.

    Each rigid body is ``(id, (x,y,z), (qx,qy,qz,qw), mean_error_m, tracked)``.
    """
    off = 0
    msg_id = struct.unpack_from("<H", data, off)[0]
    off += 4  # msgID(2) + nDataBytes(2)
    if msg_id != NAT_FRAME_OF_DATA:
        return None
    frame_number = struct.unpack_from("<i", data, off)[0]
    off += 4
    for _section in ("marker_sets", "legacy_other_markers"):  # skip count + size + payload
        off += 4  # section count (we skip the payload by its byte size instead)
        size = struct.unpack_from("<i", data, off)[0]
        off += 4 + size
    n_rb = struct.unpack_from("<i", data, off)[0]
    off += 8  # rigid-body count + section byteSize (bodies parsed directly below)
    bodies = []
    for _ in range(n_rb):
        rb_id = struct.unpack_from("<i", data, off)[0]; off += 4
        pos = struct.unpack_from("<3f", data, off); off += 12
        quat = struct.unpack_from("<4f", data, off); off += 16  # x, y, z, w
        mean_err = struct.unpack_from("<f", data, off)[0]; off += 4
        flags = struct.unpack_from("<h", data, off)[0]; off += 2
        bodies.append((rb_id, pos, quat, mean_err, bool(flags & 0x01)))
    return frame_number, bodies


class NatNetDirectReceiver:
    """ROS-free NatNet client: open Motive's stream and yield ``FrameOfData``.

    Multicast (default; matches Motive → Transmission Type = Multicast): bind the
    data port and join the group on ``local_ip``. Unicast: send a connect to the
    command port and Motive streams to this host. ``local_ip`` MUST be the address
    of the NIC on Motive's subnet — on macOS the multicast group otherwise joins
    on the wrong (default-route) interface and no frames arrive.
    """

    def __init__(
        self,
        *,
        local_ip: str,
        server: Optional[str] = None,
        multicast_group: str = DEFAULT_MULTICAST,
        data_port: int = DEFAULT_DATA_PORT,
        cmd_port: int = DEFAULT_CMD_PORT,
        unicast: bool = False,
        timeout: float = 5.0,
    ) -> None:
        self.local_ip = local_ip
        self.server = server
        self.multicast_group = None if unicast else multicast_group
        self.data_port = int(data_port)
        self.cmd_port = int(cmd_port)
        self.timeout = float(timeout)
        self._cmd = None
        self._data = None

    def connect(self) -> None:
        import socket  # noqa: PLC0415

        # Command socket bound to the NIC on Motive's subnet so the connect (and
        # any unicast stream) egress/return on the right interface.
        self._cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._cmd.bind((self.local_ip, 0))
        self._cmd.settimeout(self.timeout)
        if self.server:  # announce: required for unicast, harmless for multicast
            self._cmd.sendto(struct.pack("<HH", NAT_CONNECT, 5) + b"Ping\0", (self.server, self.cmd_port))

        self._data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._data.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._data.bind(("", self.data_port))
        if self.multicast_group:
            mreq = struct.pack(
                "4s4s", socket.inet_aton(self.multicast_group), socket.inet_aton(self.local_ip)
            )
            self._data.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self._data.settimeout(self.timeout)

    def recv_frame(self) -> Tuple[int, list]:
        """Block for the next ``FrameOfData``; raises ``socket.timeout`` if none arrives."""
        while True:
            data, _addr = self._data.recvfrom(65536)
            frame = parse_frame_of_data(data)
            if frame is not None:
                return frame

    def close(self) -> None:
        for sock in (self._cmd, self._data):
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        self._cmd = self._data = None


class NatNetShmWriter:
    """Subscribe to a natnet_ros2 ``PoseStamped`` topic and mirror it to ``mocap_state_shm``."""

    def __init__(
        self,
        *,
        topic: str = "/g1_base/pose",
        shm_name: str = ViconShmPlugin.SHM_NAME,
        z_offset: float = 0.0,
        vel_lpf: float = 0.0,
        reliable: bool = False,
        verbose: bool = True,
    ) -> None:
        from multiprocessing import shared_memory  # noqa: PLC0415

        self.topic = topic
        self.z_offset = float(z_offset)
        self.verbose = verbose
        self._twist = _TwistEstimator(alpha=vel_lpf)
        self._reliable = reliable
        self._count = 0

        try:
            self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=ViconShmPlugin.SHM_SIZE)
            self._created = True
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False, size=ViconShmPlugin.SHM_SIZE)
            self._created = False
        self._shm_name = shm_name

        self._node = None  # set in run()

    def _write_pose(self, t: float, pos: np.ndarray, quat_xyzw: np.ndarray, *, source: str) -> None:
        """Shared core: finite-diff twist + pack one ``q13d`` sample into shm.

        ``pos``/``quat_xyzw`` are in the world frame, meters / NatNet xyzw order
        (shm slots 4:8). Both the ROS callback and the direct NatNet loop feed it.
        """
        pos = np.asarray(pos, dtype=np.float64).copy()
        pos[2] += self.z_offset
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        vel, omega = self._twist.update(t, pos, quat_xyzw)
        self._shm.buf[: ViconShmPlugin.SHM_SIZE] = pack_q13d(int(t * 1e6), pos, quat_xyzw, vel, omega)

        self._count += 1
        if self.verbose and self._count % 200 == 1:
            print(f"[natnet_shm_writer] {source} → {self._shm_name}  "
                  f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  n={self._count}")

    def _on_pose(self, msg) -> None:
        p = msg.pose.position
        o = msg.pose.orientation
        t_msg = _stamp_to_sec(msg.header.stamp)
        t = t_msg if t_msg > 0.0 else time.time()  # natnet stamps are wall-clock ROS time
        self._write_pose(
            t,
            np.array([p.x, p.y, p.z], dtype=np.float64),
            np.array([o.x, o.y, o.z, o.w], dtype=np.float64),  # ROS order == shm slots 4:8
            source=self.topic,
        )

    def run_natnet(
        self,
        *,
        local_ip: str,
        server: Optional[str] = None,
        multicast_group: str = DEFAULT_MULTICAST,
        data_port: int = DEFAULT_DATA_PORT,
        cmd_port: int = DEFAULT_CMD_PORT,
        unicast: bool = False,
        rigid_body_id: Optional[int] = None,
    ) -> None:
        """Direct (ROS-free) NatNet → shm: mirror the target rigid body's base pose.

        ``rigid_body_id`` is the Motive streaming ID of the base body; if ``None``,
        the first tracked body in each frame is used. Untracked frames (markers
        occluded) are skipped, so the last good pose stays in shm.
        """
        rx = NatNetDirectReceiver(
            local_ip=local_ip, server=server, multicast_group=multicast_group,
            data_port=data_port, cmd_port=cmd_port, unicast=unicast,
        )
        rx.connect()
        if self.verbose:
            transport = "unicast" if unicast else f"multicast {multicast_group}"
            target = rigid_body_id if rigid_body_id is not None else "first tracked"
            print(f"[natnet_shm_writer] direct NatNet {transport}:{data_port} via {local_ip} "
                  f"→ shm '{self._shm_name}'. rigid body: {target}. Waiting for Motive …")
        import socket  # noqa: PLC0415 — for socket.timeout

        picked: Optional[int] = None
        try:
            while True:
                try:
                    _frame, bodies = rx.recv_frame()
                except socket.timeout:
                    print("[natnet_shm_writer] no NatNet frames in 5s — check Motive Streaming "
                          "(Enable / Local Interface / Rigid Bodies on / robot in volume).")
                    continue
                if not bodies:
                    continue
                if rigid_body_id is not None:
                    rb = next((b for b in bodies if b[0] == rigid_body_id), None)
                else:
                    rb = next((b for b in bodies if b[4]), bodies[0])  # first tracked, else first
                if rb is None:
                    continue
                rb_id, pos, quat, _err, tracked = rb
                if picked != rb_id:
                    picked = rb_id
                    if self.verbose:
                        ids = [b[0] for b in bodies]
                        print(f"[natnet_shm_writer] tracking rigid body id={rb_id} "
                              f"({'tracked' if tracked else 'NOT tracked'}); bodies in frame: {ids}")
                if not tracked:
                    continue  # hold last good sample rather than write a stale/invalid pose
                self._write_pose(time.time(), np.asarray(pos), np.asarray(quat), source=f"natnet/rb{rb_id}")
        except KeyboardInterrupt:
            pass
        finally:
            rx.close()
            self.close()

    def run(self) -> None:
        import rclpy  # noqa: PLC0415 — lazy: ROS2-only
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("natnet_shm_writer")
        qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE if self._reliable else ReliabilityPolicy.BEST_EFFORT,
        )
        self._node.create_subscription(PoseStamped, self.topic, self._on_pose, qos)
        if self.verbose:
            print(f"[natnet_shm_writer] subscribing {self.topic} "
                  f"({'reliable' if self._reliable else 'best-effort'}) → shm '{self._shm_name}'. "
                  f"Waiting for natnet_ros2 …")
        try:
            rclpy.spin(self._node)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
            if rclpy.ok():
                rclpy.shutdown()

    def run_ros1(self) -> None:
        """ROS1 (rospy) variant for the L2S-lab ``natnet_ros_cpp`` driver (ROS noetic).

        Same shm output as :meth:`run`; only the ROS runtime differs. natnet_ros_cpp
        publishes the rigid body at ``/natnet_ros/<body>/pose`` (PoseStamped), Z-up,
        meters — matching our world frame, so no axis/scale conversion needed.
        """
        import rospy  # noqa: PLC0415 — lazy: ROS1-only
        from geometry_msgs.msg import PoseStamped

        rospy.init_node("natnet_shm_writer", anonymous=True, disable_signals=True)
        rospy.Subscriber(self.topic, PoseStamped, self._on_pose, queue_size=10)
        if self.verbose:
            print(f"[natnet_shm_writer] (ROS1) subscribing {self.topic} → shm "
                  f"'{self._shm_name}'. Waiting for natnet_ros_cpp …")
        try:
            rospy.spin()
        except (KeyboardInterrupt, rospy.ROSInterruptException):
            pass
        finally:
            self.close()

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if getattr(self, "_shm", None) is not None:
            self._shm.close()
            if self._created:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
            self._shm = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--topic", default=None, help="PoseStamped topic (default: /<rigid-body>/pose).")
    g.add_argument("--rigid-body", default="g1_base", help="Motive rigid-body name → topic /<name>/pose.")
    ap.add_argument("--shm-name", default=ViconShmPlugin.SHM_NAME)
    ap.add_argument("--z-offset", type=float, default=0.0, help="Added to z (pivot-to-ground correction, m).")
    ap.add_argument("--vel-lpf", type=float, default=0.0, help="EMA alpha [0,1) on finite-diff vel/omega (0=raw).")
    ap.add_argument("--reliable", action="store_true", help="RELIABLE QoS (default best-effort). ROS2 only.")
    ap.add_argument("--ros1", action="store_true",
                    help="ROS1/rospy for natnet_ros_cpp (noetic) instead of ROS2/rclpy for natnet_ros2.")
    # ---- direct NatNet (no ROS) ----
    ap.add_argument("--natnet", action="store_true",
                    help="Direct NatNet, no ROS: read Motive's stream off the wire (e.g. on the Mac host).")
    ap.add_argument("--local-ip", default=None,
                    help="[--natnet] this host's IP on Motive's subnet (the NIC reaching Motive). Required.")
    ap.add_argument("--server", default=None,
                    help="[--natnet] Motive host IP (for connect/unicast; optional for pure multicast).")
    ap.add_argument("--multicast-group", default=DEFAULT_MULTICAST, help="[--natnet] Motive multicast group.")
    ap.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT, help="[--natnet] NatNet data port.")
    ap.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT, help="[--natnet] NatNet command port.")
    ap.add_argument("--unicast", action="store_true",
                    help="[--natnet] Motive Transmission Type = Unicast (default: multicast).")
    ap.add_argument("--rigid-body-id", type=int, default=None,
                    help="[--natnet] Motive streaming ID of the base body (default: first tracked).")
    args = ap.parse_args(argv)

    if args.natnet:
        if not args.local_ip:
            ap.error("--natnet requires --local-ip (this host's IP on Motive's subnet, e.g. 192.168.0.100).")
        writer = NatNetShmWriter(
            topic="(direct-natnet)", shm_name=args.shm_name, z_offset=args.z_offset, vel_lpf=args.vel_lpf,
        )
        writer.run_natnet(
            local_ip=args.local_ip, server=args.server, multicast_group=args.multicast_group,
            data_port=args.data_port, cmd_port=args.cmd_port, unicast=args.unicast,
            rigid_body_id=args.rigid_body_id,
        )
        return 0

    if args.topic:
        topic = args.topic
    elif args.ros1:
        topic = f"/natnet_ros/{args.rigid_body}/pose"   # natnet_ros_cpp topic prefix
    else:
        topic = f"/{args.rigid_body}/pose"              # natnet_ros2 topic
    writer = NatNetShmWriter(
        topic=topic, shm_name=args.shm_name, z_offset=args.z_offset,
        vel_lpf=args.vel_lpf, reliable=args.reliable,
    )
    if args.ros1:
        writer.run_ros1()
    else:
        writer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

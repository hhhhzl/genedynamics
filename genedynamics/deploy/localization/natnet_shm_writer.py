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

Run (in the ROS2 env, after ``source /opt/ros/<distro>/setup.bash``)::

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

    def _on_pose(self, msg) -> None:
        p = msg.pose.position
        o = msg.pose.orientation
        pos = np.array([p.x, p.y, p.z + self.z_offset], dtype=np.float64)
        quat_xyzw = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)  # ROS order == shm slots 4:8

        stamp = msg.header.stamp
        t_msg = stamp.sec + stamp.nanosec * 1e-9
        t = t_msg if t_msg > 0.0 else time.time()  # natnet stamps are wall-clock ROS time

        vel, omega = self._twist.update(t, pos, quat_xyzw)
        self._shm.buf[: ViconShmPlugin.SHM_SIZE] = pack_q13d(int(t * 1e6), pos, quat_xyzw, vel, omega)

        self._count += 1
        if self.verbose and self._count % 200 == 1:
            print(f"[natnet_shm_writer] {self.topic} → {self._shm_name}  "
                  f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  n={self._count}")

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
    ap.add_argument("--reliable", action="store_true", help="RELIABLE QoS (default best-effort, for high-rate mocap).")
    args = ap.parse_args(argv)

    topic = args.topic or f"/{args.rigid_body}/pose"
    writer = NatNetShmWriter(
        topic=topic, shm_name=args.shm_name, z_offset=args.z_offset,
        vel_lpf=args.vel_lpf, reliable=args.reliable,
    )
    writer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

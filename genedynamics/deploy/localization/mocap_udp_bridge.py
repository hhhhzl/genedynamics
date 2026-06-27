#!/usr/bin/env python
"""Mocap pose bridge: NatNet/natnet_ros2 (mocap host, ROS2) → Mac `mocap_state_shm`.

The control Mac has NO ROS2 and `mocap_state_shm` is host-local, so the mocap
pose must be carried across machines. This single file has three modes:

  --mode ros2   (run on the MOCAP HOST, has ROS2): subscribe a natnet_ros2
                geometry_msgs/PoseStamped topic (e.g. /<rigid_body>/pose) and
                send each pose to the Mac over plain UDP. (needs rclpy)
  --mode recv   (run on the MAC, no ROS2): receive UDP poses and write them into
                `mocap_state_shm` (q13d) that ViconShmPlugin / `--localization
                vicon` reads. Velocity/omega by finite difference.
  --mode fake   (run anywhere, for OFFLINE testing — no robot, no mocap, no ROS2):
                send a slowly-wobbling fake pose over UDP so you can verify the
                recv→shm→ViconShmPlugin path end-to-end on the Mac.

Wire (UDP packet): struct '<7d' = (x, y, z, qx, qy, qz, qw)   [world frame, xyzw].

Examples
--------
Offline test on the Mac (3 terminals):
  python -m genedynamics.deploy.localization.mocap_udp_bridge --mode recv  --port 9870
  python -m genedynamics.deploy.localization.mocap_udp_bridge --mode fake  --host 127.0.0.1 --port 9870
  python -c "from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin; \
import time; p=ViconShmPlugin({}); time.sleep(0.3); print(p.get_state(), p.health())"

Real (in the lab):
  # on the mocap host:
  python -m genedynamics.deploy.localization.mocap_udp_bridge --mode ros2 \
      --topic /g1_base/pose --host <MAC_IP> --port 9870
  # on the Mac:
  python -m genedynamics.deploy.localization.mocap_udp_bridge --mode recv --port 9870
  # then run_real_g1 ... --localization vicon
"""
from __future__ import annotations

import argparse
import math
import socket
import struct
import time

_PKT = struct.Struct("<7d")  # x,y,z, qx,qy,qz,qw


def _shm_consts():
    from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin
    return ViconShmPlugin.SHM_NAME, ViconShmPlugin.SHM_SIZE, ViconShmPlugin.STRUCT_FMT


# ---------------------------------------------------------------------------
def run_recv(args) -> None:
    """MAC side: UDP → mocap_state_shm (no ROS2)."""
    from multiprocessing import shared_memory

    import numpy as np
    from scipy.spatial.transform import Rotation as R

    shm_name, shm_size, fmt = _shm_consts()
    name = args.shm_name or shm_name
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=shm_size)
        created = True
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name, create=False, size=shm_size)
        created = False

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(2.0)
    print(f"[bridge:recv] UDP :{args.port} -> shm '{name}' (created={created})")

    prev_t = prev_pos = prev_quat = None
    n = 0
    while True:
        try:
            data, addr = sock.recvfrom(64)
        except socket.timeout:
            print("[bridge:recv] no UDP in 2s — sender up? same network? right port?")
            continue
        if len(data) < _PKT.size:
            continue
        x, y, z, qx, qy, qz, qw = _PKT.unpack(data[: _PKT.size])
        t = time.time()
        pos = np.array([x, y, z + args.z_offset], dtype=float)
        quat = np.array([qx, qy, qz, qw], dtype=float)  # xyzw (ViconShmPlugin's order)
        vel = np.zeros(3)
        omega = np.zeros(3)
        if prev_t is not None:
            dt = t - prev_t
            if dt > 1e-4:
                vel = (pos - prev_pos) / dt
                dq = R.from_quat(quat) * R.from_quat(prev_quat).inv()
                omega = dq.as_rotvec() / dt
        prev_t, prev_pos, prev_quat = t, pos, quat
        struct.pack_into(fmt, shm.buf, 0, int(t * 1e6), *pos, *quat, *vel, *omega)
        n += 1
        if n % 200 == 0:
            print(f"[bridge:recv] {n} pkts | last pos={pos.round(3)} from {addr[0]}")


# ---------------------------------------------------------------------------
def run_ros2(args) -> None:
    """MOCAP HOST side: natnet_ros2 PoseStamped → UDP to the Mac (needs rclpy)."""
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.host, args.port)
    print(f"[bridge:ros2] {args.topic} -> UDP {dst}")

    class _Fwd(Node):
        def __init__(self):
            super().__init__("mocap_udp_bridge")
            self.create_subscription(PoseStamped, args.topic, self._cb, 10)
            self._n = 0

        def _cb(self, msg):
            p, q = msg.pose.position, msg.pose.orientation
            sock.sendto(_PKT.pack(p.x, p.y, p.z, q.x, q.y, q.z, q.w), dst)
            self._n += 1
            if self._n % 200 == 0:
                self.get_logger().info(f"forwarded {self._n} poses")

    rclpy.init()
    node = _Fwd()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
def run_fake(args) -> None:
    """OFFLINE test: send a wobbling fake pose over UDP (no robot/mocap/ROS2)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.host, args.port)
    print(f"[bridge:fake] fake pose -> UDP {dst} @ {args.hz}Hz (Ctrl-C to stop)")
    t0 = time.time()
    while True:
        s = time.time() - t0
        x = 0.5 + 0.10 * math.sin(0.2 * s)   # wobble around scene start (0.5, 0)
        y = 0.05 * math.sin(0.5 * s)
        sock.sendto(_PKT.pack(x, y, 0.0, 0.0, 0.0, 0.0, 1.0), dst)
        time.sleep(1.0 / args.hz)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["recv", "ros2", "fake"])
    ap.add_argument("--host", default="127.0.0.1", help="Mac IP (ros2/fake send target)")
    ap.add_argument("--port", type=int, default=9870)
    ap.add_argument("--topic", default="/g1_base/pose", help="natnet_ros2 PoseStamped topic (ros2 mode)")
    ap.add_argument("--shm-name", default=None, help="override mocap_state_shm name (recv mode)")
    ap.add_argument("--z-offset", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=50.0, help="fake mode send rate")
    args = ap.parse_args(argv)
    {"recv": run_recv, "ros2": run_ros2, "fake": run_fake}[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

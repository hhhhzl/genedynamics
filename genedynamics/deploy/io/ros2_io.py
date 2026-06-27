"""ROS2 Robot IO adapter.

Implements the :class:`~genedynamics.deploy.interfaces.robot_io.RobotIO`
protocol for any robot exposed via ROS2 topics.  This is the generic
hardware adapter when the robot vendor provides a ROS2 driver — no vendor
SDK is required.

Topic conventions (all configurable)::

    Joint-space control:
        Publish: sensor_msgs/JointState → joint_cmd_topic
        Subscribe: sensor_msgs/JointState ← joint_state_topic

    Base-level twist control (mobile base / legged):
        Publish: geometry_msgs/Twist → twist_cmd_topic

    Base pose (odometry):
        Subscribe: nav_msgs/Odometry ← odom_topic

Supported :class:`ControlCommand` kinds:

* ``"joint_pos"`` — publish ``position`` array to ``joint_cmd_topic``
* ``"torque"`` — publish ``effort`` array to ``joint_cmd_topic``
* ``"loco"`` — publish ``Twist`` to ``twist_cmd_topic``
* ``"mixed"`` — publish joint_pos upper-body to ``joint_cmd_topic`` AND
  loco_cmd to ``twist_cmd_topic``

``get_state()`` / ``step()`` read the latest cached message from background
subscriber threads. :meth:`step` is a rate-limiter that sleeps until the
desired control period elapses — on a real ROS2 robot the driver publishes
state asynchronously, so we simply wait for the next cycle.

Usage::

    from genedynamics.deploy.io.ros2_io import ROS2RobotIO, ROS2IOConfig

    io = ROS2RobotIO(
        spec=my_spec,
        joint_state_topic="/joint_states",
        joint_cmd_topic="/joint_cmd",
        odom_topic="/odom",
    )
    state = io.reset()
    cmd = controller.act(state, intent)
    io.send_control(cmd)
    state = io.step(dt=0.02)
    io.close()

Requires ``rclpy``, ``sensor_msgs``, ``geometry_msgs``, ``nav_msgs`` (ROS2).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState

__all__ = ["ROS2RobotIO", "ROS2IOConfig"]

_ROS2_AVAILABLE = False
try:
    import rclpy  
    _ROS2_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ROS2IOConfig:
    """Configuration for :class:`ROS2RobotIO`.

    Attributes:
        joint_state_topic: Incoming ``sensor_msgs/JointState`` topic name.
        joint_cmd_topic: Outgoing ``sensor_msgs/JointState`` topic name.
        odom_topic: Optional incoming ``nav_msgs/Odometry`` for base pose.
        twist_cmd_topic: Outgoing ``geometry_msgs/Twist`` topic for loco cmds.
        node_name: Name of the rclpy node.
        joint_names: Ordered list of joint names to extract from JointState.
            When ``None`` the message order is used as-is.
        state_timeout_s: How long (seconds) to wait for the first state
            message during :meth:`reset`. Raises ``TimeoutError`` if exceeded.
    """

    joint_state_topic: str = "/joint_states"
    joint_cmd_topic: str = "/joint_cmd"
    odom_topic: Optional[str] = "/odom"
    twist_cmd_topic: str = "/cmd_vel"
    node_name: str = "deploy_ros2_io"
    joint_names: Optional[list[str]] = None
    state_timeout_s: float = 5.0
    extra: dict = field(default_factory=dict)


class ROS2RobotIO:
    """Generic ROS2 robot IO for any ROS2-enabled hardware.

    Args:
        spec: Robot spec object exposing at least ``actuated_joints`` and
            optionally ``joint_range``, ``joint_torque_limit``. May be ``None``
            for robots without a predefined spec.
        config: :class:`ROS2IOConfig` with topic / node settings.
        joint_state_topic: Convenience shortcut — overrides
            ``config.joint_state_topic`` when provided.
        joint_cmd_topic: Convenience shortcut.
        odom_topic: Convenience shortcut.
        twist_cmd_topic: Convenience shortcut.
    """

    physics_backend: Optional[str] = None   # real hardware
    array_runtime: str = "numpy"
    accepts: Tuple[str, ...] = ("joint_pos", "torque", "loco", "mixed")

    def __init__(
        self,
        spec: Any = None,
        config: Optional[ROS2IOConfig] = None,
        *,
        joint_state_topic: Optional[str] = None,
        joint_cmd_topic: Optional[str] = None,
        odom_topic: Optional[str] = None,
        twist_cmd_topic: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> None:
        if not _ROS2_AVAILABLE:
            raise ImportError(
                "ROS2RobotIO requires rclpy + sensor_msgs + nav_msgs. "
                "Source a ROS2 workspace and ensure rclpy is on PYTHONPATH."
            )

        self.spec = spec
        self._cfg = config or ROS2IOConfig()
        if joint_state_topic is not None:
            self._cfg.joint_state_topic = joint_state_topic
        if joint_cmd_topic is not None:
            self._cfg.joint_cmd_topic = joint_cmd_topic
        if odom_topic is not None:
            self._cfg.odom_topic = odom_topic
        if twist_cmd_topic is not None:
            self._cfg.twist_cmd_topic = twist_cmd_topic
        if node_name is not None:
            self._cfg.node_name = node_name

        # Cached state (updated by subscriber callbacks)
        self._lock = threading.Lock()
        self._joint_pos: Optional[np.ndarray] = None
        self._joint_vel: Optional[np.ndarray] = None
        self._joint_effort: Optional[np.ndarray] = None
        self._joint_names_from_msg: Optional[list[str]] = None
        self._base_pose: Optional[np.ndarray] = None   # (x,y,z,qw,qx,qy,qz)
        self._base_twist: Optional[np.ndarray] = None  # (vx,vy,vz,wx,wy,wz)
        self._t: float = 0.0
        self._msg_count: int = 0

        # Pending command (latched by send_control, dispatched in step)
        self._pending_cmd: Optional[ControlCommand] = None

        # ROS2 node / executor
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False

        self._pub_joint: Any = None
        self._pub_twist: Any = None

    # ------------------------------------------------------------------
    # RobotIO protocol
    # ------------------------------------------------------------------

    def reset(self) -> RobotState:
        """Connect to ROS2, start subscribers, wait for first state message."""
        self._init_ros()

        # Wait for first joint state message.
        deadline = time.monotonic() + self._cfg.state_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._joint_pos is not None:
                    break
            time.sleep(0.05)
        else:
            raise TimeoutError(
                f"No message received on {self._cfg.joint_state_topic} "
                f"within {self._cfg.state_timeout_s}s. Is the ROS2 driver running?"
            )

        self._pending_cmd = None
        return self._build_state()

    def get_state(self) -> RobotState:
        return self._build_state()

    def send_control(self, cmd: ControlCommand) -> None:
        """Latch a command — dispatched to ROS2 topics on the next :meth:`step`."""
        self._pending_cmd = cmd

    def step(self, dt: float) -> RobotState:
        """Publish the pending command and rate-limit to ``dt`` seconds."""
        t0 = time.monotonic()

        if self._pending_cmd is not None:
            self._dispatch_command(self._pending_cmd)
            self._pending_cmd = None

        elapsed = time.monotonic() - t0
        remaining = float(dt) - elapsed
        if remaining > 0:
            time.sleep(remaining)

        with self._lock:
            self._t += float(dt)

        return self._build_state()

    def close(self) -> None:
        """Destroy the ROS2 node and stop background threads."""
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        try:
            rclpy.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_ros(self) -> None:
        if self._node is not None:
            return

        if not rclpy.ok():
            rclpy.init()

        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState
        from geometry_msgs.msg import Twist

        node = Node(self._cfg.node_name)
        self._node = node

        # Publishers
        self._pub_joint = node.create_publisher(JointState, self._cfg.joint_cmd_topic, 10)
        self._pub_twist = node.create_publisher(Twist, self._cfg.twist_cmd_topic, 10)

        # Subscribers
        node.create_subscription(JointState, self._cfg.joint_state_topic, self._on_joint_state, 1)
        if self._cfg.odom_topic:
            from nav_msgs.msg import Odometry
            node.create_subscription(Odometry, self._cfg.odom_topic, self._on_odom, 1)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop, daemon=True, name="ros2-io-spin"
        )
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        while self._running and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.005)

    def _on_joint_state(self, msg: Any) -> None:
        names = list(msg.name) if msg.name else []
        pos = np.asarray(msg.position, dtype=np.float64) if msg.position else np.zeros(0)
        vel = np.asarray(msg.velocity, dtype=np.float64) if msg.velocity else np.zeros(len(pos))
        eff = np.asarray(msg.effort, dtype=np.float64) if msg.effort else np.zeros(len(pos))

        desired = self._cfg.joint_names
        if desired is not None and names:
            name_to_idx = {n: i for i, n in enumerate(names)}
            idx = [name_to_idx.get(n) for n in desired]
            valid = [i for i in idx if i is not None]
            pos = np.array([pos[i] if i is not None else 0.0 for i in idx], dtype=np.float64)
            vel = np.array([vel[i] if i is not None and i < len(vel) else 0.0 for i in idx], dtype=np.float64)
            eff = np.array([eff[i] if i is not None and i < len(eff) else 0.0 for i in idx], dtype=np.float64)

        stamp = msg.header.stamp
        t = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        with self._lock:
            self._joint_pos = pos
            self._joint_vel = vel
            self._joint_effort = eff
            self._joint_names_from_msg = names
            self._msg_count += 1
            if t > 0:
                self._t = t

    def _on_odom(self, msg: Any) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        with self._lock:
            # MuJoCo convention: qpos = (x,y,z, qw,qx,qy,qz)
            self._base_pose = np.array(
                [pos.x, pos.y, pos.z, ori.w, ori.x, ori.y, ori.z], dtype=np.float64
            )
            self._base_twist = np.array(
                [lv.x, lv.y, lv.z, av.x, av.y, av.z], dtype=np.float64
            )

    def _dispatch_command(self, cmd: ControlCommand) -> None:
        if cmd.kind in ("joint_pos", "torque", "mixed") and (
            cmd.joint_pos is not None or cmd.joint_torque is not None
        ):
            self._publish_joint_cmd(cmd)
        if cmd.kind in ("loco", "mixed") and cmd.loco_cmd is not None:
            self._publish_twist_cmd(cmd)

    def _publish_joint_cmd(self, cmd: ControlCommand) -> None:
        from sensor_msgs.msg import JointState
        msg = JointState()
        from builtin_interfaces.msg import Time as RosTime
        wall = time.time()
        stamp = RosTime()
        stamp.sec = int(wall)
        stamp.nanosec = int((wall - stamp.sec) * 1e9)
        msg.header.stamp = stamp
        msg.header.frame_id = ""

        names = self._cfg.joint_names or []
        if cmd.joint_pos is not None:
            arr = np.asarray(cmd.joint_pos, dtype=np.float64).reshape(-1)
            msg.name = names[:len(arr)] if names else [f"joint_{i}" for i in range(len(arr))]
            msg.position = arr.tolist()
        if cmd.joint_torque is not None:
            arr = np.asarray(cmd.joint_torque, dtype=np.float64).reshape(-1)
            msg.effort = arr.tolist()
        self._pub_joint.publish(msg)

    def _publish_twist_cmd(self, cmd: ControlCommand) -> None:
        from geometry_msgs.msg import Twist
        loco = cmd.loco_cmd
        msg = Twist()
        msg.linear.x = float(loco.vx)
        msg.linear.y = float(loco.vy)
        msg.angular.z = float(loco.yaw_rate)
        if loco.body_height is not None:
            msg.linear.z = float(loco.body_height)
        self._pub_twist.publish(msg)

    def _build_state(self) -> RobotState:
        with self._lock:
            t = float(self._t)
            qpos = self._joint_pos.copy() if self._joint_pos is not None else np.zeros(0)
            qvel = self._joint_vel.copy() if self._joint_vel is not None else np.zeros(len(qpos))
            effort = self._joint_effort.copy() if self._joint_effort is not None else None
            base_pose = self._base_pose.copy() if self._base_pose is not None else None
            base_twist = self._base_twist.copy() if self._base_twist is not None else None

        return RobotState(
            t=t,
            qpos=qpos,
            qvel=qvel,
            joint_torque=effort,
            base_pose=base_pose,
            base_twist=base_twist,
        )

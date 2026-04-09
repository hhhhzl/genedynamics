"""ROS2 publisher observer — broadcasts deploy state and commands to ROS2.

Publishes every control step to two topics:

* ``state_topic`` (``sensor_msgs/JointState``) — joint positions, velocities,
  and efforts from :class:`RobotState`.
* ``cmd_topic`` (``std_msgs/Float64MultiArray``) — the raw joint command
  vector from :class:`ControlCommand`.

Additionally publishes:

* ``base_pose_topic`` (``geometry_msgs/PoseStamped``) — base pose when
  ``state.base_pose`` is available.

This lets standard ROS2 tooling (RViz, ros2 bag, PlotJuggler) consume deploy
telemetry without any code changes to the runner.

Usage::

    obs = ROS2PublisherObserver(
        state_topic="/deploy/robot_state",
        cmd_topic="/deploy/control_cmd",
        node_name="deploy_publisher",
    )
    # plug into a preset's observers list or build manually

Requires ``rclpy``, ``sensor_msgs``, ``geometry_msgs``, ``std_msgs`` (ROS2).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping, Optional

import numpy as np

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)
from genedynamics.deploy.observers.base import BaseObserver

__all__ = ["ROS2PublisherObserver"]

_ROS2_AVAILABLE = False
try:
    import rclpy  # noqa: F401
    _ROS2_AVAILABLE = True
except ImportError:
    pass


class ROS2PublisherObserver(BaseObserver):
    """Observer that publishes robot state and commands to ROS2 topics.

    Args:
        state_topic: ROS2 topic for ``sensor_msgs/JointState``.
        cmd_topic: ROS2 topic for ``std_msgs/Float64MultiArray`` (joint cmds).
        base_pose_topic: ROS2 topic for ``geometry_msgs/PoseStamped``.
        node_name: Name of the rclpy node created internally.
        frame_id: ``header.frame_id`` used for all stamped messages.
        joint_names: Optional list of joint names for the JointState message.
            When ``None`` generic names ``["joint_0", "joint_1", ...]`` are
            generated from the array length.
        name: Display name for this observer.
    """

    def __init__(
        self,
        state_topic: str = "/deploy/robot_state",
        cmd_topic: str = "/deploy/control_cmd",
        base_pose_topic: str = "/deploy/base_pose",
        node_name: str = "deploy_publisher",
        frame_id: str = "world",
        joint_names: Optional[list[str]] = None,
        name: str = "ros2_publisher",
    ) -> None:
        super().__init__(name=name)

        if not _ROS2_AVAILABLE:
            raise ImportError(
                "ROS2PublisherObserver requires rclpy + sensor_msgs + geometry_msgs. "
                "Source a ROS2 workspace first."
            )

        self._state_topic = state_topic
        self._cmd_topic = cmd_topic
        self._base_pose_topic = base_pose_topic
        self._node_name = node_name
        self._frame_id = frame_id
        self._joint_names = joint_names

        self._node: Any = None
        self._pub_state: Any = None
        self._pub_cmd: Any = None
        self._pub_pose: Any = None
        self._executor: Any = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False

    def on_episode_start(self, episode_id: str, metadata: Mapping[str, Any]) -> None:
        self._init_ros()

    def on_step(
        self,
        t: float,
        state: RobotState,
        intent: Intent,
        cmd: ControlCommand,
        info: StepInfo,
    ) -> None:
        if self._node is None:
            return
        try:
            self._publish_state(t, state)
            self._publish_cmd(t, cmd)
            if state.base_pose is not None:
                self._publish_base_pose(t, state)
        except Exception as exc:
            # Never crash the runner due to ROS publishing errors.
            print(f"[ROS2PublisherObserver] publish error: {exc}")

    def on_episode_end(self, summary: Mapping[str, Any]) -> None:
        self._shutdown_ros()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_ros(self) -> None:
        if self._node is not None:
            return  # already running

        if not rclpy.ok():
            rclpy.init()

        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState          # noqa: F811
        from std_msgs.msg import Float64MultiArray      # noqa: F811
        from geometry_msgs.msg import PoseStamped       # noqa: F811

        node = Node(self._node_name)
        self._node = node
        self._pub_state = node.create_publisher(JointState, self._state_topic, 10)
        self._pub_cmd = node.create_publisher(Float64MultiArray, self._cmd_topic, 10)
        self._pub_pose = node.create_publisher(PoseStamped, self._base_pose_topic, 10)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop, daemon=True, name="ros2-pub-spin"
        )
        self._spin_thread.start()

    def _shutdown_ros(self) -> None:
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            self._spin_thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _spin_loop(self) -> None:
        while self._running and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.01)

    def _stamp(self, t: float) -> Any:
        from rclpy.clock import ClockType
        from rclpy.time import Time
        # Use wall time if t <= 0, else encode as nanoseconds.
        wall = time.time()
        sec = int(wall)
        nanosec = int((wall - sec) * 1e9)
        from builtin_interfaces.msg import Time as RosTime
        stamp = RosTime()
        stamp.sec = sec
        stamp.nanosec = nanosec
        return stamp

    def _publish_state(self, t: float, state: RobotState) -> None:
        from sensor_msgs.msg import JointState
        msg = JointState()
        msg.header.stamp = self._stamp(t)
        msg.header.frame_id = self._frame_id

        if state.qpos is not None:
            q = np.asarray(state.qpos, dtype=np.float64).reshape(-1)
            n = q.shape[0]
            names = self._joint_names or [f"joint_{i}" for i in range(n)]
            msg.name = list(names[:n])
            msg.position = q[:n].tolist()
        if state.qvel is not None:
            qv = np.asarray(state.qvel, dtype=np.float64).reshape(-1)
            msg.velocity = qv.tolist()
        if state.joint_torque is not None:
            tau = np.asarray(state.joint_torque, dtype=np.float64).reshape(-1)
            msg.effort = tau.tolist()

        self._pub_state.publish(msg)

    def _publish_cmd(self, t: float, cmd: ControlCommand) -> None:
        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        if cmd.joint_pos is not None:
            msg.data = np.asarray(cmd.joint_pos, dtype=np.float64).reshape(-1).tolist()
        elif cmd.joint_torque is not None:
            msg.data = np.asarray(cmd.joint_torque, dtype=np.float64).reshape(-1).tolist()
        self._pub_cmd.publish(msg)

    def _publish_base_pose(self, t: float, state: RobotState) -> None:
        from geometry_msgs.msg import PoseStamped
        pose = np.asarray(state.base_pose, dtype=np.float64).reshape(-1)
        if pose.shape[0] < 7:
            return
        msg = PoseStamped()
        msg.header.stamp = self._stamp(t)
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = float(pose[2])
        # MuJoCo / deploy quat convention: (w, x, y, z)
        # ROS2 geometry_msgs convention: (x, y, z, w)
        msg.pose.orientation.w = float(pose[3])
        msg.pose.orientation.x = float(pose[4])
        msg.pose.orientation.y = float(pose[5])
        msg.pose.orientation.z = float(pose[6])
        self._pub_pose.publish(msg)

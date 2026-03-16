"""
ROS2 Odometry localization plugin.

Subscribes to nav_msgs/Odometry and provides pose/velocity in world frame.
Requires: rclpy, nav_msgs (ROS2).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.localization.base_plugin import BaseLocalizationPlugin

ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from scipy.spatial.transform import Rotation as R
    ROS2_AVAILABLE = True
except ImportError:
    pass


class ROS2OdometryPlugin(BaseLocalizationPlugin):
    """Localization from ROS2 Odometry topic."""

    def __init__(self, config: Dict[str, Any]) -> None:
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2OdometryPlugin requires rclpy and nav_msgs. Install ROS2.")
        super().__init__(config)
        self._qpos: Optional[np.ndarray] = None
        self._qvel: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None
        self._node: Optional["Node"] = None
        self._odom_topic = config.get("odom_topic", "/odom")
        self._init_ros()

    def _init_ros(self) -> None:
        rclpy.init()
        self._node = _OdomNode(self._odom_topic, self._on_odom)
        from rclpy.executors import SingleThreadedExecutor
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = __import__("threading").Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _on_odom(self, msg: Any) -> None:
        qpos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        ], dtype=np.float64)
        vb = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ], dtype=np.float64)
        ab = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ], dtype=np.float64)
        q = R.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]])
        vw = q.apply(vb)
        aw = q.apply(ab)
        self._qpos = qpos
        self._qvel = np.concatenate([vw, aw])
        self._last_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _spin_loop(self) -> None:
        while self._node and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.01)

    def get_state(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._qpos is None or self._qvel is None:
            return None
        return self._qpos.copy(), self._qvel.copy()

    def get_last_update_time(self) -> Optional[float]:
        return self._last_time

    def shutdown(self) -> None:
        if self._node:
            self._node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if ROS2_AVAILABLE:
    class _OdomNode(Node):
        def __init__(self, topic: str, callback) -> None:
            super().__init__("ros2_odom_plugin")
            self.sub = self.create_subscription(Odometry, topic, callback, 1)

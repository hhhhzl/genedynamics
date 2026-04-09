"""ROS2 Twist input source.

Subscribes to a ``geometry_msgs/Twist`` topic and exposes the latest message
as a :class:`TeleopCommand`. This lets any ROS2 publisher (joystick_drivers,
teleop_twist_keyboard, Nav2, etc.) drive the deploy loop without modification.

Topic mapping::

    Twist.linear.x  → TeleopCommand.vx
    Twist.linear.y  → TeleopCommand.vy
    Twist.angular.z → TeleopCommand.yaw_rate
    Twist.linear.z  → TeleopCommand.body_height_delta

An optional ``std_msgs/Bool`` stop topic can be subscribed to forward
emergency-stop signals from external safety nodes.

Usage::

    src = ROS2TwistSource(twist_topic="/cmd_vel", node_name="deploy_teleop")
    src.open()         # spins background executor thread
    cmd = src.read()   # non-blocking
    src.close()

Requires ``rclpy`` and ``geometry_msgs`` (installed with ROS2).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from genedynamics.deploy.teleop.input_source import TeleopCommand

__all__ = ["ROS2TwistSource"]

_ROS2_AVAILABLE = False
try:
    import rclpy  # noqa: F401
    _ROS2_AVAILABLE = True
except ImportError:
    pass


class ROS2TwistSource:
    """ROS2 Twist subscriber as an :class:`InputSource`.

    Args:
        twist_topic: ROS2 topic name for ``geometry_msgs/Twist``.
        stop_topic: Optional ``std_msgs/Bool`` topic. When ``True`` is
            received the command's ``stop`` flag is latched until ``reset()``.
        node_name: Name of the rclpy node created internally.
        timeout_s: If no Twist message is received for this many seconds,
            ``TeleopCommand.stop`` is set (device lost). Set to ``None`` to
            disable timeout.
        vx_scale / vy_scale / yaw_scale / height_scale: Optional gain applied
            to each axis after reading from the topic.
    """

    name: str = "ros2_twist"

    def __init__(
        self,
        twist_topic: str = "/cmd_vel",
        stop_topic: Optional[str] = None,
        node_name: str = "deploy_teleop_source",
        timeout_s: Optional[float] = 1.0,
        vx_scale: float = 1.0,
        vy_scale: float = 1.0,
        yaw_scale: float = 1.0,
        height_scale: float = 1.0,
    ) -> None:
        self._twist_topic = twist_topic
        self._stop_topic = stop_topic
        self._node_name = node_name
        self._timeout_s = timeout_s
        self._vx_scale = float(vx_scale)
        self._vy_scale = float(vy_scale)
        self._yaw_scale = float(yaw_scale)
        self._height_scale = float(height_scale)

        self._lock = threading.Lock()
        self._latest = TeleopCommand()
        self._last_msg_time: Optional[float] = None
        self._stop_latched = False

        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False

    def open(self) -> None:
        """Initialise rclpy and start the spin thread."""
        if not _ROS2_AVAILABLE:
            raise ImportError(
                "ROS2TwistSource requires rclpy + geometry_msgs. "
                "Source a ROS2 workspace and ensure rclpy is on PYTHONPATH."
            )
        if not rclpy.ok():
            rclpy.init()

        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from geometry_msgs.msg import Twist

        class _TeleopNode(Node):
            def __init__(inner_self, parent: "ROS2TwistSource") -> None:
                super().__init__(parent._node_name)
                inner_self._parent = parent
                inner_self.create_subscription(Twist, parent._twist_topic, parent._on_twist, 1)
                if parent._stop_topic:
                    from std_msgs.msg import Bool
                    inner_self.create_subscription(Bool, parent._stop_topic, parent._on_stop, 1)

        self._node = _TeleopNode(self)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True, name="ros2-teleop-spin")
        self._spin_thread.start()

    def read(self) -> TeleopCommand:
        """Return the latest command (non-blocking)."""
        with self._lock:
            timed_out = (
                self._timeout_s is not None
                and self._last_msg_time is not None
                and (time.monotonic() - self._last_msg_time) > self._timeout_s
            )
            stop = self._stop_latched or timed_out
            return TeleopCommand(
                vx=self._latest.vx,
                vy=self._latest.vy,
                yaw_rate=self._latest.yaw_rate,
                body_height_delta=self._latest.body_height_delta,
                stop=stop,
            )

    def reset_stop(self) -> None:
        """Clear the latched E-stop flag."""
        with self._lock:
            self._stop_latched = False

    def close(self) -> None:
        """Shut down the spin thread and destroy the node."""
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_twist(self, msg: Any) -> None:
        with self._lock:
            self._latest = TeleopCommand(
                vx=float(msg.linear.x) * self._vx_scale,
                vy=float(msg.linear.y) * self._vy_scale,
                yaw_rate=float(msg.angular.z) * self._yaw_scale,
                body_height_delta=float(msg.linear.z) * self._height_scale,
            )
            self._last_msg_time = time.monotonic()

    def _on_stop(self, msg: Any) -> None:
        with self._lock:
            if bool(msg.data):
                self._stop_latched = True

    def _spin_loop(self) -> None:
        while self._running and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.01)

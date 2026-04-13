"""G1 teleoperation preset — real hardware (Unitree SDK).

Uses :class:`~genedynamics.deploy.io.unitree_g1_io.UnitreeG1RobotIO` for real
hardware and adds the ROS2 publisher observer for live monitoring via RViz /
rosbag.

**Before running on real hardware:**

1. Connect to the G1 over Ethernet (default ``192.168.123.161``).
2. Source your ROS2 workspace (for the :class:`ROS2PublisherObserver`).
3. Start teleoperation input::

       # gamepad
       python -m genedynamics.deploy.scripts.teleop_real --input gamepad

       # ROS2 topic (from any external publisher)
       ros2 topic pub /cmd_vel geometry_msgs/Twist ...

4. Press the E-stop button or Esc to end the episode safely.

Safety layer order: workspace clamp → joint limit → torque limit.  The
workspace filter fires first so the controller never sees a command that
would drive the end-effector out of bounds.
"""

from __future__ import annotations

from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig

__all__ = ["G1TeleopRealPreset"]


class G1TeleopRealPreset(DeployConfig):
    """G1 teleoperation — real Unitree G1 hardware."""

    control_hz: float = 50.0
    sim_dt: float = 0.02          # 50 Hz; on real HW step() is a rate-limiter
    max_steps: int = 100_000
    base_height: float = 0.75

    class runtime(ComponentConfig):
        name = "numpy"

    class robot(ComponentConfig):
        robot_type = "humanoid"
        model_id = "g1"

    class io(ComponentConfig):
        registry_key = "io.unitree_g1"
        robot_ip = "192.168.123.161"
        control_hz = 50.0

    class controller(ComponentConfig):
        registry_key = "controller.sport_mode"
        leg_kp = 100.0
        leg_kd = 4.0
        upper_body_kp = 80.0
        upper_body_kd = 3.0

        class loco_client(ComponentConfig):
            registry_key = "loco_client.real"  # Unitree LocoClient SDK

    class safety(ComponentConfig):
        registry_key = "safety.composite"
        filters = [
            # Workspace clamp FIRST — prevents controller from chasing
            # unreachable end-effector targets.
            {"registry_key": "safety.workspace",
             "x_range": [-0.6, 0.6], "y_range": [-0.5, 0.5], "z_range": [0.0, 0.5]},
            # Joint limits with tighter margin than sim.
            {"registry_key": "safety.joint_limit", "margin": 0.05},
            # Torque at 85 % of hardware limit (conservative for real).
            {"registry_key": "safety.torque_limit", "safety_margin": 0.85},
        ]

    class task(ComponentConfig):
        registry_key = "task.teleop"
        fall_height_m = 0.30
        fall_angle_deg = 35.0    # tighter than sim — stop earlier on real HW

    observers = [
        {
            "registry_key": "observer.logger",
            "out_dir": "results/teleop/g1_real",
        },
        {
            "registry_key": "observer.recorder",
            "out_dir": "results/teleop/g1_real",
        },
        {
            "registry_key": "observer.ros2_publisher",
            "state_topic": "/deploy/robot_state",
            "cmd_topic": "/deploy/control_cmd",
            "node_name": "deploy_teleop_publisher",
        },
    ]

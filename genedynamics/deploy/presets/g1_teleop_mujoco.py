"""G1 teleoperation preset — MuJoCo simulation.

Use this for debugging teleoperation control logic in sim before putting it
on real hardware.  Swap ``G1TeleopRealPreset`` (same file) for the real G1.

The preset does **not** hardcode the input source — choose one at runtime::

    from genedynamics.deploy.presets.g1_teleop_mujoco import G1TeleopMujocoPreset
    from genedynamics.deploy.teleop import TeleopFollower, GamepadSource
    from genedynamics.deploy.runner import run_preset

    follower = TeleopFollower(
        source=GamepadSource(vx_max=0.4),
        base_height_nominal=0.75,
    )
    # Monkey-patch the preset or build components manually and inject follower.
    # For a hand-rolled loop use build_from_preset() + inject follower directly.
    run_preset(G1TeleopMujocoPreset)
"""

from __future__ import annotations

from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig

__all__ = ["G1TeleopMujocoPreset"]


class G1TeleopMujocoPreset(DeployConfig):
    """G1 teleoperation (sim, SportMode controller, keyboard or gamepad)."""

    control_hz: float = 50.0
    sim_dt: float = 1.0 / 500.0
    max_steps: int = 100_000      # very long; episode ends via E-stop or fall
    base_height: float = 0.75

    class runtime(ComponentConfig):
        name = "numpy"

    class robot(ComponentConfig):
        robot_type = "humanoid"
        model_id = "g1"

    class io(ComponentConfig):
        registry_key = "io.mujoco"
        sim_dt = 1.0 / 500.0
        keyframe_name = "stand"

    class controller(ComponentConfig):
        registry_key = "controller.sport_mode"
        leg_kp = 100.0
        leg_kd = 4.0
        upper_body_kp = 80.0
        upper_body_kd = 3.0

        class loco_client(ComponentConfig):
            registry_key = "loco_client.spark_rl"
            nominal_step_period = 0.8
            cmd_clip = 0.3

    class safety(ComponentConfig):
        registry_key = "safety.composite"
        filters = [
            {"registry_key": "safety.joint_limit", "margin": 0.02},
            {"registry_key": "safety.torque_limit", "safety_margin": 0.95},
            {"registry_key": "safety.workspace", "x_range": [-0.6, 0.6],
             "y_range": [-0.5, 0.5], "z_range": [0.0, 0.5]},
        ]

    class task(ComponentConfig):
        registry_key = "task.teleop"
        fall_height_m = 0.30
        fall_angle_deg = 40.0

    observers = [
        {
            "registry_key": "observer.logger",
            "out_dir": "results/teleop/g1_mujoco",
        },
    ]

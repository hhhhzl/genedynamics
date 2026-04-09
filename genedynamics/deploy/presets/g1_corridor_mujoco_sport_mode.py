"""G1 corridor + sport-mode controller running in MuJoCo.

This is the canonical sim baseline that the corridor diagnose scripts
validated end-to-end:

* IO: ``MujocoRobotIO`` (CPU NumPy MuJoCo)
* Controller: ``SportModeController`` driven by :class:`SparkRLLocoClient`
  (loads the bundled ``g1_motion.pt`` PPO policy and walks the legs at ~50 Hz)
* Safety: composite (joint + torque limits)
* Observers: logger + recorder

Inherit and override one inner class to switch any layer. The Real-G1
sibling preset (:mod:`genedynamics.deploy.presets.g1_corridor_real_sport_mode`)
is exactly that — three lines of overrides.
"""

from __future__ import annotations

from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig

__all__ = ["G1CorridorMujocoSportModePreset"]


class G1CorridorMujocoSportModePreset(DeployConfig):
    """Default G1 narrow-corridor preset (sim, sport-mode controller)."""

    control_hz: float = 50.0
    sim_dt: float = 1.0 / 500.0
    max_steps: int = 5_000

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
            # nominal_step_period matches spark's training-time phase clock
            nominal_step_period = 0.8
            cmd_clip = 0.3

    class safety(ComponentConfig):
        registry_key = "safety.composite"
        filters = [
            {"registry_key": "safety.joint_limit", "margin": 0.02},
            {"registry_key": "safety.torque_limit", "safety_margin": 0.95},
        ]

    observers = [
        {"registry_key": "observer.logger", "out_dir": "results/g1_corridor/sport_spark_rl"},
        {"registry_key": "observer.recorder", "out_dir": "results/g1_corridor/sport_spark_rl"},
    ]

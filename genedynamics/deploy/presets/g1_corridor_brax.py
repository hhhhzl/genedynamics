"""G1 corridor + sport-mode controller running in Brax (GPU JAX).

Identical structure to :class:`G1CorridorMujocoSportModePreset` except the
IO layer uses :class:`BraxRobotIO` with ``physics_backend="brax"`` and
``array_runtime="jax"``. The WBC / sport-mode controller stays numpy; the
Phase 16 :class:`ArrayBridge` handles jax→numpy conversion transparently.

Contact observations from Brax are less rich than MuJoCo's per-geom
wrenches — see :mod:`genedynamics.deploy.io.brax_io` docstring for details.
"""

from __future__ import annotations

from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig

__all__ = ["G1CorridorBraxPreset"]


class G1CorridorBraxPreset(DeployConfig):
    """G1 narrow-corridor preset using Brax MJX backend."""

    control_hz: float = 50.0
    sim_dt: float = 1.0 / 500.0
    max_steps: int = 5_000

    class runtime(ComponentConfig):
        name = "jax"

    class robot(ComponentConfig):
        robot_type = "humanoid"
        model_id = "g1"

    class io(ComponentConfig):
        registry_key = "io.brax"
        env_name = "humanoid"
        backend = "mjx"
        sim_dt = 1.0 / 500.0
        seed = 0

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
        ]

    observers = [
        {"registry_key": "observer.logger", "out_dir": "results/g1_corridor/brax"},
        {"registry_key": "observer.recorder", "out_dir": "results/g1_corridor/brax"},
    ]

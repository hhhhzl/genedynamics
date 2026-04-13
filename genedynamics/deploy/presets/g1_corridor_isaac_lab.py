"""G1 corridor + RL controller running in Isaac Lab (GPU torch).

Uses :class:`IsaacLabRobotIO` with ``physics_backend="isaac_lab"`` and
``array_runtime="torch"``. The RL controller
(:class:`UnitreeRLGymG1Controller`) is natively torch-runtime, so **no
Phase 16 bridge is needed** for the controller path — both declare
``"torch"`` and the runtime check passes directly.

Safety filters stay numpy-based; the Phase 16 :class:`ArrayBridge`
transparently handles the torch→numpy→torch conversion at the safety
boundary.
"""

from __future__ import annotations

from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig

__all__ = ["G1CorridorIsaacLabPreset"]


class G1CorridorIsaacLabPreset(DeployConfig):
    """G1 narrow-corridor preset using Isaac Lab GPU backend."""

    control_hz: float = 50.0
    sim_dt: float = 1.0 / 200.0
    max_steps: int = 5_000

    class runtime(ComponentConfig):
        name = "torch"

    class robot(ComponentConfig):
        robot_type = "humanoid"
        model_id = "g1"

    class io(ComponentConfig):
        registry_key = "io.isaac_lab"
        task_name = "Isaac-Velocity-Flat-G1-v0"
        device = "cuda:0"
        sim_dt = 1.0 / 200.0
        headless = True

    class controller(ComponentConfig):
        registry_key = "controller.rl_unitree_rl_gym"

    class safety(ComponentConfig):
        registry_key = "safety.composite"
        filters = [
            {"registry_key": "safety.joint_limit", "margin": 0.02},
            {"registry_key": "safety.torque_limit", "safety_margin": 0.95},
        ]

    observers = [
        {"registry_key": "observer.logger", "out_dir": "results/g1_corridor/isaac_lab"},
        {"registry_key": "observer.recorder", "out_dir": "results/g1_corridor/isaac_lab"},
    ]

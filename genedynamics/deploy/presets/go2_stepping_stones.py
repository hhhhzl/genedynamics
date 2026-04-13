"""Go2 stepping-stones walk preset.

Uses :class:`~genedynamics.deploy.controllers.quadruped_stepping.QuadrupedSteppingController`
which is a *batch* controller that owns its own internal MuJoCo simulation.
It therefore does **not** go through the standard ``runner.py`` loop;
scripts call ``controller.rollout(plan_states)`` directly.

This preset serves two purposes:

1. **Config container** — scripts can instantiate
   :class:`MinimalFollowerConfig` directly from ``Go2SteppingStonesMujocoPreset``
   parameters without hard-coding magic numbers.

2. **Registry registration target** — ``"controller.quadruped_stepping"``
   is registered in :mod:`~genedynamics.deploy.registries` so that future
   pipeline extensions can build the controller via the registry.

Typical usage in a script::

    from genedynamics.deploy.presets.go2_stepping_stones import (
        Go2SteppingStonesMujocoPreset as P,
    )
    from genedynamics.deploy.controllers.quadruped_stepping import (
        MinimalFollowerConfig,
        QuadrupedSteppingController,
        load_stepping_plan_from_seed_dir,
    )

    cfg = MinimalFollowerConfig(
        gait=P.walker.gait,
        sim_dt=P.walker.sim_dt,
        phase_steps=P.walker.phase_steps,
        swing_height=P.walker.swing_height,
        step_width=P.walker.step_width,
    )
    ctrl = QuadrupedSteppingController(cfg=cfg, stepping_scene=scene_dict)
    plan = load_stepping_plan_from_seed_dir(seed_dir, step_width=P.walker.step_width)
    result = ctrl.rollout(plan["states"])
"""

from __future__ import annotations

from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig

__all__ = ["Go2SteppingStonesMujocoPreset"]


class Go2SteppingStonesMujocoPreset(DeployConfig):
    """Default Go2 stepping-stones preset (batch, internal-sim controller)."""

    # control_hz / sim_dt are unused by the batch controller but kept for
    # documentation parity with the other presets.
    control_hz: float = 100.0     # 1 / sim_dt
    sim_dt: float = 0.01
    max_steps: int = 20_000       # upper bound on total sim steps per rollout

    class runtime(ComponentConfig):
        name = "numpy"

    class robot(ComponentConfig):
        robot_type = "quadruped"
        model_id = "go2"

    class controller(ComponentConfig):
        registry_key = "controller.quadruped_stepping"

    class walker(ComponentConfig):
        """Parameters forwarded to :class:`MinimalFollowerConfig`."""
        gait = "walk"               # "walk" | "trot"
        sim_dt = 0.01
        phase_steps = 20
        settle_steps = 30
        # Geometry
        step_width = 0.30
        leg_half_length = 0.18
        x_f_nominal = 0.18
        x_r_nominal = -0.18
        y_L_nominal = 0.15
        y_R_nominal = -0.15
        # Swing
        swing_height = 0.07
        base_height_offset = 0.02
        base_z_min = 0.22
        min_foot_z = 0.015
        # Speed
        uniform_base_speed = True
        base_speed_mps = 0.16

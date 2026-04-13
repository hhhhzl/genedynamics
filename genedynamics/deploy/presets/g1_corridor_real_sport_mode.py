"""Real-G1 sibling of :class:`G1CorridorMujocoSportModePreset`.

The Phase 7 contract: switching from sim to real should be one inner-class
override per changed layer. Concretely, this preset overrides only:

* ``io``           — ``unitree_g1`` instead of ``mujoco`` (with localization plugin)
* ``loco_client``  — ``real`` instead of ``mock``
* ``observers``    — write under a real-mode results dir

Everything else (controller gains, safety filters, control_hz) inherits
from the sim preset. Run with::

    python -m genedynamics.deploy.runner \\
        --preset genedynamics.deploy.presets:G1CorridorRealSportModePreset

Localization is wired by attaching a plugin instance under
``io.localization``. The default below uses
:class:`Ros2OdometryLocalizationPlugin`; swap to ``MockLocalizationPlugin``
for hardware-in-the-loop dry runs that have no external pose source.
"""

from __future__ import annotations


def _build_default_localization():
    """Construct a localization plugin lazily.

    Kept as a factory so the preset module imports cleanly on machines
    without ROS2 installed — the lookup only fails when a runner actually
    tries to instantiate the plugin via this preset.
    """
    try:
        from genedynamics.deploy.localization.ros2_odometry_plugin import (
            Ros2OdometryLocalizationPlugin,
        )

        return Ros2OdometryLocalizationPlugin(
            config={"topic": "/odom", "frame_id": "odom", "timeout_sec": 0.5}
        )
    except Exception:
        from genedynamics.deploy.localization.mock_plugin import (
            MockLocalizationPlugin,
        )

        return MockLocalizationPlugin(config={})


from genedynamics.deploy.config_schema import ComponentConfig, Lazy
from genedynamics.deploy.presets.g1_corridor_mujoco_sport_mode import (
    G1CorridorMujocoSportModePreset,
)

__all__ = ["G1CorridorRealSportModePreset"]


class G1CorridorRealSportModePreset(G1CorridorMujocoSportModePreset):
    """G1 corridor + sport-mode on real hardware."""

    class io(ComponentConfig):
        registry_key = "io.unitree_g1"
        network_interface = "eth0"
        domain_id = 0
        msc_mode = "sport"
        control_period_s = 0.02
        torque_safety_margin = 0.85
        # ``localization`` is materialized lazily by ``initialize_class`` —
        # the factory only runs when this preset is actually built, so the
        # preset module imports cleanly even on machines without ROS2.
        localization = Lazy(_build_default_localization)

    class controller(ComponentConfig):
        registry_key = "controller.sport_mode"
        leg_kp = 60.0
        leg_kd = 3.0
        upper_body_kp = 60.0
        upper_body_kd = 2.5

        class loco_client(ComponentConfig):
            registry_key = "loco_client.real"
            nominal_step_period = 0.6
            rate_limit_hz = 50.0

    observers = [
        {"registry_key": "observer.logger", "out_dir": "results/g1_corridor/real"},
        {"registry_key": "observer.recorder", "out_dir": "results/g1_corridor/real"},
    ]

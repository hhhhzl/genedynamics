# `deploy/presets/`

Presets are nested-class config classes inheriting from
[`DeployConfig`](../config_schema.py). Each preset declares what to build
(IO + controller + safety + task + observers); the
[`runner`](../runner.py) builds the components via
[`registries.py`](../registries.py) and runs the loop.

## Available presets

| Preset | Robot | IO | Controller | Loco client |
|---|---|---|---|---|
| [`G1CorridorMujocoSportModePreset`](g1_corridor_mujoco_sport_mode.py) | G1 humanoid | `io.mujoco` (CPU) | `controller.sport_mode` | `loco_client.mock` (template walker) |
| [`G1CorridorRealSportModePreset`](g1_corridor_real_sport_mode.py) | G1 humanoid | `io.unitree_g1` | `controller.sport_mode` | `loco_client.real` (Unitree SDK) |

The two presets are paired — the real one inherits from the sim one and
overrides exactly three inner classes (`io`, `controller`, `observers`).
This is the Phase 7 contract; the diff is enforced by
[`test_preset_inheritance.py`](../../../test/unit/deploy/test_preset_inheritance.py).

## Writing a preset

```python
from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig

class MyPreset(DeployConfig):
    control_hz = 50.0
    sim_dt = 1.0 / 500.0
    max_steps = 5_000

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
        # ...

    class safety(ComponentConfig):
        registry_key = "safety.composite"
        filters = [
            {"registry_key": "safety.joint_limit", "margin": 0.02},
            {"registry_key": "safety.torque_limit", "safety_margin": 0.95},
        ]

    class task(ComponentConfig):
        registry_key = "task.corridor_follow"
        goal_xy = (5.0, 0.0)
        goal_tolerance_m = 0.2

    observers = [
        {"registry_key": "observer.logger", "out_dir": "results/my_run"},
    ]
```

Run it with:

```bash
python -m genedynamics.deploy.runner \
    --preset module.path:MyPreset
```

## Inheritance pattern

To make a variant, subclass an existing preset and override just the inner
classes that change. Inherited classes carry through automatically — see
[`g1_corridor_real_sport_mode.py`](g1_corridor_real_sport_mode.py) for the
canonical sim → real flip.

## `Lazy(...)` for optional dependencies

When a field's value depends on an optional package (ROS2, Unitree SDK,
JAX), wrap a zero-arg factory in [`Lazy`](../config_schema.py):

```python
from genedynamics.deploy.config_schema import Lazy

class io(ComponentConfig):
    registry_key = "io.unitree_g1"
    localization = Lazy(_build_default_localization)
```

The factory only runs when the preset is **actually built**, so the preset
module imports cleanly on every machine — including dev laptops without
the optional package.

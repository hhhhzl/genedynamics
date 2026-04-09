# `deploy/controllers/`

Each controller implements the [`Controller`](../interfaces/controller.py)
protocol: it consumes a [`RobotState`](../interfaces/messages.py) + an
[`Intent`](../interfaces/messages.py) and produces a
[`ControlCommand`](../interfaces/messages.py). The runner enforces that the
controller's `runtime` matches the IO's `array_runtime`.

## Layout

| Directory | Controller | Runtime | Produces |
|---|---|---|---|
| [`wbc/`](wbc/) | `HumanoidWBCController` (split from the legacy 1273-line god file) | `numpy` (osqp) | `joint_pos` (with feedforward `joint_torque`) |
| [`sport_mode/`](sport_mode/) | `SportModeController` driven by a [`LocoClient`](sport_mode/loco_client.py) | `numpy` | `joint_pos` (sim with `MockLocoClient`) / `mixed` (real with `RealLocoClient`) |
| [`rl/`](rl/) | `UnitreeRLGymG1Controller`, `PassthroughRLController` | `torch` / `numpy` | `joint_pos` |

## Adding a new controller

1. Implement the protocol — `runtime`, `produces`, `reset(io)`, `act(state, intent) -> ControlCommand`.
2. Register in [`registries.py`](../registries.py) under `controller.<name>`.
3. Add a preset (or override one) referencing `controller.<name>`.

The bar is roughly: **one file under 400 lines plus one registry entry** to
land a new controller.

## Sport-mode loco-client subprotocol

The `SportModeController` is itself controller-agnostic — it delegates leg
control to a [`LocoClient`](sport_mode/loco_client.py). Two implementations
ship in this directory:

| LocoClient | Used by | Phase |
|---|---|---|
| [`MockLocoClient`](sport_mode/mock_loco_client.py) | Sim baseline (template walker — Bezier swing + closed-form leg IK) | 5 |
| [`RealLocoClient`](sport_mode/real_loco_client.py) | Real G1 (wraps Unitree SDK `LocoClient`) | 7 |

Switching between them is one config-line change of `loco_client.registry_key`
in the preset. The controller doesn't change.

## WBC subdirectory

The whole-body controller was deliberately split across seven files to make
each piece unit-testable in isolation:

```
wbc/
├── controller.py          # ~250 lines — protocol entry point
├── task_stack.py          # COM / pelvis / swing-foot / posture tasks
├── contact_blocks.py      # support contact equality blocks
├── friction_cone.py       # inequality constraints
├── qp_builder.py          # assemble (H, g, A, b, lb, ub)
├── qp_solver_adapter.py   # delegates to core/constraints/solvers
└── config.py              # 4 grouped sub-configs (replaces the legacy 92-field flat dataclass)
```

22 unit tests cover the WBC pieces; see
[`test/unit/deploy/test_wbc_*.py`](../../test/unit/deploy/).

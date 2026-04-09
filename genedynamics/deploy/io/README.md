# `deploy/io/`

Sim/real-symmetric robot IO adapters. Every adapter implements the
[`RobotIO`](../interfaces/robot_io.py) protocol; the deploy pipeline
([`runner.py`](../runner.py)) consumes them through that contract and never
branches on sim vs real.

## Files

| File | Backend | Array runtime | Accepts |
|---|---|---|---|
| [`base.py`](base.py) | — | — | abstract base, command latching, episode bookkeeping |
| [`mujoco_io.py`](mujoco_io.py) | `mujoco` (CPU) | `numpy` | `joint_pos`, `torque`, `mixed` |
| [`mjx_io.py`](mjx_io.py) | `mjx` (GPU/TPU) | `jax` | `joint_pos`, `torque`, `mixed` |
| [`unitree_g1_io.py`](unitree_g1_io.py) | `None` (real hardware) | `numpy` | `loco`, `joint_pos`, `torque`, `mixed` |
| [`stub_io.py`](stub_io.py) | `stub` | `numpy` | every kind |

## Adding a new IO adapter

1. Subclass [`BaseRobotIO`](base.py) and implement four hooks:
   - `_reset_robot()`     — restore initial pose
   - `_read_state(t)`     — produce a [`RobotState`](../interfaces/messages.py)
   - `_apply_command(cmd)` — write `ControlCommand` to the underlying physics/SDK
   - `_step_physics(dt)`  — advance time (no-op on real hardware)
2. Set the class-level `physics_backend`, `array_runtime`, and `accepts` tuple.
3. Register in [`registries.py`](../registries.py) under `io.<name>`.
4. Add a preset under [`presets/`](../presets/) referencing `io.<name>`.

## Pipeline contract

The runner asserts the active controller's `runtime` is compatible with the
IO's `array_runtime`. WBC (`numpy`) plus an MJX IO (`jax`) trips a host↔device
copy per step — acceptable for 200 Hz, painful at 1 kHz. Pair runtime backends
deliberately.

## Diagnostics

`MujocoRobotIO` exposes a small set of physics-query methods (mass matrix,
jacobians, contact wrenches) that controllers consume directly. They are
strictly read-only and never mutate `self.data`. `UnitreeG1RobotIO` exposes
`sdk_health` returning `("ok"|"stale"|"failed", lowstate_age_s)` for monitoring
the DDS link.

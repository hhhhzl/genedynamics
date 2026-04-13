# `deploy/safety/`

Safety filters sit between the controller and the IO. Each one implements the
[`SafetyFilter`](../interfaces/safety.py) protocol: take a `RobotState` +
`ControlCommand`, return a `SafetyResult` carrying a (possibly modified)
command + intervention metadata.

## Files

| File | What it clips | Stateful? |
|---|---|---|
| [`base.py`](base.py) | abstract base + no-op `reset` | — |
| [`joint_limit.py`](joint_limit.py) | `joint_pos` → `spec.joint_range` (with optional margin) | no |
| [`torque_limit.py`](torque_limit.py) | `joint_torque` → `actuator_forcerange × safety_margin` | no |
| [`self_collision.py`](self_collision.py) | `joint_pos` → retreat toward current qpos when capsule pairs are within `min_clearance` | no |
| [`cbf_filter.py`](cbf_filter.py) | `joint_pos` → projection onto `A u >= b` defined by a barrier callback (osqp → scipy fallback) | optional warm start |
| [`composite.py`](composite.py) | chain of filters in declaration order | propagates `reset` |

## Composing safety filters

Use [`CompositeSafetyFilter`](composite.py) — it's itself a `SafetyFilter`, so
the runner doesn't know whether one or six filters are active. Violations are
namespaced as `<FilterClass>.<key>` so observers can attribute interventions:

```python
result = composite.filter(state, cmd)
result.violations
# {'JointLimitFilter.joint_limit_max_rad': 0.21,
#  'TorqueLimitFilter.torque_max_nm':       12.0}
```

## Constraint-stack bridge

[`CBFFilter`](cbf_filter.py) is the first concrete bridge to
[`core/constraints/`](../../core/constraints/). It takes a user-supplied
`barrier_fn(state, cmd) -> (A, b)` and solves the projection QP locally. Once
the per-step convexify pipeline lands on the dispatch path, the filter's
`_solve` method can be swapped for `PerStepQPFilter` from
[`core/constraints/operators/qp/per_step_filter.py`](../../core/constraints/operators/qp/per_step_filter.py)
without changing the filter's external contract.

## Adding a filter

1. Subclass [`BaseSafetyFilter`](base.py) and implement `filter(state, cmd)`.
2. Override `reset()` only if you carry per-episode state.
3. Register under `safety.<name>` in [`registries.py`](../registries.py).
4. Reference it from a preset's `safety` section, either alone or as one entry
   inside a `safety.composite` filter list.

## Tests

- 15 unit tests in [`test_safety.py`](../../../test/unit/deploy/test_safety.py)
- 5 osqp-path tests in [`test_cbf_osqp_path.py`](../../../test/unit/deploy/test_cbf_osqp_path.py) (uses a fake `osqp.OSQP`)

# Humanoid WBC Controller — Standalone Project

## Status

**Mathematically correct, behaviourally broken.**

The QP-based inverse dynamics controller solves cleanly on the unit tests and on the
standing integration test (`test/integration/test_wbc_g1_standing.py`), but the
closed-loop corridor follower falls over within the first few steps on any real
`trajectory.json`.  This is a known tuning / scheduling issue, not a fundamental
algorithmic flaw.

## Architecture

```
CorridorPlanFrame (14D)
       │
       ▼
HumanoidContactScheduler   ── ContactObservations (from MujocoRobotIO)
       │
       ▼
HumanoidFootstepPlanner    ── foot IK + swing trajectories
       │
       ▼
HumanoidTaskBuilder        ── PelvisTask, FootTask × 2, joint_hints
       │
       ▼
  HumanoidWBCController    ── QP: min ||J ddq − a_des||² s.t. contact, friction, torque
       │
       ▼
  ControlCommand (tau_ff + q_ref + kp/kd)
```

The controller speaks the same `Controller.act(state, intent) → ControlCommand` protocol
as every other controller in this repo.  The `intent.extras["humanoid_tasks"]` key carries
a `HumanoidTaskSpec` assembled by the follower stack.

## Known Issues

| Issue | Likely Cause | Fix Path |
|-------|-------------|----------|
| Falls within first step | Contact scheduler triggers swing too early | Tune `HumanoidContactSchedulerConfig.min_double_support_time` |
| Torque saturation on stance leg | `lambda_max_normal` too low for single-support | Raise to ~700 N, or lower `com_kp` |
| Pelvis drifts laterally | `pelvis` task weight too low vs swing | Raise `TaskWeightsConfig.pelvis` from 7.0 → 12.0 |
| QP falls back to SLSQP every step | OSQP warm-start lost between ticks | Re-use previous solution as OSQP warm-start |
| dt=0.02 s is too coarse for WBC | Inverse dynamics assumes small dt | Try control_dt=0.005 with sub-stepping |

## Files

```
controllers/wbc/
├── __init__.py          — public exports
├── config.py            — WBCConfig (4 sub-configs, ~92 tunable scalars)
├── controller.py        — HumanoidWBCController (Controller protocol)
├── task_stack.py        — task residual/Jacobian builders
├── contact_blocks.py    — contact constraint blocks for the QP
├── friction_cone.py     — pyramid friction cone linearisation
├── qp_builder.py        — assembles H, g, A, b, lb, ub from tasks + contacts
└── qp_solver_adapter.py — multi-method solver fallback chain
```

## Running the Test Bed

```bash
# Standing balance integration test (passes):
docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu \
    python -m pytest test/integration/test_wbc_g1_standing.py -v

# Corridor diagnose (currently: robot falls):
docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu \
    python scripts/tasks/robot/humanoid/wbc_corridor.py \
        --plan results/humanoid/corridor_2d/plan/twogo_zone_a/level_1/seed_0/trajectory/trajectory.json

# Compare against sport-mode baseline:
docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu \
    python scripts/tasks/robot/humanoid/sport_mode_corridor.py
```

## Tuning Workflow

1. Run `wbc_corridor.py` — check QP failure count and torque saturation rate.
2. If `qp_solve_failures > 0`: lower `LimitsConfig.lambda_max_normal` or raise `SolverConfig.osqp_maxiter`.
3. If `torque_saturations > 20%` of steps: lower `TaskWeightsConfig.com` or `TaskWeightsConfig.contact`.
4. If `fell_over` at first step: the contact scheduler is wrong. Inspect `phase` in `wbc.npz`.
5. Run the standing integration test after each config change to confirm it doesn't regress.

## Interface Stability

`HumanoidWBCController`, `WBCConfig`, and the four sub-configs are **stable**.
Internal modules (`task_stack`, `contact_blocks`, etc.) are implementation detail
and may change without notice.

The `Controller` protocol (`act(state, intent) → ControlCommand`) will not change.

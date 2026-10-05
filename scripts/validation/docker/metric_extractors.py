"""Arm + humanoid signal extractors -> general metrics, on real brax (docker).

Each extractor consumes the structured Brax states collected during execution
(EE pose / box x / contact residual / com / stiffness), then the SHARED
general metrics are computed via GeneralMetricsPlugin. Validates that both MGA
tasks report general metrics through tiny task-specific extractors.

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python scripts/validation/docker/metric_extractors.py"
"""

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.envs.factories import make_env
from genedynamics.experiments.plugins.metrics.extractors import (
    arm_surface_scan_metrics_plugin, ARM_METRICS,
    humanoid_box_push_metrics_plugin, HUMANOID_METRICS,
)


class _Traj:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions


def _exec(env, x0, n=6, scale=0.2, key=0):
    """Execute n small random actions, returning the executed action list."""
    states = [x0]
    acts = []
    s = x0
    rng = jax.random.PRNGKey(key)
    for _ in range(n):
        rng, k = jax.random.split(rng)
        u = scale * jax.random.normal(k, (env.action_size,))
        s = env.step(s, u)
        states.append(s)
        acts.append(np.asarray(u))
    return states, acts


def _check(name, plugin, env, expect_keys):
    x0 = env.reset(jax.random.PRNGKey(1))
    states, actions = _exec(env, x0)
    traj = _Traj(states, actions)
    out = plugin.compute(traj, env, None, None, x0=x0, planning_time=0.123)
    have = set(out)
    missing = [k for k in expect_keys if k not in have]
    finite = all(np.all(np.isfinite(np.asarray(v))) for v in out.values()
                 if isinstance(v, (int, float, np.floating, np.ndarray)))
    ok = (not missing) and finite and out.get("runtime") == 0.123
    print(f"[{name}] metrics={sorted(have)}")
    print(f"[{name}] missing={missing} finite={finite} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ok = []
    expect_arm = {"surface_tracking_error", "force_tracking_error", "control_smoothness",
                  "stiffness_smoothness", "energy", "runtime",
                  "tangential_tracking_error", "force_tracking_error_settled",
                  "force_command_violation_rate", "scan_progress_ratio",
                  "realized_path_completion", "max_realized_path_progress",
                  "trajectory_path_coverage", "gate_path_next_error_corr",
                  "gate_force_next_risk_corr"}
    ok.append(_check("arm", arm_surface_scan_metrics_plugin(),
                     make_env("manipulator_surface_scan"), expect_arm))

    expect_hum = {"goal_error", "balance_violation_rate", "balance_violation_cvar",
                  "friction_cone_violation_rate", "balance_margin", "control_smoothness",
                  "stiffness_smoothness", "energy", "runtime"}
    ok.append(_check("humanoid", humanoid_box_push_metrics_plugin(),
                     make_env("humanoid_box_push"), expect_hum))

    print("RESULT:", "ALL PASS" if all(ok) else "FAILED")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

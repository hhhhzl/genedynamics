"""
Test CFSQPFullFilter in 7D (joint-lift) mode for cfsmbd_full / D3IL avoiding.

- Uses AvoidingPlanEnv9D (state 9D, action 7D) with a frozen J_xy so that the filter
  can lift 2D SDF gradients to action space via A = J_xy^T @ grad_xy.
- Uses D3IL-style obstacles (6 circles matching d3il_avoiding_fixed layout).
- Builds nominal 7D actions from a straight-line xy path (pseudo-inverse J_xy).
- Applies CFSQPFullFilter with convexifier_name="cfs_action_joint".
- Checks that the filter runs, returns 7D actions, and that the filtered trajectory
  has SDF >= robot_radius (or violations no worse than nominal).

Run:
  pytest test/algos/test_cfs_qp_full_7d.py -v
  python test/algos/test_cfs_qp_full_7d.py   # script mode: print stats + optional plot
"""

from __future__ import annotations

import numpy as np
import time
from pathlib import Path
from typing import Any

import pytest

if __name__ != "__main__":
    pytest.importorskip("jax")

from genedynamics.envs.obstacles.base import ObstacleManager
from genedynamics.envs.obstacles.convex import SphereObstacle
from genedynamics.envs.external.d3il.avoiding_plan_env_9d import AvoidingPlanEnv9D, AvoidingPlanSpec9D
from genedynamics.core.dynamics.adapters import EnvDynamicsAdapter
from genedynamics.core.constraints.action_filters.cfs_qp_full import CFSQPFullFilter
from genedynamics.core.constraints.core.types import ScheduleState


# -----------------------------------------------------------------------------
# D3IL-style obstacles (6 circles, same layout as d3il_avoiding_fixed)
# -----------------------------------------------------------------------------

D3IL_CIRCLES_LEVEL0 = [
    {"center": np.array([0.5, -0.1], dtype=np.float32), "radius": 0.03},
    {"center": np.array([0.5 - 0.075, -0.1 + 0.18], dtype=np.float32), "radius": 0.025},
    {"center": np.array([0.5 + 0.075, -0.1 + 0.18], dtype=np.float32), "radius": 0.025},
    {"center": np.array([0.5 - 2 * 0.075, -0.1 + 2 * 0.18], dtype=np.float32), "radius": 0.025},
    {"center": np.array([0.5, -0.1 + 2 * 0.18], dtype=np.float32), "radius": 0.025},
    {"center": np.array([0.5 + 2 * 0.075, -0.1 + 2 * 0.18], dtype=np.float32), "radius": 0.025},
]


def make_d3il_obstacles() -> ObstacleManager:
    """ObstacleManager with 6 spheres matching D3IL avoiding layout (triggers use_fast_analytic)."""
    obstacles = ObstacleManager()
    for i, circ in enumerate(D3IL_CIRCLES_LEVEL0):
        obs = SphereObstacle(
            center=circ["center"],
            radius=circ["radius"],
            name=f"d3il_{i}",
        )
        obstacles.add(obs)
    return obstacles


# -----------------------------------------------------------------------------
# 9D state / 7D action helpers
# -----------------------------------------------------------------------------


def make_plan_env_9d(
    dt: float = 0.035,
    horizon: int = 64,
    control_limit: float = 0.8,
    robot_radius: float = 0.01,
) -> tuple[AvoidingPlanEnv9D, np.ndarray]:
    """
    Create AvoidingPlanEnv9D with a constant J_xy so that tcp_xy moves with first two qdot.
    J_xy = [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0]] => d(tcp_xy)/dt = [qdot0, qdot1].
    Returns (plan_env, J_xy (2,7)).
    """
    spec = AvoidingPlanSpec9D(dt=dt, horizon=horizon, qdot_limit=control_limit)
    plan_env = AvoidingPlanEnv9D(spec=spec)
    plan_env.robot_radius = float(robot_radius)
    # Simple J: first two joints drive xy (like a 2D planar arm approximation)
    J_xy = np.zeros((2, 7), dtype=np.float32)
    J_xy[0, 0] = 1.0
    J_xy[1, 1] = 1.0
    plan_env.set_linearization(J_xy)
    return plan_env, J_xy


def xy_path_to_7d_actions(
    positions_xy: np.ndarray,
    dt: float,
    J_xy: np.ndarray,
    control_limit: float,
) -> np.ndarray:
    """
    Convert (N, 2) xy path to (N-1, 7) actions so that under x' = x + dt*(J_xy @ u),
    the tcp_xy follows positions_xy. Uses pseudo-inverse: u = J_xy^+ @ (delta_xy / dt).
    positions_xy: (N, 2), dt: float, J_xy: (2, 7).
    Returns u (N-1, 7), clipped to control_limit.
    """
    deltas = np.diff(positions_xy, axis=0).astype(np.float32)  # (N-1, 2)
    u_2d = deltas / float(dt)  # (N-1, 2)
    # J_xy (2,7): we want u_7d such that J_xy @ u_7d = u_2d.T => u_7d = J_xy^+ @ u_2d.T => (7, N-1).T = (N-1, 7)
    Jpinv = np.linalg.pinv(J_xy)  # (7, 2)
    u_7d = (u_2d @ Jpinv.T)  # (N-1, 2) @ (2, 7) = (N-1, 7)
    u_7d = np.clip(u_7d, -control_limit, control_limit).astype(np.float32)
    return u_7d


def rollout_9d(
    x0: np.ndarray,
    actions: np.ndarray,
    env: Any,
) -> np.ndarray:
    """Rollout 9D states from x0 using actions. env has transition(s, a) -> s_next."""
    states = [np.asarray(x0, dtype=np.float32)]
    s = np.asarray(x0, dtype=np.float32)
    for a in actions:
        s = env.transition(s, a)
        states.append(np.asarray(s, dtype=np.float32))
    return np.stack(states, axis=0)


def compute_sdf_violations(
    positions_xy: np.ndarray,
    obstacles: ObstacleManager,
    robot_radius: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    """SDF at each position, violation count, violation indices."""
    sdf_values = np.array(
        [
            float(obstacles.sdf(p)) if np.isscalar(obstacles.sdf(p)) else float(obstacles.sdf(p)[0])
            for p in positions_xy
        ],
        dtype=np.float32,
    )
    violations = sdf_values < robot_radius
    num_violations = int(np.sum(violations))
    violation_indices = np.where(violations)[0]
    return sdf_values, num_violations, violation_indices


# -----------------------------------------------------------------------------
# Main test runner
# -----------------------------------------------------------------------------


def run_cfs_qp_full_7d_test(
    num_points: int = 50,
    dt: float = 0.035,
    robot_radius: float = 0.01,
    margin: float = 0.02,
    rho: float = 10.0,
    max_constraints_per_point: int = 8,
    constraint_margin: float = 0.25,
    use_slack: bool = False,
    cfs_outer_iters: int = 5,
    control_limit: float = 0.8,
) -> dict[str, Any]:
    """
    Run CFSQPFullFilter in 7D (joint-lift) with D3IL-style obstacles.

    - Straight-line xy path from (0.4, -0.1) to (0.5, 0.35) (through obstacles).
    - Nominal 7D u from J_xy pseudo-inverse.
    - Filter with convexifier_name="cfs_action_joint".
    - Rollout and check tcp_xy SDF.

    Returns dict with positions_filtered (xy), SDF stats, violations, timing, etc.
    """
    start_xy = np.array([0.4, -0.1], dtype=np.float32)
    goal_xy = np.array([0.5, 0.35], dtype=np.float32)
    t = np.linspace(0, 1, num_points, dtype=np.float32)
    positions_xy = start_xy[None, :] + t[:, None] * (goal_xy - start_xy)[None, :]

    plan_env, J_xy = make_plan_env_9d(
        dt=dt,
        horizon=num_points - 1,
        control_limit=control_limit,
        robot_radius=robot_radius,
    )
    env = EnvDynamicsAdapter(plan_env)
    # Adapter forwards getattr to plan_env; plan_env has robot_radius set above
    if not hasattr(env, "robot_radius"):
        env.robot_radius = robot_radius

    obstacles = make_d3il_obstacles()

    # Initial state: tcp_xy = start, q = 0
    x0 = np.zeros(9, dtype=np.float32)
    x0[0], x0[1] = start_xy[0], start_xy[1]

    u_nom = xy_path_to_7d_actions(positions_xy, dt, J_xy, control_limit)
    assert u_nom.shape[0] == num_points - 1 and u_nom.shape[1] == 7

    filtr = CFSQPFullFilter(
        max_constraints_per_point=max_constraints_per_point,
        constraint_margin=constraint_margin,
        use_slack=use_slack,
        convexifier_name="cfs_action_joint",
    )
    sched_state = ScheduleState(k=0, K=1)
    sched_params = {
        "margin": margin,
        "rho": rho,
        "qp_gate": True,
        "qp_prob": 1.0,
        "I_QP": cfs_outer_iters,
    }

    t0 = time.time()
    u_filtered = filtr.apply_actions(
        x0,
        u_nom,
        env=env,
        obstacles=obstacles,
        schedule_state=sched_state,
        schedule_params=sched_params,
    )
    elapsed = time.time() - t0

    u_filtered = np.asarray(u_filtered, dtype=np.float32)
    assert u_filtered.shape == u_nom.shape, f"expected {u_nom.shape}, got {u_filtered.shape}"

    # Rollout with plan env (adapter forwards transition)
    states_filtered = rollout_9d(x0, u_filtered, env)
    positions_filtered = states_filtered[:, :2]

    sdf_filtered, num_violations, violation_indices = compute_sdf_violations(
        positions_filtered, obstacles, robot_radius
    )

    # Nominal rollout for comparison
    states_nom = rollout_9d(x0, u_nom, env)
    pos_nom = states_nom[:, :2]
    _, nom_violations, _ = compute_sdf_violations(pos_nom, obstacles, robot_radius)

    trajectory_length = 0.0
    if len(positions_filtered) > 1:
        diffs = np.diff(positions_filtered, axis=0)
        trajectory_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    return {
        "positions_xy_ref": positions_xy,
        "u_nom": u_nom,
        "u_filtered": u_filtered,
        "positions_filtered": positions_filtered,
        "positions_nom": pos_nom,
        "sdf_values": sdf_filtered,
        "min_sdf": float(np.min(sdf_filtered)),
        "mean_sdf": float(np.mean(sdf_filtered)),
        "num_violations": num_violations,
        "nom_violations": nom_violations,
        "violation_indices": violation_indices,
        "trajectory_length": trajectory_length,
        "elapsed_time": elapsed,
        "obstacles": obstacles,
        "env": env,
        "robot_radius": robot_radius,
    }


# -----------------------------------------------------------------------------
# Pytest
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_cfs_qp_full_7d_filter_runs():
    """CFSQPFullFilter (7D joint-lift) runs without error and returns valid 7D actions."""
    res = run_cfs_qp_full_7d_test(
        num_points=40,
        dt=0.035,
        robot_radius=0.01,
        margin=0.02,
        max_constraints_per_point=8,
        constraint_margin=0.25,
        cfs_outer_iters=3,
    )
    assert res["u_filtered"].shape == res["u_nom"].shape
    assert res["u_filtered"].shape[1] == 7
    assert res["positions_filtered"].shape[0] == res["positions_xy_ref"].shape[0]
    assert np.all(np.isfinite(res["u_filtered"]))
    assert np.all(np.isfinite(res["positions_filtered"]))
    assert res["elapsed_time"] >= 0


@pytest.mark.unit
def test_cfs_qp_full_7d_reduces_or_keeps_violations():
    """Filtered 7D trajectory should have no more violations than nominal (ideally fewer)."""
    res = run_cfs_qp_full_7d_test(
        num_points=50,
        dt=0.035,
        robot_radius=0.01,
        margin=0.02,
        cfs_outer_iters=5,
    )
    assert res["num_violations"] <= res["nom_violations"], (
        f"Filter increased violations: nominal={res['nom_violations']}, filtered={res['num_violations']}"
    )


@pytest.mark.unit
def test_cfs_qp_full_7d_filter_changes_actions():
    """Filter should modify nominal actions when path crosses obstacles (7D lift is active)."""
    res = run_cfs_qp_full_7d_test(
        num_points=50,
        dt=0.035,
        robot_radius=0.01,
        margin=0.02,
        cfs_outer_iters=5,
    )
    u_nom = res["u_nom"]
    u_f = res["u_filtered"]
    diff = np.abs(u_f - u_nom)
    # At least some timesteps should be modified (straight line goes through obstacles)
    assert np.any(diff > 1e-4), (
        "Filter did not change actions; 7D CFS may not be active (check J_xy / obstacles)."
    )


# -----------------------------------------------------------------------------
# Main (script mode)
# -----------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("CFSQPFull 7D (joint-lift) — D3IL-style obstacles")
    print("=" * 80)

    res = run_cfs_qp_full_7d_test(
        num_points=50,
        dt=0.035,
        robot_radius=0.01,
        margin=0.02,
        rho=10.0,
        max_constraints_per_point=8,
        constraint_margin=0.25,
        use_slack=False,
        cfs_outer_iters=5,
        control_limit=0.8,
    )

    print()
    print("  [nominal]")
    print(f"    violations={res['nom_violations']}")
    print()
    print("  [filtered 7D]")
    print(f"    violations={res['num_violations']}, min_sdf={res['min_sdf']:.6f}, mean_sdf={res['mean_sdf']:.6f}")
    print(f"    elapsed={res['elapsed_time']:.4f}s, traj_len={res['trajectory_length']:.4f}")
    print(f"    u_filtered shape={res['u_filtered'].shape}, finite={np.all(np.isfinite(res['u_filtered']))}")
    print()

    try:
        import matplotlib.pyplot as plt

        out_dir = Path("test_results/cfs_qp_full_7d")
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        for o in res["obstacles"]:
            if hasattr(o, "center") and hasattr(o, "radius"):
                c = np.asarray(o.center)[:2]
                r = float(o.radius)
                circle = plt.Circle((c[0], c[1]), r, fill=True, facecolor="red", alpha=0.3, edgecolor="black")
                ax.add_patch(circle)
        x_ref = res["positions_xy_ref"]
        x_f = res["positions_filtered"]
        x_nom = res["positions_nom"]
        ax.plot(x_ref[:, 0], x_ref[:, 1], "b--", alpha=0.5, label="Reference (xy)")
        ax.plot(x_nom[:, 0], x_nom[:, 1], "gray", alpha=0.7, label="Nominal rollout")
        ax.plot(x_f[:, 0], x_f[:, 1], "g-", linewidth=2, label=f"Filtered 7D (v={res['num_violations']})")
        ax.scatter(x_ref[0, 0], x_ref[0, 1], c="blue", s=80, marker="s", zorder=5)
        ax.scatter(x_ref[-1, 0], x_ref[-1, 1], c="red", s=80, marker="*", zorder=5)
        if res["num_violations"] > 0:
            v = res["violation_indices"]
            ax.scatter(x_f[v, 0], x_f[v, 1], c="orange", s=60, marker="x", zorder=6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("CFSQPFull 7D joint-lift: nominal vs filtered tcp_xy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(-0.15, 0.45)
        plt.tight_layout()
        p = out_dir / "cfs_qp_full_7d_trajectory.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
        plt.close()
    except Exception as e:
        print(f"  Skipped plot: {e}")

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        import jax  # noqa: F401
    except ImportError:
        print("JAX required; install jax or run with pytest.")
        raise SystemExit(1)
    main()

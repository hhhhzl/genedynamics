"""
Test script for CFSQPFullFilter and CFSQPPerStepFilter (action-space CFS QP).

This mirrors the setup of test_cfs_jaxopt_vs_cvxopt.py but adapts for u-space:
- Reference: same straight-line trajectory in X (state) as the CFS projection test.
- Nominal u: obtained via inverse dynamics (single integrator): u_t = (x_{t+1} - x_t) / dt.
- CFSQPFullFilter (u_traj + outer loop + box) or CFSQPPerStepFilter project u onto
  CFS feasible set and return filtered actions.

Script mode runs both filters, prints comparison (elapsed, violations, min_sdf, etc.),
and plots Reference vs Full vs Perstep trajectories.

Requires: JAX (pytest skips module if missing).

Run:
  pytest test/algos/test_cfs_qp_full.py -v
  python test/algos/test_cfs_qp_full.py   # script mode: full vs perstep + viz
"""

from __future__ import annotations

import numpy as np
import time
from pathlib import Path
from typing import Any

import pytest

if __name__ != "__main__":
    pytest.importorskip("jax")

from genedynamics.envs.single_integrator_box_2d import SingleIntegratorBox2DEnv
from genedynamics.envs.obstacles.base import ObstacleManager
from genedynamics.envs.obstacles.convex import BoxObstacle
from genedynamics.core.constraints.action_filters.cfs_qp_full import CFSQPFullFilter
from genedynamics.core.constraints.action_filters.cfs_qp_perstep import CFSQPPerStepFilter
from genedynamics.core.constraints.core.types import ScheduleState


# -----------------------------------------------------------------------------
# Env wrapper: use model_transition for rollout (SingleIntegratorBox2DEnv.step
# has nonstandard signature and does not perform transition)
# -----------------------------------------------------------------------------


class _EnvForCFSFull:
    """
    Thin wrapper over SingleIntegratorBox2DEnv so that CFSQPFullFilter rollout
    uses model_transition (no `step`), and we add robot_radius.
    """

    def __init__(self, base: SingleIntegratorBox2DEnv, robot_radius: float = 0.05):
        self._base = base
        self.dt = float(base.dt)
        self.robot_radius = float(robot_radius)
        self.control_limit = float(getattr(base, "control_limit", 1.0))
        self.p_max = float(getattr(base, "p_max", 2.0))
        self.horizon = int(getattr(base, "horizon", 80))
        self.act_dim = int(getattr(base, "act_dim", 2))

    def model_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self._base.model_transition(state, action)

    def jax_transition(self, state: Any, action: Any) -> Any:
        return self._base.jax_transition(state, action)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def make_obstacles() -> ObstacleManager:
    """Same obstacle layout as test_cfs_jaxopt_vs_cvxopt.py."""
    obstacles = ObstacleManager()
    obs1_center = np.array([0.5, 0.5], dtype=np.float32)
    obs1_half = np.array([0.2, 0.2], dtype=np.float32)
    obstacles.add(BoxObstacle(center=obs1_center, half_extents=obs1_half, name="obstacle1"))
    obs2_center = np.array([-0.3, -0.3], dtype=np.float32)
    obs2_half = np.array([0.15, 0.15], dtype=np.float32)
    obstacles.add(BoxObstacle(center=obs2_center, half_extents=obs2_half, name="obstacle2"))
    return obstacles


def x_reference_straight_line(
    start: np.ndarray,
    goal: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """Positions along straight line from start to goal (same as CFS projection test)."""
    t = np.linspace(0, 1, num_points, dtype=np.float32)
    positions = start[None, :] + t[:, None] * (goal - start)[None, :]
    return positions


def inverse_dynamics_single_integrator(
    positions: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Single integrator: x_{t+1} = x_t + dt * u_t  =>  u_t = (x_{t+1} - x_t) / dt.
    positions: (N, 2), dt: float.
    Returns actions (N-1, 2).
    """
    diffs = np.diff(positions, axis=0).astype(np.float32)
    u = diffs / float(dt)
    return u


def rollout_states(
    x0: np.ndarray,
    actions: np.ndarray,
    env: _EnvForCFSFull,
) -> np.ndarray:
    """Rollout states from x0 using actions via model_transition."""
    states = [np.asarray(x0, dtype=np.float32)]
    s = np.asarray(x0, dtype=np.float32)
    for a in actions:
        s = env.model_transition(s, a)
        states.append(np.asarray(s, dtype=np.float32))
    return np.stack(states, axis=0)


def compute_sdf_and_violations(
    positions: np.ndarray,
    obstacles: ObstacleManager,
    robot_radius: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    """SDF at each position, violation count, violation indices."""
    sdf_values = np.array(
        [float(obstacles.sdf(p)) if np.isscalar(obstacles.sdf(p)) else float(obstacles.sdf(p)[0])
         for p in positions],
        dtype=np.float32,
    )
    violations = sdf_values < robot_radius
    num_violations = int(np.sum(violations))
    violation_indices = np.where(violations)[0]
    return sdf_values, num_violations, violation_indices


# -----------------------------------------------------------------------------
# Main test flow
# -----------------------------------------------------------------------------


def run_cfs_qp_full_test(
    num_points: int = 50,
    dt: float = 0.05,
    robot_radius: float = 0.05,
    margin: float = 0.1,
    rho: float = 10.0,
    max_constraints_per_point: int = 8,
    constraint_margin: float = 0.25,
    use_slack: bool = True,
    cfs_outer_iters: int = 8,
) -> dict[str, Any]:
    """
    Run CFSQPFullFilter test with X-reference + inverse-dynamics u.

    Returns dict with filtered positions, SDF stats, violations, timing, etc.
    """
    start_pos = np.array([-1.0, -1.0], dtype=np.float32)
    goal_pos = np.array([1.0, 1.0], dtype=np.float32)

    base_env = SingleIntegratorBox2DEnv(dt=dt, p_max=2.0, control_limit=1.0)
    env = _EnvForCFSFull(base_env, robot_radius=robot_radius)
    obstacles = make_obstacles()

    # 1) X-reference (straight line, same as test_cfs_jaxopt_vs_cvxopt)
    positions_x = x_reference_straight_line(start_pos, goal_pos, num_points)
    x0 = positions_x[0]

    # 2) Inverse dynamics => nominal u
    u_nom = inverse_dynamics_single_integrator(positions_x, dt)
    assert u_nom.shape[0] == num_points - 1 and u_nom.shape[1] == 2

    # 3) Filter
    filtr = CFSQPFullFilter(
        max_constraints_per_point=max_constraints_per_point,
        constraint_margin=constraint_margin,
        use_slack=use_slack,
    )
    sched_state = ScheduleState(k=0, K=1)
    sched_params = {
        "margin": margin,
        "rho": rho,
        "qp_gate": True,
        "qp_prob": 1.0,
        # Unify: I_QP is the paper-style CFS outer iterations (linearize + solve QP).
        # Keep cfs_outer_iters for backward compatibility.
        "I_QP": cfs_outer_iters,
        "cfs_outer_iters": cfs_outer_iters,
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
    assert u_filtered.shape == u_nom.shape

    # 4) Rollout with filtered u
    states_filtered = rollout_states(x0, u_filtered, env)
    positions_filtered = states_filtered[:, :2]

    # 5) SDF and violations
    sdf_filtered, num_violations, violation_indices = compute_sdf_and_violations(
        positions_filtered, obstacles, robot_radius
    )

    trajectory_length = 0.0
    if len(positions_filtered) > 1:
        diffs = np.diff(positions_filtered, axis=0)
        trajectory_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    return {
        "positions_x_ref": positions_x,
        "u_nom": u_nom,
        "u_filtered": u_filtered,
        "positions_filtered": positions_filtered,
        "sdf_values": sdf_filtered,
        "min_sdf": float(np.min(sdf_filtered)),
        "mean_sdf": float(np.mean(sdf_filtered)),
        "max_sdf": float(np.max(sdf_filtered)),
        "num_violations": num_violations,
        "violation_indices": violation_indices,
        "trajectory_length": trajectory_length,
        "elapsed_time": elapsed,
        "obstacles": obstacles,
        "env": env,
        "robot_radius": robot_radius,
    }


def run_cfs_qp_perstep_test(
    num_points: int = 50,
    dt: float = 0.05,
    robot_radius: float = 0.05,
    margin: float = 0.1,
    rho: float = 10.0,
    max_constraints_per_point: int = 8,
    constraint_margin: float = 0.25,
    use_slack: bool = True,
    cfs_outer_iters: int = 8,
) -> dict[str, Any]:
    """
    Run CFSQPPerStepFilter test with same X-reference + inverse-dynamics u.

    Returns same dict structure as run_cfs_qp_full_test for comparison.
    """
    start_pos = np.array([-1.0, -1.0], dtype=np.float32)
    goal_pos = np.array([1.0, 1.0], dtype=np.float32)

    base_env = SingleIntegratorBox2DEnv(dt=dt, p_max=2.0, control_limit=1.0)
    env = _EnvForCFSFull(base_env, robot_radius=robot_radius)
    obstacles = make_obstacles()

    positions_x = x_reference_straight_line(start_pos, goal_pos, num_points)
    x0 = positions_x[0]
    u_nom = inverse_dynamics_single_integrator(positions_x, dt)
    assert u_nom.shape[0] == num_points - 1 and u_nom.shape[1] == 2

    filtr = CFSQPPerStepFilter(
        max_constraints_per_point=max_constraints_per_point,
        constraint_margin=constraint_margin,
        use_slack=use_slack,
    )
    sched_state = ScheduleState(k=0, K=1)
    sched_params = {
        "margin": margin,
        "rho": rho,
        "qp_gate": True,
        "qp_prob": 1.0,
        # Unify: I_QP is the paper-style CFS outer iterations (linearize + solve QP).
        # Keep cfs_outer_iters for backward compatibility.
        "I_QP": cfs_outer_iters,
        "cfs_outer_iters": cfs_outer_iters,
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
    assert u_filtered.shape == u_nom.shape

    states_filtered = rollout_states(x0, u_filtered, env)
    positions_filtered = states_filtered[:, :2]
    sdf_filtered, num_violations, violation_indices = compute_sdf_and_violations(
        positions_filtered, obstacles, robot_radius
    )

    trajectory_length = 0.0
    if len(positions_filtered) > 1:
        diffs = np.diff(positions_filtered, axis=0)
        trajectory_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    return {
        "positions_x_ref": positions_x,
        "u_nom": u_nom,
        "u_filtered": u_filtered,
        "positions_filtered": positions_filtered,
        "sdf_values": sdf_filtered,
        "min_sdf": float(np.min(sdf_filtered)),
        "mean_sdf": float(np.mean(sdf_filtered)),
        "max_sdf": float(np.max(sdf_filtered)),
        "num_violations": num_violations,
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
def test_cfs_qp_full_inverse_dynamics_setup():
    """Sanity check: inverse dynamics + rollout recovers reference X (no obstacles)."""
    dt = 0.05
    num_points = 20
    start = np.array([-1.0, -1.0], dtype=np.float32)
    goal = np.array([1.0, 1.0], dtype=np.float32)
    positions = x_reference_straight_line(start, goal, num_points)
    u = inverse_dynamics_single_integrator(positions, dt)
    # IMPORTANT: For this round-trip sanity check we need an *unclipped* model transition.
    # The reference straight-line has per-step delta 2/(num_points-1), so u can exceed 1.0.
    # Use a large control_limit so env.model_transition does not clip u and the identity holds.
    base = SingleIntegratorBox2DEnv(dt=dt, p_max=2.0, control_limit=10.0)
    env = _EnvForCFSFull(base, robot_radius=0.05)
    states = rollout_states(positions[0], u, env)
    recovered = states[:, :2]
    np.testing.assert_allclose(recovered, positions, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_cfs_qp_full_filter_runs():
    """CFSQPFullFilter runs without error and returns valid actions."""
    res = run_cfs_qp_full_test(
        num_points=50,
        dt=0.05,
        robot_radius=0.05,
        margin=0.1,
        rho=10.0,
        max_constraints_per_point=8,
        constraint_margin=0.25,
        use_slack=True,
    )
    assert res["u_filtered"].shape == res["u_nom"].shape
    assert res["positions_filtered"].shape[0] == res["positions_x_ref"].shape[0]
    assert np.all(np.isfinite(res["u_filtered"]))
    assert np.all(np.isfinite(res["positions_filtered"]))
    assert res["elapsed_time"] >= 0


@pytest.mark.unit
def test_cfs_qp_full_reduces_violations():
    """
    Filtered trajectory should have no more violations than nominal rollout,
    and ideally fewer (nominal straight line crosses obstacles).
    """
    # Nominal straight-line u -> rollout -> violations
    start = np.array([-1.0, -1.0], dtype=np.float32)
    goal = np.array([1.0, 1.0], dtype=np.float32)
    dt = 0.05
    num_points = 50
    positions = x_reference_straight_line(start, goal, num_points)
    u_nom = inverse_dynamics_single_integrator(positions, dt)
    base = SingleIntegratorBox2DEnv(dt=dt, p_max=2.0, control_limit=1.0)
    env = _EnvForCFSFull(base, robot_radius=0.05)
    obstacles = make_obstacles()
    states_nom = rollout_states(positions[0], u_nom, env)
    pos_nom = states_nom[:, :2]
    _, nom_violations, _ = compute_sdf_and_violations(pos_nom, obstacles, 0.05)

    res = run_cfs_qp_full_test(
        num_points=num_points,
        dt=dt,
        robot_radius=0.05,
        margin=0.1,
        rho=10.0,
        max_constraints_per_point=8,
        constraint_margin=0.25,
        use_slack=True,
    )
    filtered_violations = res["num_violations"]

    # Filter should not increase violations; we expect it to reduce or keep same
    assert filtered_violations <= nom_violations, (
        f"Filter increased violations: nominal={nom_violations}, filtered={filtered_violations}"
    )


@pytest.mark.unit
def test_cfs_qp_perstep_filter_runs():
    """CFSQPPerStepFilter runs without error and returns valid actions."""
    res = run_cfs_qp_perstep_test(
        num_points=50,
        dt=0.05,
        robot_radius=0.05,
        margin=0.1,
        rho=10.0,
        max_constraints_per_point=8,
        constraint_margin=0.25,
        use_slack=True,
    )
    assert res["u_filtered"].shape == res["u_nom"].shape
    assert res["positions_filtered"].shape[0] == res["positions_x_ref"].shape[0]
    assert np.all(np.isfinite(res["u_filtered"]))
    assert np.all(np.isfinite(res["positions_filtered"]))
    assert res["elapsed_time"] >= 0


# -----------------------------------------------------------------------------
# Main (run as script for quick check + optional viz)
# -----------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("CFSQPFull vs CFSQPPerStep (X-reference + inverse dynamics => u)")
    print("=" * 80)

    res_full = run_cfs_qp_full_test(
        num_points=50,
        dt=0.05,
        robot_radius=0.05,
        margin=0.1,
        rho=10.0,
        max_constraints_per_point=8,
        constraint_margin=0.25,
        use_slack=True,
        cfs_outer_iters=8,
    )
    res_perstep = run_cfs_qp_perstep_test(
        num_points=50,
        dt=0.05,
        robot_radius=0.05,
        margin=0.1,
        rho=10.0,
        max_constraints_per_point=8,
        constraint_margin=0.25,
        use_slack=True,
        cfs_outer_iters=8,
    )

    # Nominal stats (same u_nom)
    x0 = res_full["positions_x_ref"][0]
    states_nom = rollout_states(x0, res_full["u_nom"], res_full["env"])
    pos_nom = states_nom[:, :2]
    sdf_nom, num_violations_nom, _ = compute_sdf_and_violations(
        pos_nom, res_full["obstacles"], res_full["robot_radius"]
    )
    min_sdf_nom = float(np.min(sdf_nom))
    max_u_nom = float(np.max(np.abs(res_full["u_nom"])))

    max_u_full = float(np.max(np.abs(res_full["u_filtered"])))
    max_u_perstep = float(np.max(np.abs(res_perstep["u_filtered"])))

    print()
    print("  [nominal]")
    print(f"    num_violations={num_violations_nom}, min_sdf={min_sdf_nom:.6f}, max|u|={max_u_nom:.6f}")
    print()
    print("  [comparison]")
    print("                    elapsed(s)   max|u|   violations   min_sdf    traj_len")
    print(f"    full        {res_full['elapsed_time']:>10.4f}   {max_u_full:>6.3f}   {res_full['num_violations']:>10}   {res_full['min_sdf']:>8.4f}   {res_full['trajectory_length']:.4f}")
    print(f"    perstep     {res_perstep['elapsed_time']:>10.4f}   {max_u_perstep:>6.3f}   {res_perstep['num_violations']:>10}   {res_perstep['min_sdf']:>8.4f}   {res_perstep['trajectory_length']:.4f}")
    print()
    print("  filter pushed u > 1?  full:", max_u_full > 1.0, "  perstep:", max_u_perstep > 1.0)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        out_dir = Path("test_results/cfs_qp_full")
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        obstacles = res_full["obstacles"]
        for o in obstacles:
            if hasattr(o, "center") and hasattr(o, "half_extents"):
                c = o.center[:2]
                h = o.half_extents[:2]
                ax.add_patch(
                    Rectangle(
                        (c[0] - h[0], c[1] - h[1]),
                        2 * h[0], 2 * h[1],
                        fill=True, facecolor="red", alpha=0.3, edgecolor="black", linewidth=1,
                    )
                )
        x_ref = res_full["positions_x_ref"]
        x_f = res_full["positions_filtered"]
        x_p = res_perstep["positions_filtered"]

        ax.plot(x_ref[:, 0], x_ref[:, 1], "b--", alpha=0.5, label="Reference (X)")
        ax.plot(x_f[:, 0], x_f[:, 1], "g-", label=f"Full (v={res_full['num_violations']})", linewidth=2, marker="o", markersize=2)
        ax.plot(x_p[:, 0], x_p[:, 1], "m-", label=f"Perstep (v={res_perstep['num_violations']})", linewidth=2, marker="s", markersize=2)
        ax.scatter(x_ref[0, 0], x_ref[0, 1], c="blue", s=100, marker="s", label="Start", zorder=5)
        ax.scatter(x_ref[-1, 0], x_ref[-1, 1], c="red", s=100, marker="*", label="Goal (ref)", zorder=5)
        if res_full["num_violations"] > 0:
            v = res_full["violation_indices"]
            ax.scatter(x_f[v, 0], x_f[v, 1], c="orange", s=80, marker="x", zorder=6, linewidths=2)
        if res_perstep["num_violations"] > 0:
            v = res_perstep["violation_indices"]
            ax.scatter(x_p[v, 0], x_p[v, 1], c="cyan", s=80, marker="+", zorder=6, linewidths=2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("CFSQPFull vs CFSQPPerStep: X-ref → inv-dyn u → filtered → rollout")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        plt.tight_layout()
        p = out_dir / "cfs_qp_full_trajectory.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"\n  Saved plot: {p}")
        plt.close()
    except Exception as e:
        print(f"\n  Skipped visualization: {e}")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        import jax  # noqa: F401
    except ImportError:
        print("JAX not installed; run with pytest or install jax.")
        raise SystemExit(1)
    main()

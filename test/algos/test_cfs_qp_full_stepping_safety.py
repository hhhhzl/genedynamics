"""
Minimal stepping-stones CFSQP safety experiment.

Goal:
- Build an intentionally unsafe reference control sequence on
  QuadrupedSteppingStones2DEnv.
- Apply CFSQPFullFilter with the SAME obstacle/safety-point constraints used by
  runtime pipeline.
- Compare safety before vs after filtering under:
  (1) weighted QP semantics: d_foot >= w_foot * clearance
  (2) hard semantics: d_foot >= clearance (all feet hard)

Run:
  pytest test/algos/test_cfs_qp_full_stepping_safety.py -v
  python test/algos/test_cfs_qp_full_stepping_safety.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import yaml
import jax.numpy as jnp

if __name__ != "__main__":
    pytest.importorskip("jax")

from genedynamics.core.constraints.action_filters.cfs_qp_full import CFSQPFullFilter
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.envs.obstacles.stepping_stones import (
    foot_stepping_violation_np,
    make_stepping_stones_obstacles,
)
from genedynamics.envs.domains.quadruped.stepping_stones import (
    LEG_ORDER,
    QuadrupedSteppingStones2DEnv,
)
from genedynamics.tasks.stepping_stones import sample_stepping_stones_scene


def _load_mdcoas_env_defaults() -> Dict[str, float]:
    cfg_path = Path("configs/quadruped/stepping_stones_2d/smoke/mdcoas_smoke.yaml")
    defaults = {
        "dt": 1.0,
        "horizon": 16,
        "control_limit": 0.45,
        "start_mid": (-1.25, 0.0),
        "goal_mid": (1.25, 0.0),
    }
    if not cfg_path.exists():
        return defaults
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        env_p = cfg.get("env_params", {}) or {}
        defaults["dt"] = float(env_p.get("dt", defaults["dt"]))
        defaults["horizon"] = int(env_p.get("horizon", defaults["horizon"]))
        defaults["control_limit"] = float(env_p.get("control_limit", defaults["control_limit"]))
        defaults["start_mid"] = tuple(env_p.get("start_mid", defaults["start_mid"]))
        defaults["goal_mid"] = tuple(env_p.get("goal_mid", defaults["goal_mid"]))
    except Exception:
        pass
    return defaults


def _make_env_and_obstacles(
    level: int = 1,
    seed: int = 0,
    dt: float | None = None,
    horizon: int | None = None,
    control_limit: float | None = None,
) -> tuple[QuadrupedSteppingStones2DEnv, Any]:
    cfg = _load_mdcoas_env_defaults()
    dt = float(cfg["dt"] if dt is None else dt)
    horizon = int(cfg["horizon"] if horizon is None else horizon)
    control_limit = float(cfg["control_limit"] if control_limit is None else control_limit)
    start_mid = tuple(cfg["start_mid"])
    goal_mid = tuple(cfg["goal_mid"])

    scene = sample_stepping_stones_scene(
        level=int(level),
        seed=int(seed),
        l_max=0.35,
        stance_width=0.30,
        start_mid=start_mid,
        goal_mid=goal_mid,
        fore_hind_offset=0.18,
        k_horizon_cap=16,
        lane_tail_margin=3,
        lane_gap_scale=0.97,
    )
    obstacles = make_stepping_stones_obstacles(scene)
    env = QuadrupedSteppingStones2DEnv(
        dt=float(dt),
        horizon=int(horizon),
        scene=scene,
        obstacles=obstacles,
    )
    # Keep this experiment strictly aligned with config horizon, instead of scene auto-expansion.
    env.horizon = int(horizon)
    # CFSQP reads control_limit from env via getattr(..., 1.0); set explicitly for this experiment.
    env.control_limit = float(control_limit)
    return env, obstacles


def _make_unsafe_reference_actions(env: QuadrupedSteppingStones2DEnv) -> np.ndarray:
    """
    Build an intentionally unsafe action sequence:
    - Move body forward aggressively every step
    - Keep residual feet offsets mostly unchanged
    - Push phase forward to force mode transitions
    """
    H = int(env.horizon)
    U = np.zeros((H, env.act_dim), dtype=np.float32)
    U[:, 0] = float(min(env.body_step_norm_max, env.control_limit))  # forward shift
    U[:, 1] = 0.0
    U[:, 2] = 0.0
    U[:, 11] = float(env.phase_rate_limit)  # fast tau progression
    return np.clip(U, -float(env.control_limit), float(env.control_limit)).astype(np.float32)


def _rollout_states(env: QuadrupedSteppingStones2DEnv, x0: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.asarray(env.rollout_actions(x0, actions), dtype=np.float32)


def _foot_sdf_matrix(states: np.ndarray, env: QuadrupedSteppingStones2DEnv, obstacles: Any) -> np.ndarray:
    vals = []
    for s in states:
        pts = np.asarray(env.get_safety_points(s), dtype=np.float32)  # (4,2)
        row = []
        for i in range(pts.shape[0]):
            d = obstacles.sdf(pts[i])
            d = float(d[0]) if isinstance(d, np.ndarray) and d.size > 0 else float(d)
            row.append(d)
        vals.append(row)
    return np.asarray(vals, dtype=np.float32)  # (T,4)


def _qp_weight_matrix(states: np.ndarray, env: QuadrupedSteppingStones2DEnv) -> np.ndarray:
    w = []
    for s in states:
        if hasattr(env, "numpy_cfs_safety_point_weights"):
            wi = np.asarray(env.numpy_cfs_safety_point_weights(s), dtype=np.float32).reshape(-1)
        else:
            wi = np.ones((4,), dtype=np.float32)
        w.append(wi[:4])
    return np.asarray(w, dtype=np.float32)  # (T,4)


def _safety_stats(
    states: np.ndarray,
    env: QuadrupedSteppingStones2DEnv,
    obstacles: Any,
    clearance: float,
) -> Dict[str, float]:
    sdf = _foot_sdf_matrix(states, env, obstacles)  # (T,4)
    w = _qp_weight_matrix(states, env)  # (T,4)

    viol_weighted = np.maximum(0.0, w * float(clearance) - sdf)
    viol_hard = np.maximum(0.0, float(clearance) - sdf)

    any_weighted = np.max(viol_weighted, axis=1) > 1e-6
    any_hard = np.max(viol_hard, axis=1) > 1e-6

    # Match trajectory visualization notion of red points (for reference)
    centers = np.asarray(env.scene.stones_centers, dtype=np.float32)
    radii = np.asarray(env.scene.stones_radii, dtype=np.float32)
    plats = np.asarray(getattr(env.scene, "support_platforms", np.zeros((0, 4))), dtype=np.float32)
    m = float(env.stone_margin)
    red_tol = max(1e-3, 0.5 * m)
    red_count = 0
    for s in states:
        feet = np.asarray(env.get_safety_points(s), dtype=np.float32)
        for i in range(4):
            v = foot_stepping_violation_np(feet[i], centers, radii, plats, m)
            if float(v) > float(red_tol):
                red_count += 1

    return {
        "max_weighted_violation": float(np.max(viol_weighted)),
        "max_hard_violation": float(np.max(viol_hard)),
        "mean_weighted_violation": float(np.mean(np.max(viol_weighted, axis=1))),
        "mean_hard_violation": float(np.mean(np.max(viol_hard, axis=1))),
        "timesteps_weighted_violating": float(np.sum(any_weighted)),
        "timesteps_hard_violating": float(np.sum(any_hard)),
        "red_point_count": float(red_count),
    }


def _debug_choose_targets_np(
    env: QuadrupedSteppingStones2DEnv,
    state_t: np.ndarray,
    pts_t: np.ndarray,
    *,
    clearance: float = 0.02,
) -> Dict[str, np.ndarray]:
    x = np.asarray(state_t, dtype=np.float32).reshape(-1)
    mode = int(np.rint(float(x[14]))) % 2
    is_m0 = mode == 0
    swing_mask = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32) if is_m0 else np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    side_mask = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    body_x = float(x[0])
    centers = np.asarray(env._foothold_centers_np, dtype=np.float32)
    radii = np.asarray(env._foothold_radii_np, dtype=np.float32)
    lanes = np.asarray(env._foothold_lane_np, dtype=np.float32)
    policy = str(getattr(env, "cfs_custom_constraint_target_policy", "legacy"))
    swing_cap = float(getattr(env, "cfs_custom_swing_forward_cap", 0.4))
    w4 = (
        np.asarray(env.numpy_cfs_safety_point_weights(state_t), dtype=np.float32).reshape(-1)[:4]
        if hasattr(env, "numpy_cfs_safety_point_weights")
        else np.ones((4,), dtype=np.float32)
    )
    c_scalar = float(clearance)

    idxs = np.zeros((4,), dtype=np.int32)
    valid = np.zeros((4,), dtype=bool)
    dist_to_target = np.zeros((4,), dtype=np.float32)
    margin_to_target = np.zeros((4,), dtype=np.float32)
    is_swing = swing_mask > 0.5

    for i in range(4):
        p = np.asarray(pts_t[i], dtype=np.float32).reshape(2)
        dvec = centers - p[None, :]
        dist = np.linalg.norm(dvec, axis=1)
        lane_ok = np.logical_or(np.abs(lanes) < 0.5, np.sign(lanes) == np.sign(side_mask[i]))
        forward_ok = centers[:, 0] >= (body_x - 0.02)
        reach_ok = centers[:, 0] <= (p[0] + swing_cap)
        valid_legacy = np.logical_and(lane_ok, np.logical_or(np.logical_not(is_swing[i]), forward_ok))
        valid_rw = np.logical_and(
            lane_ok,
            np.logical_or(np.logical_not(is_swing[i]), np.logical_and(forward_ok, reach_ok)),
        )
        wi = float(w4[i])
        need_clear = wi * c_scalar
        margin_disk = radii - dist

        if policy == "reachable":
            valid_support = lane_ok
            valid_swing_rw = np.logical_and(lane_ok, np.logical_and(forward_ok, reach_ok))
            score_sup = dist + np.where(valid_support, 0.0, 1e3)
            idx_sup = int(np.argmin(score_sup))
            ok_sup = bool(score_sup[idx_sup] < 900.0)

            if is_swing[i]:
                cand_in = np.logical_and(valid_swing_rw, margin_disk >= (need_clear - 1e-3))
                score_in = np.where(cand_in, dist, np.inf)
                idx_in = int(np.argmin(score_in))
                has_in = bool(np.min(score_in) < 1.0e6)

                margin_masked = np.where(valid_swing_rw, margin_disk, -1.0e9)
                idx_fb = int(np.argmax(margin_masked))
                has_rw = bool(np.max(margin_masked) > -1.0e8)

                score_lg = dist + np.where(valid_legacy, 0.0, 1e3)
                idx_lg = int(np.argmin(score_lg))

                if has_in:
                    idx = idx_in
                    ok = True
                elif has_rw:
                    idx = idx_fb
                    ok = True
                else:
                    idx = idx_lg
                    ok = bool(score_lg[idx] < 900.0)
            else:
                idx = idx_sup
                ok = ok_sup
        else:
            leg_valid = valid_legacy
            score = dist + np.where(leg_valid, 0.0, 1e3)
            idx = int(np.argmin(score))
            ok = bool(score[idx] < 900.0)

        idxs[i] = idx
        valid[i] = ok
        dist_to_target[i] = float(dist[idx])
        margin_to_target[i] = float(radii[idx] - dist[idx])

    return {
        "target_idx": idxs,
        "target_valid": valid,
        "target_dist": dist_to_target,
        "target_margin": margin_to_target,
        "is_swing": is_swing.astype(bool),
    }


def _custom_constraint_residual_trace(
    env: QuadrupedSteppingStones2DEnv,
    states: np.ndarray,
    actions: np.ndarray,
    clearance: float,
    max_constraints_per_point: int,
) -> Dict[str, Any]:
    if not hasattr(env, "jax_cfs_custom_safety_constraints") or not callable(getattr(env, "jax_cfs_custom_safety_constraints")):
        return {"enabled": False}

    H = int(actions.shape[0])
    k_select = int(max(1, int(max_constraints_per_point) * int(getattr(env, "cfs_qp_num_safety_points", 4))))
    act_dim = int(actions.shape[1])
    row_min = []
    row_mean = []
    row_max = []
    active_count = []
    per_t = []

    for t in range(H):
        state_prev = np.asarray(states[t], dtype=np.float32)
        state_t = np.asarray(states[t + 1], dtype=np.float32)
        u_t = np.asarray(actions[t], dtype=np.float32)
        pts_t = np.asarray(env.get_safety_points(state_t), dtype=np.float32)
        if hasattr(env, "jax_jacobian_safety_points_action_local"):
            J_pts = np.asarray(env.jax_jacobian_safety_points_action_local(state_prev, u_t), dtype=np.float32)
        else:
            J_pts = np.asarray(env.jax_jacobian_safety_points_action(state_prev), dtype=np.float32)

        out = env.jax_cfs_custom_safety_constraints(
            jnp.asarray(state_prev, dtype=jnp.float32),
            jnp.asarray(state_t, dtype=jnp.float32),
            jnp.asarray(u_t, dtype=jnp.float32),
            jnp.asarray(pts_t, dtype=jnp.float32),
            jnp.asarray(J_pts, dtype=jnp.float32),
            jnp.asarray(float(clearance), dtype=jnp.float32),
            act_dim,
            k_select,
        )
        if not isinstance(out, tuple) or len(out) < 2:
            continue
        A_t = np.asarray(out[0], dtype=np.float32)
        b_t = np.asarray(out[1], dtype=np.float32).reshape(-1)
        if len(out) >= 3:
            valid = np.asarray(out[2], dtype=bool).reshape(-1)
        else:
            valid = np.isfinite(b_t)
        mask = np.logical_and(valid, np.isfinite(b_t))
        if np.any(mask):
            lhs = A_t[mask] @ u_t
            res = lhs - b_t[mask]
            row_min.append(float(np.min(res)))
            row_mean.append(float(np.mean(res)))
            row_max.append(float(np.max(res)))
            active_count.append(int(np.sum(mask)))
            chosen = _debug_choose_targets_np(env, state_t, pts_t, clearance=float(clearance))
            per_t.append(
                {
                    "t": int(t),
                    "active": int(np.sum(mask)),
                    "res_min": float(np.min(res)),
                    "res_mean": float(np.mean(res)),
                    "res_max": float(np.max(res)),
                    "target_idx": chosen["target_idx"].tolist(),
                    "target_valid": chosen["target_valid"].astype(int).tolist(),
                    "target_margin": [float(v) for v in chosen["target_margin"].tolist()],
                    "is_swing": chosen["is_swing"].astype(int).tolist(),
                }
            )
        else:
            active_count.append(0)
            row_min.append(0.0)
            row_mean.append(0.0)
            row_max.append(0.0)

    return {
        "enabled": True,
        "k_select": k_select,
        "active_count_mean": float(np.mean(np.asarray(active_count, dtype=np.float32))) if active_count else 0.0,
        "res_min_global": float(np.min(np.asarray(row_min, dtype=np.float32))) if row_min else 0.0,
        "res_mean_global": float(np.mean(np.asarray(row_mean, dtype=np.float32))) if row_mean else 0.0,
        "res_max_global": float(np.max(np.asarray(row_max, dtype=np.float32))) if row_max else 0.0,
        "per_t": per_t,
    }


def _swing_support_violation_breakdown(
    states: np.ndarray,
    env: QuadrupedSteppingStones2DEnv,
    clearance: float,
) -> Dict[str, float]:
    hard = []
    hard_support = []
    hard_swing = []
    for s in states:
        x = np.asarray(s, dtype=np.float32).reshape(-1)
        mode = int(np.rint(float(x[14]))) % 2
        is_m0 = mode == 0
        swing_idx = (1, 2) if is_m0 else (0, 3)
        support_idx = (0, 3) if is_m0 else (1, 2)
        feet = np.asarray(env.get_safety_points(x), dtype=np.float32)
        sdf = np.asarray([float(env._numpy_scene_qp_sdf(feet[i])) for i in range(4)], dtype=np.float32)
        v = np.maximum(0.0, float(clearance) - sdf)
        hard.append(float(np.max(v)))
        hard_support.append(float(np.max(v[list(support_idx)])))
        hard_swing.append(float(np.max(v[list(swing_idx)])))
    hard = np.asarray(hard, dtype=np.float32)
    hard_support = np.asarray(hard_support, dtype=np.float32)
    hard_swing = np.asarray(hard_swing, dtype=np.float32)
    return {
        "mean_hard_all": float(np.mean(hard)),
        "mean_hard_support": float(np.mean(hard_support)),
        "mean_hard_swing": float(np.mean(hard_swing)),
        "max_hard_all": float(np.max(hard)),
        "max_hard_support": float(np.max(hard_support)),
        "max_hard_swing": float(np.max(hard_swing)),
    }


def run_stepping_cfs_safety_experiment(
    *,
    level: int = 1,
    seed: int = 0,
    clearance: float = 0.02,
    rho: float = 10.0,
    cfs_outer_iters: int = 8,
    max_constraints_per_point: int = 8,
    constraint_margin: float = 0.01,
    use_slack: bool = True,
    use_jax_filter: bool = True,
    enable_custom_constraints: bool = True,
    cfs_target_policy: Optional[str] = None,
) -> Dict[str, Any]:
    env, obstacles = _make_env_and_obstacles(level=level, seed=seed)
    if cfs_target_policy is not None:
        env.cfs_custom_constraint_target_policy = str(cfs_target_policy)
    if not bool(enable_custom_constraints):
        setattr(env, "jax_cfs_custom_safety_constraints", None)
    x0, _ = env.reset()
    x0 = np.asarray(x0, dtype=np.float32)

    u_ref = _make_unsafe_reference_actions(env)
    states_ref = _rollout_states(env, x0, u_ref)
    stats_ref = _safety_stats(states_ref, env, obstacles, clearance=float(clearance))

    filt = CFSQPFullFilter(
        max_constraints_per_point=int(max_constraints_per_point),
        constraint_margin=float(constraint_margin),
        use_slack=bool(use_slack),
        convexifier_name="cfs_action",
    )
    sched_state = ScheduleState(k=0, K=1)
    sched_params = {
        "margin": float(clearance),
        "rho": float(rho),
        "qp_gate": True,
        "qp_prob": 1.0,
        "I_QP": int(cfs_outer_iters),
        "topK": int(max_constraints_per_point),
        "eps": 1e-5,
    }

    # IMPORTANT: use JAX arrays to trigger CFSQP JAX path.
    x0_in = jnp.asarray(x0, dtype=jnp.float32) if use_jax_filter else x0
    u_ref_in = jnp.asarray(u_ref, dtype=jnp.float32) if use_jax_filter else u_ref

    t0 = time.time()
    u_f = filt.apply_actions(
        x0_in,
        u_ref_in,
        env=env,
        obstacles=obstacles,
        schedule_state=sched_state,
        schedule_params=sched_params,
    )
    elapsed = time.time() - t0
    u_f = np.asarray(u_f, dtype=np.float32)

    states_f = _rollout_states(env, x0, u_f)
    stats_f = _safety_stats(states_f, env, obstacles, clearance=float(clearance))

    # Transition analysis: did filtering turn feasible points into infeasible points?
    sdf_ref = _foot_sdf_matrix(states_ref, env, obstacles)
    sdf_f = _foot_sdf_matrix(states_f, env, obstacles)
    hard_ref = np.max(np.maximum(0.0, float(clearance) - sdf_ref), axis=1)
    hard_f = np.max(np.maximum(0.0, float(clearance) - sdf_f), axis=1)
    safe_ref = hard_ref <= 1e-6
    safe_f = hard_f <= 1e-6
    safe_to_unsafe = int(np.sum(np.logical_and(safe_ref, np.logical_not(safe_f))))
    unsafe_to_safe = int(np.sum(np.logical_and(np.logical_not(safe_ref), safe_f)))

    body_ref = np.asarray(states_ref[:, :2], dtype=np.float32)
    body_f = np.asarray(states_f[:, :2], dtype=np.float32)
    body_shift = body_f - body_ref
    body_shift_norm = np.linalg.norm(body_shift, axis=1)
    custom_trace_ref = _custom_constraint_residual_trace(
        env, states_ref, u_ref, float(clearance), int(max_constraints_per_point)
    )
    custom_trace_f = _custom_constraint_residual_trace(
        env, states_f, u_f, float(clearance), int(max_constraints_per_point)
    )
    breakdown_ref = _swing_support_violation_breakdown(states_ref, env, float(clearance))
    breakdown_f = _swing_support_violation_breakdown(states_f, env, float(clearance))

    return {
        "x0": x0,
        "u_ref": u_ref,
        "u_filtered": u_f,
        "states_ref": states_ref,
        "states_filtered": states_f,
        "stats_ref": stats_ref,
        "stats_filtered": stats_f,
        "safe_to_unsafe_timesteps_hard": safe_to_unsafe,
        "unsafe_to_safe_timesteps_hard": unsafe_to_safe,
        "body_shift_norm_mean": float(np.mean(body_shift_norm)),
        "body_shift_norm_max": float(np.max(body_shift_norm)),
        "elapsed_s": float(elapsed),
        "custom_trace_ref": custom_trace_ref,
        "custom_trace_filtered": custom_trace_f,
        "breakdown_ref": breakdown_ref,
        "breakdown_filtered": breakdown_f,
        "cfs_target_policy": str(getattr(env, "cfs_custom_constraint_target_policy", "legacy")),
    }


@pytest.mark.unit
def test_cfs_stepping_filter_runs_and_shapes():
    res = run_stepping_cfs_safety_experiment()
    assert res["u_filtered"].shape == res["u_ref"].shape
    assert res["states_filtered"].shape == res["states_ref"].shape
    assert np.all(np.isfinite(res["u_filtered"]))
    assert np.all(np.isfinite(res["states_filtered"]))


@pytest.mark.unit
def test_cfs_stepping_not_worse_than_reference():
    res = run_stepping_cfs_safety_experiment()
    a = res["stats_ref"]
    b = res["stats_filtered"]
    assert b["max_weighted_violation"] <= a["max_weighted_violation"] + 1e-6
    assert b["timesteps_weighted_violating"] <= a["timesteps_weighted_violating"] + 1e-6


def main():
    res = run_stepping_cfs_safety_experiment()
    T_ref = int(np.asarray(res["states_ref"]).shape[0])
    T_f = int(np.asarray(res["states_filtered"]).shape[0])
    print("=" * 84)
    print("Stepping-stones CFSQP safety check (unsafe ref -> filtered)")
    print("=" * 84)
    print(f"cfs_custom_constraint_target_policy: {res['cfs_target_policy']}")
    print(f"elapsed: {res['elapsed_s']:.3f}s")
    print(f"rollout steps: ref={T_ref}, filtered={T_f}, horizon={T_ref - 1}")
    print("\n[reference]")
    for k, v in res["stats_ref"].items():
        print(f"  {k}: {v}")
    print("\n[filtered]")
    for k, v in res["stats_filtered"].items():
        print(f"  {k}: {v}")
    print("\n[delta filtered - ref]")
    for k in res["stats_ref"].keys():
        dv = float(res["stats_filtered"][k] - res["stats_ref"][k])
        print(f"  {k}: {dv:+.6f}")
    print(f"  safe_to_unsafe_timesteps_hard: {res['safe_to_unsafe_timesteps_hard']}")
    print(f"  unsafe_to_safe_timesteps_hard: {res['unsafe_to_safe_timesteps_hard']}")
    print(f"  body_shift_norm_mean: {res['body_shift_norm_mean']:.6f}")
    print(f"  body_shift_norm_max: {res['body_shift_norm_max']:.6f}")
    print("\n[support vs swing hard violation]")
    for k in ["mean_hard_all", "mean_hard_support", "mean_hard_swing", "max_hard_all", "max_hard_support", "max_hard_swing"]:
        dv = float(res["breakdown_filtered"][k] - res["breakdown_ref"][k])
        print(f"  {k}: ref={res['breakdown_ref'][k]:.6f} filtered={res['breakdown_filtered'][k]:.6f} delta={dv:+.6f}")
    tr = res["custom_trace_ref"]
    tf = res["custom_trace_filtered"]
    if bool(tf.get("enabled", False)):
        print("\n[custom linear-constraint residual A u - b]")
        print(
            "  ref: "
            f"active_mean={tr['active_count_mean']:.2f}, "
            f"res_min={tr['res_min_global']:.6f}, "
            f"res_mean={tr['res_mean_global']:.6f}, "
            f"res_max={tr['res_max_global']:.6f}"
        )
        print(
            "  filtered: "
            f"active_mean={tf['active_count_mean']:.2f}, "
            f"res_min={tf['res_min_global']:.6f}, "
            f"res_mean={tf['res_mean_global']:.6f}, "
            f"res_max={tf['res_max_global']:.6f}"
        )
        print("  per-step target snapshot (filtered):")
        for row in tf.get("per_t", [])[: min(6, len(tf.get("per_t", [])))]:
            print(
                "    "
                f"t={row['t']} active={row['active']} "
                f"res[min/mean/max]=[{row['res_min']:.4f}/{row['res_mean']:.4f}/{row['res_max']:.4f}] "
                f"swing={row['is_swing']} target_idx={row['target_idx']} "
                f"target_margin={[round(v, 4) for v in row['target_margin']]}"
            )

    try:
        import matplotlib.pyplot as plt

        out_dir = Path("test_results/cfs_qp_stepping")
        out_dir.mkdir(parents=True, exist_ok=True)

        env, _ = _make_env_and_obstacles()
        fil_xy = np.asarray([np.asarray(s, dtype=np.float32)[:2] for s in res["states_filtered"]], dtype=np.float32)
        fil_feet = np.asarray([env.get_safety_points(s) for s in res["states_filtered"]], dtype=np.float32)  # (T,4,2)

        # Display reference: desired body path from start deck center to goal deck center
        # (ends exactly at goal center, independent of env transition limits).
        T = int(np.asarray(res["states_ref"]).shape[0])
        start_mid = np.asarray(env.scene.start_mid, dtype=np.float32).reshape(2)
        goal_mid = np.asarray(env.scene.goal_mid, dtype=np.float32).reshape(2)
        alpha = np.linspace(0.0, 1.0, T, dtype=np.float32)
        ref_xy = (1.0 - alpha)[:, None] * start_mid[None, :] + alpha[:, None] * goal_mid[None, :]
        psi_ref = float(np.arctan2(goal_mid[1] - start_mid[1], goal_mid[0] - start_mid[0]))
        ref_feet_list = []
        for t in range(T):
            feet_t = env._nominal_feet(ref_xy[t], psi_ref)
            ref_feet_list.append(np.stack([feet_t[leg] for leg in LEG_ORDER], axis=0))
        ref_feet = np.asarray(ref_feet_list, dtype=np.float32)  # (T,4,2)

        scene = env.scene
        centers = np.asarray(scene.stones_centers, dtype=np.float32)
        radii = np.asarray(scene.stones_radii, dtype=np.float32)
        platforms = np.asarray(getattr(scene, "support_platforms", np.zeros((0, 4))), dtype=np.float32)
        red_tol = max(1e-3, 0.5 * float(env.stone_margin))

        def _red_mask(feet_traj: np.ndarray) -> np.ndarray:
            mask = np.zeros((feet_traj.shape[0], feet_traj.shape[1]), dtype=bool)
            for t in range(feet_traj.shape[0]):
                for i in range(feet_traj.shape[1]):
                    v = foot_stepping_violation_np(
                        feet_traj[t, i], centers, radii, platforms, float(env.stone_margin)
                    )
                    mask[t, i] = float(v) > red_tol
            return mask

        ref_red = _red_mask(ref_feet)
        fil_red = _red_mask(fil_feet)

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        for c, r in zip(centers, radii):
            circ = plt.Circle((float(c[0]), float(c[1])), float(r), facecolor="#7fb069", edgecolor="#335c33", alpha=0.45)
            ax.add_patch(circ)
        for rect in platforms:
            x0, x1, y0, y1 = map(float, rect.tolist())
            ax.add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#89b4fa", edgecolor="#1e3a8a", alpha=0.25)
            )

        leg_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
        for i, leg in enumerate(LEG_ORDER):
            ax.plot(
                ref_feet[:, i, 0], ref_feet[:, i, 1],
                linestyle="--", linewidth=1.2, color=leg_colors[i], alpha=0.55,
                label=f"ref {leg}" if i == 0 else None,
            )
            ax.scatter(
                ref_feet[:, i, 0], ref_feet[:, i, 1],
                s=14, marker="o", facecolors="none", edgecolors=leg_colors[i], linewidths=0.8, alpha=0.35,
            )
            ax.plot(
                fil_feet[:, i, 0], fil_feet[:, i, 1],
                linestyle="-", linewidth=1.6, color=leg_colors[i], alpha=0.95,
                label=f"filtered {leg}" if i == 0 else None,
            )

            rr = np.where(ref_red[:, i])[0]
            if rr.size > 0:
                ax.scatter(
                    ref_feet[rr, i, 0], ref_feet[rr, i, 1],
                    s=26, marker="x", color="#cc5500", linewidths=0.9,
                )
            rf = np.where(fil_red[:, i])[0]
            if rf.size > 0:
                ax.scatter(
                    fil_feet[rf, i, 0], fil_feet[rf, i, 1],
                    s=30, marker="o", facecolors="#d62728", edgecolors="#6a0000", linewidths=0.6,
                )

        ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", linewidth=1.2, alpha=0.65, label="reference body (to goal center)")
        ax.plot(fil_xy[:, 0], fil_xy[:, 1], "k-", linewidth=1.8, alpha=0.9, label="filtered body")
        ax.scatter(ref_xy[0, 0], ref_xy[0, 1], s=42, marker="s", color="black", alpha=0.7)
        ax.scatter(fil_xy[-1, 0], fil_xy[-1, 1], s=64, marker="*", color="#f04f88", alpha=0.9)

        ax.set_title("Stepping-stones: body + 4-feet trajectories, stones, and red points")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9, ncol=1)
        p = out_dir / "stepping_body_ref_vs_filtered.png"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {p}")

        # Vector-difference figure: nominal rollout (unsafe reference actions) -> filtered rollout.
        nom_xy = np.asarray([np.asarray(s, dtype=np.float32)[:2] for s in res["states_ref"]], dtype=np.float32)
        fil_xy_nomref = np.asarray([np.asarray(s, dtype=np.float32)[:2] for s in res["states_filtered"]], dtype=np.float32)
        nom_feet = np.asarray([env.get_safety_points(s) for s in res["states_ref"]], dtype=np.float32)
        fil_feet_nomref = np.asarray([env.get_safety_points(s) for s in res["states_filtered"]], dtype=np.float32)

        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
        for c, r in zip(centers, radii):
            ax2.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), facecolor="#7fb069", edgecolor="#335c33", alpha=0.35))
        for rect in platforms:
            x0, x1, y0, y1 = map(float, rect.tolist())
            ax2.add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#89b4fa", edgecolor="#1e3a8a", alpha=0.18)
            )

        # Body displacement vectors
        dxy = fil_xy_nomref - nom_xy
        ax2.quiver(
            nom_xy[:, 0], nom_xy[:, 1],
            dxy[:, 0], dxy[:, 1],
            angles="xy", scale_units="xy", scale=1.0,
            color="black", width=0.0025, alpha=0.85, label="body delta (nominal->filtered)"
        )
        ax2.plot(nom_xy[:, 0], nom_xy[:, 1], "k--", linewidth=1.0, alpha=0.5, label="nominal body")
        ax2.plot(fil_xy_nomref[:, 0], fil_xy_nomref[:, 1], "k-", linewidth=1.6, alpha=0.8, label="filtered body")

        # Foot displacement vectors (downsample to reduce clutter)
        step = max(1, nom_feet.shape[0] // 12)
        for i, leg in enumerate(LEG_ORDER):
            c = leg_colors[i]
            p0 = nom_feet[::step, i, :]
            d = fil_feet_nomref[::step, i, :] - p0
            ax2.quiver(
                p0[:, 0], p0[:, 1],
                d[:, 0], d[:, 1],
                angles="xy", scale_units="xy", scale=1.0,
                color=c, width=0.0018, alpha=0.65,
                label=f"{leg} delta" if i == 0 else None,
            )

        ax2.set_title("Nominal -> Filtered displacement vectors (body + feet)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_aspect("equal")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9, ncol=1)
        p2 = out_dir / "stepping_nominal_vs_filtered_vectors.png"
        fig2.savefig(p2, dpi=180, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {p2}")
    except Exception as e:
        print(f"\nSkipped plot: {e}")

    print("=" * 84)


if __name__ == "__main__":
    try:
        import jax  # noqa: F401
    except Exception:
        print("JAX is required to run this script.")
        raise SystemExit(1)
    main()


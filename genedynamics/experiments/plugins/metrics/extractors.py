"""Task signal extractors for the general metrics library.

An *extractor* is the ONLY task-specific glue: it maps a task's ``Trajectory`` +
``env`` (+ rollout diagnostics) to the generic ``signals`` dict the
task-agnostic metrics consume (positions / controls / inequality residual ``g`` /
equality residual ``h`` / forces / contact / com / stiffness / ...). The metric
math itself lives entirely in ``genedynamics.evaluation`` and is reused unchanged
across tasks — only the extractor differs.

Each ``*_metrics_plugin`` factory pairs a list of GENERAL metric names with the
task's extractor, yielding a ``GeneralMetricsPlugin`` the experiment runner can
use. Compare ``corridor_signals`` (a ~10-line replacement for the monolithic
``CorridorMetricsPlugin.compute`` whose metric math is now shared).

Flat-state tasks (corridor / stepping) extract directly from the trajectory and
are pure-numpy. Brax contact tasks consume the actual structured states collected
by ``run_receding``. Replay is retained only as a backward-compatible fallback
for legacy trajectories that contain actions but no structured states.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from genedynamics.experiments.plugins.metrics.general import GeneralMetricsPlugin


# ---------------------------------------------------------------------------
# Flat-state tasks (pure numpy)
# ---------------------------------------------------------------------------

def corridor_signals(trajectory, env, obstacles, constraints, **kw) -> Dict[str, Any]:
    """Corridor: position is ``state[:2]``, violation is negative clearance.
    Replaces the hardcoded math of the old monolithic plugin with shared metrics."""
    states = np.stack([np.ravel(np.asarray(s, np.float64)) for s in trajectory.states])
    actions = (np.stack([np.ravel(np.asarray(a, np.float64)) for a in trajectory.actions])
               if trajectory.actions else np.zeros((0, 1)))
    pos = states[:, :2]
    clear = np.array([float(env.get_min_clearance(states[t])) for t in range(states.shape[0])])
    return {
        "positions": pos, "final_pos": pos[-1], "start_pos": pos[0],
        "target": np.asarray(env.target, np.float64).ravel()[:2],
        "controls": actions,
        "g": -clear,                      # clearance < 0 => g > 0 (collision = violation)
        "seconds": float(kw.get("planning_time", 0.0)),
    }


CORRIDOR_METRICS = [
    "goal_error", "max_violation", "violation_rate", "violation_cvar",
    {"name": "success", "bind": {"violation": "max_violation"}},
    "path_length", "path_efficiency", "control_smoothness",
    "progress_ratio", "completion_step", "runtime",
]


def corridor_metrics_plugin(margin: float = 0.2, name: str = "corridor_metrics"):
    """General corridor metrics (success/violation-CVaR/path/smoothness/...) via
    the shared library + the tiny ``corridor_signals`` extractor."""
    return GeneralMetricsPlugin(
        [*CORRIDOR_METRICS], extractor=corridor_signals, name=name,
        config={"success": {"margin": margin}, "completion_step": {"margin": margin}},
    )


# ---------------------------------------------------------------------------
# brax tasks: roll the env to recover per-step signals (mjx; docker)
# ---------------------------------------------------------------------------

def _roll_brax(env, x0, actions, per_step: Callable[[Any, Any, Any], Dict[str, Any]]):
    """Roll ``env.step`` over executed ``actions`` (from ``x0``), collecting the
    per-step signal dict ``per_step(env, state, action)`` into stacked arrays."""
    import jax
    import jax.numpy as jnp

    def f(s, u):
        s2 = env.step(s, u)
        return s2, per_step(env, s2, u)

    _, diag = jax.lax.scan(f, x0, jnp.asarray(actions))
    return {k: np.asarray(v, np.float64) for k, v in diag.items()}


def _x0(env, kw):
    if "x0" in kw and kw["x0"] is not None:
        return kw["x0"]
    import jax
    return env.reset(jax.random.PRNGKey(int(kw.get("seed", 0))))


def _contact_step_signals(trajectory, env, actions, per_step, kw):
    """Stack signals from executed states, falling back only for legacy data."""
    import jax.numpy as jnp

    actual_states = list(getattr(trajectory, "states", ()))
    if (
        len(actual_states) == len(actions) + 1
        and actions.shape[0] > 0
        and hasattr(actual_states[0], "pipeline_state")
    ):
        rows = [per_step(env, state, action)
                for state, action in zip(actual_states[1:], actions)]
        return {
            key: np.asarray(jnp.stack([row[key] for row in rows]), np.float64)
            for key in rows[0]
        }
    return _roll_brax(env, _x0(env, kw), actions, per_step)


def arm_surface_scan_signals(trajectory, env, obstacles, constraints, **kw) -> Dict[str, Any]:
    """Surface-scan: roll the arm to collect EE pose, the surface/normal tracking
    residual ``h``, the commanded normal force, and the stiffness chart."""
    import jax
    import jax.numpy as jnp
    actions = jnp.asarray([np.ravel(np.asarray(a, np.float32)) for a in trajectory.actions])

    cfg = env._config

    def per_step(e, s, u):
        h, _ = e.constraint_residual(s, u)
        _, s_vec, f_cmd = e._unpack(u)                     # physical stiffness svec + commanded force
        xi, eta = s.info["xi"], s.info["eta"]
        _, n_s, p_d, _ = e._desired_pose(xi, eta)
        ee = s.pipeline_state.site_xpos[e._ee_site]
        h_rest = ee - p_d
        h_tangent = h_rest - jnp.dot(h_rest, n_s) * n_s
        normal_offset = jnp.dot(h_rest, n_s)
        # contact force at the scan coords: real mjx contact (rigid/soft) or the Winkler
        # reaction (hybrid). in_contact = a nonzero contact force.
        f_real = e._contact_force_at(s.pipeline_state, xi, eta)
        in_contact = (f_real > 0.5).astype(jnp.float32)
        # on_surface = the EE is doing the scan properly: on the path (|h_surf| small) AND
        # actually pressing (in contact). Force-tracking precision is only defined while
        # pressing on-path -- a baseline that sits off the path, or a contact-loss step
        # (force=0, scored separately by contact_loss_rate), must not skew force tracking.
        on_surface = ((jnp.linalg.norm(h[:3]) < e._config.track_tol) & (f_real > 0.5)).astype(jnp.float32)
        # surface deformation = penetration; k_surf = local surface stiffness (spatial on hybrid).
        pen = e._penetration_at(s.pipeline_state, xi, eta)
        k_surf = e._k_surf_fn(xi, eta)
        reliability = e.geometry_reliability(s)
        realization_offset = e.realization_coordinate_offset(s)
        _, _, jacp, _, _, _ = e._ee_kin(s.pipeline_state)
        g_ee = jnp.linalg.solve(
            jacp.T @ jacp + 1e-6 * jnp.eye(3),
            jacp.T @ s.pipeline_state.qfrc_bias[:7],
        )
        gravity_normal = jnp.dot(g_ee, n_s)
        residual_gravity = (1.0 - cfg.grav_comp) * gravity_normal
        effective_press = (
            f_cmd - residual_gravity
            + cfg.kp_force * (f_cmd - f_real)
            + s.info["force_int"]
        )
        # This mask is invariant to h_surf_tangential/soft_contact_manifold and
        # therefore comparable across methods.
        on_path_common = (
            (jnp.linalg.norm(h_tangent) < e._config.track_tol) & (f_real > 0.5)
        ).astype(jnp.float32)
        K = e._stiffness(s_vec)
        return {"ee": ee, "h": h, "h_rest": h_rest, "h_tangent": h_tangent,
                "normal_offset": normal_offset[None], "xi": xi[None],
                # Metrics must use the executed SPD matrix, not a sampled chart
                # coordinate that is ignored by fixed/no-stiffness ablations.
                "stiffness": K.reshape(-1), "on_surface": on_surface[None],
                "on_path_common": on_path_common[None],
                "force": f_real[None], "force_cmd": f_cmd[None], "contact": in_contact[None],
                "penetration": pen[None], "k_surf": k_surf[None],
                "gate_scalar": reliability["scalar"][None],
                "gate_path": reliability["path"][None],
                "gate_normal": reliability["normal"][None],
                "gate_stiffness": reliability["stiffness"][None],
                "gate_force": reliability["force"][None],
                "gate_path_error": reliability["path_error"][None],
                "gate_normal_error": reliability["normal_error"][None],
                "gate_force_error": reliability["force_error"][None],
                "gate_deformation_risk": reliability["deformation_risk"][None],
                "gate_contact_loss": reliability["contact_loss"][None],
                "realization_offset": realization_offset,
                "force_int": s.info["force_int"][None],
                "gravity_normal": gravity_normal[None],
                "effective_press": effective_press[None]}

    d = _contact_step_signals(trajectory, env, actions, per_step, kw)
    cfg = env._config
    f_target = float(getattr(cfg, "f_target", 0.0))
    h = np.asarray(d["h"])                                 # (T, 6) = [h_surf(3); h_normal(3)]
    n = h.shape[0]
    acquisition_steps = min(int(getattr(cfg, "metric_acquisition_steps", 5)), n)
    settled = np.arange(n) >= acquisition_steps
    xi = d["xi"].reshape(-1)
    xi0 = float(np.asarray(getattr(env, "_xi0", xi[0] if xi.size else 0.0)))
    scan_span = float(getattr(cfg, "scan_span", 1.0))
    scan_rate = float(getattr(cfg, "scan_rate", scan_span))
    # Evaluate the reference segment that was actually scheduled within this
    # finite episode. Legacy runs that reach scan_span are unchanged.
    scan_target = xi0 + min(scan_span, max(n - 1, 0) * scan_rate)
    ref_xi = jnp.linspace(xi0, scan_target, 101, dtype=jnp.float32)
    ref_eta = jnp.asarray(getattr(env, "_eta0", 0.5), jnp.float32)
    reference_path = np.asarray(
        jax.vmap(lambda x: env._desired_pose(x, ref_eta)[2])(ref_xi),
        np.float64,
    )
    path_err = np.linalg.norm(np.asarray(d["h_tangent"]), axis=1)
    force_err = np.abs(np.asarray(d["force"]).reshape(-1) - f_target)
    f_span = max(float(getattr(cfg, "f_max", 0.0) - getattr(cfg, "f_min", 0.0)), 1e-6)
    def_scale = max(float(getattr(cfg, "deformation_scale", 0.005)), 1e-6)
    def_safe = float(getattr(cfg, "deformation_safe", 0.005))
    force_risk = (
        force_err / f_span
        + np.maximum(np.asarray(d["penetration"]).reshape(-1) - def_safe, 0.0) / def_scale
        + (np.asarray(d["contact"]).reshape(-1) <= 0.5).astype(np.float64)
    )
    return {
        "positions": d["ee"], "final_pos": d["ee"][-1],
        "reference_path": reference_path,
        # Coverage/progress use the complete executed EE trajectory.  The
        # contact mask, rather than an arbitrary acquisition-time crop, decides
        # whether a visited point belongs to the physical scan.
        "coverage_positions": d["ee"],
        "coverage_valid_mask": d["contact"].reshape(-1),
        "path_tolerance": float(getattr(cfg, "metric_path_tolerance", 0.005)),
        "controls": np.asarray(actions, np.float64),
        "h": h.reshape(-1),                                # all equality residuals
        "h_surf": h[:, :3], "h_normal": h[:, 3:],          # surface / normal tracking
        "h_rest": d["h_rest"], "h_tangent": d["h_tangent"],
        "normal_offset": d["normal_offset"].reshape(-1),
        "stiffness": d["stiffness"], "force": d["force"].reshape(-1),
        "force_cmd": d["force_cmd"].reshape(-1), "in_contact": d["contact"].reshape(-1),
        "on_surface": d["on_surface"].reshape(-1),
        "on_path_common": d["on_path_common"].reshape(-1),
        "settled": settled.astype(np.float64),
        "in_contact_settled": d["contact"].reshape(-1)[acquisition_steps:],
        "deformation": d["penetration"].reshape(-1),   # per-step surface deformation (soft/hybrid)
        "k_surf": d["k_surf"].reshape(-1),              # local surface stiffness (spatial on hybrid)
        "scan_start": np.array([xi0]), "scan_final": np.array([np.max(xi)]),
        "scan_target": np.array([scan_target]), "scan_xi": xi,
        # Gate at post-action state t predicts consistency after the next action.
        # Shift by one step to avoid the tautological same-state correlation.
        "gate_path_predictor": d["gate_path"].reshape(-1)[:-1],
        "next_path_quality": -path_err[1:],
        "gate_force_predictor": d["gate_force"].reshape(-1)[:-1],
        "next_force_quality": -force_risk[1:],
        "gate_scalar": d["gate_scalar"].reshape(-1),
        "gate_path": d["gate_path"].reshape(-1),
        "gate_normal": d["gate_normal"].reshape(-1),
        "gate_stiffness": d["gate_stiffness"].reshape(-1),
        "gate_force": d["gate_force"].reshape(-1),
        "gate_path_error": d["gate_path_error"].reshape(-1),
        "gate_normal_error": d["gate_normal_error"].reshape(-1),
        "gate_force_error": d["gate_force_error"].reshape(-1),
        "gate_deformation_risk": d["gate_deformation_risk"].reshape(-1),
        "gate_contact_loss": d["gate_contact_loss"].reshape(-1),
        "realization_offset": d["realization_offset"],
        "force_int": d["force_int"].reshape(-1),
        "gravity_normal": d["gravity_normal"].reshape(-1),
        "effective_press": d["effective_press"].reshape(-1),
        "force_des": np.full(h.shape[0], f_target),
        "f_min": float(getattr(cfg, "f_min", 0.0)), "f_max": float(getattr(cfg, "f_max", 0.0)),
        "seconds": float(kw.get("planning_time", 0.0)),
    }


# idea.txt Exp I metrics; surface/normal tracking reuse equality_residual_rms
# bound to the h_surf / h_normal residual components (no task-specific registry).
ARM_METRICS = [
    {"name": "equality_residual_rms", "as": "surface_tracking_error", "bind": {"h": "h_surf"}},
    {"name": "equality_residual_rms", "as": "rest_surface_error", "bind": {"h": "h_rest"}},
    {"name": "equality_residual_rms", "as": "tangential_tracking_error", "bind": {"h": "h_tangent"}},
    {"name": "equality_residual_rms", "as": "normal_offset_rms", "bind": {"h": "normal_offset"}},
    {"name": "equality_residual_rms", "as": "normal_alignment_error", "bind": {"h": "h_normal"}},
    "force_tracking_error", "contact_loss_rate",
    # force precision scored ONLY while actually tracking the scan path (on_surface) ->
    # a baseline that lags off the path can't earn a steady-force advantage by not scanning.
    {"name": "force_tracking_error_tracked", "as": "force_tracking_error_tracked",
     "bind": {"force": "force", "force_des": "force_des", "mask": "on_surface"}},
    {"name": "force_tracking_error_tracked", "as": "force_tracking_error_tracked_common",
     "bind": {"force": "force", "force_des": "force_des", "mask": "on_path_common"}},
    {"name": "force_tracking_error_tracked", "as": "force_tracking_error_settled",
     "bind": {"force": "force", "force_des": "force_des", "mask": "settled"}},
    {"name": "rate", "as": "on_path_rate", "bind": {"mask": "on_path_common"}},
    {"name": "contact_loss_rate", "as": "contact_loss_rate_settled",
     "bind": {"in_contact": "in_contact_settled"}},
    {"name": "force_violation_rate", "as": "force_violation_rate", "bind": {"force": "force"}},
    {"name": "force_violation_rate", "as": "force_command_violation_rate",
     "bind": {"force": "force_cmd"}},
    {"name": "cvar", "as": "force_cvar95", "bind": {"x": "force"}},   # worst-case high force tail
    # soft/hybrid medium: surface deformation + compliant-contact transients (~0 on rigid).
    "deformation_depth", "deformation_peak",
    {"name": "cvar", "as": "deformation_cvar95", "bind": {"x": "deformation"}},
    "force_overshoot", "contact_chatter",
    {"name": "progress_ratio", "as": "scan_progress_ratio",
     "bind": {"start_pos": "scan_start", "final_pos": "scan_final", "target": "scan_target"}},
    {"name": "path_completion", "as": "realized_path_completion",
     "bind": {"positions": "positions", "reference_path": "reference_path"}},
    {"name": "max_path_progress", "as": "max_realized_path_progress",
     "bind": {"positions": "coverage_positions", "reference_path": "reference_path",
              "valid_mask": "coverage_valid_mask"}},
    {"name": "trajectory_path_coverage", "as": "trajectory_path_coverage",
     "bind": {"positions": "coverage_positions", "reference_path": "reference_path",
              "valid_mask": "coverage_valid_mask", "tolerance": "path_tolerance"}},
    {"name": "pearson_correlation", "as": "gate_path_next_error_corr",
     "bind": {"x": "gate_path_predictor", "y": "next_path_quality"}},
    {"name": "pearson_correlation", "as": "gate_force_next_risk_corr",
     "bind": {"x": "gate_force_predictor", "y": "next_force_quality"}},
    "stiffness_adaptation_corr",        # RQ2: commanded-K vs local k_surf (hybrid spatial map)
    "control_smoothness", "stiffness_smoothness", "energy", "runtime",
]


def arm_surface_scan_metrics_plugin(name: str = "arm_surface_scan_metrics"):
    return GeneralMetricsPlugin([*ARM_METRICS], extractor=arm_surface_scan_signals, name=name)


def peg_insert_signals(trajectory, env, obstacles, constraints, **kw) -> Dict[str, Any]:
    """Insertion wrench, progress, contact-mode, jam, and recovery signals."""
    import jax
    import jax.numpy as jnp

    actions = jnp.asarray(
        [np.ravel(np.asarray(a, np.float32)) for a in trajectory.actions]
    )

    def per_step(e, s, u):
        pose, angle_vec = e._actual_pose(s.pipeline_state)
        comp = e.wrench_components(s.info["true_wrench"])
        measured_comp = e.wrench_components(s.info["measured_wrench"])
        delta_wrench = s.info["wrench_queue"][0] - s.info["wrench_queue"][1]
        reliability = e.geometry_reliability(s)
        force_violation = (
            (comp["lateral_force"] > e._config.lateral_force_limit)
            | (comp["axial_force"] > e._config.f_max)
        ).astype(jnp.float32)
        torque_violation = (
            (comp["bending_torque"] > e._config.bending_torque_limit)
            | (comp["torsional_torque"] > e._config.torsional_torque_limit)
        ).astype(jnp.float32)
        safety_cost = (
            jax.nn.relu(
                comp["lateral_force"] / e._config.lateral_force_limit - 1.0
            ) ** 2
            + jax.nn.relu(comp["axial_force"] / e._config.f_max - 1.0) ** 2
            + jax.nn.relu(
                comp["bending_torque"] / e._config.bending_torque_limit - 1.0
            ) ** 2
            + jax.nn.relu(
                comp["torsional_torque"] / e._config.torsional_torque_limit - 1.0
            ) ** 2
        )
        return {
            "pose": pose,
            "angle_vec": angle_vec,
            "insertion_depth": s.info["prev_depth"][None],
            "lateral_error": jnp.linalg.norm(pose[:2])[None],
            "angle_error": jnp.linalg.norm(angle_vec)[None],
            "lateral_force": comp["lateral_force"][None],
            "axial_force": comp["axial_force"][None],
            "bending_torque": comp["bending_torque"][None],
            "torsional_torque": comp["torsional_torque"][None],
            "force_violation": force_violation[None],
            "torque_violation": torque_violation[None],
            "safety_cost": safety_cost[None],
            "jammed": s.info["jammed"][None],
            "jammed_once": s.info["jammed_once"][None],
            "recovered": s.info["recovered"][None],
            "success": s.info["success"][None],
            "contact_mode": s.info["contact_mode"][None],
            "contact": (s.info["contact_count"] > 0).astype(jnp.float32)[None],
            "penetration": s.info["penetration"][None],
            "measured_lateral_force": measured_comp["lateral_force"][None],
            "measured_axial_force": measured_comp["axial_force"][None],
            "measured_bending_torque": measured_comp["bending_torque"][None],
            "measured_wrench_delta": jnp.linalg.norm(delta_wrench[:3])[None],
            "contact_count": s.info["contact_count"][None],
            "stall_steps": s.info["stall_steps"][None],
            "gate_scalar": reliability["scalar"][None],
            "gate_wrench": reliability["wrench"][None],
            "gate_force": reliability["force"][None],
            "stiffness": u[e.spec.s_slice],
        }

    d = _contact_step_signals(trajectory, env, actions, per_step, kw)

    depth = d["insertion_depth"].reshape(-1)
    safety = d["safety_cost"].reshape(-1)
    max_depth = np.maximum.accumulate(depth) if depth.size else depth
    backout = max_depth - depth
    unsafe_seen = np.maximum.accumulate(np.maximum.reduce([
        d["force_violation"].reshape(-1),
        d["torque_violation"].reshape(-1),
        d["jammed_once"].reshape(-1),
    ]))
    strict_safe_success = d["success"].reshape(-1) * (unsafe_seen <= 0.5)
    return {
        "controls": np.asarray(actions, np.float64),
        "stiffness": d["stiffness"],
        "pose": d["pose"],
        "angle_vec": d["angle_vec"],
        "insertion_depth": depth,
        "lateral_error": d["lateral_error"].reshape(-1),
        "angle_error": d["angle_error"].reshape(-1),
        "lateral_force": d["lateral_force"].reshape(-1),
        "axial_force": d["axial_force"].reshape(-1),
        "bending_torque": d["bending_torque"].reshape(-1),
        "torsional_torque": d["torsional_torque"].reshape(-1),
        "force_violation": d["force_violation"].reshape(-1),
        "torque_violation": d["torque_violation"].reshape(-1),
        "force_torque_violation": np.maximum(
            d["force_violation"].reshape(-1),
            d["torque_violation"].reshape(-1),
        ),
        "safety_cost": safety,
        "jammed": d["jammed"].reshape(-1),
        "jammed_once": d["jammed_once"].reshape(-1),
        "recovered": d["recovered"].reshape(-1),
        "success": d["success"].reshape(-1),
        "strict_safe_success": strict_safe_success,
        "contact_mode": d["contact_mode"].reshape(-1),
        "in_contact": d["contact"].reshape(-1),
        "penetration": d["penetration"].reshape(-1),
        "measured_lateral_force": d["measured_lateral_force"].reshape(-1),
        "measured_axial_force": d["measured_axial_force"].reshape(-1),
        "measured_bending_torque": d["measured_bending_torque"].reshape(-1),
        "measured_wrench_delta": d["measured_wrench_delta"].reshape(-1),
        "contact_count": d["contact_count"].reshape(-1),
        "stall_steps": d["stall_steps"].reshape(-1),
        "backout": backout,
        "gate_scalar": d["gate_scalar"].reshape(-1),
        "gate_wrench": d["gate_wrench"].reshape(-1),
        "gate_force": d["gate_force"].reshape(-1),
        "gate_predictor": d["gate_scalar"].reshape(-1)[:-1],
        "next_safety_quality": -safety[1:],
        "dt": float(env.dt),
        "seconds": float(kw.get("planning_time", 0.0)),
    }


PEG_INSERT_METRICS = [
    {"name": "event_occurred", "as": "insertion_success", "bind": {"mask": "success"}},
    {"name": "event_occurred", "as": "safe_insertion_success",
     "bind": {"mask": "strict_safe_success"}},
    {"name": "maximum", "as": "max_insertion_depth", "bind": {"x": "insertion_depth"}},
    {"name": "terminal_value", "as": "terminal_insertion_depth", "bind": {"x": "insertion_depth"}},
    {"name": "first_event_time", "as": "completion_time", "bind": {"mask": "success"}},
    {"name": "maximum", "as": "peak_lateral_force", "bind": {"x": "lateral_force"}},
    {"name": "maximum", "as": "peak_axial_force", "bind": {"x": "axial_force"}},
    {"name": "maximum", "as": "peak_bending_torque", "bind": {"x": "bending_torque"}},
    {"name": "maximum", "as": "peak_torsional_torque", "bind": {"x": "torsional_torque"}},
    {"name": "rate", "as": "force_torque_violation_rate", "bind": {"mask": "force_torque_violation"}},
    {"name": "event_occurred", "as": "any_force_torque_violation",
     "bind": {"mask": "force_torque_violation"}},
    {"name": "rate", "as": "torque_violation_rate", "bind": {"mask": "torque_violation"}},
    {"name": "event_occurred", "as": "jam_rate", "bind": {"mask": "jammed_once"}},
    {"name": "conditional_event_success", "as": "recovery_success",
     "bind": {"trigger": "jammed", "outcome": "recovered"}},
    {"name": "time_integral", "as": "cumulative_safety_cost", "bind": {"x": "safety_cost"}},
    {"name": "cvar", "as": "lateral_force_cvar95", "bind": {"x": "lateral_force"}},
    {"name": "cvar", "as": "axial_force_cvar95", "bind": {"x": "axial_force"}},
    {"name": "maximum", "as": "max_backout_distance", "bind": {"x": "backout"}},
    {"name": "transition_count", "as": "contact_mode_transitions", "bind": {"x": "contact_mode"}},
    {"name": "pearson_correlation", "as": "gate_next_safety_corr",
     "bind": {"x": "gate_predictor", "y": "next_safety_quality"}},
    "control_smoothness", "stiffness_smoothness", "energy", "runtime",
]


def peg_insert_metrics_plugin(name: str = "peg_insert_metrics"):
    return GeneralMetricsPlugin(
        [*PEG_INSERT_METRICS], extractor=peg_insert_signals, name=name
    )


def humanoid_box_push_signals(trajectory, env, obstacles, constraints, **kw) -> Dict[str, Any]:
    """Box-push/unjam (idea.txt Exp II): roll the H1 to collect box x (vs goal),
    the full contact manifold (h_box/h_hand/h_hand_R/h_foot, g_bal/g_fric/g_tip),
    hand contact/slip/force, CoM/feet, fall flag, and the stiffness chart."""
    import jax.numpy as jnp
    from brax import math as bmath
    actions = jnp.asarray([np.ravel(np.asarray(a, np.float32)) for a in trajectory.actions])
    goal_x = float(np.asarray(_x0(env, kw).info["box_goal_x"]))
    ez = jnp.array([0.0, 0.0, 1.0])

    def per_step(e, s, u):
        ps = s.pipeline_state
        c = e._hand_contact(ps, u, s.info)
        h, g = e._manifold(ps, u, s.info)                  # fixed stance h(14), walk h(10), g(3)
        x = ps.x
        box_x = x.pos[e._box_idx - 1, 0]
        box_yaw = bmath.quat_to_euler(x.rot[e._box_idx - 1])[2]
        corridor_clearance = e._corridor_clearance(ps)
        com = x.pos[e._pelvis_idx - 1, :2]
        feet_c = ps.site_xpos[e._feet_site_id].mean(axis=0)[:2]
        up = jnp.dot(bmath.rotate(ez, x.rot[e._torso_idx - 1]), ez)
        fell = ((up < 0) | (x.pos[e._torso_idx - 1, 2] < 0.5)).astype(jnp.float32)
        # REAL hand->box push force from the mjx contact (not the commanded f_n);
        # in_contact = a nonzero physical push force (not the kinematic h_hand proxy).
        box_forces = e._box_contact_forces(ps)
        f_real = box_forces["hand"]
        wall_force = box_forces["wall"]
        wall_contact = (wall_force > 0.5).astype(jnp.float32)
        nonhand_contact = (box_forces["nonhand"] > 0.5).astype(jnp.float32)
        in_contact = (f_real > 0.5).astype(jnp.float32)
        elapsed = jnp.asarray(s.info["step"], jnp.float32) * e.dt
        force_scale = jnp.clip(
            (elapsed - e._bcfg.approach_time)
            / max(float(e._bcfg.force_ramp_time), 1e-6),
            0.0, 1.0,
        )
        task_active = s.info.get("task_success", jnp.float32(0.0)) < 0.5
        force_des = jnp.float32(e._bcfg.f_target) * force_scale * task_active
        expected_contact = (force_scale >= 0.99) & task_active
        in_contact_expected = jnp.where(expected_contact, in_contact, 1.0)
        force_violation = jnp.maximum(f_real - e._bcfg.f_max, 0.0)
        safety_g = jnp.maximum(
            jnp.maximum(jnp.maximum(g[0], fell), nonhand_contact),
            force_violation / max(float(e._bcfg.f_max), 1e-6),
        )
        return {"box_x": box_x[None], "box_yaw": box_yaw[None],
                "corridor_clearance": corridor_clearance[None],
                "com": com, "feet": feet_c, "fell": fell[None],
                "g_bal": g[0][None], "g_fric": g[1][None], "tip_series": (-g[2])[None],
                "ft": jnp.linalg.norm(c["f_t"])[None], "fn": c["f_n"][None],
                "slip": c["slip"][None], "force": f_real[None],
                "force_des": force_des[None], "safety_g": safety_g[None],
                "force_expected": expected_contact.astype(jnp.float32)[None],
                "wall_force": wall_force[None],
                "wall_contact": wall_contact[None],
                "nonhand_contact": nonhand_contact[None],
                "in_contact": in_contact[None],
                "in_contact_expected": in_contact_expected[None],
                "stiffness": u[e.spec.s_slice]}

    d = _contact_step_signals(trajectory, env, actions, per_step, kw)
    final_yaw = float(np.asarray(d["box_yaw"][-1]).reshape(-1)[0])
    x_error = abs(float(np.asarray(d["box_x"][-1]).reshape(-1)[0]) - goal_x)
    if str(env._bcfg.level).lower() == "unjam":
        task_goal_error = max(
            x_error,
            float(env._bcfg.goal_eps) * abs(final_yaw) / max(float(env._bcfg.unjam_yaw_eps), 1e-6),
        )
    else:
        task_goal_error = x_error
    return {
        "positions": d["box_x"], "box_x": d["box_x"].reshape(-1),
        "box_yaw": d["box_yaw"].reshape(-1),
        "start_pos": np.array([float(np.asarray(_x0(env, kw).info["box_x0"]))]),
        "final_pos": d["box_x"][-1], "target": np.array([goal_x]),
        "goal_tolerance": float(env._bcfg.goal_eps),
        "task_goal_error": task_goal_error,
        "final_yaw": np.array([final_yaw]), "target_yaw": np.array([0.0]),
        "corridor_clearance": d["corridor_clearance"].reshape(-1),
        "controls": np.asarray(actions, np.float64), "stiffness": d["stiffness"],
        "com_xy": d["com"], "support_center": d["feet"].mean(axis=0),
        "support_radius": float(getattr(env._bcfg, "support_radius", 0.25)),
        "fell": d["fell"].reshape(-1),
        "g_bal": d["g_bal"].reshape(-1), "g_fric": d["g_fric"].reshape(-1),
        "tip_series": d["tip_series"].reshape(-1),
        "f_tangential": d["ft"].reshape(-1), "f_normal": d["fn"].reshape(-1),
        "mu": float(np.asarray(env._mu)),
        "slip_speed": d["slip"].reshape(-1), "force": d["force"].reshape(-1),
        "force_des": d["force_des"].reshape(-1),
        "force_expected": d["force_expected"].reshape(-1),
        "g_safety": d["safety_g"].reshape(-1),
        "f_min": 0.0, "f_max": float(env._bcfg.f_max), "dt": float(env.dt),
        "wall_force": d["wall_force"].reshape(-1),
        "wall_contact": d["wall_contact"].reshape(-1),
        "nonhand_contact": d["nonhand_contact"].reshape(-1),
        "in_contact": d["in_contact"].reshape(-1),
        "in_contact_expected": d["in_contact_expected"].reshape(-1),
        "seconds": float(kw.get("planning_time", 0.0)),
    }


# idea.txt Exp II metric set (signal keys named to match metric params -> bare
# requests; a few bound to specific g-components / force).
HUMANOID_METRICS = [
    "goal_error",                                          # box pose error vs goal
    "progress_ratio",
    {"name": "completion_step", "as": "completion_step",
     "bind": {"margin": "goal_tolerance"}},
    {"name": "max_violation", "as": "safety_violation", "bind": {"g": "g_safety"}},
    {"name": "success", "as": "safe_success",
     "bind": {"goal_error": "task_goal_error", "violation": "safety_violation",
              "margin": "goal_tolerance"}},
    {"name": "goal_error", "as": "yaw_error",
     "bind": {"final_pos": "final_yaw", "target": "target_yaw"}},
    {"name": "tip_margin", "as": "corridor_clearance_min",
     "bind": {"margin_series": "corridor_clearance"}},
    "fall_rate", "balance_margin", "tangential_slip",
    {"name": "contact_loss_rate", "as": "contact_loss_rate",
     "bind": {"in_contact": "in_contact_expected"}},
    "friction_cone_violation_rate",
    {"name": "tip_margin", "as": "tip_margin", "bind": {"margin_series": "tip_series"}},
    {"name": "violation_rate", "as": "balance_violation_rate", "bind": {"g": "g_bal"}},
    {"name": "violation_cvar", "as": "balance_violation_cvar", "bind": {"g": "g_bal"}},
    {"name": "rate", "as": "nonhand_collision_rate", "bind": {"mask": "nonhand_contact"}},
    {"name": "rate", "as": "wall_contact_rate", "bind": {"mask": "wall_contact"}},
    {"name": "cvar", "as": "wall_force_cvar95", "bind": {"x": "wall_force"}},
    {"name": "cvar", "as": "force_cvar95", "bind": {"x": "force"}},
    "force_tracking_error", "force_tracking_mae", "normalized_force_tracking_mae",
    {"name": "force_tracking_error_tracked", "as": "steady_force_tracking_error",
     "bind": {"mask": "force_expected"}},
    {"name": "force_tracking_mae_tracked", "as": "steady_force_tracking_mae",
     "bind": {"mask": "force_expected"}},
    {"name": "normalized_force_tracking_mae_tracked", "as": "steady_normalized_force_tracking_mae",
     "bind": {"mask": "force_expected"}},
    "force_peak", "force_overshoot", "force_overshoot_ratio",
    "force_excess_impulse", "force_settling_time", "force_violation_rate",
    "contact_chatter",
    "control_smoothness", "stiffness_smoothness", "energy", "runtime",
]


def humanoid_box_push_metrics_plugin(name: str = "humanoid_box_push_metrics"):
    return GeneralMetricsPlugin(
        [*HUMANOID_METRICS], extractor=humanoid_box_push_signals, name=name,
        config={
            "force_settling_time": {
                "band_fraction": 0.1, "band_absolute": 1.0, "hold_steps": 10,
            },
        },
    )


__all__ = [
    "corridor_signals", "corridor_metrics_plugin", "CORRIDOR_METRICS",
    "arm_surface_scan_signals", "arm_surface_scan_metrics_plugin", "ARM_METRICS",
    "peg_insert_signals", "peg_insert_metrics_plugin", "PEG_INSERT_METRICS",
    "humanoid_box_push_signals", "humanoid_box_push_metrics_plugin", "HUMANOID_METRICS",
]

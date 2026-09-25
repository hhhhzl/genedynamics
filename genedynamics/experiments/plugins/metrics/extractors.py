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
                "stiffness": K.reshape(-1),
                "normal_stiffness": (n_s @ K @ n_s)[None],
                "surface_normal": n_s,
                "on_surface": on_surface[None],
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
        "stiffness": d["stiffness"],
        "normal_stiffness": d["normal_stiffness"].reshape(-1),
        "surface_normal": d["surface_normal"],
        "force": d["force"].reshape(-1),
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
    {"name": "stiffness_adaptation_corr", "as": "stiffness_adaptation_corr",
     "bind": {"stiffness": "normal_stiffness", "k_surf": "k_surf"}},
    "control_smoothness", "stiffness_smoothness", "energy", "runtime",
]


def arm_surface_scan_metrics_plugin(name: str = "arm_surface_scan_metrics"):
    return GeneralMetricsPlugin(
        [*ARM_METRICS], extractor=arm_surface_scan_signals, name=name,
        persist_signals=True,
    )


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
        rho = jnp.max(jnp.asarray([
            comp["lateral_force"] / max(e._config.lateral_force_limit, 1.0e-6),
            comp["axial_force"] / max(e._config.f_max, 1.0e-6),
            comp["bending_torque"] / max(e._config.bending_torque_limit, 1.0e-6),
            comp["torsional_torque"] / max(e._config.torsional_torque_limit, 1.0e-6),
        ]))
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
            "rho": rho[None],
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
    # Historical PegInsert results used a prefix-safe event: completion counted
    # when no violation had happened *yet*, even if the fixed-length execution
    # violated a wrench limit afterwards.  Preserve that diagnostic explicitly,
    # while the clean-room formal SSR requires the entire recorded execution to
    # remain inside the physical force/torque and jam limits.
    prefix_safe_success = d["success"].reshape(-1) * (unsafe_seen <= 0.5)
    whole_execution_safe = bool(unsafe_seen.size) and bool(unsafe_seen[-1] <= 0.5)
    whole_safe_success = (
        d["success"].reshape(-1) * float(whole_execution_safe)
    )
    rho = d["rho"].reshape(-1)
    infos = [row for row in kw.get("infos", ()) if isinstance(row, dict)]
    diagnostic_signals = {}
    for key in (
        "incumbent_revalidated_safe", "refined_revalidated_safe",
        "selected_revalidated_safe", "emergency_selected",
        "emergency_revalidated_safe", "emergency_incumbent_active",
        "emergency_unrecoverable", "reliability_abstained",
    ):
        values = [row.get(key, np.nan) for row in infos]
        if values:
            diagnostic_signals[key] = np.asarray(values, np.float64).reshape(-1)
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
        "rho": rho,
        "rho_violation": (rho > 1.0).astype(np.float64),
        "jammed": d["jammed"].reshape(-1),
        "jammed_once": d["jammed_once"].reshape(-1),
        "recovered": d["recovered"].reshape(-1),
        "success": d["success"].reshape(-1),
        # Backward-compatible signal name for archived plotting tools.
        "strict_safe_success": prefix_safe_success,
        "prefix_safe_success": prefix_safe_success,
        "whole_safe_success": whole_safe_success,
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
        **diagnostic_signals,
    }


PEG_INSERT_METRICS = [
    {"name": "event_occurred", "as": "insertion_success", "bind": {"mask": "success"}},
    {"name": "event_occurred", "as": "safe_insertion_success",
     "bind": {"mask": "whole_safe_success"}},
    {"name": "event_occurred", "as": "prefix_safe_insertion_success",
     "bind": {"mask": "prefix_safe_success"}},
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
    {"name": "maximum", "as": "peak_rho", "bind": {"x": "rho"}},
    {"name": "cvar", "as": "rho_cvar95", "bind": {"x": "rho"}},
    {"name": "rate", "as": "rho_violation_rate", "bind": {"mask": "rho_violation"}},
    {"name": "time_integral", "as": "time_above_rho_one",
     "bind": {"x": "rho_violation"}},
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
        [*PEG_INSERT_METRICS], extractor=peg_insert_signals, name=name,
        persist_signals=True,
        execution_failure_metrics={
            "insertion_success": 0.0,
            "safe_insertion_success": 0.0,
            "prefix_safe_insertion_success": 0.0,
        },
    )


def _humanoid_goal_signals(box_x, box_yaw, task_success, goal_x, cfg):
    """Keep physical box error separate from the task's completion contract.

    Retain the legacy geometric residual separately from true task completion.
    Locomotion P4 must satisfy the environment's walking and dwell conditions;
    reaching the line with the box alone is not a completed walking task.
    """
    residual = np.abs(np.asarray(box_x).reshape(-1) - goal_x)
    task_error = float(residual[-1])
    if str(cfg.level).lower() == "unjam":
        task_error = max(
            task_error,
            float(cfg.goal_eps) * abs(float(np.asarray(box_yaw).reshape(-1)[-1]))
            / max(float(cfg.unjam_yaw_eps), 1e-6),
        )
    if (str(cfg.level).lower() == "push_walk"
            and getattr(cfg, "walk_success_mode", "legacy") == "locomotion"):
        incomplete = np.asarray(task_success).reshape(-1) < 0.5
        residual = np.maximum(residual, incomplete * (2.0 * float(cfg.goal_eps)))
        task_error = float(residual[-1])
    return {
        "task_goal_error": task_error,
        "completion_residual": residual[:, None],
        "task_completion_residual": (np.asarray(task_success).reshape(-1) < 0.5).astype(float)[:, None],
        "completion_target": np.zeros(1),
    }


_HUMANOID_WALK_DIAGNOSTIC_KEYS = (
    "foot_normal_loads", "foot_floor_clearance", "foot_ground_contact",
    "robot_subtree_com", "torso_up", "torso_height", "box_forward_velocity",
)


def _humanoid_walk_diagnostics(env, pipeline_state):
    """Persist physical support evidence without changing legacy safety metrics."""
    cfg = env._bcfg
    if not (str(cfg.level).lower() == "push_walk"
            and getattr(cfg, "walk_success_mode", "legacy") == "locomotion"
            and getattr(cfg, "walk_leg_control", "legacy") == "joint_target"):
        return {}
    return env._walk_contact_diagnostics(pipeline_state)


_HUMANOID_LEGACY_EVALUATION_CONTRACT = {
    "version": 1,
    "physical_samples": "exclude_only_contiguous_success_padding_suffix",
    "completion_event": "post_transition_task_success",
    "completion_step_indexing": "zero_based_post_transition_row",
}

_HUMANOID_EVALUATION_CONTRACT = {
    "version": 3,
    "physical_samples": "active_prefix_then_actual_post_physics_substeps",
    "completion_event": "post_transition_task_success",
    "completion_step_indexing": "zero_based_post_transition_row",
    "physics_clock": "control_start_plus_j_plus_one_times_physics_dt",
    "physics_integral": "right_rectangle_excluding_initial_state_and_padding",
    "primary_force_tail": "physics_force_normalized_cvar95",
    "primary_force_peak": "physics_force_peak",
    "primary_force_violation_rate": "physics_force_violation_rate",
    "safe_success": "initial_and_active_substep_margins_with_complete_coverage",
    "failure_tail": "retain_first_fall_transition_then_exclude_post_terminal_rollout",
    "endpoint_force_metrics": "retained_20ms_diagnostics_not_primary_safety",
}

_HUMANOID_PHYSICS_ARRAYS = (
    "physics_hand_force", "physics_nonhand_force", "physics_safety_margins",
)

# Only physical time-series consumed by H1 metrics are shortened.  In particular,
# reliability arrays describe the original rollout contract and remain untouched.
_HUMANOID_PHYSICAL_SERIES = (
    "positions", "box_x", "box_y", "box_yaw", "completion_residual",
    "task_completion_residual", "corridor_clearance", "controls", "stiffness",
    "com_xy", "support_center", "pelvis_position", "feet_positions",
    "walk_body_progress", "walk_support_progress", "walk_forward_steps",
    "walk_left_steps", "walk_right_steps", "walk_swing_seen", "walk_foot_loaded",
    "walk_swing_eligible", "walk_landing_x", "walk_goal_ready", "walk_goal_hold_time",
    "walk_force_scale", "fell", "g_bal", "g_fric", "tip_series", "f_tangential",
    "f_normal", "slip_speed", "force", "force_des", "task_force_reference",
    "execution_force_reference", "force_effective", "contact_acquired", "contact_step",
    "force_int", "unjam_released", "task_success", "task_fallen", "success_padding",
    "safety_margins", "force_rise_mask", "force_expected", "g_safety", "wall_force",
    "wall_contact", "nonhand_contact", "in_contact", "in_contact_expected", "g",
    *_HUMANOID_WALK_DIAGNOSTIC_KEYS,
    *_HUMANOID_PHYSICS_ARRAYS, "physics_samples_valid",
    "mga_execution_mode", "mga_emergency_zero_force",
    "reliability_execution_applicable",
)


def _humanoid_evaluation_view(signals):
    """Exclude terminal suffixes while retaining the terminal transition.

    Successful H1 rollouts already mark their unexecuted absorbing suffix.
    Fixed-length diagnostic drivers can also keep calling ``step`` after a
    fall, however.  Those later states are not part of the episode and must not
    create locomotion progress or completed foot steps.  Historical signals
    need an explicitly documented offline audit; absence of the collection-time
    contract must not silently select a new metric protocol.
    """
    metadata = signals.get("task_metadata") or {}
    is_h1 = str(metadata.get("robot", "h1")).lower() == "h1"
    contract = (_HUMANOID_EVALUATION_CONTRACT if is_h1
                else _HUMANOID_LEGACY_EVALUATION_CONTRACT)
    if (metadata.get("evaluation_contract") != contract
            or "success_padding" not in signals):
        raise ValueError("H1 evaluation contract/padding missing; historical signals are audit-only")
    raw_padding = np.asarray(signals["success_padding"]).reshape(-1)
    if not np.all(np.isin(raw_padding, [False, True])):
        raise ValueError("H1 success_padding must be finite binary flags")
    padding = raw_padding.astype(bool)
    count = len(padding)
    starts = np.flatnonzero(padding)
    success_active = int(starts[0]) if starts.size else count
    if success_active == 0 or (starts.size and not np.all(padding[success_active:])):
        raise ValueError("H1 padding must be a contiguous suffix after an actual transition")
    success = np.asarray(signals["task_success"]).reshape(-1)
    if len(success) != count or (starts.size and not np.all(success[success_active - 1:] > 0.5)):
        raise ValueError("H1 padding must follow the retained successful transition")
    fallen = np.asarray(signals.get("task_fallen", signals.get("fell", np.zeros(count)))).reshape(-1)
    if fallen.shape != (count,) or not np.all(np.isin(fallen, [0.0, 1.0, False, True])):
        raise ValueError("H1 task_fallen/fell must be finite binary flags")
    fall_events = np.flatnonzero(fallen > 0.5)
    # Retain the physical transition that first detects the fall, then discard
    # all fixed-horizon calls after termination.  Taking the minimum also makes
    # simultaneous success/fall obey the environment's fall-priority rule.
    fall_active = int(fall_events[0]) + 1 if fall_events.size else count
    active = min(success_active, fall_active)
    if "physics_samples_valid" in signals:
        physics_valid = np.asarray(signals["physics_samples_valid"]).reshape(-1)
        if (physics_valid.shape != (count,)
                or not np.all(np.isin(physics_valid, [False, True]))):
            raise ValueError("H1 physics_samples_valid must be finite binary transition flags")
        if np.any(physics_valid[padding]):
            raise ValueError("H1 success padding cannot claim executed physics samples")
    view = dict(signals)
    for name in _HUMANOID_PHYSICAL_SERIES:
        if name not in signals:
            continue
        values = np.asarray(signals[name])
        if values.ndim == 0 or len(values) != count:
            raise ValueError(f"H1 physical series {name!r} must match transition count")
        view[name] = values[:active]
    # Scalar endpoint aliases are constructed by the extractor before the
    # evaluation view.  Rebind them to the retained episode endpoint so that
    # secondary pose/progress diagnostics cannot use post-fall motion.
    if active > 0 and "positions" in view:
        view["final_pos"] = np.asarray(view["positions"])[-1]
    if active > 0 and "box_yaw" in view:
        view["final_yaw"] = np.atleast_1d(np.asarray(view["box_yaw"])[-1])
    if "task_goal_error" in view and not bool(view.get("force_step_applicable", False)):
        # A required alias prevents raw/legacy compute_metrics calls from
        # bypassing this view and using success.violation's default zero.
        view["safe_success_goal_error"] = view["task_goal_error"]
    if not is_h1:
        # G1 retains its endpoint protocol; it must not be mistaken for an H1
        # rollout with missing physical coverage or receive fabricated samples.
        view["safe_success_violation"] = float(np.max(np.maximum(view["g_safety"], 0.0)))
        view["safe_success_violation_tol"] = 1e-6
        return view
    return _humanoid_physics_evaluation_view(view)


def _humanoid_physics_evaluation_view(view):
    """Bind actual substep statistics without changing the endpoint time grid.

    Missing/partial coverage retains diagnostics but cannot certify success.
    In particular, an absent optional ``success.violation`` would otherwise
    default to zero in the general metric library.  The internal gate is always
    present; its infinity is not exported as an observed physical violation.
    """
    count = len(view["success_padding"])
    valid = np.asarray(view.get("physics_samples_valid", np.zeros(count))).reshape(-1)
    if valid.shape != (count,) or not np.all(np.isin(valid, [False, True])):
        raise ValueError("H1 physics_samples_valid must be finite binary transition flags")
    view["physics_samples_valid"] = valid
    view["safe_success_violation"] = np.inf
    view["safe_success_violation_tol"] = 0.0
    if not np.all(valid):
        return view
    if any(key not in view for key in _HUMANOID_PHYSICS_ARRAYS):
        raise ValueError("H1 valid physics coverage requires actual substep arrays")
    force = np.asarray(view["physics_hand_force"], dtype=float)
    nonhand = np.asarray(view["physics_nonhand_force"], dtype=float)
    margins = np.asarray(view["physics_safety_margins"], dtype=float)
    if (force.ndim != 2 or force.shape[0] != count or force.shape[1] < 1
            or nonhand.shape != force.shape or margins.shape != (*force.shape, 4)):
        raise ValueError("H1 physics arrays must have shapes [T,n_frames] and [T,n_frames,4]")
    physics_dt = float(view.get("physics_dt", 0.0))
    dt = float(view["dt"])
    f_max = float(view["f_max"])
    initial = np.asarray(view.get("physics_initial_safety_margins", ()), dtype=float)
    if (not np.isfinite(physics_dt) or physics_dt <= 0.0
            or not np.isfinite(dt) or dt <= 0.0
            or not np.isclose(force.shape[1] * physics_dt, dt, rtol=1e-6, atol=1e-10)):
        raise ValueError("H1 n_frames * physics_dt must equal the control dt")
    if not np.isfinite(f_max) or f_max <= 0.0:
        raise ValueError("H1 physical force normalization requires a positive finite f_max")
    if (initial.shape != (4,) or not all(np.all(np.isfinite(value))
            for value in (force, nonhand, margins, initial))
            or np.any(force < 0.0) or np.any(nonhand < 0.0)):
        raise ValueError("H1 claimed-valid physics and initial safety samples must be finite")
    samples = force.reshape(-1)
    reference = np.asarray(
        view.get("task_force_reference", view.get("force_des", ())), dtype=float
    ).reshape(-1)
    if reference.shape != (count,):
        raise ValueError("H1 physical force evaluation requires a per-transition reference")
    default_expected = (
        np.abs(reference) >= 0.99 * float(np.max(np.abs(reference)))
        if reference.size and np.max(np.abs(reference)) > 0.0
        else np.zeros(reference.shape, dtype=bool)
    )
    expected = np.asarray(view.get("force_expected", default_expected), dtype=float).reshape(-1)
    if expected.shape != (count,):
        raise ValueError("H1 physical force evaluation requires a per-transition tracking mask")
    reference_samples = np.repeat(reference, force.shape[1])
    expected_samples = np.repeat(expected, force.shape[1])
    physics_g = np.max(margins.reshape(-1, 4), axis=1)
    view.update({
        "physics_hand_force_samples": samples,
        "physics_hand_force_ratio_samples": samples / f_max,
        "physics_force_reference_samples": reference_samples,
        "physics_force_expected_samples": expected_samples,
        "physics_force_rise_mask_samples": (
            samples >= float(view.get("force_step_rise_fraction", 0.9))
            * float(view.get("force_step_target", np.max(np.abs(reference_samples))))
        ).astype(float),
        "physics_force_limit_profile": np.full(samples.shape, f_max),
        "physics_nonhand_collision_samples": nonhand.reshape(-1) > 0.5,
        "physics_g_safety_samples": physics_g,
        "physics_safety_margin_envelope": np.append(physics_g, np.max(initial)),
        # Initial conditions have zero duration: include them in certification,
        # not in CVaR, time fractions, or impulse quadrature.
        "safe_success_violation": max(float(np.max(initial)), float(np.max(physics_g)), 0.0),
    })
    if bool(view.get("force_step_applicable", False)):
        target = float(view["force_step_target"])
        band = max(
            float(view["force_step_band_absolute"]),
            float(view["force_step_band_fraction"]) * target,
        )
        hold = max(1, int(np.ceil(
            float(view["force_step_hold_time"]) / physics_dt - 1.0e-9
        )))
        tracked = expected_samples > 0.5
        tracked_indices = np.flatnonzero(tracked)
        terminal_hold = (
            tracked_indices.size >= hold
            and np.all(np.abs(
                samples[tracked_indices[-hold:]] - reference_samples[tracked_indices[-hold:]]
            ) <= band)
        )
        rose = bool(np.any(
            view["physics_force_rise_mask_samples"] * tracked
        ))
        safe = float(view["safe_success_violation"]) <= 0.0
        view["force_step_pass"] = np.asarray([
            rose and terminal_hold and safe
        ], dtype=float)
    return view


def humanoid_box_push_signals(trajectory, env, obstacles, constraints, **kw) -> Dict[str, Any]:
    """Box-push/unjam (idea.txt Exp II): roll the H1 to collect box x (vs goal),
    the full contact manifold (h_box/h_hand/h_hand_R/h_foot, g_bal/g_fric/g_tip),
    hand contact/slip/force, CoM/feet, fall flag, and the stiffness chart."""
    import jax
    import jax.numpy as jnp
    from brax import math as bmath
    actions = jnp.asarray([np.ravel(np.asarray(a, np.float32)) for a in trajectory.actions])
    is_h1 = str(getattr(env._bcfg, "robot", "h1")).lower() == "h1"
    states = list(getattr(trajectory, "states", ()))
    actual_states = (len(states) == len(actions) + 1 and len(actions) > 0
                     and hasattr(states[0], "pipeline_state"))
    initial = states[0] if actual_states else _x0(env, kw)
    goal_x = float(np.asarray(initial.info["box_goal_x"]))
    initial_safety = np.full(4, np.nan)
    if is_h1 and actual_states:
        ps0 = initial.pipeline_state
        balance0 = (jnp.linalg.norm(ps0.x.pos[env._pelvis_idx - 1, :2]
                    - ps0.site_xpos[env._feet_site_id, :2].mean(axis=0))
                    - env._bcfg.support_radius)
        initial_safety = np.asarray(env._safety_margins(
            ps0, env._box_contact_forces(ps0), balance0,
        ), dtype=float)

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
        fell = e._has_fallen(ps).astype(jnp.float32)
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
        # The transition that first reaches success was physically executed.
        # Only later frozen padding lacks an executed force reference.
        task_active = ~jnp.asarray(s.info["success_padding"], jnp.bool_)
        walk_force_scale = e.walk_force_scale(ps, s.info)
        benchmark_force = e.requested_force_reference(ps, s.info)
        force_des = benchmark_force * task_active
        contact_acquired = jnp.asarray(s.info.get("contact_acquired", 0.0), jnp.float32)
        contact_step = jnp.asarray(s.info.get("contact_step", -1), jnp.float32)
        execution_elapsed = jnp.maximum(
            jnp.asarray(s.info["step"], jnp.float32) - contact_step, 0.0
        ) * e.dt
        execution_scale = contact_acquired * jnp.clip(
            execution_elapsed / max(float(e._bcfg.force_ramp_time), 1e-6),
            0.0, 1.0,
        )
        if getattr(e._bcfg, "walk_force_startup_mode", "legacy") == "synchronized":
            # Nominal post-state reference on the selected execution clock;
            # this is not the action-selected F_n or the measured hand force.
            execution_scale = e._normal_force_startup_scale(s.info)
        execution_force_des = (
            jnp.float32(e._bcfg.f_target) * execution_scale * task_active * walk_force_scale
        )
        expected_contact = (force_scale >= 0.99) & task_active & (walk_force_scale > 0.0)
        in_contact_expected = jnp.where(expected_contact, in_contact, 1.0)
        safety_margins = e._safety_margins(ps, box_forces, g[0])
        safety_g = jnp.max(safety_margins)
        _, _, _, _, K_hand, _ = e._unpack(u)
        if is_h1 and callable(getattr(e, "realized_hand_stiffness", None)):
            K_hand = e.realized_hand_stiffness(s, u)
        # This is the same state/action/context contact realization as ``c``
        # above.  Reuse it rather than repeating an expensive host-side MJX
        # contact evaluation for every executed state.
        contact_command = c
        reliability_features = e.reliability_features(s, u)
        transition_margins = (e._transition_safety_margins(s, safety_margins)
                              if is_h1 else safety_margins)
        reliability_risk = jnp.asarray([
            (transition_margins[0] > 0.0).astype(jnp.float32),
            ((transition_margins[1] > 0.0) | (transition_margins[3] > 0.0)).astype(jnp.float32),
            jax.nn.relu(transition_margins[2]),
            # This performance head remains explicitly endpoint-based; only
            # the three physical safety heads use the substep envelope.
            jnp.abs(f_real - benchmark_force)
            / max(float(e._bcfg.f_target), 1.0),
        ])
        physics_signals = {}
        if is_h1:
            physics_valid = (
                jnp.asarray(s.info.get("physics_samples_valid", False), jnp.bool_)
                & actual_states
            )
            # Unknown coverage is not an observed violation label.  The
            # existing fitter rejects these nonfinite rows explicitly.  A
            # frozen success endpoint remains known without invented substeps.
            # Missing mode is unknown, not permission to back-fill NORMAL
            # labels into historical/abnormal states with known physical tape.
            execution_mode = jnp.asarray(s.info.get("mga_execution_mode", -1), jnp.int32)
            execution_applicable = execution_mode == 0
            # UNLOAD is observed physics but outside the normal-only learned
            # model's label population. Do not call it unknown physics.
            risk_valid = actual_states & ((~task_active) | physics_valid)
            risk_valid = risk_valid & execution_applicable & jnp.all(jnp.isfinite(reliability_risk))
            execution_force_des = jnp.where(execution_mode == 1, 0.0, execution_force_des)
            reliability_risk = jnp.where(risk_valid, reliability_risk,
                                         jnp.full_like(reliability_risk, jnp.nan))
            physics_signals = {
                "physics_samples_valid": physics_valid[None],
                "reliability_risk_valid": jnp.asarray(risk_valid)[None],
                "mga_execution_mode": execution_mode[None],
                "mga_emergency_zero_force": jnp.asarray(
                    s.info.get("mga_emergency_zero_force", False), jnp.bool_
                )[None],
                "reliability_execution_applicable": execution_applicable[None],
                **{key: s.info[key] for key in _HUMANOID_PHYSICS_ARRAYS if key in s.info},
            }
        return {"box_x": box_x[None], "box_y": x.pos[e._box_idx - 1, 1][None],
                "box_yaw": box_yaw[None],
                "corridor_clearance": corridor_clearance[None],
                "com": com, "feet": feet_c, "fell": fell[None],
                "pelvis_position": x.pos[e._pelvis_idx - 1],
                "feet_positions": ps.site_xpos[e._feet_site_id],
                "walk_body_progress": jnp.asarray(s.info.get("walk_body_progress", 0.0))[None],
                "walk_support_progress": jnp.asarray(s.info.get("walk_support_progress", 0.0))[None],
                "walk_forward_steps": s.info.get("walk_forward_steps", jnp.zeros(2, jnp.int32)),
                "walk_swing_seen": s.info.get("walk_swing_seen", jnp.zeros(2, jnp.bool_)),
                "walk_foot_loaded": s.info.get("walk_foot_loaded", jnp.zeros(2, jnp.bool_)),
                "walk_swing_eligible": s.info.get("walk_swing_eligible", jnp.zeros(2, jnp.bool_)),
                "walk_landing_x": s.info.get("walk_landing_x", jnp.zeros(2)),
                "walk_goal_ready": jnp.asarray(s.info.get("walk_goal_ready", False))[None],
                "walk_goal_hold_time": jnp.asarray(s.info.get("walk_goal_hold_time", 0.0))[None],
                "walk_force_scale": walk_force_scale[None],
                "g_bal": g[0][None], "g_fric": g[1][None], "tip_series": (-g[2])[None],
                "ft": jnp.linalg.norm(c["f_t"])[None], "fn": c["f_n"][None],
                "slip": c["slip"][None], "force": f_real[None],
                "force_des": force_des[None],
                "execution_force_des": execution_force_des[None],
                "force_effective": contact_command["F_eff"][None],
                "contact_acquired": contact_acquired[None],
                "contact_step": contact_step[None],
                "force_int": s.info.get("force_int", jnp.float32(0.0))[None],
                "unjam_released": s.info.get("unjam_released", jnp.float32(0.0))[None],
                "task_success": s.info.get("task_success", jnp.float32(0.0))[None],
                "task_fallen": jnp.asarray(s.info.get("task_fallen", 0.0))[None],
                "success_padding": jnp.asarray(s.info.get("success_padding", False))[None],
                "safety_margins": safety_margins,
                "safety_g": safety_g[None],
                "force_expected": expected_contact.astype(jnp.float32)[None],
                "wall_force": wall_force[None],
                "wall_contact": wall_contact[None],
                "nonhand_contact": nonhand_contact[None],
                "in_contact": in_contact[None],
                "in_contact_expected": in_contact_expected[None],
                "stiffness": K_hand.reshape(-1),
                "g": g,
                "reliability_features": reliability_features,
                "reliability_risk": reliability_risk,
                **physics_signals,
                **{key: jnp.atleast_1d(value) for key, value in
                   _humanoid_walk_diagnostics(e, ps).items()}}

    d = _contact_step_signals(trajectory, env, actions, per_step, kw)
    final_yaw = float(np.asarray(d["box_yaw"][-1]).reshape(-1)[0])
    goal_signals = _humanoid_goal_signals(
        d["box_x"], d["box_yaw"], d["task_success"], goal_x, env._bcfg,
    )
    return {
        "positions": d["box_x"], "box_x": d["box_x"].reshape(-1),
        "box_y": d["box_y"].reshape(-1), "box_yaw": d["box_yaw"].reshape(-1),
        "start_pos": np.array([float(np.asarray(initial.info["box_x0"]))]),
        "final_pos": d["box_x"][-1], "target": np.array([goal_x]),
        "goal_tolerance": float(env._bcfg.goal_eps),
        **goal_signals,
        "final_yaw": np.array([final_yaw]), "target_yaw": np.array([0.0]),
        "corridor_clearance": d["corridor_clearance"].reshape(-1),
        "controls": np.asarray(actions, np.float64), "stiffness": d["stiffness"],
        "com_xy": d["com"], "support_center": d["feet"],
        "pelvis_position": d["pelvis_position"], "feet_positions": d["feet_positions"],
        "walk_body_progress": d["walk_body_progress"].reshape(-1),
        "walk_support_progress": d["walk_support_progress"].reshape(-1),
        "walk_forward_steps": d["walk_forward_steps"],
        "walk_left_steps": d["walk_forward_steps"][:, 0],
        "walk_right_steps": d["walk_forward_steps"][:, 1],
        "walk_swing_seen": d["walk_swing_seen"],
        "walk_foot_loaded": d["walk_foot_loaded"],
        "walk_swing_eligible": d["walk_swing_eligible"],
        "walk_landing_x": d["walk_landing_x"],
        "walk_goal_ready": d["walk_goal_ready"].reshape(-1),
        "walk_goal_hold_time": d["walk_goal_hold_time"].reshape(-1),
        "walk_force_scale": d["walk_force_scale"].reshape(-1),
        "support_radius": float(getattr(env._bcfg, "support_radius", 0.25)),
        "fell": d["fell"].reshape(-1),
        "g_bal": d["g_bal"].reshape(-1), "g_fric": d["g_fric"].reshape(-1),
        "tip_series": d["tip_series"].reshape(-1),
        "f_tangential": d["ft"].reshape(-1), "f_normal": d["fn"].reshape(-1),
        "mu": float(np.asarray(env._mu)),
        "slip_speed": d["slip"].reshape(-1), "force": d["force"].reshape(-1),
        "force_des": d["force_des"].reshape(-1),
        "task_force_reference": d["force_des"].reshape(-1),
        "execution_force_reference": d["execution_force_des"].reshape(-1),
        "force_effective": d["force_effective"].reshape(-1),
        "contact_acquired": d["contact_acquired"].reshape(-1),
        "contact_step": d["contact_step"].reshape(-1),
        "force_int": d["force_int"].reshape(-1),
        "unjam_released": d["unjam_released"].reshape(-1),
        "task_success": d["task_success"].reshape(-1),
        "task_fallen": d["task_fallen"].reshape(-1),
        "success_padding": d["success_padding"].reshape(-1),
        "safety_margins": d["safety_margins"],
        "force_rise_mask": (
            d["force"].reshape(-1) >= 0.9 * float(env._bcfg.f_target)
        ).astype(np.float64),
        "force_expected": d["force_expected"].reshape(-1),
        "force_step_applicable": bool(getattr(env._bcfg, "fixed_force_target", False)),
        "force_step_target": float(env._bcfg.f_target),
        "force_step_rise_fraction": float(
            getattr(env._bcfg, "force_step_rise_fraction", 0.9)
        ),
        "force_step_band_fraction": float(
            getattr(env._bcfg, "force_step_band_fraction", 0.1)
        ),
        "force_step_band_absolute": float(
            getattr(env._bcfg, "force_step_band_absolute", 1.0)
        ),
        "force_step_hold_time": float(
            getattr(env._bcfg, "force_step_hold_time", 0.2)
        ),
        "g_safety": d["safety_g"].reshape(-1),
        "f_min": 0.0, "f_max": float(env._bcfg.f_max), "dt": float(env.dt),
        "wall_force": d["wall_force"].reshape(-1),
        "wall_contact": d["wall_contact"].reshape(-1),
        "nonhand_contact": d["nonhand_contact"].reshape(-1),
        "in_contact": d["in_contact"].reshape(-1),
        "in_contact_expected": d["in_contact_expected"].reshape(-1),
        "g": d["g"],
        "reliability_features": d["reliability_features"],
        "reliability_risk": d["reliability_risk"],
        **({
            "physics_samples_valid": d["physics_samples_valid"].reshape(-1),
            "reliability_risk_valid": d["reliability_risk_valid"].reshape(-1),
            "mga_execution_mode": d["mga_execution_mode"].reshape(-1),
            "mga_emergency_zero_force": d["mga_emergency_zero_force"].reshape(-1),
            "reliability_execution_applicable": d["reliability_execution_applicable"].reshape(-1),
            "physics_dt": float(getattr(env._bcfg, "timestep", 0.0)),
            "physics_initial_safety_margins": initial_safety,
            **{key: d[key] for key in _HUMANOID_PHYSICS_ARRAYS if key in d},
        } if is_h1 else {}),
        "task_metadata": {
            "reliability_contract": env.reliability_contract(),
            "reliability_source_sha256": env.reliability_source_sha256(),
            "robot": str(getattr(env._bcfg, "robot", "h1")).lower(),
            "evaluation_contract": dict(_HUMANOID_EVALUATION_CONTRACT if is_h1
                                        else _HUMANOID_LEGACY_EVALUATION_CONTRACT),
            **({"physics_evidence_source": "executed_states" if actual_states else "legacy_action_replay"}
               if is_h1 else {}),
        },
        **{key: d[key] for key in _HUMANOID_WALK_DIAGNOSTIC_KEYS if key in d},
        "seconds": float(kw.get("planning_time", 0.0)),
    }


# idea.txt Exp II metric set (signal keys named to match metric params -> bare
# requests; a few bound to specific g-components / force).
HUMANOID_METRICS = [
    "goal_error",                                          # box pose error vs goal
    "progress_ratio",
    {"name": "completion_step", "as": "completion_step",
     "bind": {"positions": "task_completion_residual", "target": "completion_target"},
     "config": {"margin": 0.5}},
    {"name": "completion_step", "as": "box_completion_step",
     "bind": {"positions": "positions", "target": "target", "margin": "goal_tolerance"}},
    {"name": "event_occurred", "as": "task_success", "bind": {"mask": "task_success"}},
    {"name": "event_occurred", "as": "force_step_success",
     "bind": {"mask": "force_step_pass"}},
    {"name": "maximum", "as": "walk_body_progress_max", "bind": {"x": "walk_body_progress"}},
    {"name": "maximum", "as": "walk_support_progress_max", "bind": {"x": "walk_support_progress"}},
    {"name": "terminal_value", "as": "walk_body_progress_final", "bind": {"x": "walk_body_progress"}},
    {"name": "terminal_value", "as": "walk_support_progress_final", "bind": {"x": "walk_support_progress"}},
    {"name": "maximum", "as": "walk_left_steps", "bind": {"x": "walk_left_steps"}},
    {"name": "maximum", "as": "walk_right_steps", "bind": {"x": "walk_right_steps"}},
    {"name": "max_violation", "as": "safety_violation", "bind": {"g": "g_safety"}},
    {"name": "success", "as": "safe_success",
     "bind": {"goal_error": "safe_success_goal_error", "violation": "safe_success_violation",
              "margin": "goal_tolerance", "violation_tol": "safe_success_violation_tol"}},
    {"name": "rate", "as": "physics_sample_coverage", "bind": {"mask": "physics_samples_valid"}},
    {"name": "max_violation", "as": "physics_safety_violation",
     "bind": {"g": "physics_safety_margin_envelope"}},
    {"name": "force_peak", "as": "physics_force_peak", "bind": {"force": "physics_hand_force_samples"}},
    {"name": "force_tracking_error_tracked", "as": "physics_steady_force_tracking_error",
     "bind": {"force": "physics_hand_force_samples",
              "force_des": "physics_force_reference_samples",
              "mask": "physics_force_expected_samples"}},
    {"name": "force_tracking_mae_tracked", "as": "physics_steady_force_tracking_mae",
     "bind": {"force": "physics_hand_force_samples",
              "force_des": "physics_force_reference_samples",
              "mask": "physics_force_expected_samples"}},
    {"name": "normalized_force_tracking_mae_tracked",
     "as": "physics_steady_normalized_force_tracking_mae",
     "bind": {"force": "physics_hand_force_samples",
              "force_des": "physics_force_reference_samples",
              "mask": "physics_force_expected_samples"}},
    {"name": "force_overshoot", "as": "physics_force_overshoot",
     "bind": {"force": "physics_hand_force_samples",
              "force_des": "physics_force_reference_samples"}},
    {"name": "force_overshoot_ratio", "as": "physics_force_overshoot_ratio",
     "bind": {"force": "physics_hand_force_samples",
              "force_des": "physics_force_reference_samples"}},
    {"name": "force_settling_time", "as": "physics_force_settling_time",
     "bind": {"force": "physics_hand_force_samples",
              "force_des": "physics_force_reference_samples", "dt": "physics_dt"},
     "config": {"band_fraction": 0.1, "band_absolute": 1.0, "hold_steps": 50}},
    {"name": "first_event_time", "as": "physics_force_rise_time",
     "bind": {"mask": "physics_force_rise_mask_samples", "dt": "physics_dt"}},
    {"name": "cvar", "as": "physics_force_cvar95", "bind": {"x": "physics_hand_force_samples"},
     "config": {"alpha": 0.95}},
    {"name": "cvar", "as": "physics_force_normalized_cvar95",
     "bind": {"x": "physics_hand_force_ratio_samples"}, "config": {"alpha": 0.95}},
    {"name": "force_violation_rate", "as": "physics_force_violation_rate",
     "bind": {"force": "physics_hand_force_samples"}},
    {"name": "time_integral", "as": "physics_force_impulse",
     "bind": {"x": "physics_hand_force_samples", "dt": "physics_dt"}},
    {"name": "force_excess_impulse", "as": "physics_force_limit_excess_impulse",
     "bind": {"force": "physics_hand_force_samples", "force_des": "physics_force_limit_profile",
              "dt": "physics_dt"}},
    {"name": "rate", "as": "physics_nonhand_collision_rate",
     "bind": {"mask": "physics_nonhand_collision_samples"}},
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
    {"name": "first_event_time", "as": "force_rise_time",
     "bind": {"mask": "force_rise_mask"}},
    "contact_chatter",
    "control_smoothness", "stiffness_smoothness", "energy", "runtime",
]


def humanoid_box_push_metrics_plugin(name: str = "humanoid_box_push_metrics"):
    return GeneralMetricsPlugin(
        [*HUMANOID_METRICS], extractor=humanoid_box_push_signals, name=name,
        persist_signals=True, evaluation_view=_humanoid_evaluation_view,
        execution_failure_metrics={"task_success": 0.0, "safe_success": 0.0},
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

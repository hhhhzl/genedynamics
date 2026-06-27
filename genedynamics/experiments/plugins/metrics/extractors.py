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
are pure-numpy. brax tasks (surface-scan / box-push) roll the env over the
executed actions to recover per-step signals lost in a flattened trajectory
(surface coords, contact residuals, com) — that rolling is brax/mjx (docker).
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


def arm_surface_scan_signals(trajectory, env, obstacles, constraints, **kw) -> Dict[str, Any]:
    """Surface-scan: roll the arm to collect EE pose, the surface/normal tracking
    residual ``h``, the commanded normal force, and the stiffness chart."""
    import jax.numpy as jnp
    actions = jnp.asarray([np.ravel(np.asarray(a, np.float32)) for a in trajectory.actions])

    cfg = env._config

    def per_step(e, s, u):
        h, _ = e.constraint_residual(s, u)
        _, s_vec, f_cmd = e._unpack(u)                     # physical stiffness svec + commanded force
        xi, eta = s.info["xi"], s.info["eta"]
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
        ee = s.pipeline_state.site_xpos[e._ee_site]
        return {"ee": ee, "h": h, "stiffness": s_vec, "on_surface": on_surface[None],
                "force": f_real[None], "force_cmd": f_cmd[None], "contact": in_contact[None],
                "penetration": pen[None], "k_surf": k_surf[None]}

    d = _roll_brax(env, _x0(env, kw), actions, per_step)
    cfg = env._config
    f_target = float(getattr(cfg, "f_target", 0.0))
    h = np.asarray(d["h"])                                 # (T, 6) = [h_surf(3); h_normal(3)]
    return {
        "positions": d["ee"], "final_pos": d["ee"][-1],
        "controls": np.asarray(actions, np.float64),
        "h": h.reshape(-1),                                # all equality residuals
        "h_surf": h[:, :3], "h_normal": h[:, 3:],          # surface / normal tracking
        "stiffness": d["stiffness"], "force": d["force"].reshape(-1),
        "force_cmd": d["force_cmd"].reshape(-1), "in_contact": d["contact"].reshape(-1),
        "on_surface": d["on_surface"].reshape(-1),
        "deformation": d["penetration"].reshape(-1),   # per-step surface deformation (soft/hybrid)
        "k_surf": d["k_surf"].reshape(-1),              # local surface stiffness (spatial on hybrid)
        "force_des": np.full(h.shape[0], f_target),
        "f_min": float(getattr(cfg, "f_min", 0.0)), "f_max": float(getattr(cfg, "f_max", 0.0)),
        "seconds": float(kw.get("planning_time", 0.0)),
    }


# idea.txt Exp I metrics; surface/normal tracking reuse equality_residual_rms
# bound to the h_surf / h_normal residual components (no task-specific registry).
ARM_METRICS = [
    {"name": "equality_residual_rms", "as": "surface_tracking_error", "bind": {"h": "h_surf"}},
    {"name": "equality_residual_rms", "as": "normal_alignment_error", "bind": {"h": "h_normal"}},
    "force_tracking_error", "contact_loss_rate",
    # force precision scored ONLY while actually tracking the scan path (on_surface) ->
    # a baseline that lags off the path can't earn a steady-force advantage by not scanning.
    {"name": "force_tracking_error_tracked", "as": "force_tracking_error_tracked",
     "bind": {"force": "force", "force_des": "force_des", "mask": "on_surface"}},
    {"name": "force_violation_rate", "as": "force_violation_rate", "bind": {"force": "force_cmd"}},
    {"name": "cvar", "as": "force_cvar95", "bind": {"x": "force"}},   # worst-case high force tail
    # soft/hybrid medium: surface deformation + compliant-contact transients (~0 on rigid).
    "deformation_depth", "deformation_peak",
    {"name": "cvar", "as": "deformation_cvar95", "bind": {"x": "deformation"}},
    "force_overshoot", "contact_chatter",
    "stiffness_adaptation_corr",        # RQ2: commanded-K vs local k_surf (hybrid spatial map)
    "control_smoothness", "stiffness_smoothness", "energy", "runtime",
]


def arm_surface_scan_metrics_plugin(name: str = "arm_surface_scan_metrics"):
    return GeneralMetricsPlugin([*ARM_METRICS], extractor=arm_surface_scan_signals, name=name)


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
        c = e._hand_contact(ps, u)
        h, g = e._manifold(ps, u, s.info)                  # h(11), g(3)
        x = ps.x
        box_x = x.pos[e._box_idx - 1, 0]
        com = x.pos[e._pelvis_idx - 1, :2]
        feet_c = ps.site_xpos[e._feet_site_id].mean(axis=0)[:2]
        up = jnp.dot(bmath.rotate(ez, x.rot[e._torso_idx - 1]), ez)
        fell = ((up < 0) | (x.pos[e._torso_idx - 1, 2] < 0.5)).astype(jnp.float32)
        # REAL hand->box push force from the mjx contact (not the commanded f_n);
        # in_contact = a nonzero physical push force (not the kinematic h_hand proxy).
        f_real = e._box_contact_force(ps)
        in_contact = (f_real > 0.5).astype(jnp.float32)
        return {"box_x": box_x[None], "com": com, "feet": feet_c, "fell": fell[None],
                "g_bal": g[0][None], "g_fric": g[1][None], "tip_series": (-g[2])[None],
                "ft": jnp.linalg.norm(c["f_t"])[None], "fn": c["f_n"][None],
                "slip": c["slip"][None], "force": f_real[None],
                "in_contact": in_contact[None], "stiffness": u[e.spec.s_slice]}

    d = _roll_brax(env, _x0(env, kw), actions, per_step)
    return {
        "final_pos": d["box_x"][-1], "target": np.array([goal_x]),
        "controls": np.asarray(actions, np.float64), "stiffness": d["stiffness"],
        "com_xy": d["com"], "support_center": d["feet"].mean(axis=0),
        "support_radius": float(getattr(env._bcfg, "support_radius", 0.25)),
        "fell": d["fell"].reshape(-1),
        "g_bal": d["g_bal"].reshape(-1), "g_fric": d["g_fric"].reshape(-1),
        "tip_series": d["tip_series"].reshape(-1),
        "f_tangential": d["ft"].reshape(-1), "f_normal": d["fn"].reshape(-1),
        "mu": float(np.asarray(env._mu)),
        "slip_speed": d["slip"].reshape(-1), "force": d["force"].reshape(-1),
        "in_contact": d["in_contact"].reshape(-1),
        "seconds": float(kw.get("planning_time", 0.0)),
    }


# idea.txt Exp II metric set (signal keys named to match metric params -> bare
# requests; a few bound to specific g-components / force).
HUMANOID_METRICS = [
    "goal_error",                                          # box pose error vs goal
    "fall_rate", "balance_margin", "tangential_slip",
    "contact_loss_rate", "friction_cone_violation_rate",
    {"name": "tip_margin", "as": "tip_margin", "bind": {"margin_series": "tip_series"}},
    {"name": "violation_rate", "as": "balance_violation_rate", "bind": {"g": "g_bal"}},
    {"name": "violation_cvar", "as": "balance_violation_cvar", "bind": {"g": "g_bal"}},
    {"name": "cvar", "as": "force_cvar95", "bind": {"x": "force"}},
    "control_smoothness", "stiffness_smoothness", "energy", "runtime",
]


def humanoid_box_push_metrics_plugin(name: str = "humanoid_box_push_metrics"):
    return GeneralMetricsPlugin([*HUMANOID_METRICS], extractor=humanoid_box_push_signals, name=name)


__all__ = [
    "corridor_signals", "corridor_metrics_plugin", "CORRIDOR_METRICS",
    "arm_surface_scan_signals", "arm_surface_scan_metrics_plugin", "ARM_METRICS",
    "humanoid_box_push_signals", "humanoid_box_push_metrics_plugin", "HUMANOID_METRICS",
]

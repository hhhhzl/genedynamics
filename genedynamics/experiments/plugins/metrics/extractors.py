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

    def per_step(e, s, u):
        h, _ = e.constraint_residual(s, u)
        _, s_vec, f_n = e._unpack(u)                       # physical stiffness svec + force
        ee = s.pipeline_state.site_xpos[e._ee_site]
        return {"ee": ee, "h": h, "stiffness": s_vec, "force": f_n[None]}

    d = _roll_brax(env, _x0(env, kw), actions, per_step)
    f_target = float(getattr(env._config, "f_target", 0.0))
    return {
        "positions": d["ee"], "final_pos": d["ee"][-1],
        "controls": np.asarray(actions, np.float64),
        "h": d["h"].reshape(-1),                           # all equality residuals
        "stiffness": d["stiffness"], "force": d["force"].reshape(-1),
        "force_des": np.full(d["force"].shape[0], f_target),
        "seconds": float(kw.get("planning_time", 0.0)),
    }


ARM_METRICS = [
    "equality_residual_rms",          # surface + normal tracking (the task constraint)
    "force_tracking_error",
    "control_smoothness", "stiffness_smoothness", "energy", "runtime",
]


def arm_surface_scan_metrics_plugin(name: str = "arm_surface_scan_metrics"):
    return GeneralMetricsPlugin([*ARM_METRICS], extractor=arm_surface_scan_signals, name=name)


def humanoid_box_push_signals(trajectory, env, obstacles, constraints, **kw) -> Dict[str, Any]:
    """Box-push: roll the H1 to collect box x (vs goal), the balance inequality
    ``g`` and hand-contact residual ``h``, CoM, and the stiffness chart."""
    import jax.numpy as jnp
    actions = jnp.asarray([np.ravel(np.asarray(a, np.float32)) for a in trajectory.actions])
    goal_x = float(np.asarray(_x0(env, kw).info["box_goal_x"]))

    def per_step(e, s, u):
        h, g = e.constraint_residual(s, u)                 # h=h_hand(3), g=g_bal(1)
        x = s.pipeline_state.x
        box_x = x.pos[e._box_idx - 1, 0]
        com = x.pos[e._pelvis_idx - 1, :2]
        feet = s.pipeline_state.site_xpos[e._feet_site_id].mean(axis=0)[:2]
        return {"box_x": box_x[None], "com": com, "feet": feet, "h": h, "g": g,
                "stiffness": u[e.spec.s_slice]}

    d = _roll_brax(env, _x0(env, kw), actions, per_step)
    return {
        "final_pos": d["box_x"][-1], "target": np.array([goal_x]),
        "controls": np.asarray(actions, np.float64),
        "g": d["g"].reshape(-1),                           # balance violations
        "h": d["h"].reshape(-1),                           # hand-contact residual
        "com_xy": d["com"], "support_center": d["feet"].mean(axis=0),
        "stiffness": d["stiffness"],
        "seconds": float(kw.get("planning_time", 0.0)),
    }


HUMANOID_METRICS = [
    "goal_error",                     # box x vs goal x
    "max_violation", "violation_rate", "violation_cvar",   # balance (g)
    "equality_residual_rms",          # hand contact (h)
    "balance_margin",
    "control_smoothness", "stiffness_smoothness", "energy", "runtime",
]


def humanoid_box_push_metrics_plugin(name: str = "humanoid_box_push_metrics"):
    return GeneralMetricsPlugin([*HUMANOID_METRICS], extractor=humanoid_box_push_signals, name=name)


__all__ = [
    "corridor_signals", "corridor_metrics_plugin", "CORRIDOR_METRICS",
    "arm_surface_scan_signals", "arm_surface_scan_metrics_plugin", "ARM_METRICS",
    "humanoid_box_push_signals", "humanoid_box_push_metrics_plugin", "HUMANOID_METRICS",
]

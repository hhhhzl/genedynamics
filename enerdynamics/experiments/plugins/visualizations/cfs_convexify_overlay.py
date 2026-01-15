"""
CFS convexify overlay utilities for visualizations.

This module draws the *convexified* (linearized) obstacle constraints produced by CFS:
    A x >= b
For 2D position-space environments, each constraint row corresponds to a halfspace:
    g^T p_t >= b

We use it as an overlay on existing trajectory/diffusion plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from enerdynamics.core.constraints.convexify import CFSConvexifier
from enerdynamics.core.constraints.core.array_interface import BackendArray
from enerdynamics.core.constraints.core.types import ScheduleParams, ScheduleState
from enerdynamics.core.types import Trajectory


def _is_cfs_enabled(exp_cfg: Any, obstacles: Any) -> bool:
    """Best-effort check: only draw when CFS is enabled and obstacles exist."""
    try:
        if obstacles is None or len(obstacles) == 0:
            return False
    except Exception:
        # If obstacles doesn't implement __len__, assume present
        if obstacles is None:
            return False

    cc = getattr(exp_cfg, "constraint_config", None) or {}
    cfs_cfg = cc.get("cfs", {}) if isinstance(cc, dict) else {}
    if isinstance(cfs_cfg, dict) and cfs_cfg.get("enabled", True) is False:
        return False
    return True


def _infer_margin(exp_cfg: Any) -> float:
    """
    Infer a reasonable margin for CFS convexification.

    This repo often treats margin as "clearance" or "robot radius".
    We default to robot_radius if nothing else is configured.
    """
    obstacle_cfg = getattr(exp_cfg, "obstacle_config", None) or {}
    robot_radius = float(obstacle_cfg.get("robot_radius", 0.05))

    cc = getattr(exp_cfg, "constraint_config", None) or {}
    if isinstance(cc, dict):
        hard = cc.get("hard_constraint", {}) or {}
        sched = cc.get("schedule", {}) or {}
        # Prefer explicit clearances if present
        if isinstance(sched, dict) and "hard_clearance_end" in sched:
            return float(sched.get("hard_clearance_end")) + robot_radius
        if isinstance(hard, dict) and "clearance" in hard:
            return float(hard.get("clearance")) + robot_radius

    return robot_radius


def _to_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, BackendArray):
        return arr.to_numpy()
    return np.asarray(arr, dtype=np.float32)


def _group_cfs_halfspaces_by_t(
    A: np.ndarray,
    b: np.ndarray,
    *,
    H: int,
    state_dim: int,
    pos_dim: int = 2,
    eps: float = 1e-10,
) -> Dict[int, List[Tuple[np.ndarray, float]]]:
    """
    CFS constraints are returned in trajectory-flattened space.
    Each row is non-zero only for one timestep's position block.

    Returns dict[t] = [(g(pos_dim,), b_scalar), ...]
    """
    m = int(A.shape[0])
    if m == 0:
        return {t: [] for t in range(H)}

    A3 = A.reshape(m, H, state_dim)[:, :, :pos_dim]  # (m, H, pos_dim)
    groups: Dict[int, List[Tuple[np.ndarray, float]]] = {t: [] for t in range(H)}

    norms = np.linalg.norm(A3, axis=-1)  # (m, H)
    t_idx = np.argmax(norms, axis=1)  # (m,)
    max_norm = norms[np.arange(m), t_idx]

    for i in range(m):
        if not np.isfinite(max_norm[i]) or float(max_norm[i]) <= eps:
            continue
        t = int(t_idx[i])
        g = A3[i, t].astype(np.float32, copy=True)
        groups[t].append((g, float(b[i])))
    return groups


@dataclass
class CFSOverlayConfig:
    color: str = "#2ca02c"  # green-ish
    alpha: float = 0.22
    linewidth: float = 0.9
    fill: bool = True              # whether to fill feasible region
    fill_alpha: float = 0.15
    grid_n: int = 220             # grid resolution for fill mask
    max_timesteps: int = 12        # cap number of timesteps we draw to avoid clutter
    max_constraints_per_t: int = 4  # cap constraints per timestep
    show_ref_points: bool = True


def draw_cfs_convexify_overlay(
    ax: Any,
    *,
    env: Any,
    obstacles: Any,
    initial_state: np.ndarray,
    action_sequence: np.ndarray,
    env_plugin: Any,
    exp_cfg: Any,
    diffusion_step: int,
    diffusion_total_steps: int,
    overlay_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Draw CFS convexified halfspaces on an existing 2D axis for a given diffusion step.
    """
    if not _is_cfs_enabled(exp_cfg, obstacles):
        return

    # Only support 2D position plots
    try:
        p0 = np.asarray(env_plugin.extract_position(np.asarray(initial_state)), dtype=np.float32)
    except Exception:
        return
    if p0.shape[0] != 2:
        return

    cfg = CFSOverlayConfig()
    if isinstance(overlay_cfg, dict):
        for k, v in overlay_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Build reference trajectory from this diffusion step
    try:
        states = env.rollout_actions(initial_state, action_sequence)
    except Exception:
        return
    if states is None or len(states) < 2:
        return

    states_list = [np.asarray(s, dtype=np.float32) for s in states]
    actions_list = [np.asarray(a, dtype=np.float32) for a in action_sequence]
    ref_traj = Trajectory(states=states_list, actions=actions_list)

    H = len(ref_traj.states)
    state_dim = len(ref_traj.states[0])

    margin = float(cfg.__dict__.get("margin", _infer_margin(exp_cfg))) if hasattr(cfg, "margin") else _infer_margin(exp_cfg)
    margin = float(margin)

    backend = str(getattr(exp_cfg, "backend", "numpy"))
    # Convexifier backend selection (fallback to numpy if backend unavailable)
    try:
        convexifier = CFSConvexifier(
            obstacles=obstacles,
            backend=backend,
            max_constraints_per_point=int(getattr(exp_cfg, "method_params", {}).get("cfs_max_constraints_per_point", 8)),
            constraint_margin=float(getattr(exp_cfg, "method_params", {}).get("cfs_constraint_margin", 0.25)),
            use_jit=bool(getattr(exp_cfg, "method_params", {}).get("cfs_use_jit", False)),
        )
    except Exception:
        convexifier = CFSConvexifier(
            obstacles=obstacles,
            backend="numpy",
            max_constraints_per_point=8,
            constraint_margin=0.25,
        )

    constraints = convexifier.build_constraints(
        ref_traj,
        ScheduleParams(margin=margin),
        ScheduleState(k=int(diffusion_step), K=int(diffusion_total_steps), H=H),
    )

    A = _to_numpy(constraints.A)
    b = _to_numpy(constraints.b)
    if A.size == 0 or b.size == 0:
        return

    groups = _group_cfs_halfspaces_by_t(A, b, H=H, state_dim=state_dim, pos_dim=2)

    # Select timesteps to draw (evenly spaced, capped)
    timesteps = np.linspace(0, H - 1, num=min(cfg.max_timesteps, H), dtype=int) if H > 1 else np.array([0], dtype=int)

    # Axis bounds for drawing infinite lines
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xs = np.linspace(x_min, x_max, 200, dtype=np.float32)
    ys = np.linspace(y_min, y_max, 200, dtype=np.float32)

    for t in timesteps:
        hs = groups.get(int(t), [])
        if not hs:
            continue
        # Cap constraints per timestep (closest to ref point tends to be first in generation, but not guaranteed)
        hs = hs[: cfg.max_constraints_per_t]

        # Optionally mark reference point at this timestep
        if cfg.show_ref_points:
            pt = np.asarray(env_plugin.extract_position(np.asarray(ref_traj.states[int(t)])), dtype=np.float32)
            ax.scatter([pt[0]], [pt[1]], s=8, color=cfg.color, alpha=min(1.0, cfg.alpha + 0.15), zorder=3)

        # Fill feasible region for this timestep only (intersection of its halfspaces)
        if cfg.fill and len(hs) > 0:
            gx = np.linspace(x_min, x_max, cfg.grid_n, dtype=np.float32)
            gy = np.linspace(y_min, y_max, cfg.grid_n, dtype=np.float32)
            GX, GY = np.meshgrid(gx, gy)
            P = np.stack([GX.ravel(), GY.ravel()], axis=-1)
            feas = np.ones((P.shape[0],), dtype=bool)
            for g, bb in hs:
                feas &= (P @ g >= bb - 1e-9)
            feas = feas.reshape(cfg.grid_n, cfg.grid_n)
            if feas.any():
                ax.contourf(GX, GY, feas.astype(float), levels=[0.5, 1.5], colors=[cfg.color], alpha=cfg.fill_alpha, zorder=1.5)

        for g, bb in hs:
            a0, a1 = float(g[0]), float(g[1])
            if abs(a1) > 1e-9:
                yy = (bb - a0 * xs) / a1
                ax.plot(xs, yy, color=cfg.color, alpha=cfg.alpha, linewidth=cfg.linewidth, zorder=2)
            elif abs(a0) > 1e-9:
                xx = np.full_like(ys, bb / a0)
                ax.plot(xx, ys, color=cfg.color, alpha=cfg.alpha, linewidth=cfg.linewidth, zorder=2)



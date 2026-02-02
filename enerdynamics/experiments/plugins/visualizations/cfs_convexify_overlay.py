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
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

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


def _draw_local_cutting_line(
    ax: Any,
    *,
    p: np.ndarray,
    g_unit: np.ndarray,
    ratio: float,
    R_viz: float,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    zorder: float,
) -> None:
    """
    Draw a long cutting line segment in the local disk at point p.

    Disk is centered at p with radius R_viz. Half-plane boundary is:
        g^T Δ >= ratio * R_viz
    so the line passes through p_cut = p + g_unit * (ratio * R_viz),
    and is oriented along the tangent direction n = [-g_y, g_x].
    """
    n = np.array([-g_unit[1], g_unit[0]], dtype=float)
    p_cut = p + g_unit * (ratio * R_viz)
    L = 1.15 * R_viz
    q1 = p_cut - n * L
    q2 = p_cut + n * L
    ax.plot(
        [q1[0], q2[0]], [q1[1], q2[1]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


@dataclass
class CFSOverlayConfig:
    """
    Visualization config for CFS convexified halfspaces.

    Defaults are intentionally aligned with the MDOC CBF fan style in
    `DiffusionVisualizationPlugin._draw_mdoc_fans()`:
    - orange-ish edge strokes
    - light orange fill
    - dotted reachability disk
    """

    # MDOC fan palette (aligned with diffusion.py)
    facecolor: str = "#FFDCC2"
    edgecolor: str = "#C46A2D"
    scale: float = 1.0

    # Fan / boundary styling
    wedge_alpha: float = 0.5
    wedge_linewidth: float = 0.9
    chord_linewidth: float = 1.4
    chord_alpha: float = 0.85
    cutline_linestyle: str = "-"
    cutline_linewidth: float = 1.7
    cutline_alpha: float = 0.95

    # Reachability disk styling
    disk_alpha: float = 0.5
    disk_linestyle: str = "-"
    disk_linewidth: float = 0.8

    # Sampling of which constraints/timesteps to show
    # Match MDOC: avoid clutter by drawing only informative constraints
    informative_band: float = 1   # only draw if |ratio| < band (ratio = b/r in [-1,1])
    max_constraints_per_t: int = 2   # default: one fan per timestep
    show_ref_points: bool = False
    draw_disk: bool = True


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

    # Select timesteps to draw: exactly aligned with request / MDOC style (H // 12)
    step = max(1, H // 32)
    timesteps = np.arange(0, H, step, dtype=int)
    if timesteps.size == 0:
        timesteps = np.array([0], dtype=int)
    if timesteps[-1] < H - 1:
        timesteps = np.append(timesteps, H - 1)

    # Get action limit for disk visualization (default 1.0)
    action_limit = float(getattr(exp_cfg, "env_params", {}).get("control_limit", 1.0))
    dt = float(getattr(exp_cfg, "env_params", {}).get("dt", 0.05))
    # Visualization radius (aligned with MDOC): R = scale * dt * u_max
    r_viz = float(cfg.scale) * dt * action_limit

    for t in timesteps:
        hs = groups.get(int(t), [])
        if not hs:
            continue
        pt = np.asarray(env_plugin.extract_position(np.asarray(ref_traj.states[int(t)])), dtype=np.float32)

        # Optionally mark reference point at this timestep
        if cfg.show_ref_points:
            ax.scatter([pt[0]], [pt[1]], s=12, color=cfg.edgecolor, alpha=0.8, zorder=4)

        # Precompute ratios for this timestep and keep only the most informative ones
        candidates: list[tuple[float, float, float]] = []
        # candidates elements: (abs(ratio), ratio, idx)
        for i_h, (g, bb) in enumerate(hs):
            a0, a1 = float(g[0]), float(g[1])
            norm_g = np.sqrt(a0**2 + a1**2) + 1e-9
            # CFS constraint: g^T x_{t+1} >= bb
            # Single integrator: x_{t+1} = x_t + u * dt
            b_rel_state = float(bb) - (a0 * float(pt[0]) + a1 * float(pt[1]))
            b_rel_action = b_rel_state / (dt + 1e-9)
            phys_ratio = b_rel_action / (norm_g * action_limit + 1e-9)
            ratio = float(np.clip(phys_ratio, -1.0, 1.0))
            if abs(ratio) >= float(cfg.informative_band):
                continue
            candidates.append((abs(ratio), ratio, float(i_h)))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])  # most informative: closest to 0 (visible cut)
        keep = candidates[: int(cfg.max_constraints_per_t)]

        # Draw the background disk (action bound) ONCE per timestep (MDOC-style)
        if bool(cfg.draw_disk):
            circle = plt.Circle(
                tuple(pt), r_viz,
                color=cfg.edgecolor,
                fill=False,
                linestyle=str(cfg.disk_linestyle),
                linewidth=float(cfg.disk_linewidth),
                alpha=float(cfg.disk_alpha),
                zorder=2.5,
            )
            ax.add_patch(circle)

        for _, ratio, i_h in keep:
            g, bb = hs[int(i_h)]
            a0, a1 = float(g[0]), float(g[1])
            norm_g = np.sqrt(a0**2 + a1**2) + 1e-9
            gx, gy = a0 / norm_g, a1 / norm_g  # Unit normal

            phi = np.arctan2(gy, gx)
            alpha = np.arccos(ratio)

            theta_center = np.degrees(phi)
            theta_half_span = np.degrees(alpha)
            theta_start = theta_center - theta_half_span
            theta_end = theta_center + theta_half_span

            # Draw margin cutting line (same visual cue as MDOC fans)
            _draw_local_cutting_line(
                ax=ax,
                p=pt,
                g_unit=np.array([gx, gy], dtype=float),
                ratio=ratio,
                R_viz=r_viz,
                color=cfg.edgecolor,
                linestyle=str(cfg.cutline_linestyle),
                linewidth=float(cfg.cutline_linewidth),
                alpha=float(cfg.cutline_alpha),
                zorder=3.15,
            )

            # 2. Draw the Wedge (The Admissible Fan) - aligned with MDOC fan style
            wedge = Wedge(
                center=tuple(pt),
                r=r_viz,
                theta1=theta_start,
                theta2=theta_end,
                facecolor=cfg.facecolor,
                alpha=float(cfg.wedge_alpha),
                edgecolor=cfg.edgecolor,
                linewidth=float(cfg.wedge_linewidth),
                zorder=3.0
            )
            ax.add_patch(wedge)
            
            # 3. Draw the Chord (The Linearized Boundary Line)
            p1 = pt + r_viz * np.array([np.cos(np.radians(theta_start)), np.sin(np.radians(theta_start))])
            p2 = pt + r_viz * np.array([np.cos(np.radians(theta_end)), np.sin(np.radians(theta_end))])
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color=cfg.edgecolor,
                linewidth=float(cfg.chord_linewidth),
                alpha=float(cfg.chord_alpha),
                zorder=3.1
            )
            
            # disk is drawn once per timestep above



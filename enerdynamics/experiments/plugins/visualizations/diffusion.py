"""
Diffusion steps visualization plugin.
"""

from pathlib import Path as PathLib
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path

from ...framework.base import VisualizationPlugin
from ...common.visualization import draw_obstacles, EDOC_COLOR, MAX_SAMPLE_TRAJ_PLOT
from .cfs_convexify_overlay import draw_cfs_convexify_overlay

def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-9
    return v / n

def _add_cap_rectangle(
        self, ax: Any, p: np.ndarray, grad: np.ndarray,
        r: float, b: float, dt: float, scale: float,
        facecolor: str, edgecolor: str, alpha: float, zorder: float = 4.0
    ) -> None:
        """Draw CAP as a rectangle in workspace: depth = dt*(r-b), width = 2*dt*sqrt(r^2-b^2).
        Center between cutting plane and farthest point: p + g * (scale*dt*(r+b)/2).
        """
        if b >= r:
            return
        if b <= -r:
            return
        g = grad / (np.linalg.norm(grad) + 1e-9)
        n = np.array([-g[1], g[0]])
        R = scale * dt * r
        length = scale * dt * (r - b)
        width = scale * dt * (2.0 * np.sqrt(max(r * r - b * b, 0)))
        center = p + g * (scale * dt * (r + b) / 2.0)
        c1 = center + 0.5 * length * g + 0.5 * width * n
        c2 = center + 0.5 * length * g - 0.5 * width * n
        c3 = center - 0.5 * length * g - 0.5 * width * n
        c4 = center - 0.5 * length * g + 0.5 * width * n
        poly = Polygon(
            [c1, c2, c3, c4],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.8,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(poly)

def _draw_cutting_line(
    ax,
    p: np.ndarray,
    g: np.ndarray,
    n: np.ndarray,
    b_val: float,
    dt: float,
    scale: float,
    R_viz: float,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    zorder: float,
):
    """
    Draw the cutting line in workspace corresponding to: g^T u = b
    with workspace mapping: Δx = dt * u, so g^T Δx = dt*b.
    Line passes through p_cut = p + g*(scale*dt*b).
    """
    p_cut = p + g * (scale * dt * b_val)
    L = 1.15 * R_viz
    q1 = p_cut - n * L
    q2 = p_cut + n * L
    ax.plot(
        [q1[0], q2[0]], [q1[1], q2[1]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder
    )

def _add_cap_polygon(
    ax,
    p: np.ndarray,
    g: np.ndarray,
    n: np.ndarray,
    r: float,
    b_val: float,
    dt: float,
    scale: float,
    R_viz: float,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    hatch: str | None,
    zorder: float,
    n_arc: int = 48,
):
    """
    Strict CAP in workspace: disk(Δx; R_viz) ∩ {g^T Δx >= scale*dt*b}.
    Here R_viz = scale*dt*r.
    """
    # Feasibility of half-plane within disk in control-space:
    # ratio = b/r in [-1,1] gives non-empty non-full cap
    ratio = np.clip(b_val / (r + 1e-9), -1.0, 1.0)
    if ratio <= -1.0:
        # constraint too loose -> full disk feasible; skip filling to avoid clutter
        return
    if ratio >= 1.0:
        # infeasible -> empty; skip
        return

    # In disk coordinates around p, boundary is circle, cut is line normal g.
    # Arc endpoints correspond to angles centered at phi = atan2(g_y, g_x)
    phi = np.arctan2(g[1], g[0])
    alpha_angle = np.arccos(ratio)
    theta_start = phi - alpha_angle
    theta_end   = phi + alpha_angle

    thetas = np.linspace(theta_start, theta_end, n_arc)
    arc_pts = np.stack([np.cos(thetas), np.sin(thetas)], axis=1) * R_viz + p[None, :]

    # Close with chord explicitly (p2 -> p1)
    p1 = arc_pts[0]
    p2 = arc_pts[-1]
    cap_pts = np.vstack([arc_pts, p2[None, :], p1[None, :]])

    cap = Polygon(
        cap_pts,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
        alpha=alpha,
        hatch=hatch,
        zorder=zorder,
    )
    ax.add_patch(cap)

    # chord highlight
    ax.plot(
        [p1[0], p2[0]], [p1[1], p2[1]],
        color=edgecolor,
        linewidth=1.4,
        alpha=min(1.0, alpha + 0.35),
        zorder=zorder + 0.05
    )

def _add_cap_rectangle(
    ax,
    p: np.ndarray,
    g: np.ndarray,
    n: np.ndarray,
    r: float,
    b_val: float,
    dt: float,
    scale: float,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    hatch: str | None,
    zorder: float,
):
    """
    Rectangle approximation of CAP in workspace (Δx-space).
    depth  = scale*dt*(r - b)
    width  = scale*dt*2*sqrt(r^2 - b^2)
    center between cutting plane and farthest point:
        p + g * (scale*dt*(r+b)/2)
    """
    if b_val >= r:
        return
    if b_val <= -r:
        return

    length = scale * dt * (r - b_val)
    width  = scale * dt * (2.0 * np.sqrt(max(r * r - b_val * b_val, 0.0)))

    center = p + g * (scale * dt * (r + b_val) / 2.0)

    c1 = center + 0.5 * length * g + 0.5 * width * n
    c2 = center + 0.5 * length * g - 0.5 * width * n
    c3 = center - 0.5 * length * g - 0.5 * width * n
    c4 = center - 0.5 * length * g + 0.5 * width * n

    poly = Polygon(
        [c1, c2, c3, c4],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=alpha,
        hatch=hatch,
        zorder=zorder,
    )
    ax.add_patch(poly)


class DiffusionVisualizationPlugin(VisualizationPlugin):
    """
    Plugin for visualizing diffusion steps during planning.
    """
    
    @property
    def name(self) -> str:
        """Visualization name identifier."""
        return "diffusion"
    
    def visualize(self, fig: Any, axes: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Create diffusion steps visualization.
        
        Args:
            fig: Matplotlib figure
            axes: Array of matplotlib axes (one for each diffusion step)
            data: Dictionary containing:
                - result: Planning result dictionary
                - env: Environment instance
                - obstacles: Obstacle manager
                - initial_state: Initial state array
                - env_plugin: Environment plugin
            config: Visualization configuration dictionary
        """
        result = data['result']
        env = data['env']
        obstacles = data['obstacles']
        initial_state = data['initial_state']
        env_plugin = data['env_plugin']
        exp_cfg = config.get('config', None)
        overlay_cfg = config.get('cfs_overlay', None)
        method_name = getattr(exp_cfg, 'method', None) if exp_cfg is not None else None
        method_params = getattr(exp_cfg, 'method_params', {}) or {}
        is_ebmbd = (method_name == "ebmbd")  # Only EB-MBD draws barrier; MBD does not
        is_mbd = (method_name == "mbd")  # MBD: no sample-rollout fan, clean plot
        # MDOC: exact method name or ablation config (mdoc_constraint_mode / cbf_eta in method_params)
        is_mdoc = (
            method_name == "mdoc"
            or "mdoc_constraint_mode" in method_params
            or "cbf_eta" in method_params
        )
        if is_ebmbd:
            overlay_cfg = None  # fully disable half-space overlays
        
        # When qp_gate is False and qp_prob is 0, skip CFS convexified halfspace (fan) overlay
        draw_cfs_fan = True
        if exp_cfg is not None:
            scheduler_config = getattr(exp_cfg, 'scheduler_config', None)
            if scheduler_config is not None:
                cs_list = getattr(scheduler_config, 'constraint_schedulers', None)
                if cs_list is None and isinstance(scheduler_config, dict):
                    cs_list = scheduler_config.get('constraint_schedulers', [])
                if cs_list and len(cs_list) > 0:
                    first_cs = cs_list[0]
                    qp_gate = first_cs.get('qp_gate', True) if isinstance(first_cs, dict) else getattr(first_cs, 'qp_gate', True)
                    qp_prob = first_cs.get('qp_prob', 1.0) if isinstance(first_cs, dict) else getattr(first_cs, 'qp_prob', 1.0)
                    if qp_gate is False and (qp_prob == 0 or qp_prob == 0.0):
                        draw_cfs_fan = False
        
        # Get map bounds
        obstacle_config = config.get('config', {}).obstacle_config or {}
        map_bounds = obstacle_config.get('map_bounds', {})
        x_min = map_bounds.get('x_min', -2.0)
        x_max = map_bounds.get('x_max', 2.0)
        y_min = map_bounds.get('y_min', -2.0)
        y_max = map_bounds.get('y_max', 2.0)
        
        # Get diffusion data
        diffusion_actions = result.get('diffusion_actions_traj', None)
        diffusion_samples = result.get('diffusion_sampled_actions', None)
        diffusion_fractions = config.get('fractions', (0.1, 0.5, 0.9))  # 90%, 50%, 10%
        if diffusion_actions is not None and len(diffusion_actions) > 0:
            diffusion_actions = np.asarray(diffusion_actions, dtype=np.float32)
            Ndiffuse = diffusion_actions.shape[0]
            diffusion_total_steps = max(0, Ndiffuse - 1)
            
            for ax, frac in zip(axes, diffusion_fractions):
                # Convention: config fractions are (0.1, 0.5, 0.9) meaning (90%, 50%, 10%) noise remaining.
                # Our diffusion histories are stored with index 0 = final (least noisy), largest index = initial (most noisy).
                step_idx = int((1.0 - frac) * (Ndiffuse - 1))
                step_idx = max(0, min(step_idx, Ndiffuse - 1))
                
                action_seq = diffusion_actions[step_idx]
                sample_acts = None
                if diffusion_samples is not None and len(diffusion_samples) > 0:
                    diffusion_samples_arr = np.asarray(diffusion_samples, dtype=np.float32)
                    if diffusion_samples_arr.ndim == 4 and diffusion_samples_arr.shape[0] > step_idx:
                        sample_acts = diffusion_samples_arr[step_idx]

                pct = round((1 - frac) * 100)
                self._visualize_single_step(
                    ax, env, obstacles, initial_state, action_seq, sample_acts,
                    env_plugin, x_min, x_max, y_min, y_max,
                    title=f"Diffusion {pct}%",
                    exp_cfg=exp_cfg,
                    diffusion_step=step_idx,
                    diffusion_total_steps=diffusion_total_steps,
                    cfs_overlay_cfg=overlay_cfg,
                    is_ebmbd=is_ebmbd,
                    is_mdoc=is_mdoc,
                    is_mbd=is_mbd,
                    draw_cfs_fan=draw_cfs_fan,
                )
        else:
            # Fallback: show final trajectory
            final_actions = result.get('actions', None)
            for ax, frac in zip(axes, diffusion_fractions):
                self._visualize_single_step(
                    ax, env, obstacles, initial_state, final_actions, None,
                    env_plugin, x_min, x_max, y_min, y_max,
                    title=f"Diffusion {int(frac * 100)}% (Final)",
                    exp_cfg=exp_cfg,
                    diffusion_step=0,
                    diffusion_total_steps=0,
                    cfs_overlay_cfg=overlay_cfg,
                    is_ebmbd=is_ebmbd,
                    is_mdoc=is_mdoc,
                    is_mbd=is_mbd,
                    draw_cfs_fan=draw_cfs_fan,
                )
    
    def _visualize_single_step(
        self, ax: Any, env: Any, obstacles: Any, initial_state: np.ndarray,
        action_sequence: np.ndarray, sample_actions: np.ndarray,
        env_plugin: Any, x_min: float, x_max: float, y_min: float, y_max: float,
        title: str = None,
        exp_cfg: Any = None,
        diffusion_step: int = 0,
        diffusion_total_steps: int = 0,
        cfs_overlay_cfg: Any = None,
        is_ebmbd: bool = False,
        is_mdoc: bool = False,
        is_mbd: bool = False,
        draw_cfs_fan: bool = True,
        show_title: bool = True,
        show_axis_labels: bool = True,
        show_grid: bool = True,
    ) -> None:
        """Visualize a single diffusion step."""
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # EB-MBD: white background
        if is_ebmbd:
            ax.set_facecolor('white')
        # Draw obstacles
        draw_obstacles(ax, obstacles)
        # Draw barrier field for EB-MBD (contour circles only, no heatmap; color = MDOC fan orange)
        if is_ebmbd:
            self._draw_barrier_field(ax, obstacles, x_min, x_max, y_min, y_max)
        
        # Draw sample rollouts (skip for MBD: clean plot without fan)
        if not is_mbd and sample_actions is not None and len(sample_actions) > 0:
            if is_ebmbd:
                num_samples = len(sample_actions)
            else:
                num_samples = min(len(sample_actions), MAX_SAMPLE_TRAJ_PLOT)
                if num_samples < len(sample_actions):
                    indices = np.linspace(0, len(sample_actions) - 1, num_samples, dtype=int)
                    sample_actions = sample_actions[indices]
            
            light_rgba = mcolors.to_rgba(EDOC_COLOR, alpha=0.15)
            for acts in sample_actions:
                states = env.rollout_actions(initial_state, acts)
                if len(states) > 1:
                    positions = np.array([env_plugin.extract_position(s) for s in states])
                    ax.plot(positions[:, 0], positions[:, 1], color=light_rgba, linewidth=0.8)
        
        # Draw main trajectory
        if action_sequence is not None and len(action_sequence) > 0:
            states = env.rollout_actions(initial_state, action_sequence)
            if len(states) > 1:
                positions = np.array([env_plugin.extract_position(s) for s in states])
                # Trajectory below fans so MDOC fans are visible (fans use zorder 5.5+)
                ax.plot(positions[:, 0], positions[:, 1], color=EDOC_COLOR, linewidth=2.5, zorder=4.5)
                
                # Draw start
                ax.scatter(
                    positions[0, 0],
                    positions[0, 1],
                    marker='o',
                    s=30,
                    facecolors='white',
                    edgecolors=EDOC_COLOR,
                    linewidths=1.0,
                    zorder=6
                )
                # Draw end
                ax.scatter(
                    positions[-1, 0],
                    positions[-1, 1],
                    marker='*',
                    s=55,
                    facecolors=EDOC_COLOR,
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=6
                )

                # Draw MDOC conservative fan visualization
                if is_mdoc:
                    self._draw_mdoc_fans(ax, env, obstacles, positions, exp_cfg)

                # Overlay CFS convexified halfspaces for this diffusion step (if enabled)
                # For MDOC, we show the specialized Fans instead of full halfspaces.
                # Skip when qp_gate is False and qp_prob is 0 (no QP filter used).
                if exp_cfg is not None and not is_ebmbd and not is_mdoc and draw_cfs_fan:
                    try:
                        draw_cfs_convexify_overlay(
                            ax,
                            env=env,
                            obstacles=obstacles,
                            initial_state=np.asarray(initial_state, dtype=np.float32),
                            action_sequence=np.asarray(action_sequence, dtype=np.float32),
                            env_plugin=env_plugin,
                            exp_cfg=exp_cfg,
                            diffusion_step=int(diffusion_step),
                            diffusion_total_steps=int(diffusion_total_steps),
                            overlay_cfg=cfs_overlay_cfg if isinstance(cfs_overlay_cfg, dict) else None,
                        )
                    except Exception:
                        # Never fail the diffusion visualization due to overlay issues
                        pass
        
        # Draw target
        target = np.asarray(env.target)
        target_pos = env_plugin.extract_position(target)
        ax.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target', zorder=10)
        
        if show_title and title:
            ax.set_title(title, fontsize=12)
        if not show_axis_labels:
            ax.set_xticks([])
            ax.set_yticks([])
            # Subtle spine styling
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#888888')
                spine.set_linewidth(1.0)
        if show_grid:
            ax.grid(True, alpha=0.2)
    
    def _draw_barrier_field(
        self, ax: Any, obstacles: Any, 
        x_min: float, x_max: float, y_min: float, y_max: float,
        res: int = 60
    ) -> None:
        """Draw emerging barrier contour circles only (no heatmap, no solid obstacles). Color matches MDOC fan edge."""
        if obstacles is None or len(obstacles) == 0:
            return

        # MDOC fan edge color for consistency
        barrier_color = "#C46A2D"
        # Create grid for SDF sampling
        xs = np.linspace(x_min, x_max, res)
        ys = np.linspace(y_min, y_max, res)
        X, Y = np.meshgrid(xs, ys)
        points = np.stack([X.ravel(), Y.ravel()], axis=-1)

        # Batch SDF computation
        try:
            sdf_vals = obstacles.sdf(points).reshape(res, res)
        except Exception:
            return

        # Equipotential lines only (barrier circles), no heatmap
        levels = [0.05, 0.1, 0.2, 0.4]
        ax.contour(
            X, Y, sdf_vals,
            levels=levels,
            colors=barrier_color,
            alpha=0.4,
            linewidths=1.0,
            linestyles='--',
            zorder=1.0,
        )
    
    def _draw_mdoc_fans(
        self,
        ax,
        env,
        obstacles,
        positions: np.ndarray,
        exp_cfg
    ) -> None:
        """
        Paper-friendly visualization of MDOC local feasible sets:
        - Two layers: margin=0 (reference) vs margin=cbf_margin (barrier margin)
        - Draw cutting line g^T Δx = dt*b as a long line segment (most important cue)
        - Draw either strict CAP polygon or rectangle approximation
        - Only draw at informative points to reduce clutter (|b| not too extreme)
        """
        if obstacles is None or len(positions) < 2:
            return

        method_params = getattr(exp_cfg, "method_params", {}) or {}
        eta = float(method_params.get("cbf_eta", 1.5))
        margin = float(method_params.get("cbf_margin", 0.1))
        base_beta = float(method_params.get("base_beta", 0.05))

        # viz controls
        viz_style = str(method_params.get("mdoc_viz_style", "cap"))      # "cap" or "rectangle"
        scale = float(method_params.get("mdoc_viz_scale", 4.0))          # workspace scaling for visibility
        informative_band = float(method_params.get("mdoc_viz_band", 0.85))  # draw only if |b/r| < band
        use_beta_in_viz = bool(method_params.get("mdoc_viz_use_beta", False))  # often False makes margin effect clearer
        step_override = method_params.get("mdoc_viz_step", None)

        obstacle_config = getattr(exp_cfg, "obstacle_config", None) or {}
        robot_radius = float(obstacle_config.get("robot_radius", 0.05))
        env_params = getattr(exp_cfg, "env_params", None) or {}
        dt = float(env_params.get("dt", 0.05))
        r = float(env_params.get("control_limit", 1.0))

        # colors (keep one hue, but distinguish layers via hatch + line style)
        face = "#FFDCC2"
        edge = "#C46A2D"

        # workspace reachable disk radius
        R_viz = scale * dt * r

        # index selection
        H = len(positions)
        if step_override is not None:
            step = max(1, int(step_override))
        else:
            step = max(1, H // 32)
        indices = np.arange(0, H - 1, step, dtype=int)
        if len(indices) == 0:
            return
        if indices[-1] < H - 2:
            indices = np.append(indices, H - 2)

        for idx in indices:
            p = positions[idx]

            # gradient of sdf
            try:
                eps = 1e-4
                sdf_p = float(obstacles.sdf(p))
                grad_x = (float(obstacles.sdf(p + np.array([eps, 0.0]))) - sdf_p) / eps
                grad_y = (float(obstacles.sdf(p + np.array([0.0, eps]))) - sdf_p) / eps
                grad = np.array([grad_x, grad_y], dtype=float)
                g = _safe_unit(grad)

                # sanity check: g should point to safer direction (sdf increases)
                sdf_forward = float(obstacles.sdf(p + 0.02 * g))
                if sdf_forward < sdf_p:
                    g = -g
            except Exception:
                continue

            n = np.array([-g[1], g[0]], dtype=float)

            # two layers: reference vs margin
            # reference is hatched + dashed line; margin is solid + thicker line.
            layers = [
                dict(margin_vis=0.0,  alpha=0.5, hatch="////", line_ls="--", line_lw=1.1, z=4.0),
                dict(margin_vis=margin, alpha=0.5, hatch=None,  line_ls="-",  line_lw=1.7, z=4.2),
            ]

            for layer in layers:
                margin_vis = layer["margin_vis"]

                # CBF value (using sdf): h = sdf - (robot_radius + margin)
                h = sdf_p - (robot_radius + margin_vis)

                # your current b form: b = -(eta/dt)*h + base_beta
                beta_use = base_beta if use_beta_in_viz else 0.0
                b_val = -(eta / dt) * h + beta_use

                # Informative-only filter: avoid drawing full/empty cases and super extreme cuts
                # cap only changes visibly when |b/r| is not near 0 or 1.
                br = abs(b_val / (r + 1e-9))
                if br >= informative_band:
                    continue

                # 1) cutting line (best cue for "margin pushes the cut")
                _draw_cutting_line(
                    ax=ax, p=p, g=g, n=n,
                    b_val=b_val, dt=dt, scale=scale, R_viz=R_viz,
                    color=edge, linestyle=layer["line_ls"], linewidth=layer["line_lw"],
                    alpha=0.95, zorder=layer["z"] + 0.15
                )

                # 2) feasible region: cap or rectangle
                if viz_style == "rectangle":
                    _add_cap_rectangle(
                        ax=ax, p=p, g=g, n=n,
                        r=r, b_val=b_val, dt=dt, scale=scale,
                        facecolor=face, edgecolor=edge,
                        alpha=layer["alpha"], hatch=layer["hatch"],
                        zorder=layer["z"]
                    )
                else:
                    _add_cap_polygon(
                        ax=ax, p=p, g=g, n=n,
                        r=r, b_val=b_val, dt=dt, scale=scale, R_viz=R_viz,
                        facecolor=face, edgecolor=edge,
                        alpha=layer["alpha"], hatch=layer["hatch"],
                        zorder=layer["z"],
                        n_arc=48
                    )

            # optional: draw g arrow once per point (to avoid clutter)
            if bool(method_params.get("mdoc_viz_draw_grad", True)):
                arrow_len = 0.22 * R_viz
                ax.arrow(
                    p[0], p[1],
                    g[0] * arrow_len, g[1] * arrow_len,
                    head_width=0.06 * R_viz,
                    head_length=0.09 * R_viz,
                    fc=edge, ec=edge,
                    alpha=0.75, zorder=4.9,
                    length_includes_head=True
                )

            # optional: show reachable disk lightly (helps interpret "cap is cut from disk")
            if bool(method_params.get("mdoc_viz_draw_disk", True)):
                circ = plt.Circle(
                    tuple(p), R_viz,
                    fill=False, linestyle=":", linewidth=0.8,
                    color=edge, alpha=0.22, zorder=3.6
                )
                ax.add_patch(circ)

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """Save visualization to file."""
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

    def save_individual_and_gif(
        self,
        output_dir: PathLib | str,
        data: Dict[str, Any],
        config: Dict[str, Any],
        dpi: int = 150,
        gif_duration_ms: float = 150,
        gif_loop: int = 0,
    ) -> None:
        """
        Save each diffusion fraction (90%, 50%, 10%) as separate PNGs
        and generate a GIF from K-1 diffusion steps.
        """
        output_dir = PathLib(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = data['result']
        env = data['env']
        obstacles = data['obstacles']
        initial_state = data.get('initial_state')
        if initial_state is None:
            initial_state = result.get('initial_state')
        env_plugin = data['env_plugin']
        exp_cfg = config.get('config', None)
        overlay_cfg = config.get('cfs_overlay', None)
        method_name = getattr(exp_cfg, 'method', None) if exp_cfg is not None else None
        method_params = getattr(exp_cfg, 'method_params', {}) or {}
        is_ebmbd = (method_name == "ebmbd")  # Only EB-MBD draws barrier; MBD does not
        is_mbd = (method_name == "mbd")  # MBD: no sample-rollout fan, clean plot
        is_mdoc = (
            method_name == "mdoc"
            or "mdoc_constraint_mode" in method_params
            or "cbf_eta" in method_params
        )
        if is_ebmbd:
            overlay_cfg = None

        draw_cfs_fan = True
        if exp_cfg is not None:
            scheduler_config = getattr(exp_cfg, 'scheduler_config', None)
            if scheduler_config is not None:
                cs_list = getattr(scheduler_config, 'constraint_schedulers', None)
                if cs_list is None and isinstance(scheduler_config, dict):
                    cs_list = scheduler_config.get('constraint_schedulers', [])
                if cs_list and len(cs_list) > 0:
                    first_cs = cs_list[0]
                    qp_gate = first_cs.get('qp_gate', True) if isinstance(first_cs, dict) else getattr(first_cs, 'qp_gate', True)
                    qp_prob = first_cs.get('qp_prob', 1.0) if isinstance(first_cs, dict) else getattr(first_cs, 'qp_prob', 1.0)
                    if qp_gate is False and (qp_prob == 0 or qp_prob == 0.0):
                        draw_cfs_fan = False

        exp_cfg_for_obs = config.get('config', None)
        obstacle_config = (getattr(exp_cfg_for_obs, 'obstacle_config', None) or {}) if exp_cfg_for_obs else {}
        if not isinstance(obstacle_config, dict):
            obstacle_config = {}
        map_bounds = obstacle_config.get('map_bounds', {})
        x_min = map_bounds.get('x_min', -2.0)
        x_max = map_bounds.get('x_max', 2.0)
        y_min = map_bounds.get('y_min', -2.0)
        y_max = map_bounds.get('y_max', 2.0)

        diffusion_actions = result.get('diffusion_actions_traj', None)
        diffusion_samples = result.get('diffusion_sampled_actions', None)
        diffusion_fractions = config.get('fractions', (0.1, 0.5, 0.9))

        if diffusion_actions is not None and len(diffusion_actions) > 0:
            diffusion_actions = np.asarray(diffusion_actions, dtype=np.float32)
            Ndiffuse = diffusion_actions.shape[0]
            diffusion_total_steps = max(0, Ndiffuse - 1)

            # 1. Save individual PNGs: percentage = noise level (90%=high noise, 10%=low noise)
            # frac in config = noise fraction; step_idx 0 = least noisy, Ndiffuse-1 = most noisy
            for frac in diffusion_fractions:
                step_idx = int(frac * (Ndiffuse - 1))
                step_idx = max(0, min(step_idx, Ndiffuse - 1))
                action_seq = diffusion_actions[step_idx]
                sample_acts = None
                if diffusion_samples is not None and len(diffusion_samples) > 0:
                    diffusion_samples_arr = np.asarray(diffusion_samples, dtype=np.float32)
                    if diffusion_samples_arr.ndim == 4 and diffusion_samples_arr.shape[0] > step_idx:
                        sample_acts = diffusion_samples_arr[step_idx]
                pct = round(frac * 100)
                fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 6))
                self._visualize_single_step(
                    ax_single, env, obstacles, initial_state, action_seq, sample_acts,
                    env_plugin, x_min, x_max, y_min, y_max,
                    title=None,
                    exp_cfg=exp_cfg,
                    diffusion_step=step_idx,
                    diffusion_total_steps=diffusion_total_steps,
                    cfs_overlay_cfg=overlay_cfg,
                    is_ebmbd=is_ebmbd,
                    is_mdoc=is_mdoc,
                    is_mbd=is_mbd,
                    draw_cfs_fan=draw_cfs_fan,
                    show_title=False,
                    show_axis_labels=False,
                    show_grid=False,
                )
                out_path = output_dir / f"diffusion_steps_{pct}.png"
                if is_ebmbd:
                    fig_single.set_facecolor('white')
                    ax_single.set_facecolor('white')
                fig_single.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor=fig_single.get_facecolor())
                plt.close(fig_single)

            # 2. Generate GIF: 100% -> 1% (high noise to low noise)
            # Order: step_idx Ndiffuse-1 (most noisy) -> 0 (least noisy)
            temp_files = []
            try:
                for step_idx in range(Ndiffuse - 1, -1, -1):
                    action_seq = diffusion_actions[step_idx]
                    sample_acts = None
                    if diffusion_samples is not None and len(diffusion_samples) > 0:
                        diffusion_samples_arr = np.asarray(diffusion_samples, dtype=np.float32)
                        if diffusion_samples_arr.ndim == 4 and diffusion_samples_arr.shape[0] > step_idx:
                            sample_acts = diffusion_samples_arr[step_idx]
                    # pct = noise level: step Ndiffuse-1 -> 100%, step 0 -> 0%
                    denom = max(1, Ndiffuse - 1)
                    pct = round(100.0 * step_idx / denom)
                    fig_frame, ax_frame = plt.subplots(1, 1, figsize=(6, 6))
                    self._visualize_single_step(
                        ax_frame, env, obstacles, initial_state, action_seq, sample_acts,
                        env_plugin, x_min, x_max, y_min, y_max,
                        title=f"Diffusion {pct}%",
                        exp_cfg=exp_cfg,
                        diffusion_step=step_idx,
                        diffusion_total_steps=diffusion_total_steps,
                        cfs_overlay_cfg=overlay_cfg,
                        is_ebmbd=is_ebmbd,
                        is_mdoc=is_mdoc,
                        is_mbd=is_mbd,
                        draw_cfs_fan=draw_cfs_fan,
                        show_title=True,
                        show_axis_labels=False,
                        show_grid=False,
                    )
                    tmp_path = output_dir / f"_gif_frame_{step_idx}.png"
                    if is_ebmbd:
                        fig_frame.set_facecolor('white')
                        ax_frame.set_facecolor('white')
                    fig_frame.savefig(tmp_path, dpi=dpi, bbox_inches='tight', facecolor=fig_frame.get_facecolor())
                    plt.close(fig_frame)
                    temp_files.append(tmp_path)
                if temp_files:
                    frames = [imageio.v3.imread(p) for p in temp_files]
                    gif_path = output_dir / "diffusion_steps.gif"
                    imageio.v3.imwrite(
                        gif_path,
                        frames,
                        duration=gif_duration_ms,
                        loop=gif_loop,
                    )
            finally:
                for p in temp_files:
                    try:
                        p.unlink()
                    except OSError:
                        pass
        else:
            # Fallback: only save the 3 fraction PNGs with final trajectory
            final_actions = result.get('actions', None)
            for frac in diffusion_fractions:
                pct = round(frac * 100)
                fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 6))
                self._visualize_single_step(
                    ax_single, env, obstacles, initial_state, final_actions, None,
                    env_plugin, x_min, x_max, y_min, y_max,
                    title=None,
                    exp_cfg=exp_cfg,
                    diffusion_step=0,
                    diffusion_total_steps=0,
                    cfs_overlay_cfg=overlay_cfg,
                    is_ebmbd=is_ebmbd,
                    is_mdoc=is_mdoc,
                    is_mbd=is_mbd,
                    draw_cfs_fan=draw_cfs_fan,
                    show_title=False,
                    show_axis_labels=False,
                    show_grid=False,
                )
                out_path = output_dir / f"diffusion_steps_{pct}.png"
                if is_ebmbd:
                    fig_single.set_facecolor('white')
                    ax_single.set_facecolor('white')
                fig_single.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor=fig_single.get_facecolor())
                plt.close(fig_single)


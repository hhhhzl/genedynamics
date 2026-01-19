"""
Diffusion steps visualization plugin.
"""

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path

from ...framework.base import VisualizationPlugin
from ...common.visualization import draw_obstacles, EDOC_COLOR, MAX_SAMPLE_TRAJ_PLOT
from .cfs_convexify_overlay import draw_cfs_convexify_overlay


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
        exp_cfg = config.get('config', None)  # ExperimentConfig
        # Determine method to handle EB-MBD special-casing
        overlay_cfg = config.get('cfs_overlay', None)
        method_name = getattr(exp_cfg, 'method', None) if exp_cfg is not None else None
        is_ebmbd = (method_name == "ebmbd" or method_name == "mbd")
        is_mdoc = (method_name == "mdoc")
        if is_ebmbd:
            overlay_cfg = None  # fully disable half-space overlays
        
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
                
                self._visualize_single_step(
                    ax, env, obstacles, initial_state, action_seq, sample_acts,
                    env_plugin, x_min, x_max, y_min, y_max,
                    title=f"Diffusion {int((1-frac) * 100)}%",
                    exp_cfg=exp_cfg,
                    diffusion_step=step_idx,
                    diffusion_total_steps=diffusion_total_steps,
                    cfs_overlay_cfg=overlay_cfg,
                    is_ebmbd=is_ebmbd,
                    is_mdoc=is_mdoc,
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
    ) -> None:
        """Visualize a single diffusion step."""
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Draw obstacles
        draw_obstacles(ax, obstacles)
        
        # Draw barrier field for EB-MBD (Heatmap and Equipotential lines)
        if is_ebmbd:
            self._draw_barrier_field(ax, obstacles, x_min, x_max, y_min, y_max)
        
        # Draw sample rollouts
        if sample_actions is not None and len(sample_actions) > 0:
            # For EB-MBD, show all samples (Nsample typically moderate).
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
                # Increase zorder to 5 to stay on top of fans/half-spaces
                ax.plot(positions[:, 0], positions[:, 1], color=EDOC_COLOR, linewidth=2.5, zorder=5)
                
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
                if exp_cfg is not None and not is_ebmbd and not is_mdoc:
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
        
        if title:
            ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.2)
    
    def _draw_barrier_field(
        self, ax: Any, obstacles: Any, 
        x_min: float, x_max: float, y_min: float, y_max: float,
        res: int = 60
    ) -> None:
        """Draw background heatmap and equipotential lines for obstacles."""
        if obstacles is None or len(obstacles) == 0:
            return
            
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
            
        # 1. Draw Equipotential Lines (3-5 levels of distance)
        # We draw them for positive SDF (outside obstacles)
        levels = [0.05, 0.1, 0.2, 0.4]
        ax.contour(
            X, Y, sdf_vals, 
            levels=levels, 
            colors='orange', 
            alpha=0.15, 
            linewidths=0.8,
            linestyles='--'
        )
        
        # 2. Draw light heatmap (more 'hot' near obstacles)
        # Using a exponential decay for the heatmap intensity
        heatmap_intensity = np.exp(-4.0 * np.maximum(0, sdf_vals))
        ax.imshow(
            heatmap_intensity, 
            extent=[x_min, x_max, y_min, y_max], 
            origin='lower', 
            cmap='YlOrRd', 
            alpha=0.06, 
            zorder=0.5,
            interpolation='bilinear'
        )

    def _draw_mdoc_fans(
        self, ax: Any, env: Any, obstacles: Any, 
        positions: np.ndarray, exp_cfg: Any
    ) -> None:
        """Draw conservative feasible direction fans (Disk cut by Half-plane) for MDOC."""
        if obstacles is None or len(positions) < 2:
            return
            
        # Extract CBF/MDOC params
        method_params = getattr(exp_cfg, "method_params", {})
        eta = float(method_params.get("cbf_eta", 1.5))
        margin = float(method_params.get("cbf_margin", 0.1))
        base_beta = float(method_params.get("base_beta", 0.05))
        robot_radius = float(getattr(exp_cfg, "obstacle_config", {}).get("robot_radius", 0.05))
        dt = float(getattr(exp_cfg, "env_params", {}).get("dt", 0.05))
        action_limit = float(getattr(exp_cfg, "env_params", {}).get("control_limit", 1.0))
        
        # 1. Key point selection
        # 1. Representative point selection: regular interval along the whole trajectory
        try:
            H = len(positions)
            # Draw every 8 steps to cover the whole trajectory without too much overlap
            step = max(1, H // 12) 
            indices = np.arange(0, H - 1, step, dtype=int)
            # Always include the last point before the target
            if indices[-1] < H - 2:
                indices = np.append(indices, H - 2)
        except Exception:
            indices = np.linspace(0, len(positions) - 2, num=8, dtype=int)
        
        for idx in indices:
            p = positions[idx]
            try:
                eps = 1e-4
                sdf_p = float(obstacles.sdf(p))
                grad_x = (float(obstacles.sdf(p + np.array([eps, 0]))) - sdf_p) / eps
                grad_y = (float(obstacles.sdf(p + np.array([0, eps]))) - sdf_p) / eps
                grad = np.array([grad_x, grad_y])
                norm_grad = np.linalg.norm(grad) + 1e-9
                grad = grad / norm_grad
            except Exception:
                continue
                
            h = sdf_p - (robot_radius + margin)
            b = -(eta / dt) * h + base_beta
            r = action_limit
            
            # Visualization radius (local scale)
            r_viz = 0.22 
            
            # Ratio of offset. 
            # To emphasize the "Conservative Fan", we ensure it's always narrower than a half-disk.
            phys_ratio = b / (r + 1e-9)
            # Map to [0.2, 0.85] range to show a clear conservative cut compared to EDOC.
            ratio = np.clip(phys_ratio, 0.2, 0.85)
            
            phi = np.arctan2(grad[1], grad[0])
            alpha = np.arccos(ratio)
            
            # Convert to degrees for matplotlib Wedge
            theta_center = np.degrees(phi)
            theta_half_span = np.degrees(alpha)
            
            theta_start = theta_center - theta_half_span
            theta_end = theta_center + theta_half_span
            
            color_mdoc = "#2ca02c" # Classic green
            
            # 1. Draw the Wedge (The Fan) - Originates from p
            wedge = Wedge(
                center=tuple(p),
                r=r_viz,
                theta1=theta_start,
                theta2=theta_end,
                facecolor=color_mdoc,
                alpha=0.3, # Consistent alpha with EDOC
                edgecolor='#666666',
                linewidth=0.7,
                zorder=4.0
            )
            ax.add_patch(wedge)
            
            # 2. Draw the Chord (The 'Line Boundary')
            # This shows the linear constraint g^Tu >= b
            p1 = p + r_viz * np.array([np.cos(np.radians(theta_start)), np.sin(np.radians(theta_start))])
            p2 = p + r_viz * np.array([np.cos(np.radians(theta_end)), np.sin(np.radians(theta_end))])
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color='#444444', 
                linewidth=1.2, # Thicker chord
                alpha=0.8,
                zorder=4.1
            )
            
            # 3. Draw Normal Arrow (Pointing to safe side)
            # Use the same style as EDOC for perfect alignment
            arrow_len = 0.12
            ax.arrow(
                p[0], p[1],
                grad[0] * arrow_len, grad[1] * arrow_len,
                head_width=0.03,
                head_length=0.04,
                fc='#444444', 
                ec='#444444',
                alpha=0.8, # More opaque arrow
                zorder=4.5
            )
            
            # 4. Draw a very light full circle to show the original action bound (velocity disk)
            circle = plt.Circle(
                tuple(p), r_viz, 
                color='#777777', 
                fill=False, 
                linestyle=':', 
                linewidth=0.6, 
                alpha=0.3, # Slightly more visible
                zorder=3.5
            )
            ax.add_patch(circle)
    
    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """Save visualization to file."""
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)


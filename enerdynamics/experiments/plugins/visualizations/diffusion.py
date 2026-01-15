"""
Diffusion steps visualization plugin.
"""

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    ) -> None:
        """Visualize a single diffusion step."""
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Draw obstacles
        draw_obstacles(ax, obstacles)
        
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
                ax.plot(positions[:, 0], positions[:, 1], color=EDOC_COLOR, linewidth=2.5)
                
                # Draw start
                ax.scatter(
                    positions[0, 0],
                    positions[0, 1],
                    marker='o',
                    s=30,
                    facecolors='white',
                    edgecolors=EDOC_COLOR,
                    linewidths=1.0,
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
                )

                # Overlay CFS convexified halfspaces for this diffusion step (if enabled)
                if exp_cfg is not None and not is_ebmbd:
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
    
    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """Save visualization to file."""
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)


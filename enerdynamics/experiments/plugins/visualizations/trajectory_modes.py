"""
Trajectory modes visualization plugin.

Visualizes multiple trajectory modes (C paths) found during planning,
with the best (lowest cost) trajectory highlighted.
"""

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from ...framework.base import VisualizationPlugin
from ...common.visualization import (
    draw_obstacles,
    EDOC_COLOR,
    is_d3il_experiment,
    D3IL_BG_YELLOW,
    D3IL_OBSTACLE_RED,
    D3IL_TARGET_GREEN,
)


class TrajectoryModesVisualizationPlugin(VisualizationPlugin):
    """
    Plugin for visualizing multiple trajectory modes (C paths).
    
    Shows all candidate trajectories with different colors/opacities,
    and highlights the best (lowest cost) trajectory.
    """
    
    @property
    def name(self) -> str:
        """Visualization name identifier."""
        return "trajectory_modes"
    
    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Create trajectory modes visualization.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes
            data: Dictionary containing:
                - result: Planning result dictionary with candidate_states, candidate_actions, candidate_costs, best_idx
                - env: Environment instance
                - obstacles: Obstacle manager
                - env_plugin: Environment plugin
            config: Visualization configuration dictionary
        """
        result = data['result']
        env = data['env']
        obstacles = data['obstacles']
        env_plugin = data['env_plugin']
        
        # Get candidate trajectories from result
        candidate_states = result.get('candidate_states', None)
        candidate_actions = result.get('candidate_actions', None)
        candidate_costs = result.get('candidate_costs', None)
        best_idx = result.get('best_idx', 0)
        
        # If no candidate data, fall back to single trajectory
        if candidate_states is None or len(candidate_states) == 0:
            # Fallback: try to extract from single trajectory
            states = result.get('states', [])
            if len(states) > 0:
                candidate_states = [states]
                candidate_costs = [result.get('energies', [0.0])]
                if isinstance(candidate_costs[0], (list, np.ndarray)):
                    candidate_costs = [float(np.sum(c)) if len(c) > 0 else 0.0 for c in candidate_costs]
                best_idx = 0
            else:
                # No data available
                ax.text(0.5, 0.5, 'No trajectory data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
        
        # Convert to numpy arrays
        candidate_states = np.asarray(candidate_states, dtype=object)
        if candidate_costs is not None:
            candidate_costs = np.asarray(candidate_costs, dtype=np.float32)
        else:
            candidate_costs = np.zeros(len(candidate_states), dtype=np.float32)
        
        # Ensure best_idx is valid
        best_idx = int(best_idx)
        if best_idx < 0 or best_idx >= len(candidate_states):
            best_idx = 0
        
        # Get map bounds from config
        obstacle_config = config.get('config', {}).obstacle_config or {}
        map_bounds = obstacle_config.get('map_bounds', {})
        x_min = map_bounds.get('x_min', -2.0)
        x_max = map_bounds.get('x_max', 2.0)
        y_min = map_bounds.get('y_min', -2.0)
        y_max = map_bounds.get('y_max', 2.0)
        
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        exp_cfg = config.get("config")
        is_d3il = is_d3il_experiment(exp_cfg)
        if is_d3il:
            ax.set_facecolor(D3IL_BG_YELLOW)
        # Draw obstacles (D3IL: red; single2d/default: gray)
        if is_d3il:
            draw_obstacles(ax, obstacles, obstacle_color=D3IL_OBSTACLE_RED, obstacle_alpha=1.0)
        else:
            draw_obstacles(ax, obstacles)
        
        # Visualization config: all modes use unified best color
        viz_config = config.get('trajectory_modes', {})
        linewidth_best = float(viz_config.get('linewidth_best', 3.0))
        linewidth_other = float(viz_config.get('linewidth_other', 1.5))
        alpha_best = float(viz_config.get('alpha_best', 0.9))
        alpha_other = float(viz_config.get('alpha_other', 0.35))
        best_color = viz_config.get('best_color', EDOC_COLOR)
        max_modes_to_show = viz_config.get('max_modes_to_show', None)
        
        if max_modes_to_show is None:
            num_modes = int(len(candidate_states))
        else:
            num_modes = int(min(len(candidate_states), int(max_modes_to_show)))
        
        # Optional: draw only up to this time step (for GIF frames)
        partial_until_step = data.get('partial_until_step', None)

        # Draw each candidate trajectory (all same best color)
        for c in range(num_modes):
            states_c = candidate_states[c]
            if len(states_c) == 0:
                continue
            
            try:
                positions = np.array([env_plugin.extract_position(np.asarray(s)) for s in states_c])
            except Exception:
                positions = np.asarray(states_c)
                if positions.ndim > 2:
                    positions = positions[:, :2]
            
            if partial_until_step is not None:
                end_idx = min(partial_until_step + 1, len(positions))
                positions = positions[:end_idx]
            
            if len(positions) < 2:
                if len(positions) == 1:
                    ax.plot(positions[0:1, 0], positions[0:1, 1], color=best_color,
                            linewidth=linewidth_other if c != best_idx else linewidth_best,
                            alpha=alpha_other if c != best_idx else alpha_best,
                            marker='o', markersize=4)
                continue
            
            is_best = (c == best_idx)
            lw = linewidth_best if is_best else linewidth_other
            alpha = alpha_best if is_best else alpha_other
            ax.plot(positions[:, 0], positions[:, 1],
                    color=best_color, linewidth=lw, alpha=alpha)
        
        # Draw start point (same size as diffusion_steps: scatter s=30)
        if len(candidate_states) > best_idx and len(candidate_states[best_idx]) > 0:
            best_states = candidate_states[best_idx]
            try:
                start_pos = env_plugin.extract_position(np.asarray(best_states[0]))
            except Exception:
                start_pos = np.asarray(best_states[0])[:2]
            ax.scatter(
                start_pos[0], start_pos[1],
                marker='o',
                s=30,
                facecolors='white',
                edgecolors=best_color,
                linewidths=1.0,
                zorder=10
            )
        
        # Draw target (no margin circle). D3IL: green; single2d: red.
        target = np.asarray(env.target, dtype=np.float32)
        target_pos = env_plugin.extract_position(target)
        target_color = D3IL_TARGET_GREEN if is_d3il else 'r'
        ax.plot(target_pos[0], target_pos[1], color=target_color, marker='*', markersize=15, linestyle='', zorder=10)
        
        title = config.get('title', f'Trajectory Modes ({num_modes} paths)')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('0.4')
            spine.set_linewidth(0.8)
        ax.grid(True, alpha=0.3)
        if not is_d3il:
            ax.set_facecolor('white')
    
    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """
        Save visualization to file.
        
        Args:
            output_path: Path to save figure
            fig: Matplotlib figure
            **kwargs: Additional save options
        """
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

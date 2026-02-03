"""
Trajectory modes visualization plugin.

Visualizes multiple trajectory modes (C paths) found during planning,
with the best (lowest cost) trajectory highlighted.
"""

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from ...framework.base import VisualizationPlugin
from ...common.visualization import draw_obstacles, EDOC_COLOR


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
        
        # Draw obstacles
        draw_obstacles(ax, obstacles)
        
        # Visualization config
        viz_config = config.get('trajectory_modes', {})
        alpha_other = float(viz_config.get('alpha_other', 0.35))
        alpha_best = float(viz_config.get('alpha_best', 0.9))
        linewidth_other = float(viz_config.get('linewidth_other', 1.5))
        linewidth_best = float(viz_config.get('linewidth_best', 3.0))
        show_labels = bool(viz_config.get('show_labels', True))
        # By default, show *all* returned modes; keep an optional cap for readability.
        max_modes_to_show = viz_config.get('max_modes_to_show', None)
        best_color = viz_config.get('best_color', EDOC_COLOR)
        
        # Get colormap for different modes
        if max_modes_to_show is None:
            num_modes = int(len(candidate_states))
        else:
            num_modes = int(min(len(candidate_states), int(max_modes_to_show)))
        cmap_name = viz_config.get('cmap', 'turbo' if num_modes > 20 else 'tab20')
        if num_modes > 1:
            colors = cm.get_cmap(cmap_name)(np.linspace(0, 1, num_modes))
        else:
            colors = [best_color]
        
        # Draw each candidate trajectory
        robot_radius = float(obstacle_config.get('robot_radius', 0.05))
        
        for c in range(num_modes):
            states_c = candidate_states[c]
            if len(states_c) == 0:
                continue
            
            # Extract positions
            try:
                positions = np.array([env_plugin.extract_position(np.asarray(s)) for s in states_c])
            except Exception:
                # Fallback: assume states are already positions
                positions = np.asarray(states_c)
                if positions.ndim > 2:
                    positions = positions[:, :2]  # Take first 2 dims
            
            if len(positions) < 2:
                continue
            
            is_best = (c == best_idx)
            color = best_color if is_best else colors[c % len(colors)]
            lw = linewidth_best if is_best else linewidth_other
            alpha = alpha_best if is_best else alpha_other

            # Do not label each mode in the legend (avoids huge legends for many modes).
            label = None
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=color, linewidth=lw, alpha=alpha, label=label)
        
        # Draw start point (from best trajectory)
        if len(candidate_states) > best_idx and len(candidate_states[best_idx]) > 0:
            best_states = candidate_states[best_idx]
            try:
                start_pos = env_plugin.extract_position(np.asarray(best_states[0]))
            except Exception:
                start_pos = np.asarray(best_states[0])[:2]
            
            start_circle = plt.Circle(
                tuple(start_pos),
                robot_radius,
                facecolor='white',
                edgecolor=colors[best_idx % len(colors)],
                linewidth=2.0,
                label='Start',
                zorder=10
            )
            ax.add_patch(start_circle)
        
        # Draw target
        target = np.asarray(env.target, dtype=np.float32)
        target_pos = env_plugin.extract_position(target)
        target_circle = plt.Circle(
            tuple(target_pos),
            robot_radius * 2,  # Robot diameter
            facecolor='none',
            edgecolor='red',
            linewidth=1.5,
            linestyle='--',
            label='Target (margin)',
            zorder=10
        )
        ax.add_patch(target_circle)
        (target_star,) = ax.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target', zorder=10)
        
        title = config.get('title', f'Trajectory Modes ({num_modes} paths)')
        ax.set_title(title, fontsize=12)
        
        if show_labels:
            # Compact legend: avoid per-mode labels; show only aggregate entries.
            mode_handle = Line2D([0], [0], color='gray', linewidth=linewidth_other, alpha=alpha_other, label='Modes')
            best_handle = Line2D([0], [0], color=best_color, linewidth=linewidth_best, alpha=alpha_best, label='Best')

            handles = [mode_handle, best_handle]
            labels = ['Modes', 'Best']

            # Pull labeled handles from the axes (Start / Target / Target margin)
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            for h, l in zip(ax_handles, ax_labels):
                if l in ('Start', 'Target (margin)', 'Target') and l not in labels:
                    handles.append(h)
                    labels.append(l)

            ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=8, ncol=1, framealpha=0.8)
        ax.grid(True, alpha=0.3)
    
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

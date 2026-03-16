"""
State components visualization plugin.
"""

from typing import Dict, Any, List
import numpy as np

from genedynamics.core.types import Trajectory
from ...framework.base import VisualizationPlugin
from ...common.visualization import EDOC_COLOR


class StatesVisualizationPlugin(VisualizationPlugin):
    """
    Plugin for visualizing state components over time.
    """
    
    @property
    def name(self) -> str:
        """Visualization name identifier."""
        return "states"
    
    def visualize(self, fig: Any, axes: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Create state components plots.
        
        Args:
            fig: Matplotlib figure
            axes: Array of matplotlib axes (one per state component)
            data: Dictionary containing:
                - trajectory: Trajectory object
                - env: Environment instance
            config: Visualization configuration dictionary
        """
        trajectory = data['trajectory']
        env = data['env']
        
        if len(trajectory.states) == 0:
            return
        
        states_array = np.array([np.asarray(s) for s in trajectory.states], dtype=np.float32)
        dt = getattr(env, 'dt', 0.1)
        time = np.arange(len(trajectory.states)) * dt
        
        # Generate labels based on state dimension
        state_dim = states_array.shape[1]
        if state_dim == 2:
            labels = [("x position", 0), ("y position", 1)]
        elif state_dim == 4:
            labels = [
                ("x position", 0),
                ("y position", 1),
                ("x velocity", 2),
                ("y velocity", 3),
            ]
        else:
            # Generic labels
            labels = [(f"State {i}", i) for i in range(state_dim)]
        
        for ax, (title, idx) in zip(axes[:len(labels)], labels):
            ax.plot(time, states_array[:, idx], color=EDOC_COLOR, linewidth=2.0)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
    
    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """Save visualization to file."""
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)


"""
SSR (Safety Success Rate) metrics plugin.
"""

from typing import Dict, Any, Optional
import numpy as np

from enerdynamics.core.types import Trajectory
from enerdynamics.envs.obstacles.base import ObstacleManager
from ...framework.base import MetricsPlugin


class SSRMetricsPlugin(MetricsPlugin):
    """
    Plugin for computing Safety Success Rate (SSR) metrics.
    
    SSR = 1 if trajectory is safe, constraint-feasible, and reaches target, else 0.
    """
    
    @property
    def name(self) -> str:
        """Metric name identifier."""
        return "ssr"
    
    def compute(self, trajectory: Trajectory, env: Any, obstacles: ObstacleManager,
                constraints: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Compute SSR metrics.
        
        Args:
            trajectory: Computed trajectory
            env: Environment instance
            obstacles: Obstacle manager instance
            constraints: Constraint manager instance (optional)
            **kwargs: Additional parameters:
                - robot_radius: Robot radius for collision checking (default: 0.05)
                - success_margin: Distance threshold for success (default: 2 * robot_radius)
                
        Returns:
            Dictionary containing SSR and component metrics
        """
        robot_radius = float(kwargs.get('robot_radius', 0.05))
        success_margin = kwargs.get('success_margin', None)
        if success_margin is None:
            success_margin = 2 * robot_radius  # Robot diameter
        
        env_plugin = kwargs.get("env_plugin", None)

        # Helper: extract 2D position
        def extract_pos_2d(x: np.ndarray) -> np.ndarray:
            if env_plugin is not None and hasattr(env_plugin, "extract_position"):
                p = np.asarray(env_plugin.extract_position(x), dtype=np.float32).reshape(-1)
                return p[:2]
            x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
            return x_arr[:2] if x_arr.size >= 2 else x_arr

        # Check safety: no collisions (considering robot radius)
        safe = True
        if len(obstacles) > 0:
            for state in trajectory.states:
                pos_2d = extract_pos_2d(np.asarray(state, dtype=np.float32))
                
                # Check if robot (with radius) collides with obstacles
                sdf = obstacles.sdf(pos_2d)
                sdf_val = float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf)
                if sdf_val < robot_radius:
                    safe = False
                    break
                
                if obstacles.contains(pos_2d):
                    safe = False
                    break
        
        # Check constraint feasibility
        feasible = True
        constraint_name = "unknown"
        if constraints is not None and hasattr(constraints, 'hard_constraints'):
            # Check all hard constraints
            for hard_constraint in constraints.hard_constraints:
                if hasattr(hard_constraint, 'is_feasible'):
                    if not hard_constraint.is_feasible(trajectory):
                        feasible = False
                        constraint_name = type(hard_constraint).__name__
                        break
        
        # Check task success: reached target within margin
        task_success = False
        distance_to_target = float('inf')
        if len(trajectory.states) > 0:
            final_state = np.asarray(trajectory.states[-1], dtype=np.float32)
            final_pos = extract_pos_2d(final_state)
            target = np.asarray(env.target, dtype=np.float32)
            target_pos = extract_pos_2d(target)
            distance_to_target = float(np.linalg.norm(final_pos - target_pos))
            task_success = bool(distance_to_target < success_margin)
        
        # SSR: safe AND feasible AND task success
        ssr = 1.0 if (safe and feasible and task_success) else 0.0
        
        return {
            'ssr': float(ssr),
            'safe': bool(safe),
            'feasible': bool(feasible),
            'task_success': bool(task_success),
            'distance_to_target': float(distance_to_target),
            'constraint_name': constraint_name,
        }


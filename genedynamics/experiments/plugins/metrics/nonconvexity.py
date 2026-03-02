"""
Nonconvexity metrics plugin.
"""

from typing import Dict, Any
import numpy as np

from genedynamics.core.types import Trajectory
from genedynamics.envs.obstacles.base import ObstacleManager
from genedynamics.envs.obstacles.nonconvex import UnionObstacle
from ...framework.base import MetricsPlugin


class NonconvexityMetricsPlugin(MetricsPlugin):
    """
    Plugin for computing nonconvexity metrics.
    
    Computes both simple nonconvexity metrics (union fraction, primitive counts)
    and geometric SDF-based nonconvexity scores.
    """
    
    @property
    def name(self) -> str:
        """Metric name identifier."""
        return "nonconvexity"
    
    def compute(self, trajectory: Trajectory, env: Any, obstacles: ObstacleManager,
                constraints: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Compute nonconvexity metrics.
        
        Args:
            trajectory: Computed trajectory (not used, but required by interface)
            env: Environment instance
            obstacles: Obstacle manager instance
            constraints: Constraint manager (not used)
            **kwargs: Additional parameters:
                - robot_radius: Robot radius (default: 0.05)
                - level: Obstacle level (for seed, default: 0)
                - obstacle_config: Obstacle configuration dict (for map bounds)
                - n_pairs: Number of point pairs for SDF metric (default: 20000)
                
        Returns:
            Dictionary containing nonconvexity metrics
        """
        # Simple metrics based on obstacle structure
        simple_metrics = self._compute_simple_metrics(obstacles)
        
        # SDF-based geometric metrics
        sdf_metrics = self._compute_sdf_metrics(obstacles, env, kwargs)
        
        return {
            **simple_metrics,
            'sdf': sdf_metrics,
        }
    
    def _compute_simple_metrics(self, obstacles: ObstacleManager) -> Dict[str, Any]:
        """
        Compute simple nonconvexity metrics based on obstacle structure.
        
        Args:
            obstacles: Obstacle manager
            
        Returns:
            Dictionary with simple metrics
        """
        obstacles_list = list(obstacles)
        obstacle_count = int(len(obstacles_list))
        
        if obstacle_count == 0:
            return {
                "obstacle_count": 0,
                "union_count": 0,
                "total_primitives": 0,
                "union_fraction": 0.0,
                "avg_primitives_per_union": 0.0,
                "nonconvexity_score": 0.0,
            }
        
        union_count = 0
        union_primitives = 0
        total_primitives = 0
        
        for obs in obstacles_list:
            if isinstance(obs, UnionObstacle):
                union_count += 1
                k = int(len(obs.obstacles))
                union_primitives += k
                total_primitives += k
            else:
                total_primitives += 1
        
        union_fraction = float(union_count / obstacle_count) if obstacle_count > 0 else 0.0
        avg_prims = float(union_primitives / union_count) if union_count > 0 else 0.0
        nonconvexity_score = float(union_fraction * max(0.0, avg_prims - 1.0))
        
        return {
            "obstacle_count": obstacle_count,
            "union_count": union_count,
            "total_primitives": int(total_primitives),
            "union_fraction": union_fraction,
            "avg_primitives_per_union": avg_prims,
            "nonconvexity_score": nonconvexity_score,
        }
    
    def _compute_sdf_metrics(self, obstacles: ObstacleManager, env: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute SDF-based geometric nonconvexity metrics.
        
        Args:
            obstacles: Obstacle manager
            env: Environment instance
            kwargs: Additional parameters
            
        Returns:
            Dictionary with SDF-based metrics
        """
        if len(obstacles) == 0:
            return {
                "score_raw": 0.0,
                "violation_rate": 0.0,
                "mean_violation": 0.0,
                "n_pairs": 0,
                "robot_radius": 0.0,
            }
        
        robot_radius = float(kwargs.get('robot_radius', 0.05))
        level = int(kwargs.get('level', 0))
        n_pairs = int(kwargs.get('n_pairs', 20000))
        obstacle_config = kwargs.get('obstacle_config', {})
        
        # Get map bounds
        map_bounds = obstacle_config.get('map_bounds', {})
        p_max = float(getattr(env, 'p_max', 2.0))
        x_min = map_bounds.get('x_min', -p_max)
        x_max = map_bounds.get('x_max', p_max)
        y_min = map_bounds.get('y_min', -p_max)
        y_max = map_bounds.get('y_max', p_max)
        
        # Use deterministic seed for reproducibility
        seed = int(level) + 54321
        rng = np.random.RandomState(seed)
        
        n = int(max(1, n_pairs))
        xlo = float(x_min)
        xhi = float(x_max)
        ylo = float(y_min)
        yhi = float(y_max)
        
        # Sample random point pairs
        x = np.stack(
            [
                rng.uniform(xlo, xhi, size=(n,)).astype(np.float32),
                rng.uniform(ylo, yhi, size=(n,)).astype(np.float32),
            ],
            axis=-1,
        )
        y = np.stack(
            [
                rng.uniform(xlo, xhi, size=(n,)).astype(np.float32),
                rng.uniform(ylo, yhi, size=(n,)).astype(np.float32),
            ],
            axis=-1,
        )
        mid = 0.5 * (x + y)
        
        # Compute SDF values (with robot radius inflation)
        dx = np.asarray(obstacles.sdf(x), dtype=np.float32).reshape(-1) - robot_radius
        dy = np.asarray(obstacles.sdf(y), dtype=np.float32).reshape(-1) - robot_radius
        dm = np.asarray(obstacles.sdf(mid), dtype=np.float32).reshape(-1) - robot_radius
        
        # Violation: d(mid) < min(d(x), d(y)) (nonconvexity indicator)
        viol = np.maximum(0.0, np.minimum(dx, dy) - dm)
        mean_viol = float(np.mean(viol))
        rate = float(np.mean(viol > 1e-6))
        
        return {
            "score_raw": mean_viol,
            "violation_rate": rate,
            "mean_violation": mean_viol,
            "n_pairs": int(n),
            "robot_radius": float(robot_radius),
        }


"""
NumPy backend implementation of CBF double integrator 2D action filter.
"""

import numpy as np
from typing import Optional

from enerdynamics.core.constraints.action_filters.backend_impl import ActionFilterBase
from enerdynamics.core.registry.action_filters import register_action_filter
from enerdynamics.core.types import State
from enerdynamics.envs.obstacles.base import ObstacleManager


@register_action_filter("cbf_double_integrator_2d", "numpy")
class CBFActionFilterNumpy(ActionFilterBase):
    """
    NumPy backend implementation of CBF double integrator 2D action filter.
    
    Uses CBF-style barrier h(p) = sdf(p) - (robot_radius + clearance) with
    relative-degree-2 condition: n^T a >= -(k1 * hdot + k0 * h)
    """
    
    def __init__(self, obstacles: ObstacleManager, **config):
        """
        Initialize NumPy backend CBF action filter.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            **config: Configuration parameters:
                - robot_radius: Robot radius
                - dt: Time step
                - tau: Activation threshold (only filter when h < tau)
                - k0: CBF gain k0
                - k1: CBF gain k1
                - schedule_manager: Optional ConstraintScheduleManager
        """
        super().__init__(obstacles, **config)
        self.robot_radius = config.get('robot_radius', 0.05)
        self.dt = config.get('dt', 0.1)
        self.tau = config.get('tau', 0.05)
        self.k0 = config.get('k0', 1.0)
        self.k1 = config.get('k1', 4.0)
        self.schedule_manager = config.get('schedule_manager', None)
    
    def filter_actions(
        self,
        x0: State,
        actions: np.ndarray,
        step: Optional[int],
        total_steps: Optional[int],
        **kwargs
    ) -> np.ndarray:
        """
        Filter actions to ensure safety using NumPy implementation.
        
        Args:
            x0: Initial state
            actions: Actions to filter, shape (T, action_dim)
            step: Current step (for scheduling)
            total_steps: Total steps (for scheduling)
            **kwargs: Additional parameters
            
        Returns:
            Filtered actions, shape (T, action_dim)
        """
        actions = np.asarray(actions, dtype=np.float32).copy()
        if actions.ndim != 2 or actions.shape[1] != 2:
            return actions
        
        if self.obstacles.get_sdf_texture_2d() is None:
            # No texture: fallback to do nothing
            return actions
        
        tex = self.obstacles.get_sdf_texture_2d()
        clearance = self._get_clearance(self.schedule_manager, step, total_steps)
        margin = float(self.robot_radius + clearance)
        dt = float(self.dt)
        
        x = np.asarray(x0, dtype=np.float32).copy()
        for t in range(actions.shape[0]):
            p = x[:2]
            v = x[2:4]
            a = actions[t]
            sdf, grad = tex.sample(p, backend="numpy")
            sdf = float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf)
            grad = np.asarray(grad, dtype=np.float32).reshape(2)
            gnorm = float(np.linalg.norm(grad))
            
            if gnorm > 1e-6:
                n = grad / gnorm
                h = sdf - margin
                if h < self.tau:
                    hdot = float(np.dot(n, v))
                    b = -(self.k1 * hdot + self.k0 * h)
                    lhs = float(np.dot(n, a))
                    delta = max(0.0, b - lhs)  # ||n||^2 ~ 1
                    a = a + delta * n
                    actions[t] = a.astype(np.float32)
            
            # Integrate one step (double integrator)
            # NOTE: We intentionally do not clamp here; caller/environment handles bounds.
            v_next = v + dt * a
            p_next = p + dt * v_next
            x = np.concatenate([p_next, v_next]).astype(np.float32)
        
        return actions


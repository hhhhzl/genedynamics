"""
PyTorch backend implementation of CBF double integrator 2D action filter.
"""

import numpy as np
from typing import Optional, Callable

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from enerdynamics.core.constraints.action_filters.backend_impl import ActionFilterBase
from enerdynamics.core.registry.action_filters import register_action_filter
from enerdynamics.core.types import State
from enerdynamics.envs.obstacles.base import ObstacleManager


if TORCH_AVAILABLE:
    @register_action_filter("cbf_double_integrator_2d", "torch")
    class CBFActionFilterTorch(ActionFilterBase):
        """
        PyTorch backend implementation of CBF double integrator 2D action filter.
        
        This implementation uses PyTorch operations optimized for GPU acceleration.
        """
        
        def __init__(self, obstacles: ObstacleManager, **config):
            """
            Initialize PyTorch backend CBF action filter.
            
            Args:
                obstacles: ObstacleManager containing obstacles
                **config: Configuration parameters:
                    - robot_radius: Robot radius
                    - dt: Time step
                    - tau: Activation threshold
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
            
            # Build PyTorch filter function
            self._torch_filter_fn = None
            self._build_torch_filter()
        
        def _build_torch_filter(self):
            """Build PyTorch-compatible filter function."""
            if torch is None:
                return
            
            tex = self.obstacles.get_sdf_texture_2d()
            if tex is None:
                return
            
            tau = float(self.tau)
            k0 = float(self.k0)
            k1 = float(self.k1)
            robot_radius = float(self.robot_radius)
            
            def filter_fn(state, action, hard_clearance, hard_enabled):
                """PyTorch filter function."""
                # state: (..., 4), action: (..., 2)
                device = action.device
                p = state[..., 0:2]
                v = state[..., 2:4]
                sdf, grad = tex.sample(p, backend="torch", device=device)
                gnorm = torch.linalg.norm(grad, dim=-1).clamp_min(1e-12)
                n = grad / gnorm.unsqueeze(-1)
                margin = robot_radius + hard_clearance
                h = sdf - margin
                hdot = (n * v).sum(dim=-1)
                b = -(k1 * hdot + k0 * h)
                lhs = (n * action).sum(dim=-1)
                delta = (b - lhs).clamp_min(0.0)
                do = (gnorm > 1e-6) & (h < tau) & hard_enabled
                action_new = action + delta.unsqueeze(-1) * n
                return torch.where(do.unsqueeze(-1), action_new, action)
            
            self._torch_filter_fn = filter_fn
        
        def filter_actions(
            self,
            x0: State,
            actions: np.ndarray,
            step: Optional[int],
            total_steps: Optional[int],
            **kwargs
        ) -> np.ndarray:
            """
            Filter actions to ensure safety using PyTorch implementation.
            
            Args:
                x0: Initial state
                actions: Actions to filter, shape (T, action_dim)
                step: Current step (for scheduling)
                total_steps: Total steps (for scheduling)
                **kwargs: Additional parameters
                
            Returns:
                Filtered actions, shape (T, action_dim)
            """
            if self._torch_filter_fn is None:
                # Fallback: return actions unchanged if PyTorch filter not available
                # (This should not happen if obstacles support PyTorch)
                return np.asarray(actions, dtype=np.float32)
            
            # Convert to PyTorch tensors
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x0_torch = torch.as_tensor(x0, dtype=torch.float32, device=device)
            actions_torch = torch.as_tensor(actions, dtype=torch.float32, device=device)
            clearance = self._get_clearance(self.schedule_manager, step, total_steps)
            hard_enabled = torch.tensor(True, device=device)
            
            # Apply filter step by step
            x = x0_torch
            filtered_actions = []
            for t in range(actions.shape[0]):
                action = actions_torch[t:t+1]  # (1, 2)
                state = x[None]  # (1, 4)
                
                # Apply filter
                filtered_action = self._torch_filter_fn(
                    state, action, clearance, hard_enabled
                )
                filtered_action = filtered_action[0]  # Remove batch dimension
                filtered_actions.append(filtered_action)
                
                # Integrate one step (double integrator)
                p = x[:2]
                v = x[2:4]
                a = filtered_action
                v_next = v + self.dt * a
                p_next = p + self.dt * v_next
                x = torch.cat([p_next, v_next])
            
            # Stack and convert back to NumPy
            filtered_torch = torch.stack(filtered_actions, dim=0)
            return filtered_torch.cpu().numpy()
        
        def make_torch_filter(self) -> Optional[Callable]:
            """
            Return a PyTorch-callable filter function.
            
            Returns:
                PyTorch function (state, action, hard_clearance, hard_enabled) -> filtered_action,
                or None if not available
            """
            return self._torch_filter_fn

else:
    # PyTorch not available - don't register the class
    pass


"""
Obstacle constraint implementations for soft and hard constraints.

This module provides:
- ObstacleSoftConstraint: Barrier energy for collision avoidance
- ObstacleHardConstraint: Strict collision avoidance

For projection operators (CFSProjection), see enerdynamics.core.constraints.projections.cfs
"""

import numpy as np
from typing import Callable, Optional, List
from enerdynamics.core.constraints.base import SoftConstraint, HardConstraint
from enerdynamics.core.types import Trajectory, State
from enerdynamics.envs.obstacles.base import ObstacleManager

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


class ObstacleSoftConstraint(SoftConstraint):
    """
    Soft obstacle constraint: differentiable barrier energy.
    
    S(τ) = Σ_h Σ_m α * exp(-β * g_m(x^h))
    where g_m(x^h) is clearance function (SDF) for obstacle m at timestep h.
    
    This provides smooth gradients for optimization while encouraging
    obstacle avoidance.
    """
    
    def __init__(
        self,
        obstacles: ObstacleManager,
        alpha: float = 1.0,
        beta: float = 10.0,
        position_extractor: Optional[Callable[[State], np.ndarray]] = None,
    ):
        """
        Initialize soft obstacle constraint.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            alpha: Barrier strength (higher = stronger penalty)
            beta: Barrier sharpness (higher = sharper barrier)
            position_extractor: Function to extract position from state
                              (if None, uses default: first 2 dims for 2D, first 1 dim for 1D)
        """
        self.obstacles = obstacles
        self.alpha = alpha
        self.beta = beta
        self.position_extractor = position_extractor or self._default_extract_position
    
    def evaluate(self, trajectory: Trajectory) -> float:
        """Compute barrier energy: Σ_h Σ_m α * exp(-β * sdf(x^h))"""
        total = 0.0
        for state in trajectory.states:
            pos = self.position_extractor(state)
            sdf = self.obstacles.sdf(pos)
            # Ensure sdf is a scalar
            sdf = float(np.asarray(sdf).item() if hasattr(sdf, 'item') else sdf)
            # Barrier: high when close to obstacles (negative or small SDF)
            # Using exponential barrier: exp(-β * sdf) grows as sdf becomes negative
            total += self.alpha * np.exp(-self.beta * sdf)
        return float(total)
    
    @staticmethod
    def _default_extract_position(state: State) -> np.ndarray:
        """Default position extractor: assumes first dims are position."""
        state_np = np.asarray(state, dtype=np.float32)
        # For double integrator: first dim(s) are positions
        if len(state_np) == 4:  # 2D double integrator [px, py, vx, vy]
            return state_np[:2]
        elif len(state_np) == 2:  # 1D double integrator [p, v]
            return state_np[:1]
        else:
            # Fallback: assume first 2 dims
            return state_np[:min(2, len(state_np))]


class ObstacleHardConstraint(HardConstraint):
    """
    Hard obstacle constraint: strict collision avoidance.
    
    Feasible set: F = {τ: g_m(x^h) ≥ clearance, ∀m, h}
    where g_m(x^h) is SDF for obstacle m at timestep h.
    """
    
    def __init__(
        self,
        obstacles: ObstacleManager,
        clearance: float = 0.0,
        position_extractor: Optional[Callable[[State], np.ndarray]] = None,
    ):
        """
        Initialize hard obstacle constraint.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            clearance: Safety margin (minimum SDF required)
            position_extractor: Function to extract position from state
        """
        self.obstacles = obstacles
        self.clearance = clearance
        self.position_extractor = position_extractor or self._default_extract_position
    
    def is_feasible(self, trajectory: Trajectory) -> bool:
        """Check if trajectory avoids all obstacles with clearance."""
        for state in trajectory.states:
            pos = self.position_extractor(state)
            # Check collision
            if self.obstacles.contains(pos):
                return False
            # Check clearance
            sdf = self.obstacles.sdf(pos)
            # Ensure sdf is a scalar
            sdf = float(np.asarray(sdf).item() if hasattr(sdf, 'item') else sdf)
            if sdf < self.clearance:
                return False
        return True
    
    def violations(self, trajectory: Trajectory) -> np.ndarray:
        """Return clearance violations: max(0, clearance - sdf(x^h)) for each timestep."""
        violations = []
        for state in trajectory.states:
            pos = self.position_extractor(state)
            sdf = self.obstacles.sdf(pos)
            # Ensure sdf is a scalar
            sdf = float(np.asarray(sdf).item() if hasattr(sdf, 'item') else sdf)
            violation = max(0.0, self.clearance - sdf)
            violations.append(violation)
        return np.array(violations, dtype=np.float32)
    
    def project(
        self,
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> Trajectory:
        """
        Project trajectory onto feasible set.
        
        Simple projection: push states away from obstacles until feasible.
        For more sophisticated projection (CFS-QP), use CFSProjection instead.
        
        Args:
            trajectory: Trajectory to project
            step: Current step (optional, for scheduled clearance)
            total_steps: Total steps (optional, for scheduled clearance)
            
        Returns:
            Projected trajectory
        """
        # Get clearance (can be scheduled)
        clearance = self._get_clearance(step, total_steps)
        
        projected_states = []
        for state in trajectory.states:
            pos = self.position_extractor(state)
            projected_pos = self._project_point_simple(pos, clearance)
            projected_state = self._reconstruct_state(state, projected_pos)
            projected_states.append(projected_state)
        
        return Trajectory(states=projected_states, actions=trajectory.actions)
    
    def _get_clearance(self, step: Optional[int], total_steps: Optional[int]) -> float:
        """Get clearance, optionally scheduled."""
        # Default: use fixed clearance
        # Can override to implement schedule: relaxed early, strict late
        return self.clearance
    
    def _project_point_simple(self, point: np.ndarray, clearance: float) -> np.ndarray:
        """Simple projection: push away from closest obstacle."""
        point = np.asarray(point, dtype=np.float32)
        sdf = self.obstacles.sdf(point)
        # Ensure sdf is a scalar
        sdf = float(np.asarray(sdf).item() if hasattr(sdf, 'item') else sdf)
        
        # If feasible, return as-is
        if sdf >= clearance:
            return point
        
        # Find closest obstacle and push away
        min_sdf = float('inf')
        closest_grad = None
        
        for obstacle in self.obstacles:
            obs_sdf = obstacle.sdf(point)
            # Ensure sdf is a scalar
            obs_sdf = float(np.asarray(obs_sdf).item() if hasattr(obs_sdf, 'item') else obs_sdf)
            if obs_sdf < min_sdf:
                min_sdf = obs_sdf
                if hasattr(obstacle, 'gradient'):
                    grad = obstacle.gradient(point)
                    if grad is not None:
                        grad = np.asarray(grad, dtype=np.float32)
                        # SDF gradient points in direction of increasing SDF (away from obstacle)
                        # If inside obstacle (negative SDF), we want to move in gradient direction
                        # If outside but too close, we also move in gradient direction
                        closest_grad = grad
        
        # Push along gradient to satisfy clearance
        if closest_grad is not None:
            grad_norm = np.linalg.norm(closest_grad)
            if grad_norm > 1e-8:
                violation = clearance - min_sdf
                # Move along gradient direction (increasing SDF = away from obstacle)
                point = point + violation * closest_grad / grad_norm
            else:
                # No gradient available: push away from obstacle center if possible
                for obstacle in self.obstacles:
                    if hasattr(obstacle, 'center'):
                        center = np.asarray(obstacle.center, dtype=np.float32)
                        direction = point - center
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm > 1e-8:
                            violation = clearance - min_sdf
                            point = point + violation * direction / direction_norm
                            break
        
        return point
    
    @staticmethod
    def _default_extract_position(state: State) -> np.ndarray:
        """Default position extractor."""
        state_np = np.asarray(state, dtype=np.float32)
        if len(state_np) == 4:
            return state_np[:2]
        elif len(state_np) == 2:
            return state_np[:1]
        return state_np[:min(2, len(state_np))]
    
    @staticmethod
    def _reconstruct_state(original_state: State, new_pos: np.ndarray) -> np.ndarray:
        """Reconstruct state with new position."""
        state_np = np.asarray(original_state, dtype=np.float32)
        new_state = state_np.copy()
        if len(state_np) == 4:
            new_state[:2] = new_pos
        elif len(state_np) == 2:
            new_state[0] = new_pos[0]
        return new_state
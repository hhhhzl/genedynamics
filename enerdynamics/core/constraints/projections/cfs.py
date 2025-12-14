"""
Convex Feasible Set (CFS) projection implementation.

This module provides CFSProjection: A feasibility operator that implements
CFS-QP (Convex Feasible Set via Quadratic Programming) for non-convex obstacle
constraints using linearization.

Optimized with batch tensor operations for efficiency.
"""

import numpy as np
from typing import Callable, Optional
from enerdynamics.core.constraints.base import FeasibilityOperator
from enerdynamics.core.types import Trajectory, State
from enerdynamics.envs.obstacles.base import ObstacleManager

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


class CFSProjection(FeasibilityOperator):
    """
    Convex Feasible Set (CFS) projection via linearization.
    
    For non-convex obstacles, linearizes constraints around reference trajectory
    to create convex inner approximation. Implements QP-based projection.
    
    This implements the CFS-QP method from the theory for efficient projection
    onto non-convex obstacle-free space.
    
    Optimized with batch tensor operations to avoid nested loops.
    """
    
    def __init__(
        self,
        obstacles: ObstacleManager,
        clearance_schedule: Optional[Callable[[Optional[int], Optional[int]], float]] = None,
        position_extractor: Optional[Callable[[State], np.ndarray]] = None,
        use_late_stage_only: bool = False,
        late_stage_ratio: float = 0.3,  # Only apply in last 30% of steps
        max_iterations: int = 5,
        convergence_tol: float = 1e-6,
    ):
        """
        Initialize CFS projection operator.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            clearance_schedule: Function (step, total_steps) -> clearance
                               (if None, uses fixed clearance from hard constraints)
            position_extractor: Function to extract position from state
            use_late_stage_only: If True, only apply projection in late stages
            late_stage_ratio: Ratio of steps for late-stage application
            max_iterations: Maximum iterations for iterative projection
            convergence_tol: Convergence tolerance for iterative projection
        """
        self.obstacles = obstacles
        self.clearance_schedule = clearance_schedule
        self.position_extractor = position_extractor or self._default_extract_position
        self.use_late_stage_only = use_late_stage_only
        self.late_stage_ratio = late_stage_ratio
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
    
    def should_apply(self, step: Optional[int] = None, total_steps: Optional[int] = None) -> bool:
        """Check if projection should be applied at current step."""
        if not self.use_late_stage_only:
            return True
        
        if step is None or total_steps is None:
            return True
        
        # Only apply in late stages
        progress = step / total_steps if total_steps > 0 else 1.0
        return progress >= (1.0 - self.late_stage_ratio)
    
    def project(
        self,
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> Trajectory:
        """
        Project using CFS: linearize obstacles around trajectory and solve QP.
        
        Optimized batch version that processes all states in parallel.
        
        Args:
            trajectory: Trajectory to project
            step: Current diffusion step
            total_steps: Total diffusion steps
            
        Returns:
            Projected trajectory
        """
        # Get clearance (scheduled if provided)
        if self.clearance_schedule is not None:
            clearance = self.clearance_schedule(step, total_steps)
        else:
            clearance = 0.0  # Default
        
        # Extract all positions at once (batch processing)
        # Use list comprehension then stack for better performance
        position_list = [self.position_extractor(state) for state in trajectory.states]
        if len(position_list) > 0:
            # Stack into array: more efficient than creating from list
            positions = np.stack(position_list, axis=0).astype(np.float32)
        else:
            positions = np.zeros((0, 2), dtype=np.float32)  # Empty trajectory
        
        # Batch project all positions
        projected_positions = self._project_cfs_batch(
            positions, 
            clearance, 
            step
        )
        
        # Reconstruct states with projected positions
        projected_states = []
        for i, state in enumerate(trajectory.states):
            projected_state = self._reconstruct_state(state, projected_positions[i])
            projected_states.append(projected_state)
        
        return Trajectory(states=projected_states, actions=trajectory.actions)
    
    def _project_cfs_batch(
        self,
        positions: np.ndarray,
        clearance: float,
        step: Optional[int]
    ) -> np.ndarray:
        """
        Batch project multiple points onto CFS using linearized constraints.
        
        This uses vectorized operations to process all points simultaneously,
        avoiding nested loops.
        
        Args:
            positions: Points to project, shape (N, dim)
            clearance: Minimum clearance required
            step: Current step (for logging/debugging)
            
        Returns:
            Projected positions, shape (N, dim)
        """
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        N, dim = positions.shape
        current = positions.copy()
        
        # Iteratively project until all points satisfy constraints
        for iteration in range(self.max_iterations):
            # Batch compute SDF for all points and all obstacles
            # Shape: (N,) - minimum SDF across all obstacles for each point
            sdfs = self.obstacles.sdf(current)
            
            # Ensure sdfs is 1D array
            if not isinstance(sdfs, np.ndarray):
                sdfs = np.array([sdfs] * N, dtype=np.float32)
            elif sdfs.ndim == 0:
                sdfs = np.full(N, float(sdfs), dtype=np.float32)
            
            # Check which points violate clearance
            violations = clearance - sdfs  # Positive = violation
            violating_mask = violations > 0
            
            # If no violations, we're done
            if not np.any(violating_mask):
                break
            
            # For violating points, compute correction vectors
            corrections = np.zeros_like(current)
            
            # Process each obstacle to accumulate corrections
            for obstacle in self.obstacles:
                # Compute SDF for this obstacle (batch operation)
                obs_sdfs = obstacle.sdf(current)
                if not isinstance(obs_sdfs, np.ndarray):
                    obs_sdfs = np.array([obs_sdfs] * N, dtype=np.float32)
                elif obs_sdfs.ndim == 0:
                    obs_sdfs = np.full(N, float(obs_sdfs), dtype=np.float32)
                
                # Find points that violate this obstacle's clearance (vectorized)
                obs_violations = clearance - obs_sdfs
                obs_violating_mask = obs_violations > 0
                
                if not np.any(obs_violating_mask):
                    continue
                
                violating_indices = np.where(obs_violating_mask)[0]
                
                # Compute gradients for violating points
                if hasattr(obstacle, 'gradient'):
                    # Compute gradients for all violating points
                    # Note: gradient() typically only accepts single points, so we need a loop
                    # but this is minimal since we only process violating points
                    violating_points = current[violating_indices]
                    violations = obs_violations[violating_indices]
                    
                    for i, idx in enumerate(violating_indices):
                        grad = obstacle.gradient(violating_points[i])
                        if grad is not None:
                            grad = np.asarray(grad, dtype=np.float32).flatten()
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-8:
                                # Compute correction: push along gradient
                                violation = violations[i]
                                corrections[idx] += violation * grad / grad_norm
                else:
                    # Fallback: push away from center (fully vectorized)
                    if hasattr(obstacle, 'center'):
                        center = np.asarray(obstacle.center, dtype=np.float32)
                        # Vectorized: compute direction for all violating points at once
                        violating_points = current[violating_indices]
                        violations = obs_violations[violating_indices]
                        
                        # Compute directions (vectorized)
                        directions = violating_points - center  # Shape: (M, dim)
                        direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)  # Shape: (M, 1)
                        valid_mask = (direction_norms.squeeze() > 1e-8)  # Shape: (M,)
                        
                        if np.any(valid_mask):
                            # Normalize directions (vectorized)
                            normalized = directions / (direction_norms + 1e-8)  # Shape: (M, dim)
                            # Apply corrections (vectorized)
                            corrections[violating_indices[valid_mask]] += (
                                violations[valid_mask, None] * normalized[valid_mask]
                            )
            
            # Apply corrections only to violating points
            if np.any(violating_mask):
                current[violating_mask] += corrections[violating_mask]
                
                # Check convergence: stop if corrections are small
                correction_norms = np.linalg.norm(corrections[violating_mask], axis=1)
                if len(correction_norms) > 0:
                    max_correction = np.max(correction_norms)
                    if max_correction < self.convergence_tol:
                        break
        
        return current
    
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

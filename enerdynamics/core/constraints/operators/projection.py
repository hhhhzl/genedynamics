"""
Euclidean projection operator (Projected Diffusion / PDM).

This operator projects trajectories onto the feasible set using Euclidean projection.
Useful for Projected Diffusion Models (PDM) and other projection-based methods.
"""

import time
from typing import Tuple, Optional, Dict, Any
import numpy as np

from enerdynamics.core.constraints.operators.base import Operator
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    OperatorInfo,
)
from enerdynamics.core.constraints.core.registry import get_registry, register
from enerdynamics.core.types import Trajectory


@register("operator", "projection", "numpy")
class ProjectionOperator(Operator):
    """
    Euclidean projection operator.
    
    Projects trajectories onto the feasible set defined by convex constraints.
    This is the core operator for Projected Diffusion Models (PDM).
    
    The projection is computed as:
        min ||τ - τ_nom||^2
        s.t. A τ >= b
    
    where τ is the trajectory (states and/or actions).
    """
    
    def __init__(
        self,
        project_states: bool = True,
        project_actions: bool = True,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        **kwargs
    ):
        """
        Initialize projection operator.
        
        Args:
            project_states: Whether to project states
            project_actions: Whether to project actions
            max_iterations: Maximum iterations for iterative projection
            tolerance: Convergence tolerance
            **kwargs: Additional arguments
        """
        self.project_states = project_states
        self.project_actions = project_actions
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.kwargs = kwargs
    
    def apply(
        self,
        nominal: Trajectory,
        constraints: ConvexConstraint,
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[Trajectory, OperatorInfo]:
        """
        Project trajectory onto feasible set.
        
        Args:
            nominal: Nominal trajectory to project
            constraints: Convex constraints (A, b)
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            Tuple of (projected trajectory, operator info)
        """
        start_time = time.time()
        
        # Get constraints
        A = np.asarray(constraints.A, dtype=np.float32)
        b = np.asarray(constraints.b, dtype=np.float32)
        
        if A.size == 0:
            # No constraints - return nominal
            info = OperatorInfo(
                success=True,
                violation_before=0.0,
                violation_after=0.0,
                iterations=0,
                time=time.time() - start_time
            )
            return nominal, info
        
        # Project trajectory
        if self.project_states and self.project_actions:
            # Project both states and actions
            projected = self._project_full_trajectory(nominal, A, b)
        elif self.project_states:
            # Project only states
            projected = self._project_states_only(nominal, A, b)
        elif self.project_actions:
            # Project only actions
            projected = self._project_actions_only(nominal, A, b)
        else:
            # Nothing to project
            projected = nominal
        
        # Compute violations
        violation_before = self._compute_violation(nominal, A, b)
        violation_after = self._compute_violation(projected, A, b)
        
        elapsed_time = time.time() - start_time
        
        info = OperatorInfo(
            success=True,
            violation_before=violation_before,
            violation_after=violation_after,
            iterations=self.max_iterations,
            time=elapsed_time,
            extra={
                "method": "euclidean_projection",
                "project_states": self.project_states,
                "project_actions": self.project_actions,
            }
        )
        
        return projected, info
    
    def _project_full_trajectory(
        self,
        nominal: Trajectory,
        A: np.ndarray,
        b: np.ndarray
    ) -> Trajectory:
        """
        Project full trajectory (states + actions) onto feasible set.
        
        Args:
            nominal: Nominal trajectory
            A: Constraint matrix
            b: Constraint RHS
            
        Returns:
            Projected trajectory
        """
        # Flatten trajectory: [s0, s1, ..., sH, a0, a1, ..., aH-1]
        H = len(nominal.states)
        state_dim = len(nominal.states[0])
        action_dim = len(nominal.actions[0]) if nominal.actions else 0
        
        states_flat = np.stack([np.asarray(s) for s in nominal.states]).flatten()  # (H*state_dim,)
        actions_flat = np.stack([np.asarray(a) for a in nominal.actions]).flatten() if nominal.actions else np.array([])  # (H-1)*action_dim,)
        
        traj_flat = np.concatenate([states_flat, actions_flat])  # (n,)
        
        # Project onto feasible set: min ||x - x_nom||^2 s.t. A x >= b
        # Use iterative projection for general constraints
        x_proj = self._iterative_projection(traj_flat, A, b)
        
        # Unflatten
        states_proj = x_proj[:H*state_dim].reshape(H, state_dim)
        actions_proj = x_proj[H*state_dim:].reshape(H-1, action_dim) if nominal.actions else []
        
        return Trajectory(
            states=[states_proj[t] for t in range(H)],
            actions=[actions_proj[t] for t in range(H-1)] if nominal.actions else [],
            info=nominal.info
        )
    
    def _project_states_only(
        self,
        nominal: Trajectory,
        A: np.ndarray,
        b: np.ndarray
    ) -> Trajectory:
        """Project only states."""
        H = len(nominal.states)
        state_dim = len(nominal.states[0])
        
        states_flat = np.stack([np.asarray(s) for s in nominal.states]).flatten()
        states_proj_flat = self._iterative_projection(states_flat, A, b)
        states_proj = states_proj_flat.reshape(H, state_dim)
        
        return Trajectory(
            states=[states_proj[t] for t in range(H)],
            actions=nominal.actions,
            info=nominal.info
        )
    
    def _project_actions_only(
        self,
        nominal: Trajectory,
        A: np.ndarray,
        b: np.ndarray
    ) -> Trajectory:
        """Project only actions."""
        H = len(nominal.actions)
        action_dim = len(nominal.actions[0])
        
        actions_flat = np.stack([np.asarray(a) for a in nominal.actions]).flatten()
        actions_proj_flat = self._iterative_projection(actions_flat, A, b)
        actions_proj = actions_proj_flat.reshape(H, action_dim)
        
        return Trajectory(
            states=nominal.states,
            actions=[actions_proj[t] for t in range(H)],
            info=nominal.info
        )
    
    def _iterative_projection(
        self,
        x_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """
        Iteratively project onto feasible set.
        
        Uses Dykstra's projection algorithm or simple iterative projection.
        
        Args:
            x_nom: Nominal point
            A: Constraint matrix (m, n)
            b: Constraint RHS (m,)
            
        Returns:
            Projected point
        """
        x = x_nom.copy()
        
        for iteration in range(self.max_iterations):
            # Check violations
            violations = np.maximum(0, b - A @ x)
            
            if violations.max() < self.tolerance:
                break
            
            # Project onto violated constraints
            for i in range(len(b)):
                if violations[i] > 0:
                    # Project onto halfspace: a_i^T x >= b_i
                    a_i = A[i]
                    b_i = b[i]
                    
                    # Check if already feasible
                    if a_i @ x >= b_i:
                        continue
                    
                    # Project onto halfspace
                    a_norm_sq = np.dot(a_i, a_i)
                    if a_norm_sq < 1e-10:
                        continue
                    
                    # Projection: x = x + lambda * a_i, where lambda = (b_i - a_i^T x) / ||a_i||^2
                    lambda_val = (b_i - a_i @ x) / a_norm_sq
                    x = x + lambda_val * a_i
        
        return x
    
    def _compute_violation(
        self,
        trajectory: Trajectory,
        A: np.ndarray,
        b: np.ndarray
    ) -> float:
        """
        Compute constraint violation for trajectory.
        
        Args:
            trajectory: Trajectory to evaluate
            A: Constraint matrix
            b: Constraint RHS
            
        Returns:
            Maximum violation
        """
        if A.size == 0:
            return 0.0
        
        # Flatten trajectory
        H = len(trajectory.states)
        state_dim = len(trajectory.states[0])
        action_dim = len(trajectory.actions[0]) if trajectory.actions else 0
        
        states_flat = np.stack([np.asarray(s) for s in trajectory.states]).flatten()
        actions_flat = np.stack([np.asarray(a) for a in trajectory.actions]).flatten() if trajectory.actions else np.array([])
        traj_flat = np.concatenate([states_flat, actions_flat])
        
        # Compute violations
        violations = np.maximum(0, b - A @ traj_flat)
        return float(violations.max())



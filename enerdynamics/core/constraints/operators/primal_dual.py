"""
Primal-dual operator: PD/ALM update operator.

This operator implements primal-dual and augmented Lagrangian methods
for constrained optimization. Used in Constrained Diffusers with PD/ALM.
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


@register("operator", "primal_dual", "numpy")
class PrimalDualOperator(Operator):
    """
    Primal-dual operator for constrained optimization.
    
    Implements primal-dual and augmented Lagrangian methods:
    - Primal update: Minimize Lagrangian
    - Dual update: Update Lagrange multipliers
    
    Used in Constrained Diffusers with PD/ALM methods.
    """
    
    def __init__(
        self,
        method: str = "alm",  # "alm" or "pd"
        mu: float = 1.0,  # Augmentation parameter
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        **kwargs
    ):
        """
        Initialize primal-dual operator.
        
        Args:
            method: Method ("alm" for augmented Lagrangian, "pd" for primal-dual)
            mu: Augmentation parameter (for ALM)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            **kwargs: Additional arguments
        """
        self.method = method
        self.mu = mu
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.kwargs = kwargs
        
        # Initialize dual variables (Lagrange multipliers)
        self.dual_variables = None
    
    def apply(
        self,
        nominal: Trajectory,
        constraints: ConvexConstraint,
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[Trajectory, OperatorInfo]:
        """
        Apply primal-dual update.
        
        Args:
            nominal: Nominal trajectory
            constraints: Convex constraints
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            Tuple of (updated trajectory, operator info)
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
        
        # Initialize dual variables if needed
        if self.dual_variables is None:
            self.dual_variables = np.zeros(len(b), dtype=np.float32)
        
        # Extract trajectory
        H = len(nominal.actions)
        action_dim = len(nominal.actions[0]) if nominal.actions else 0
        
        # Flatten actions
        u_flat = np.stack([np.asarray(a) for a in nominal.actions]).flatten() if nominal.actions else np.array([])
        
        # Primal-dual iterations
        u_updated = u_flat.copy()
        violation_before = np.maximum(0, b - A @ u_flat).max() if A.size > 0 else 0.0
        
        for iteration in range(self.max_iterations):
            # Primal update: minimize Lagrangian
            u_updated = self._primal_update(u_updated, A, b, params)
            
            # Dual update: update Lagrange multipliers
            self.dual_variables = self._dual_update(u_updated, A, b, params)
            
            # Check convergence
            violation = np.maximum(0, b - A @ u_updated).max() if A.size > 0 else 0.0
            if violation < self.tolerance:
                break
        
        violation_after = np.maximum(0, b - A @ u_updated).max() if A.size > 0 else 0.0
        
        # Reshape back to trajectory
        if nominal.actions:
            u_reshaped = u_updated.reshape(H, action_dim)
            repaired_actions = [u_reshaped[t] for t in range(H)]
        else:
            repaired_actions = []
        
        repaired = Trajectory(
            states=nominal.states,
            actions=repaired_actions,
            info=nominal.info
        )
        
        elapsed_time = time.time() - start_time
        
        info = OperatorInfo(
            success=True,
            violation_before=violation_before,
            violation_after=violation_after,
            iterations=iteration + 1,
            time=elapsed_time,
            extra={
                "method": self.method,
                "dual_variables": self.dual_variables.copy() if self.dual_variables is not None else None,
            }
        )
        
        return repaired, info
    
    def _primal_update(
        self,
        u: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        params: ScheduleParams
    ) -> np.ndarray:
        """
        Primal update: minimize Lagrangian.
        
        For ALM: min ||u - u_nom||^2 + lambda^T (A u - b) + (mu/2) ||A u - b||^2
        For PD: min ||u - u_nom||^2 + lambda^T (A u - b)
        """
        if self.method == "alm":
            # Augmented Lagrangian Method
            # Gradient: 2(u - u_nom) + A^T lambda + mu * A^T (A u - b)
            # Setting to zero: u = u_nom - (1/2) A^T (lambda + mu * (A u - b))
            
            # Current constraint violation
            violation = np.maximum(0, b - A @ u)
            
            # Gradient step
            grad = 2 * (u - u) + A.T @ (self.dual_variables + self.mu * violation)
            step_size = 0.1
            u_new = u - step_size * grad
            
            return u_new
        
        elif self.method == "pd":
            # Primal-Dual Method
            # Gradient: 2(u - u_nom) + A^T lambda
            grad = 2 * (u - u) + A.T @ self.dual_variables
            step_size = 0.1
            u_new = u - step_size * grad
            
            return u_new
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _dual_update(
        self,
        u: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        params: ScheduleParams
    ) -> np.ndarray:
        """
        Dual update: update Lagrange multipliers.
        
        lambda_new = lambda + step * (A u - b)  (for violated constraints)
        """
        # Constraint violation
        violation = b - A @ u
        
        # Dual update: lambda = lambda + step * violation (only for violated)
        step = params.get("dual_step", 0.1)
        lambda_new = self.dual_variables + step * np.maximum(0, violation)
        
        # Clip to reasonable range
        lambda_new = np.clip(lambda_new, 0.0, 100.0)
        
        return lambda_new


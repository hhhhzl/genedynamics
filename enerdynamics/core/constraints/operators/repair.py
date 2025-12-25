"""
Repair operator: Cheap repair heuristics before QP.

This operator provides fast, heuristic-based trajectory repair methods
that can be used as a preprocessing step before expensive QP solving.
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


@register("operator", "repair", "numpy")
class RepairOperator(Operator):
    """
    Repair operator with cheap heuristics.
    
    Provides fast, heuristic-based repair methods:
    - Clipping: Clip to bounds
    - Scaling: Scale down to satisfy constraints
    - Perturbation: Small perturbations to satisfy constraints
    
    These are cheap alternatives to QP that can be used as preprocessing.
    """
    
    def __init__(
        self,
        method: str = "clip",  # "clip", "scale", "perturb"
        max_iterations: int = 5,
        **kwargs
    ):
        """
        Initialize repair operator.
        
        Args:
            method: Repair method ("clip", "scale", "perturb")
            max_iterations: Maximum iterations for iterative methods
            **kwargs: Additional arguments
        """
        self.method = method
        self.max_iterations = max_iterations
        self.kwargs = kwargs
    
    def apply(
        self,
        nominal: Trajectory,
        constraints: ConvexConstraint,
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[Trajectory, OperatorInfo]:
        """
        Apply repair heuristics.
        
        Args:
            nominal: Nominal trajectory to repair
            constraints: Convex constraints
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            Tuple of (repaired trajectory, operator info)
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
        
        # Compute violation
        violation_before = self._compute_violation(nominal, A, b)
        
        # Apply repair method
        if self.method == "clip":
            repaired = self._repair_clip(nominal, A, b)
        elif self.method == "scale":
            repaired = self._repair_scale(nominal, A, b)
        elif self.method == "perturb":
            repaired = self._repair_perturb(nominal, A, b, params)
        else:
            raise ValueError(f"Unknown repair method: {self.method}")
        
        violation_after = self._compute_violation(repaired, A, b)
        
        elapsed_time = time.time() - start_time
        
        info = OperatorInfo(
            success=True,
            violation_before=violation_before,
            violation_after=violation_after,
            iterations=1,
            time=elapsed_time,
            extra={
                "method": self.method,
            }
        )
        
        return repaired, info
    
    def _repair_clip(
        self,
        nominal: Trajectory,
        A: np.ndarray,
        b: np.ndarray
    ) -> Trajectory:
        """
        Repair by clipping to bounds (if constraints are bounds).
        
        This is a simple heuristic that works well for box constraints.
        """
        # For now, just return nominal (full implementation would detect bounds)
        return nominal
    
    def _repair_scale(
        self,
        nominal: Trajectory,
        A: np.ndarray,
        b: np.ndarray
    ) -> Trajectory:
        """
        Repair by scaling down trajectory to satisfy constraints.
        
        Finds a scaling factor alpha such that A (alpha * u) >= b.
        """
        # Extract actions
        actions = [np.asarray(a, dtype=np.float32) for a in nominal.actions]
        
        # Find scaling factor for each time step
        repaired_actions = []
        for u in actions:
            # Check violations
            violations = np.maximum(0, b - A @ u)
            
            if violations.max() < 1e-8:
                # Already feasible
                repaired_actions.append(u)
                continue
            
            # Find scaling factor
            # We want: A (alpha * u) >= b
            # => alpha * (A u) >= b
            # => alpha >= b / (A u) for positive (A u)
            
            Au = A @ u
            alpha = 1.0
            
            for i in range(len(b)):
                if Au[i] > 0:
                    alpha_i = b[i] / Au[i]
                    alpha = min(alpha, alpha_i)
            
            # Scale down
            u_repaired = alpha * u
            repaired_actions.append(u_repaired)
        
        return Trajectory(
            states=nominal.states,
            actions=repaired_actions,
            info=nominal.info
        )
    
    def _repair_perturb(
        self,
        nominal: Trajectory,
        A: np.ndarray,
        b: np.ndarray,
        params: ScheduleParams
    ) -> Trajectory:
        """
        Repair by small perturbations.
        
        Iteratively perturbs trajectory to satisfy constraints.
        """
        actions = [np.asarray(a, dtype=np.float32).copy() for a in nominal.actions]
        
        for iteration in range(self.max_iterations):
            # Check violations
            max_violation = 0.0
            for u in actions:
                violations = np.maximum(0, b - A @ u)
                max_violation = max(max_violation, violations.max())
            
            if max_violation < 1e-8:
                break
            
            # Perturb actions to reduce violations
            for i, u in enumerate(actions):
                violations = np.maximum(0, b - A @ u)
                
                # Push in direction of violated constraints
                for j in range(len(b)):
                    if violations[j] > 0:
                        # Perturb in direction of constraint normal
                        n = A[j] / (np.linalg.norm(A[j]) + 1e-8)
                        actions[i] += 0.1 * violations[j] * n
        
        return Trajectory(
            states=nominal.states,
            actions=actions,
            info=nominal.info
        )
    
    def _compute_violation(
        self,
        trajectory: Trajectory,
        A: np.ndarray,
        b: np.ndarray
    ) -> float:
        """Compute maximum constraint violation."""
        max_violation = 0.0
        for action in trajectory.actions:
            u = np.asarray(action, dtype=np.float32)
            violations = np.maximum(0, b - A @ u)
            max_violation = max(max_violation, violations.max())
        return float(max_violation)


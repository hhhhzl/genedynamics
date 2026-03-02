"""
Reweight operator: Modify importance weights based on constraints.

This operator modifies importance weights (for importance sampling) based on
constraint violations. Used in EB-MBD (Energy-Based Model-Based Diffusion)
and JM2D (Joint Model-Based Diffusion) methods.
"""

import time
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

from genedynamics.core.constraints.operators.base import Operator
from genedynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    OperatorInfo,
)
from genedynamics.core.constraints.core.registry import get_registry, register
from genedynamics.core.types import Trajectory


@register("operator", "reweight", "numpy")
class ReweightOperator(Operator):
    """
    Reweight operator for importance sampling.
    
    Modifies importance weights based on constraint violations:
        w_new = w_old * exp(-beta * violation)
    
    This is used in:
    - EB-MBD: Energy-based reweighting
    - JM2D: Joint model-based reweighting
    - Barrier methods: Barrier-based reweighting
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        reweight_method: str = "exponential",
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize reweight operator.
        
        Args:
            beta: Reweighting strength (higher = stronger penalty)
            reweight_method: Method ("exponential", "linear", "barrier")
            normalize: Whether to normalize weights after reweighting
            **kwargs: Additional arguments
        """
        self.beta = beta
        self.reweight_method = reweight_method
        self.normalize = normalize
        self.kwargs = kwargs
    
    def apply(
        self,
        nominal: Trajectory,
        constraints: ConvexConstraint,
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[Trajectory, OperatorInfo]:
        """
        Apply reweighting to trajectory.
        
        Note: This operator doesn't modify the trajectory itself,
        but modifies weights stored in trajectory.info.
        
        Args:
            nominal: Nominal trajectory (with weights in info)
            constraints: Convex constraints
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            Tuple of (trajectory with updated weights, operator info)
        """
        start_time = time.time()
        
        # Get current weight (default: 1.0)
        current_weight = nominal.info.get("weight", 1.0) if nominal.info else 1.0
        
        # Compute constraint violation
        violation = self._compute_violation(nominal, constraints)
        
        # Compute new weight
        beta = params.get("beta", self.beta)
        new_weight = self._compute_new_weight(current_weight, violation, beta)
        
        # Update trajectory info
        new_info = nominal.info.copy() if nominal.info else {}
        new_info["weight"] = new_weight
        new_info["violation"] = violation
        new_info["weight_ratio"] = new_weight / (current_weight + 1e-10)
        
        repaired = Trajectory(
            states=nominal.states,
            actions=nominal.actions,
            info=new_info
        )
        
        elapsed_time = time.time() - start_time
        
        info = OperatorInfo(
            success=True,
            violation_before=violation,
            violation_after=violation,  # Reweighting doesn't change violation
            iterations=1,
            time=elapsed_time,
            extra={
                "method": "reweight",
                "reweight_method": self.reweight_method,
                "old_weight": current_weight,
                "new_weight": new_weight,
                "beta": beta,
            }
        )
        
        return repaired, info
    
    def apply_batch(
        self,
        nominals: List[Trajectory],
        constraints: List[ConvexConstraint],
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """
        Apply reweighting to batch of trajectories.
        
        This also normalizes weights across the batch if normalize=True.
        
        Args:
            nominals: List of nominal trajectories
            constraints: List of constraints (one per trajectory)
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            Tuple of (list of reweighted trajectories, merged info)
        """
        # Apply reweighting to each trajectory
        results = [self.apply(n, c, params, state) for n, c in zip(nominals, constraints)]
        repaired = [r[0] for r in results]
        
        # Normalize weights across batch if requested
        if self.normalize:
            weights = [traj.info.get("weight", 1.0) for traj in repaired if traj.info]
            if weights:
                total_weight = sum(weights)
                if total_weight > 1e-10:
                    for traj in repaired:
                        if traj.info:
                            traj.info["weight"] = traj.info.get("weight", 1.0) / total_weight
        
        # Merge info
        info = self._merge_info([r[1] for r in results])
        
        return repaired, info
    
    def _compute_violation(
        self,
        trajectory: Trajectory,
        constraints: ConvexConstraint
    ) -> float:
        """
        Compute constraint violation for trajectory.
        
        Args:
            trajectory: Trajectory to evaluate
            constraints: Convex constraints
            
        Returns:
            Maximum violation
        """
        A = np.asarray(constraints.A, dtype=np.float32)
        b = np.asarray(constraints.b, dtype=np.float32)
        
        if A.size == 0:
            return 0.0
        
        # Extract trajectory states/actions
        # For simplicity, evaluate constraints on states
        violations = []
        for state in trajectory.states:
            state_arr = np.asarray(state, dtype=np.float32)
            # Check if constraint applies to this state
            if A.shape[1] == len(state_arr):
                violation = np.maximum(0, b - A @ state_arr).max()
                violations.append(violation)
        
        return float(max(violations) if violations else 0.0)
    
    def _compute_new_weight(
        self,
        old_weight: float,
        violation: float,
        beta: float
    ) -> float:
        """
        Compute new weight based on violation.
        
        Args:
            old_weight: Current weight
            violation: Constraint violation
            beta: Reweighting strength
            
        Returns:
            New weight
        """
        if self.reweight_method == "exponential":
            # Exponential: w_new = w_old * exp(-beta * violation)
            return old_weight * np.exp(-beta * violation)
        
        elif self.reweight_method == "linear":
            # Linear: w_new = w_old * (1 - beta * violation)
            return old_weight * max(0.0, 1.0 - beta * violation)
        
        elif self.reweight_method == "barrier":
            # Barrier: w_new = w_old * exp(-beta * barrier(violation))
            # barrier(v) = -log(max(epsilon, 1 - v))
            epsilon = 1e-6
            barrier = -np.log(max(epsilon, 1.0 - violation))
            return old_weight * np.exp(-beta * barrier)
        
        else:
            raise ValueError(f"Unknown reweight method: {self.reweight_method}")



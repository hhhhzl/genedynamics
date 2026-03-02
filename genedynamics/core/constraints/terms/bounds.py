"""
State and control bounds constraint terms.

This module provides bounds constraints for states and controls.
"""

from typing import Optional, Union
import numpy as np

from .base import ConstraintTerm
from genedynamics.core.types import Trajectory, State, Action


class BoundsTerm(ConstraintTerm):
    """
    Bounds constraint term for states and/or controls.
    
    Defines box constraints:
    - State bounds: x_min <= x <= x_max
    - Control bounds: u_min <= u <= u_max
    
    Energy: Quadratic penalty for violations
    Feasible: All states/controls within bounds
    Violation: Distance outside bounds
    """
    
    def __init__(
        self,
        state_lower: Optional[np.ndarray] = None,
        state_upper: Optional[np.ndarray] = None,
        control_lower: Optional[np.ndarray] = None,
        control_upper: Optional[np.ndarray] = None,
        alpha: float = 10.0,
    ):
        """
        Initialize bounds constraint term.
        
        Args:
            state_lower: Lower bounds for states (None = no constraint)
            state_upper: Upper bounds for states (None = no constraint)
            control_lower: Lower bounds for controls (None = no constraint)
            control_upper: Upper bounds for controls (None = no constraint)
            alpha: Penalty weight for energy computation
        """
        self.state_lower = np.asarray(state_lower, dtype=np.float32) if state_lower is not None else None
        self.state_upper = np.asarray(state_upper, dtype=np.float32) if state_upper is not None else None
        self.control_lower = np.asarray(control_lower, dtype=np.float32) if control_lower is not None else None
        self.control_upper = np.asarray(control_upper, dtype=np.float32) if control_upper is not None else None
        self.alpha = alpha
    
    def energy(
        self,
        trajectory: Trajectory,
        **kwargs
    ) -> float:
        """
        Compute quadratic penalty for bounds violations.
        
        Args:
            trajectory: Trajectory to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Total penalty energy
        """
        total = 0.0
        
        # State bounds
        if self.state_lower is not None or self.state_upper is not None:
            for state in trajectory.states:
                state_arr = np.asarray(state, dtype=np.float32)
                
                if self.state_lower is not None:
                    below = np.maximum(0.0, self.state_lower - state_arr)
                    total += self.alpha * np.sum(below ** 2)
                
                if self.state_upper is not None:
                    above = np.maximum(0.0, state_arr - self.state_upper)
                    total += self.alpha * np.sum(above ** 2)
        
        # Control bounds
        if self.control_lower is not None or self.control_upper is not None:
            for action in trajectory.actions:
                action_arr = np.asarray(action, dtype=np.float32)
                
                if self.control_lower is not None:
                    below = np.maximum(0.0, self.control_lower - action_arr)
                    total += self.alpha * np.sum(below ** 2)
                
                if self.control_upper is not None:
                    above = np.maximum(0.0, action_arr - self.control_upper)
                    total += self.alpha * np.sum(above ** 2)
        
        return float(total)
    
    def feasible(
        self,
        trajectory: Trajectory,
        **kwargs
    ) -> bool:
        """
        Check if trajectory satisfies all bounds.
        
        Args:
            trajectory: Trajectory to check
            **kwargs: Additional parameters
            
        Returns:
            True if all states/controls within bounds
        """
        # Check states
        if self.state_lower is not None or self.state_upper is not None:
            for state in trajectory.states:
                state_arr = np.asarray(state, dtype=np.float32)
                
                if self.state_lower is not None:
                    if np.any(state_arr < self.state_lower):
                        return False
                
                if self.state_upper is not None:
                    if np.any(state_arr > self.state_upper):
                        return False
        
        # Check controls
        if self.control_lower is not None or self.control_upper is not None:
            for action in trajectory.actions:
                action_arr = np.asarray(action, dtype=np.float32)
                
                if self.control_lower is not None:
                    if np.any(action_arr < self.control_lower):
                        return False
                
                if self.control_upper is not None:
                    if np.any(action_arr > self.control_upper):
                        return False
        
        return True
    
    def violation(
        self,
        trajectory: Trajectory,
        **kwargs
    ) -> np.ndarray:
        """
        Compute bounds violations.
        
        Returns violations per state/action.
        
        Args:
            trajectory: Trajectory to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Array of violations, shape (H+1 + H,) for states + actions
        """
        violations = []
        
        # State violations
        for state in trajectory.states:
            state_arr = np.asarray(state, dtype=np.float32)
            violation = 0.0
            
            if self.state_lower is not None:
                below = np.maximum(0.0, self.state_lower - state_arr)
                violation += np.sum(below)
            
            if self.state_upper is not None:
                above = np.maximum(0.0, state_arr - self.state_upper)
                violation += np.sum(above)
            
            violations.append(violation)
        
        # Control violations
        for action in trajectory.actions:
            action_arr = np.asarray(action, dtype=np.float32)
            violation = 0.0
            
            if self.control_lower is not None:
                below = np.maximum(0.0, self.control_lower - action_arr)
                violation += np.sum(below)
            
            if self.control_upper is not None:
                above = np.maximum(0.0, action_arr - self.control_upper)
                violation += np.sum(above)
            
            violations.append(violation)
        
        return np.array(violations, dtype=np.float32)



"""
Composite constraint terms: combine multiple terms.

This module provides utilities to combine multiple constraint terms:
- Sum energy: E_total = Σ_i E_i
- AND feasibility: feasible_total = AND_i feasible_i
- Aggregate violations: violation_total = max/mean/sum violations
"""

from typing import List, Optional, Union
import numpy as np

from .base import ConstraintTerm
from enerdynamics.core.types import Trajectory


class CompositeTerm(ConstraintTerm):
    """
    Composite constraint term that combines multiple terms.
    
    Combines multiple constraint terms:
    - Energy: Sum of all term energies
    - Feasibility: AND of all term feasibilities (all must be feasible)
    - Violation: Aggregate of all term violations
    """
    
    def __init__(
        self,
        terms: List[ConstraintTerm],
        violation_aggregation: str = "max",
    ):
        """
        Initialize composite constraint term.
        
        Args:
            terms: List of constraint terms to combine
            violation_aggregation: How to aggregate violations ("max", "mean", "sum")
        """
        self.terms = terms
        self.violation_aggregation = violation_aggregation
    
    def energy(
        self,
        trajectory: Trajectory,
        **kwargs
    ) -> float:
        """
        Compute total energy: sum of all term energies.
        
        Args:
            trajectory: Trajectory to evaluate
            **kwargs: Additional parameters (passed to all terms)
            
        Returns:
            Total energy (sum of all term energies)
        """
        total = 0.0
        for term in self.terms:
            total += term.energy(trajectory, **kwargs)
        return float(total)
    
    def feasible(
        self,
        trajectory: Trajectory,
        **kwargs
    ) -> bool:
        """
        Check feasibility: all terms must be feasible (AND).
        
        Args:
            trajectory: Trajectory to check
            **kwargs: Additional parameters (passed to all terms)
            
        Returns:
            True if all terms are feasible
        """
        for term in self.terms:
            if not term.feasible(trajectory, **kwargs):
                return False
        return True
    
    def violation(
        self,
        trajectory: Trajectory,
        **kwargs
    ) -> np.ndarray:
        """
        Compute aggregated violations.
        
        Args:
            trajectory: Trajectory to evaluate
            **kwargs: Additional parameters (passed to all terms)
            
        Returns:
            Aggregated violation values
        """
        violations_list = []
        
        for term in self.terms:
            violation = term.violation(trajectory, **kwargs)
            violations_list.append(violation)
        
        # Stack violations
        violations_array = np.stack(violations_list)  # (num_terms, ...)
        
        # Aggregate
        if self.violation_aggregation == "max":
            return np.max(violations_array, axis=0)
        elif self.violation_aggregation == "mean":
            return np.mean(violations_array, axis=0)
        elif self.violation_aggregation == "sum":
            return np.sum(violations_array, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.violation_aggregation}")
    
    def energy_batch(
        self,
        trajectories: List[Trajectory],
        **kwargs
    ) -> np.ndarray:
        """
        Batch compute energy for multiple trajectories.
        
        Args:
            trajectories: List of trajectories
            **kwargs: Additional parameters
            
        Returns:
            Array of total energies, shape (len(trajectories),)
        """
        total_energies = np.zeros(len(trajectories), dtype=np.float32)
        
        for term in self.terms:
            term_energies = term.energy_batch(trajectories, **kwargs)
            total_energies += term_energies
        
        return total_energies
    
    def feasible_batch(
        self,
        trajectories: List[Trajectory],
        **kwargs
    ) -> np.ndarray:
        """
        Batch check feasibility for multiple trajectories.
        
        Args:
            trajectories: List of trajectories
            **kwargs: Additional parameters
            
        Returns:
            Array of feasibility flags, shape (len(trajectories),)
        """
        # Start with all True
        feasible_flags = np.ones(len(trajectories), dtype=bool)
        
        for term in self.terms:
            term_feasible = term.feasible_batch(trajectories, **kwargs)
            feasible_flags = feasible_flags & term_feasible
        
        return feasible_flags


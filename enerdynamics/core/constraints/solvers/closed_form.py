"""
Closed-form QP solvers for special cases.

This module provides fast closed-form solutions for special QP cases:
- Single halfspace projection
- Box constraint clipping
- Unconstrained QP
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

from .base import QPSolver


class ClosedFormSolver(QPSolver):
    """
    Closed-form solver for special QP cases.
    
    Provides fast solutions for:
    - Single halfspace: min ||u - u_nom||^2 s.t. a^T u >= b
    - Box constraints: min ||u - u_nom||^2 s.t. lb <= u <= ub
    - Unconstrained: min ||u - u_nom||^2
    """
    
    def solve_qp(
        self,
        P: np.ndarray,
        q: np.ndarray,
        G: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve QP using closed-form methods when possible.
        
        Falls back to iterative methods for complex cases.
        """
        n = len(q)
        
        # Case 1: Unconstrained QP
        if (G is None or G.size == 0) and (A_eq is None or A_eq.size == 0) and \
           (lb is None) and (ub is None):
            # min (1/2) x^T P x + q^T x
            # Solution: x = -P^{-1} q
            try:
                P_inv = np.linalg.inv(P)
                solution = -P_inv @ q
                return solution, {'status': 'optimal', 'iterations': 0, 'method': 'unconstrained'}
            except:
                pass
        
        # Case 2: Single halfspace constraint
        if G is not None and G.shape[0] == 1 and (A_eq is None or A_eq.size == 0) and \
           (lb is None) and (ub is None):
            # min ||u - u_nom||^2 s.t. a^T u >= b
            # where u_nom = -P^{-1} q (unconstrained solution)
            try:
                P_inv = np.linalg.inv(P)
                u_nom = -P_inv @ q
                a = G[0]
                b_val = h[0]
                
                # Check if already feasible
                if a @ u_nom >= b_val:
                    return u_nom, {'status': 'optimal', 'iterations': 0, 'method': 'halfspace_feasible'}
                
                # Project onto halfspace
                # Projection: u = u_nom + lambda * a, where lambda = (b - a^T u_nom) / ||a||^2
                a_norm_sq = np.dot(a, a)
                if a_norm_sq < 1e-10:
                    return u_nom, {'status': 'optimal', 'iterations': 0, 'method': 'halfspace_degenerate'}
                
                lambda_val = (b_val - a @ u_nom) / a_norm_sq
                solution = u_nom + lambda_val * a
                
                return solution, {'status': 'optimal', 'iterations': 0, 'method': 'halfspace'}
            except:
                pass
        
        # Case 3: Box constraints only
        if (G is None or G.size == 0) and (A_eq is None or A_eq.size == 0) and \
           (lb is not None or ub is not None):
            # min ||u - u_nom||^2 s.t. lb <= u <= ub
            try:
                P_inv = np.linalg.inv(P)
                u_nom = -P_inv @ q
                
                if lb is None:
                    lb = np.full(n, -np.inf)
                if ub is None:
                    ub = np.full(n, np.inf)
                
                # Clip to box
                solution = np.clip(u_nom, lb, ub)
                
                return solution, {'status': 'optimal', 'iterations': 0, 'method': 'box'}
            except:
                pass
        
        # Fallback: cannot solve in closed form
        raise ValueError(
            "ClosedFormSolver cannot solve this QP. "
            "Use QPAXSolver or OSQPSolver for general QP problems."
        )
    
    def solve_single_halfspace(
        self,
        u_nom: np.ndarray,
        a: np.ndarray,
        b: float
    ) -> np.ndarray:
        """
        Fast closed-form solution for single halfspace projection.
        
        min ||u - u_nom||^2 s.t. a^T u >= b
        
        Args:
            u_nom: Nominal solution
            a: Constraint normal vector
            b: Constraint RHS
            
        Returns:
            Projected solution
        """
        # Check feasibility
        if a @ u_nom >= b:
            return u_nom
        
        # Project onto halfspace
        a_norm_sq = np.dot(a, a)
        if a_norm_sq < 1e-10:
            return u_nom
        
        lambda_val = (b - a @ u_nom) / a_norm_sq
        return u_nom + lambda_val * a
    
    def solve_box(
        self,
        u_nom: np.ndarray,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fast closed-form solution for box constraint projection.
        
        min ||u - u_nom||^2 s.t. lb <= u <= ub
        
        Args:
            u_nom: Nominal solution
            lb: Lower bounds
            ub: Upper bounds
            
        Returns:
            Projected solution
        """
        if lb is None and ub is None:
            return u_nom
        
        n = len(u_nom)
        if lb is None:
            lb = np.full(n, -np.inf)
        if ub is None:
            ub = np.full(n, np.inf)
        
        return np.clip(u_nom, lb, ub)


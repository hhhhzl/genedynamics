"""
Base class for QP solvers.

A QP solver only solves QP problems, without defining or interpreting constraints.
Solvers are backend-agnostic and can be used by any operator.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np


class QPSolver(ABC):
    """
    Base class for QP solvers.
    
    A QP solver solves the standard QP problem:
        min (1/2) x^T P x + q^T x
        s.t. G x <= h
             A_eq x = b_eq
             lb <= x <= ub
    
    Solvers are backend-agnostic and only handle numerical solving,
    not constraint interpretation.
    """
    
    @abstractmethod
    def solve_qp(
        self,
        P: np.ndarray,  # (n, n) - quadratic cost matrix
        q: np.ndarray,  # (n,) - linear cost vector
        G: Optional[np.ndarray] = None,  # (m, n) - inequality constraints
        h: Optional[np.ndarray] = None,  # (m,) - inequality RHS
        A_eq: Optional[np.ndarray] = None,  # (p, n) - equality constraints
        b_eq: Optional[np.ndarray] = None,  # (p,) - equality RHS
        lb: Optional[np.ndarray] = None,  # (n,) - lower bounds
        ub: Optional[np.ndarray] = None,  # (n,) - upper bounds
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve QP problem.
        
        Args:
            P: Quadratic cost matrix (n, n)
            q: Linear cost vector (n,)
            G: Inequality constraint matrix (m, n), G x <= h
            h: Inequality constraint RHS (m,)
            A_eq: Equality constraint matrix (p, n), A_eq x = b_eq
            b_eq: Equality constraint RHS (p,)
            lb: Lower bounds (n,)
            ub: Upper bounds (n,)
            **kwargs: Solver-specific options
            
        Returns:
            Tuple of (solution, info)
            - solution: Optimal solution (n,)
            - info: Dictionary with solver information (status, iterations, etc.)
        """
        pass
    
    def solve_least_squares_with_constraints(
        self,
        u_nom: np.ndarray,  # (n,) - nominal solution
        A: np.ndarray,  # (m, n) - constraint matrix, A u >= b
        b: np.ndarray,  # (m,) - constraint RHS
        rho: Optional[float] = None,  # Slack penalty weight
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve least-squares problem with constraints: min ||u - u_nom||^2 s.t. A u >= b.
        
        This is a convenience method for the common case of projecting onto
        a half-space constraint.
        
        Args:
            u_nom: Nominal solution to project
            A: Constraint matrix (A u >= b)
            b: Constraint RHS
            rho: Slack penalty weight (if None, use hard constraints)
            **kwargs: Solver-specific options
            
        Returns:
            Tuple of (solution, info)
        """
        n = len(u_nom)
        
        # QP formulation: min (1/2) u^T I u - u_nom^T u
        # Equivalent to: min ||u - u_nom||^2
        P = np.eye(n, dtype=np.float32)
        q = -u_nom
        
        if rho is not None:
            # Slack-QP: min ||u - u_nom||^2 + rho ||xi||^2 s.t. A u >= b - xi, xi >= 0
            # Reformulate as: min ||u - u_nom||^2 + rho ||xi||^2 s.t. A u + xi >= b, xi >= 0
            m = len(b)
            n_total = n + m
            
            # Extended P matrix
            P_ext = np.zeros((n_total, n_total), dtype=np.float32)
            P_ext[:n, :n] = np.eye(n)
            P_ext[n:, n:] = rho * np.eye(m)
            
            # Extended q vector
            q_ext = np.zeros(n_total, dtype=np.float32)
            q_ext[:n] = -u_nom
            
            # Extended constraints: A u + xi >= b, xi >= 0
            # => -A u - xi <= -b, -xi <= 0
            G_ext = np.zeros((2 * m, n_total), dtype=np.float32)
            h_ext = np.zeros(2 * m, dtype=np.float32)
            
            # -A u - xi <= -b
            G_ext[:m, :n] = -A
            G_ext[:m, n:] = -np.eye(m)
            h_ext[:m] = -b
            
            # -xi <= 0 (i.e., xi >= 0)
            G_ext[m:, n:] = -np.eye(m)
            h_ext[m:] = 0.0
            
            solution_ext, info = self.solve_qp(P_ext, q_ext, G_ext, h_ext, **kwargs)
            solution = solution_ext[:n]
            info['slack'] = solution_ext[n:]
            return solution, info
        else:
            # Hard-QP: min ||u - u_nom||^2 s.t. A u >= b
            # Convert A u >= b to -A u <= -b
            G = -A if A.size > 0 else None
            h = -b if b.size > 0 else None
            
            return self.solve_qp(P, q, G, h, **kwargs)
    
    def solve_traj_qp_with_smoothness(
        self,
        x_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        rho: Optional[float],
        T: int,
        dim: int,
        smoothness_weight: float,
        use_slack: bool = True,
        fix_initial_state: bool = True,
        initial_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve trajectory QP with smoothness term.
        
        min 0.5 ||x - x0||^2 + 0.5 * w * ||D2 x||^2 + ρ||ξ||^2
        s.t. A x >= b - ξ, ξ >= 0
        
        where D2 is the second-difference operator for smoothness.
        
        This method should be implemented by subclasses using backend-native operations.
        
        Args:
            x_nom: Nominal flattened positions, shape (T * dim,)
            A: Constraint matrix, shape (m, T * dim)
            b: Constraint RHS, shape (m,)
            rho: Slack penalty weight (if None, use hard constraints)
            T: Number of time steps
            dim: Position dimension (usually 2 for 2D)
            smoothness_weight: Weight for smoothness regularization
            use_slack: If True, use slack-QP; if False, use hard-QP
            fix_initial_state: If True, add equality constraint to fix initial state
            initial_state: Initial state position, shape (dim,). Required if fix_initial_state=True
            **kwargs: Solver-specific options
            
        Returns:
            Tuple of (optimized positions, info)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement solve_traj_qp_with_smoothness. "
            f"Subclasses should implement this method using backend-native operations."
        )



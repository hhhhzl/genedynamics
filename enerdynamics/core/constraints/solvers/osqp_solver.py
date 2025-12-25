"""
OSQP QP solver backend.

This solver uses OSQP (Operator Splitting Quadratic Program) for CPU-based
QP solving. OSQP is fast and reliable for medium-sized problems.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False
    osqp = None

from .base import QPSolver


class OSQPSolver(QPSolver):
    """
    QP solver using OSQP.
    
    OSQP is a fast, reliable QP solver suitable for CPU-based solving.
    """
    
    def __init__(
        self,
        **kwargs
    ):
        """
        Initialize OSQP solver.
        
        Args:
            **kwargs: OSQP solver options (eps_abs, eps_rel, max_iter, etc.)
        """
        if not OSQP_AVAILABLE:
            raise RuntimeError("OSQP not available. Install osqp to use OSQPSolver.")
        
        self.kwargs = kwargs
    
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
        Solve QP using OSQP.
        
        Args:
            P: Quadratic cost matrix (n, n)
            q: Linear cost vector (n,)
            G: Inequality constraint matrix (m, n)
            h: Inequality constraint RHS (m,)
            A_eq: Equality constraint matrix (p, n)
            b_eq: Equality constraint RHS (p,)
            lb: Lower bounds (n,)
            ub: Upper bounds (n,)
            **kwargs: Additional OSQP options
            
        Returns:
            Tuple of (solution, info)
        """
        n = len(q)
        
        # OSQP expects: min (1/2) x^T P x + q^T x
        #                s.t. l <= A x <= u
        
        # Build constraint matrix A and bounds l, u
        A_list = []
        l_list = []
        u_list = []
        
        # Inequality constraints: G x <= h
        if G is not None and G.size > 0:
            A_list.append(G)
            l_list.append(np.full(len(h), -np.inf))
            u_list.append(h)
        
        # Equality constraints: A_eq x = b_eq
        if A_eq is not None and A_eq.size > 0:
            A_list.append(A_eq)
            l_list.append(b_eq)
            u_list.append(b_eq)
        
        # Bounds: lb <= x <= ub
        if lb is not None or ub is not None:
            if lb is None:
                lb = np.full(n, -np.inf)
            if ub is None:
                ub = np.full(n, np.inf)
            
            # Add identity matrix for bounds
            A_list.append(np.eye(n))
            l_list.append(lb)
            u_list.append(ub)
        
        # Combine constraints
        if A_list:
            A = np.vstack(A_list)
            l = np.concatenate(l_list)
            u = np.concatenate(u_list)
        else:
            # No constraints: unconstrained QP
            A = np.zeros((0, n))
            l = np.zeros(0)
            u = np.zeros(0)
        
        # Ensure P is sparse (OSQP prefers sparse)
        try:
            from scipy import sparse
            P_sparse = sparse.csc_matrix(P)
            A_sparse = sparse.csc_matrix(A)
        except ImportError:
            # Fallback: use dense
            P_sparse = P
            A_sparse = A
        
        # Merge kwargs
        solver_kwargs = {
            'eps_abs': 1e-5,
            'eps_rel': 1e-5,
            'max_iter': 10000,
            **self.kwargs,
            **kwargs
        }
        
        # Create and solve
        prob = osqp.OSQP()
        prob.setup(P_sparse, q, A_sparse, l, u, **solver_kwargs)
        result = prob.solve()
        
        if result.info.status_val == 1:  # OSQP_SOLVED
            solution = result.x
            info = {
                'status': 'optimal',
                'iterations': result.info.iter,
                'solve_time': result.info.solve_time,
            }
        else:
            # Use solution anyway (may be suboptimal)
            solution = result.x if result.x is not None else np.zeros(n)
            info = {
                'status': 'suboptimal',
                'iterations': result.info.iter,
                'solve_time': result.info.solve_time,
                'status_val': result.info.status_val,
            }
        
        return solution, info


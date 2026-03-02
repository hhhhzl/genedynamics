"""
CVXOPT QP solver backend.

This solver uses CVXOPT for CPU-based QP solving.
CVXOPT is a reliable, well-tested QP solver suitable for general QP problems.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import cvxopt
    from cvxopt import matrix as cvx_matrix, solvers as cvx_solvers
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False
    cvxopt = None
    cvx_matrix = None
    cvx_solvers = None

try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    sp = None

from .base import QPSolver
from genedynamics.core.constraints.core.registry import register


@register("solver", "cvxopt", "numpy")
class CVXOPTSolver(QPSolver):
    """
    QP solver using CVXOPT.
    
    CVXOPT is a reliable, well-tested QP solver suitable for CPU-based solving.
    """
    
    def __init__(
        self,
        **kwargs
    ):
        """
        Initialize CVXOPT solver.
        
        Args:
            **kwargs: CVXOPT solver options (abstol, reltol, maxiters, etc.)
        """
        if not CVXOPT_AVAILABLE:
            raise RuntimeError("CVXOPT not available. Install cvxopt to use CVXOPTSolver.")
        
        # CVXOPT solver options
        # Note: CVXOPT uses abstol/reltol instead of tol, and maxiters instead of maxiter
        self.kwargs = {
            'abstol': 1e-6,  # Aligned with JAXOPT tol=1e-6
            'reltol': 1e-6,  # Aligned with JAXOPT tol=1e-6
            'feastol': 1e-6,
            'maxiters': 30,  # Aligned with JAXOPT maxiter=30
            'refinement': 2,
            'show_progress': False,
            **kwargs
        }
    
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
        Solve QP using CVXOPT.
        
        CVXOPT expects: min (1/2) x^T P x + q^T x
                        s.t. G x <= h (inequality)
                             A x = b (equality)
        
        Args:
            P: Quadratic cost matrix (n, n)
            q: Linear cost vector (n,)
            G: Inequality constraint matrix (m, n)
            h: Inequality constraint RHS (m,)
            A_eq: Equality constraint matrix (p, n)
            b_eq: Equality constraint RHS (p,)
            lb: Lower bounds (n,)
            ub: Upper bounds (n,)
            **kwargs: Additional CVXOPT options
            
        Returns:
            Tuple of (solution, info)
        """
        n = len(q)
        
        # Convert to float64 (CVXOPT requires float64)
        # Check if P is sparse first - if so, skip dense conversion
        if sp is not None and hasattr(P, 'tocoo'):
            # P is sparse, keep it sparse (will be converted to CVXOPT format later)
            pass
        else:
            # P is dense, convert to float64
            P = np.asarray(P, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        
        # Build constraint matrices
        # CVXOPT format: min (1/2) x^T P x + q^T x
        #                 s.t. G x <= h
        #                      A x = b
        
        G_list = []
        h_list = []
        
        # Inequality constraints: G x <= h
        if G is not None and G.size > 0:
            # Ensure float64 and contiguous for CVXOPT
            G_conv = np.asarray(G, dtype=np.float64, order='C')
            h_conv = np.asarray(h, dtype=np.float64, order='C')
            G_list.append(G_conv)
            h_list.append(h_conv)
        
        # Bounds: lb <= x <= ub  =>  -I x <= -lb  and  I x <= ub
        if lb is not None or ub is not None:
            I = np.eye(n, dtype=np.float64)
            if lb is not None:
                G_list.append(-I)
                h_list.append(-np.asarray(lb, dtype=np.float64))
            if ub is not None:
                G_list.append(I)
                h_list.append(np.asarray(ub, dtype=np.float64))
        
        # Combine inequality constraints
        if G_list:
            # Ensure float64 and contiguous for CVXOPT
            G_combined = np.ascontiguousarray(np.vstack(G_list), dtype=np.float64)
            h_combined = np.ascontiguousarray(np.concatenate(h_list), dtype=np.float64)
        else:
            G_combined = None
            h_combined = None
        
        # Equality constraints
        A_eq_cvx = None
        b_eq_cvx = None
        if A_eq is not None and A_eq.size > 0:
            # Ensure float64 and contiguous for CVXOPT
            A_eq_cvx = np.ascontiguousarray(A_eq, dtype=np.float64)
            b_eq_cvx = np.ascontiguousarray(b_eq, dtype=np.float64).flatten()
            # Verify shapes
            if A_eq_cvx.shape[0] != b_eq_cvx.shape[0]:
                raise ValueError(
                    f"Equality constraint shape mismatch: A_eq.shape[0]={A_eq_cvx.shape[0]}, "
                    f"b_eq.shape[0]={b_eq_cvx.shape[0]}"
                )
        
        # Convert to CVXOPT matrix format
        # If P is sparse, convert to COO and then to CVXOPT spmatrix
        if sp is not None and hasattr(P, 'tocoo'):
            # P is a scipy sparse matrix
            # Ensure float64 dtype for CVXOPT
            P_coo = P.tocoo()
            if P_coo.dtype != np.float64:
                P_coo = P_coo.astype(np.float64)
            P_cvx = self._coo_to_cvx_spmatrix(P_coo.data, P_coo.row, P_coo.col, P_coo.shape)
        else:
            # P is a dense NumPy array
            P_cvx = cvx_matrix(P)
        q_cvx = cvx_matrix(q)
        
        G_cvx = cvx_matrix(G_combined) if G_combined is not None else None
        h_cvx = cvx_matrix(h_combined) if h_combined is not None else None
        
        A_cvx = cvx_matrix(A_eq_cvx) if A_eq_cvx is not None else None
        b_cvx = cvx_matrix(b_eq_cvx) if b_eq_cvx is not None else None
        
        # Set CVXOPT solver options (these are set via cvx_solvers.options, not as kwargs)
        # Save old options
        old_options = {}
        option_keys = ['abstol', 'reltol', 'feastol', 'maxiters', 'refinement', 'show_progress']
        for key in option_keys:
            old_options[key] = cvx_solvers.options.get(key, None)
        
        # Set new options (merge self.kwargs and kwargs)
        # Map JAXOPT-style kwargs to CVXOPT options (aligned with JAXOPT)
        merged_kwargs = {**self.kwargs, **kwargs}
        # Map JAXOPT-style kwargs to CVXOPT options
        if 'tol' in merged_kwargs:
            tol = merged_kwargs.pop('tol')
            merged_kwargs.setdefault('abstol', tol)
            merged_kwargs.setdefault('reltol', tol)
        if 'maxiter' in merged_kwargs:
            merged_kwargs['maxiters'] = merged_kwargs.pop('maxiter')
        
        for key in option_keys:
            if key in merged_kwargs:
                cvx_solvers.options[key] = merged_kwargs[key]
        
        # Solve
        try:
            result = cvx_solvers.qp(
                P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx
            )
            
            # Check result
            if result is None:
                raise RuntimeError("CVXOPT QP solver returned None")
            
            if result['status'] == 'optimal':
                solution = np.asarray(result['x'], dtype=np.float32).reshape(-1)
                
                # Verify equality constraints are satisfied (for debugging)
                if A_eq_cvx is not None and b_eq_cvx is not None:
                    eq_residual = np.abs(A_eq_cvx @ solution - b_eq_cvx)
                    max_eq_error = np.max(eq_residual)
                    if max_eq_error > 1e-5:
                        # Warning: equality constraints not satisfied (but continue anyway)
                        import warnings
                        warnings.warn(
                            f"CVXOPT equality constraints not satisfied: max_error={max_eq_error:.2e}. "
                            f"This may indicate a solver issue."
                        )
                
                info = {
                    'status': 'optimal',
                    'iterations': result.get('iterations', 0),
                }
            else:
                # Use solution anyway (may be suboptimal)
                solution = np.asarray(result['x'], dtype=np.float32).reshape(-1) if result['x'] is not None else np.zeros(n, dtype=np.float32)
                info = {
                    'status': result.get('status', 'suboptimal'),
                    'iterations': result.get('iterations', 0),
                }
            
            return solution, info
        except Exception as e:
            raise RuntimeError(f"CVXOPT QP solver failed: {e}")
        finally:
            # Restore original options
            for key, value in old_options.items():
                if value is not None:
                    cvx_solvers.options[key] = value
                elif key in cvx_solvers.options:
                    del cvx_solvers.options[key]
    
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
        Solve trajectory QP with smoothness term using NumPy native operations.
        
        min 0.5 ||x - x0||^2 + 0.5 * w * ||D2 x||^2 + ρ||ξ||^2
        s.t. A x >= b - ξ, ξ >= 0
        
        where D2 is the second-difference operator: x_t - 2*x_{t+1} + x_{t+2}
        
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
            **kwargs: Additional CVXOPT options
            
        Returns:
            Tuple of (optimized positions, info)
        """
        n = len(x_nom)
        w = float(max(0.0, smoothness_weight))
        
        # Build Hessian: P = I + w * (D2^T D2) kron I_dim
        # Use scipy.sparse (aligned with legacy NumPy implementation)
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy.sparse not available. Install scipy to use CVXOPTSolver with smoothness.")
        
        P = sp.eye(n, format="csc", dtype=np.float64)
        if w > 0.0 and T >= 3:
            # Build second-difference operator D2: (T-2) x T
            # Pattern: [1, -2, 1] per row (aligned with legacy)
            r = []
            c = []
            d = []
            for t in range(T - 2):
                # Second difference: x_t - 2 x_{t+1} + x_{t+2}
                r.extend([t, t, t])
                c.extend([t, t + 1, t + 2])
                d.extend([1.0, -2.0, 1.0])
            D2 = sp.coo_matrix((d, (r, c)), shape=(T - 2, T), dtype=np.float64).tocsr()
            K = (D2.T @ D2).tocsc()
            P_time = (sp.eye(T, format="csc", dtype=np.float64) + w * K).tocsc()
            P = sp.kron(P_time, sp.eye(dim, format="csc", dtype=np.float64), format="csc")
        
        # Linear term: q = -x_nom (for min 0.5 x^T P x + q^T x)
        q = -x_nom.astype(np.float64)
        
        # Build equality constraints for fixing initial state (if enabled)
        # Align with legacy implementation: A_eq x = b_eq where A_eq = [I_dim, 0, 0, ...]
        A_eq_np = None
        b_eq_np = None
        if fix_initial_state and initial_state is not None:
            initial_state_arr = np.asarray(initial_state, dtype=np.float64).flatten()
            if initial_state_arr.shape[0] != dim:
                raise ValueError(
                    f"initial_state has shape {initial_state_arr.shape}, expected ({dim},)"
                )
            
            # Equality constraint: x[0:dim] = initial_state
            # A_eq: [I_dim, 0, 0, ...] where I_dim is identity matrix for first dim variables
            A_eq_np = np.zeros((dim, n), dtype=np.float64)
            A_eq_np[:, :dim] = np.eye(dim, dtype=np.float64)
            b_eq_np = initial_state_arr.copy()  # Make a copy to avoid issues
        
        # Solve QP with smoothness
        # Align solver options with JAXOPT: tol=1e-6, maxiter=30
        solver_kwargs = {
            'tol': 1e-6,  # Aligned with JAXOPT
            'maxiter': 30,  # Aligned with JAXOPT
            **self.kwargs,
            **kwargs
        }
        
        if use_slack and rho is not None:
            # Slack-QP: min 0.5 x^T P x + q^T x + ρ||ξ||^2 s.t. A x >= b - ξ, ξ >= 0
            m = len(b) if b.size > 0 else 0
            n_total = n + m
            
            # Extended P matrix
            # If P is sparse, build P_ext as sparse; otherwise use dense
            if sp is not None and hasattr(P, 'tocoo'):
                # P is sparse, build P_ext as sparse block diagonal
                # P_ext = [P, 0; 0, rho*I]
                # Use scipy.sparse.bmat or manual construction
                try:
                    # Try block_diag (available in scipy >= 0.11.0)
                    P_ext = sp.block_diag([P, rho * sp.eye(m, format="csc", dtype=np.float64)], format="csc")
                except AttributeError:
                    # Fallback: manual block diagonal construction
                    P_ext = sp.bmat([[P, None], [None, rho * sp.eye(m, format="csc", dtype=np.float64)]], format="csc")
            else:
                # P is dense
                P_ext = np.zeros((n_total, n_total), dtype=np.float64)
                P_ext[:n, :n] = P
                P_ext[n:, n:] = rho * np.eye(m, dtype=np.float64)
            
            # Extended q vector
            q_ext = np.zeros(n_total, dtype=np.float64)
            q_ext[:n] = q
            
            # Extended constraints: A x + ξ >= b, ξ >= 0
            # => -A x - ξ <= -b, -ξ <= 0
            # Align with JAXOPT version: check if A is empty before building constraints
            G_ext = np.zeros((2 * m, n_total), dtype=np.float64)
            h_ext = np.zeros(2 * m, dtype=np.float64)
            
            # -A x - ξ <= -b (only if A is not empty)
            if A.size > 0 and m > 0:
                G_ext[:m, :n] = -A.astype(np.float64)
                G_ext[:m, n:] = -np.eye(m, dtype=np.float64)
                h_ext[:m] = -b.astype(np.float64)
            
            # -ξ <= 0 (i.e., ξ >= 0)
            if m > 0:
                G_ext[m:, n:] = -np.eye(m, dtype=np.float64)
                h_ext[m:] = 0.0
            
            # Extended equality constraints for initial state (if enabled)
            # For slack-QP: [A_eq, 0] [x; ξ] = b_eq
            A_eq_ext = None
            b_eq_ext = None
            if A_eq_np is not None and A_eq_np.size > 0:
                # A_eq_np is (dim, n), extend to (dim, n_total) with zeros for slack variables
                A_eq_ext = np.zeros((A_eq_np.shape[0], n_total), dtype=np.float64)
                A_eq_ext[:, :n] = A_eq_np
                b_eq_ext = b_eq_np
            
            solution_ext, info = self.solve_qp(P_ext, q_ext, G_ext, h_ext, A_eq=A_eq_ext, b_eq=b_eq_ext, **solver_kwargs)
            solution = solution_ext[:n]
            info['slack'] = solution_ext[n:]
        else:
            # Hard-QP: min 0.5 x^T P x + q^T x s.t. A x >= b
            # Convert A x >= b to -A x <= -b
            G = -A.astype(np.float64) if A.size > 0 else None
            h = -b.astype(np.float64) if b.size > 0 else None
            
            solution, info = self.solve_qp(P, q, G, h, A_eq=A_eq_np, b_eq=b_eq_np, **solver_kwargs)
        
        return solution, info
    
    @staticmethod
    def _coo_to_cvx_spmatrix(
        data: np.ndarray, row: np.ndarray, col: np.ndarray, shape: Tuple[int, int]
    ):
        """Convert scipy sparse COO matrix to cvxopt sparse matrix (aligned with legacy)."""
        if cvxopt is None:
            raise RuntimeError("cvxopt is not available")
        try:
            from cvxopt import spmatrix as cvx_spmatrix
        except ImportError:
            raise RuntimeError("cvxopt.spmatrix not available")
        return cvx_spmatrix(
            data.astype(np.float64).tolist(),
            row.astype(int).tolist(),
            col.astype(int).tolist(),
            size=shape,
        )


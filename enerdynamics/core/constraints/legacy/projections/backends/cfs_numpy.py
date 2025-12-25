"""
NumPy backend implementation of CFS projection.

This module provides the NumPy-based implementation of CFS projection,
using cvxopt for trajectory-level QP solving and enumeration for pointwise QP.
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations

try:
    import cvxopt  # type: ignore
    from cvxopt import matrix as cvx_matrix, spmatrix as cvx_spmatrix, solvers as cvx_solvers  # type: ignore
except Exception:  # pragma: no cover
    cvxopt = None
    cvx_matrix = None
    cvx_spmatrix = None
    cvx_solvers = None

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from enerdynamics.core.constraints.legacy.projections.backend_impl import CFSProjectionBase
from enerdynamics.core.registry.projections import register_projection
from enerdynamics.envs.obstacles.base import ObstacleManager


@register_projection("cfs", "numpy")
class CFSProjectionNumpy(CFSProjectionBase):
    """
    NumPy backend implementation of CFS projection using cvxopt.
    
    This implementation uses cvxopt for trajectory-level QP solving and
    enumeration-based method for pointwise QP solving.
    """
    
    def __init__(self, obstacles: ObstacleManager, **config):
        """
        Initialize NumPy backend CFS projection.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            **config: Configuration parameters:
                - max_iterations: Maximum iterations for iterative projection
                - convergence_tol: Convergence tolerance
                - max_constraints_per_point: Maximum constraints per point
                - constraint_margin: Margin for constraint selection
                - use_trajectory_qp: Whether to use trajectory-level QP
                - smoothness_weight: Smoothness regularization weight
        """
        super().__init__(obstacles, **config)
        self.max_iterations = config.get('max_iterations', 5)
        self.convergence_tol = config.get('convergence_tol', 1e-6)
        self.max_constraints_per_point = config.get('max_constraints_per_point', 8)
        self.constraint_margin = config.get('constraint_margin', 0.25)
        self.use_trajectory_qp = config.get('use_trajectory_qp', True)
        self.smoothness_weight = config.get('smoothness_weight', 0.0)
    
    def project_batch(
        self,
        positions: np.ndarray,
        clearance: float,
        step: Optional[int],
        **kwargs
    ) -> np.ndarray:
        """
        Batch project multiple points onto CFS using linearized constraints and QP.
        
        This is the main projection method that implements the CFS-QP algorithm
        using NumPy and cvxopt.
        
        Args:
            positions: Points to project, shape (N, dim)
            clearance: Minimum clearance required
            step: Current step (for logging/debugging)
            **kwargs: Additional parameters (overrides config)
            
        Returns:
            Projected positions, shape (N, dim)
        """
        # Override config with kwargs if provided
        max_iterations = kwargs.get('max_iterations', self.max_iterations)
        convergence_tol = kwargs.get('convergence_tol', self.convergence_tol)
        max_constraints_per_point = kwargs.get('max_constraints_per_point', self.max_constraints_per_point)
        constraint_margin = kwargs.get('constraint_margin', self.constraint_margin)
        use_trajectory_qp = kwargs.get('use_trajectory_qp', self.use_trajectory_qp)
        smoothness_weight = kwargs.get('smoothness_weight', self.smoothness_weight)
        
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        N, dim = positions.shape
        current = positions.copy()
        
        if N == 0:
            return current
        
        obstacles_list = list(self.obstacles)
        if len(obstacles_list) == 0:
            return current
        
        # Check if trajectory QP is available
        can_solve_trajectory_qp = (
            bool(use_trajectory_qp)
            and cvx_solvers is not None
            and cvx_matrix is not None
            and cvx_spmatrix is not None
            and sp is not None
        )
        
        # Iteratively linearize + solve QP (CFS outer loop)
        for iteration in range(max_iterations):
            # Compute per-obstacle SDF for all points ONCE (batch)
            sdf_rows = []
            for obs in obstacles_list:
                vals = obs.sdf(current)
                vals = np.asarray(vals, dtype=np.float32)
                if vals.ndim == 0:
                    vals = np.full((N,), float(vals), dtype=np.float32)
                sdf_rows.append(vals)
            sdf_matrix = np.stack(sdf_rows, axis=0)  # (M, N)
            
            # Union SDF: closest obstacle
            union_sdf = np.min(sdf_matrix, axis=0)
            violating_mask = (union_sdf < clearance)
            
            if not np.any(violating_mask):
                break  # All feasible
            
            prev = current.copy()
            
            # Try trajectory QP if available and we have multiple points
            if can_solve_trajectory_qp and N > 1:
                active_mask = union_sdf < float(clearance + constraint_margin)
                active_indices = np.where(active_mask)[0]
                if active_indices.size == 0:
                    break  # Nothing near obstacles
                
                # Build constraint matrix for trajectory QP
                A_data: List[float] = []
                A_row: List[int] = []
                A_col: List[int] = []
                b_rows: List[float] = []
                row = 0
                
                for idx in active_indices:
                    x_ref = current[idx]
                    d0_all = sdf_matrix[:, idx]  # (M,)
                    threshold = float(clearance + constraint_margin)
                    cand_mask = d0_all < threshold
                    cand_indices = np.where(cand_mask)[0]
                    if cand_indices.size == 0:
                        k = min(max_constraints_per_point, d0_all.shape[0])
                        cand_indices = np.argsort(d0_all)[:k]
                    else:
                        k = min(max_constraints_per_point, cand_indices.size)
                        cand_indices = cand_indices[np.argsort(d0_all[cand_indices])[:k]]
                    
                    for j in cand_indices:
                        obstacle = obstacles_list[int(j)]
                        d0 = float(d0_all[int(j)])
                        
                        grad = None
                        if hasattr(obstacle, "gradient"):
                            try:
                                grad = obstacle.gradient(x_ref)
                            except Exception:
                                grad = None
                        if grad is None:
                            grad = self._finite_difference_gradient(obstacle, x_ref)
                        grad = np.asarray(grad, dtype=np.float32).flatten()
                        
                        gnorm = float(np.linalg.norm(grad))
                        if not np.isfinite(gnorm) or gnorm < 1e-8:
                            continue
                        
                        g = grad / gnorm
                        b = float((clearance - d0) / gnorm + float(np.dot(g, x_ref)))
                        
                        for kdim in range(dim):
                            A_data.append(float(g[kdim]))
                            A_row.append(row)
                            A_col.append(int(idx * dim + kdim))
                        b_rows.append(b)
                        row += 1
                
                # Solve trajectory QP
                x_flat = prev.reshape(-1).astype(np.float64)
                b_vec = np.asarray(b_rows, dtype=np.float64)
                try:
                    x_new_flat = self._solve_trajectory_qp_identity_cvxopt(
                        x0=x_flat,
                        A_data=A_data,
                        A_row=A_row,
                        A_col=A_col,
                        b=b_vec,
                        n_vars=int(N * dim),
                        n_cons=int(len(b_rows)),
                        dim=int(dim),
                        T=int(N),
                        smoothness_weight=float(smoothness_weight),
                    )
                    current = x_new_flat.reshape(N, dim).astype(np.float32)
                except RuntimeError as e:
                    # If trajectory QP fails, fall back to pointwise
                    error_str = str(e).lower()
                    if "infeasible" in error_str:
                        # Problem is infeasible, disable trajectory QP for this iteration
                        can_solve_trajectory_qp = False
                    else:
                        # Numerical issue, fall back to pointwise for this iteration
                        pass
                    current = prev
                except Exception:
                    # Other errors: fall back to pointwise
                    current = prev
            
            # Pointwise projection fallback
            if not can_solve_trajectory_qp:
                for idx in np.where(violating_mask)[0]:
                    x_ref = current[idx]
                    d0_all = sdf_matrix[:, idx]  # (M,)
                    threshold = float(clearance + constraint_margin)
                    cand_mask = d0_all < threshold
                    cand_indices = np.where(cand_mask)[0]
                    if cand_indices.size == 0:
                        k = min(max_constraints_per_point, d0_all.shape[0])
                        cand_indices = np.argsort(d0_all)[:k]
                    else:
                        k = min(max_constraints_per_point, cand_indices.size)
                        cand_indices = cand_indices[np.argsort(d0_all[cand_indices])[:k]]
                    
                    A, b = self._build_linearized_halfspaces_from_candidates(
                        x_ref,
                        clearance,
                        obstacles_list,
                        cand_indices,
                        d0_all,
                    )
                    if A.size == 0:
                        continue
                    current[idx] = self.solve_projection_qp(x_ref, A, b)
            
            # Check convergence
            max_step = float(np.max(np.linalg.norm(current - prev, axis=1)))
            if max_step < convergence_tol:
                # Final feasibility check
                union_sdf2 = np.asarray(self.obstacles.sdf(current), dtype=np.float32)
                if union_sdf2.ndim == 0:
                    union_sdf2 = np.full((N,), float(union_sdf2), dtype=np.float32)
                if np.any(union_sdf2 < clearance):
                    break  # Stagnating, stop iterating
        
        return current
    
    def solve_projection_qp(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve pointwise projection QP: min 0.5||x - x0||^2 s.t. A x >= b.
        
        Uses enumeration-based method optimized for 2D/3D problems.
        
        Args:
            x0: Reference point, shape (dim,)
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            
        Returns:
            Projected point, shape (dim,)
        """
        x0 = np.asarray(x0, dtype=np.float32).flatten()
        A = np.asarray(A, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32).flatten()
        m, dim = A.shape
        
        # If already feasible, nothing to do
        if self._is_feasible(A, b, x0):
            return x0
        
        best_x = None
        best_obj = float("inf")
        
        # Active set size 1: projection onto a single hyperplane
        for i in range(m):
            a = A[i]
            denom = float(np.dot(a, a))
            if denom < 1e-12:
                continue
            alpha = float((b[i] - np.dot(a, x0)) / denom)
            x = x0 + alpha * a
            if self._is_feasible(A, b, x):
                obj = float(np.sum((x - x0) ** 2))
                if obj < best_obj:
                    best_obj = obj
                    best_x = x
        
        # Active sets of size 2..dim: project onto equalities A_I x = b_I
        max_k = min(dim, m)
        for k in range(2, max_k + 1):
            for idxs in combinations(range(m), k):
                AI = A[list(idxs)]
                bI = b[list(idxs)]
                rhs = bI - (AI @ x0)
                M = AI @ AI.T  # (k, k)
                
                # Skip near-singular active sets
                try:
                    rank = int(np.linalg.matrix_rank(M))
                except Exception:
                    rank = 0
                if rank < k:
                    continue
                
                try:
                    lam = np.linalg.solve(M, rhs)
                except Exception:
                    lam, *_ = np.linalg.lstsq(M, rhs, rcond=None)
                x = x0 + (AI.T @ lam)
                if self._is_feasible(A, b, x):
                    obj = float(np.sum((x - x0) ** 2))
                    if obj < best_obj:
                        best_obj = obj
                        best_x = x.astype(np.float32)
        
        # Fallback: step along most violated constraint
        if best_x is None:
            violations = b - (A @ x0)
            i = int(np.argmax(violations))
            a = A[i]
            denom = float(np.dot(a, a))
            if denom > 1e-12:
                alpha = float((b[i] - np.dot(a, x0)) / denom)
                best_x = (x0 + alpha * a).astype(np.float32)
            else:
                best_x = x0
        
        return best_x.astype(np.float32)
    
    def _build_linearized_halfspaces_from_candidates(
        self,
        x_ref: np.ndarray,
        clearance: float,
        obstacles_list: List,
        cand_indices: np.ndarray,
        d0_all: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build halfspace constraints A x >= b from linearized obstacle SDF constraints.
        
        Args:
            x_ref: Reference point
            clearance: Clearance value
            obstacles_list: List of obstacles
            cand_indices: Candidate obstacle indices
            d0_all: SDF values for all obstacles at x_ref
            
        Returns:
            Tuple of (A, b) where A is constraint matrix and b is constraint vector
        """
        x_ref = np.asarray(x_ref, dtype=np.float32).flatten()
        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []
        
        for j in cand_indices:
            obstacle = obstacles_list[int(j)]
            d0 = float(d0_all[int(j)])
            
            grad = None
            if hasattr(obstacle, "gradient"):
                try:
                    grad = obstacle.gradient(x_ref)
                except Exception:
                    grad = None
            
            if grad is None:
                grad = self._finite_difference_gradient(obstacle, x_ref)
            grad = np.asarray(grad, dtype=np.float32).flatten()
            
            gnorm = float(np.linalg.norm(grad))
            if not np.isfinite(gnorm) or gnorm < 1e-8:
                continue
            
            # Linearized constraint: grad^T x >= clearance - d0 + grad^T x_ref
            g = grad / gnorm
            b = float((clearance - d0) / gnorm + float(np.dot(g, x_ref)))
            A_rows.append(g)
            b_rows.append(b)
        
        if not A_rows:
            return np.zeros((0, x_ref.shape[0]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        A = np.stack(A_rows, axis=0).astype(np.float32)
        b = np.asarray(b_rows, dtype=np.float32)
        return A, b
    
    def _solve_trajectory_qp_identity_cvxopt(
        self,
        *,
        x0: np.ndarray,
        A_data: List[float],
        A_row: List[int],
        A_col: List[int],
        b: np.ndarray,
        n_vars: int,
        n_cons: int,
        dim: int,
        T: int,
        smoothness_weight: float,
    ) -> np.ndarray:
        """
        Solve trajectory-level CFS-QP using cvxopt.
        
        Minimizes: 0.5 ||x - x0||^2 + 0.5 * w * ||D2 x||^2
        Subject to: A x >= b
        
        where x is flattened trajectory and D2 is second-difference operator.
        
        Args:
            x0: Flattened reference trajectory, shape (n_vars,)
            A_data, A_row, A_col: Sparse constraint matrix in COO format
            b: Constraint vector, shape (n_cons,)
            n_vars: Number of variables (T * dim)
            n_cons: Number of constraints
            dim: Dimension of each point
            T: Number of time steps
            smoothness_weight: Weight for smoothness regularization
            
        Returns:
            Flattened projected trajectory, shape (n_vars,)
        """
        if cvx_solvers is None or cvx_matrix is None or sp is None:
            raise RuntimeError("Trajectory QP solver dependencies not available (cvxopt/scipy).")
        if n_cons <= 0:
            return np.asarray(x0, dtype=np.float64).reshape(-1)
        
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0.shape[0] != n_vars:
            raise ValueError(f"x0 has shape {x0.shape}, expected ({n_vars},)")
        
        # Build Hessian: I + w * (D2^T D2) kron I_dim
        P = sp.eye(n_vars, format="csc", dtype=np.float64)
        w = float(max(0.0, smoothness_weight))
        if w > 0.0 and T >= 3:
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
        
        P_coo = P.tocoo()
        P_cvx = self._coo_to_cvx_spmatrix(P_coo.data, P_coo.row, P_coo.col, P_coo.shape)
        
        # Linear term: 0.5||x-x0||^2 -> q = -x0
        q_cvx = cvx_matrix((-x0).astype(np.float64))
        
        # Constraints: A x >= b  <=>  (-A) x <= (-b)
        A_sp = sp.coo_matrix(
            (np.asarray(A_data, dtype=np.float64), (np.asarray(A_row), np.asarray(A_col))),
            shape=(n_cons, n_vars),
            dtype=np.float64,
        )
        
        # Check for numerical issues
        if n_cons > 0 and n_vars > 0:
            A_dense = A_sp.toarray()
            b_arr = np.asarray(b, dtype=np.float64)
            if np.any(~np.isfinite(A_dense)) or np.any(~np.isfinite(b_arr)):
                raise RuntimeError("Non-finite values in QP constraints")
        
        G_sp = (-A_sp).tocoo()
        G_cvx = self._coo_to_cvx_spmatrix(G_sp.data, G_sp.row, G_sp.col, G_sp.shape)
        h_cvx = cvx_matrix((-np.asarray(b, dtype=np.float64)).reshape(-1))
        
        # Solve QP with improved numerical stability options
        old_show = cvx_solvers.options.get("show_progress", True)
        old_maxiters = cvx_solvers.options.get("maxiters", 100)
        old_abstol = cvx_solvers.options.get("abstol", 1e-7)
        old_reltol = cvx_solvers.options.get("reltol", 1e-6)
        old_feastol = cvx_solvers.options.get("feastol", 1e-7)
        old_refinement = cvx_solvers.options.get("refinement", 1)
        
        # Set improved options for better numerical stability
        cvx_solvers.options["show_progress"] = False
        cvx_solvers.options["maxiters"] = 200
        cvx_solvers.options["abstol"] = 1e-6
        cvx_solvers.options["reltol"] = 1e-5
        cvx_solvers.options["feastol"] = 1e-6
        cvx_solvers.options["refinement"] = 2
        
        try:
            sol = cvx_solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        finally:
            # Restore original options
            cvx_solvers.options["show_progress"] = old_show
            cvx_solvers.options["maxiters"] = old_maxiters
            cvx_solvers.options["abstol"] = old_abstol
            cvx_solvers.options["reltol"] = old_reltol
            cvx_solvers.options["feastol"] = old_feastol
            cvx_solvers.options["refinement"] = old_refinement
        
        if sol is None:
            raise RuntimeError("cvxopt qp failed: solver returned None")
        
        status = sol.get("status", "")
        
        # Handle "unknown" status: try to use solution if it exists and is finite
        if status == "unknown":
            if "x" in sol and sol["x"] is not None:
                x_candidate = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
                if np.all(np.isfinite(x_candidate)):
                    return x_candidate
            raise RuntimeError(
                f"cvxopt qp failed: status=unknown. "
                f"This may indicate numerical issues. "
                f"Consider using pointwise projection or checking constraint conditioning."
            )
        
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"cvxopt qp failed: status={status}")
        
        x = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
        return x
    
    @staticmethod
    def _coo_to_cvx_spmatrix(
        data: np.ndarray, row: np.ndarray, col: np.ndarray, shape: Tuple[int, int]
    ):
        """Convert scipy sparse COO matrix to cvxopt sparse matrix."""
        if cvx_spmatrix is None:
            raise RuntimeError("cvxopt is not available")
        return cvx_spmatrix(
            data.astype(np.float64).tolist(),
            row.astype(int).tolist(),
            col.astype(int).tolist(),
            size=shape,
        )


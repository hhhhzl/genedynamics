"""
JAX backend implementation of CFS projection.

This module provides the JAX-based implementation of CFS projection,
using JAX operations and qpax for QP solving, optimized for GPU acceleration.
"""

import numpy as np
from typing import Optional, Callable, Tuple, List

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

try:
    # Optional: JAX-based QP solver (supports general inequalities)
    import qpax
    QPAX_AVAILABLE = True
except ImportError:
    qpax = None
    QPAX_AVAILABLE = False

from enerdynamics.core.constraints.projections.backend_impl import CFSProjectionBase
from enerdynamics.core.registry.projections import register_projection
from enerdynamics.envs.obstacles.base import ObstacleManager


if JAX_AVAILABLE:
    # ============================================================================
    # JAX Helper Functions
    # ============================================================================
    
    def _finite_difference_gradient_jax(obstacle, point: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
        """
        Numerically approximate ∇sdf(x) with central differences using JAX.
        
        This function requires the obstacle to have JAX-compatible SDF method.
        It cannot use NumPy conversion inside JIT functions.
        
        Args:
            obstacle: Obstacle with jax_sdf or sample_sdf_and_grad_2d method
            point: Point to evaluate, shape (dim,) as JAX array
            eps: Step size for finite differences
            
        Returns:
            Gradient, shape (dim,)
        """
        point = jnp.asarray(point, dtype=jnp.float32).flatten()
        dim = point.shape[0]
        
        # Use JAX-compatible SDF computation (cannot convert to NumPy in JIT)
        grad = jnp.zeros((dim,), dtype=jnp.float32)
        
        for k in range(dim):
            # Create shifted points using JAX operations
            xp = point.at[k].add(eps)
            xm = point.at[k].add(-eps)
            
            # Compute SDF using JAX-compatible method
            if hasattr(obstacle, "jax_sdf"):
                dp = obstacle.jax_sdf(xp[None, :])  # Add batch dimension
                dm = obstacle.jax_sdf(xm[None, :])
                dp_val = dp[0] if dp.ndim > 0 else dp
                dm_val = dm[0] if dm.ndim > 0 else dm
            elif hasattr(obstacle, "sample_sdf_and_grad_2d"):
                dp, _ = obstacle.sample_sdf_and_grad_2d(xp[None, :], backend="jax")
                dm, _ = obstacle.sample_sdf_and_grad_2d(xm[None, :], backend="jax")
                dp_val = dp[0] if dp.ndim > 0 else dp
                dm_val = dm[0] if dm.ndim > 0 else dm
            else:
                # This should not happen if check above passed
                raise RuntimeError(f"Obstacle {type(obstacle).__name__} does not support JAX for finite differences")
            
            # Compute gradient using JAX operations
            grad = grad.at[k].set((dp_val - dm_val) / (2.0 * eps))
        
        return grad
    
    
    def _is_feasible_jax(A: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray, tol: float = 1e-7) -> bool:
        """
        Check A x >= b within a tolerance (JAX version).
        
        Args:
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            x: Point to check, shape (dim,)
            tol: Tolerance
            
        Returns:
            Boolean indicating if all constraints are satisfied
        """
        if A.size == 0:
            return True
        lhs = A @ x
        return bool(jnp.all(lhs + tol >= b).item())
    
    
    def _solve_trajectory_qp_identity_jax(
        x0: jnp.ndarray,
        A_data: jnp.ndarray,
        A_row: jnp.ndarray,
        A_col: jnp.ndarray,
        b: jnp.ndarray,
        n_vars: int,
        n_cons: int,
        dim: int,
        T: int,
        smoothness_weight: float,
    ) -> jnp.ndarray:
        """
        Solve trajectory-level CFS-QP using JAX-compatible QP solver (qpax):
        
            minimize   0.5 ||x - x0||^2 + 0.5 * w * ||D2 x||^2
            subject to A x >= b
        
        where x is the flattened trajectory [p0, p1, ..., p_{T-1}] and D2 is a
        second-difference operator along time (applied per dimension).
        
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
        if not QPAX_AVAILABLE or qpax is None:
            raise RuntimeError("qpax is required for JAX trajectory QP but not available")
        
        x0 = jnp.asarray(x0, dtype=jnp.float32).flatten()
        b = jnp.asarray(b, dtype=jnp.float32).flatten()
        
        if n_cons <= 0:
            return x0
        
        # Build Hessian: I + w * (D2^T D2) kron I_dim
        # For identity case (w=0), P = I
        w = jnp.asarray(max(0.0, smoothness_weight), dtype=jnp.float32)
        
        if w > 0.0 and T >= 3:
            # Build second-difference operator D2: (T-2) x T
            # D2[i, t] = 1 if t==i, -2 if t==i+1, 1 if t==i+2, else 0
            def build_D2_row(i):
                """Build row i of D2 matrix."""
                row = jnp.zeros(T, dtype=jnp.float32)
                row = row.at[i].set(1.0)
                row = row.at[i+1].set(-2.0)
                row = row.at[i+2].set(1.0)
                return row
            
            D2 = jax.vmap(build_D2_row)(jnp.arange(T - 2))  # (T-2, T)
            K = D2.T @ D2  # (T, T)
            P_time = jnp.eye(T, dtype=jnp.float32) + w * K
            
            # Kronecker product: P_time kron I_dim
            P = jnp.kron(P_time, jnp.eye(dim, dtype=jnp.float32))  # (n_vars, n_vars)
        else:
            # Identity Hessian
            P = jnp.eye(n_vars, dtype=jnp.float32)
        
        # Build constraint matrix A from sparse format (COO)
        # JAX doesn't have sparse matrices, so we build dense A
        # Use scatter-add pattern to accumulate values
        A = jnp.zeros((n_cons, n_vars), dtype=jnp.float32)
        
        # Scatter-add: for each (row, col, value) triple, add value to A[row, col]
        # Use a loop to accumulate (JAX-compatible)
        def scatter_entry(carry, idx):
            """Scatter entry at index idx into A matrix."""
            A_mat = carry
            row = A_row[idx]
            col = A_col[idx]
            val = A_data[idx]
            # Use index_add to accumulate (handles duplicate entries)
            A_mat = A_mat.at[row, col].add(val)
            return A_mat
        
        # Process all entries using scan
        A = jax.lax.scan(
            scatter_entry,
            A,
            jnp.arange(len(A_data))
        )[0]
        
        # Convert to qpax format: min 0.5 x^T P x + q^T x s.t. G x <= h
        # Original: min 0.5||x-x0||^2 + 0.5*w*||D2 x||^2 = 0.5 x^T P x - x0^T x + const
        Q = P
        q = -x0  # Linear term
        G = -A  # A x >= b  =>  -A x <= -b
        h = -b
        
        # No equality constraints
        A_eq = jnp.zeros((0, n_vars), dtype=jnp.float32)
        b_eq = jnp.zeros((0,), dtype=jnp.float32)
        
        # Solve with qpax
        try:
            x, s, z, y, converged, iters = qpax.solve_qp(
                Q, q, A_eq, b_eq, G, h,
                solver_tol=1e-7,  # Tight tolerance for accuracy
                max_iter=200,  # Increased iterations
            )
            
            if not converged:
                # If not converged, try with looser tolerance
                x, s, z, y, converged, iters = qpax.solve_qp(
                    Q, q, A_eq, b_eq, G, h,
                    solver_tol=1e-6,
                    max_iter=200,
                )
            
            if not converged:
                raise RuntimeError(f"qpax trajectory QP did not converge after {iters} iterations")
            
            return x.astype(jnp.float32)
        except Exception as e:
            raise RuntimeError(f"qpax trajectory QP failed: {e}")
    
    
    def _solve_projection_qp_identity_jax(
        x0: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Solve: min 0.5||x - x0||^2 s.t. A x >= b (JAX version).
        
        Priority:
        1. Try qpax (JAX-compatible QP solver, supports general inequalities)
        2. Fallback to optimized JAX enumeration method (fast for small problems)
        
        Args:
            x0: Reference point, shape (dim,)
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            
        Returns:
            Projected point, shape (dim,)
        """
        x0 = jnp.asarray(x0, dtype=jnp.float32).flatten()
        A = jnp.asarray(A, dtype=jnp.float32)
        b = jnp.asarray(b, dtype=jnp.float32).flatten()
        m, dim = A.shape
        
        # Filter out zero rows (invalid constraints) before solving
        # Align with NumPy: completely mask invalid rows (A=0, b=-1e6) to avoid qpax seeing ill-conditioned constraints
        row_norms = jnp.linalg.norm(A, axis=1)
        valid_mask = row_norms > 1e-8
        # Mask A: set invalid rows to zero
        A_masked = jnp.where(valid_mask[:, None], A, jnp.zeros((m, dim), dtype=jnp.float32))
        # Mask b: set invalid rows to very negative (satisfied constraint)
        b_masked = jnp.where(valid_mask, b, jnp.full((m,), -1e6, dtype=jnp.float32))
        
        # Check if already feasible
        lhs = A_masked @ x0
        feasible = jnp.all(lhs + 1e-7 >= b_masked)
        
        def solve_qp():
            # Try qpax first (supports general inequalities, fully JAX-compatible)
            num_valid = jnp.sum(valid_mask.astype(jnp.int32))
            
            # Only try qpax if we have valid constraints
            def try_qpax():
                if QPAX_AVAILABLE and qpax is not None:
                    try:
                        # Convert to qpax format: min 0.5 x^T Q x + q^T x s.t. G x <= h
                        # Use masked A to avoid ill-conditioned constraints
                        Q = jnp.eye(dim, dtype=jnp.float32)
                        q = -x0
                        G = -A_masked  # A x >= b  =>  -A x <= -b, use masked A
                        h = -b_masked
                        
                        # No equality constraints - use empty arrays instead of None
                        A_eq = jnp.zeros((0, dim), dtype=jnp.float32)  # Empty matrix
                        b_eq = jnp.zeros((0,), dtype=jnp.float32)  # Empty vector
                        
                        # Use tighter tolerance to match NumPy enumeration accuracy
                        # Increased max iterations for better convergence
                        x, s, z, y, converged, iters = qpax.solve_qp(
                            Q, q, A_eq, b_eq, G, h,
                            solver_tol=1e-7,  # Tighter tolerance to match NumPy precision
                            max_iter=200,  # Increase max iterations for better convergence
                        )
                        # Return result and convergence flag (ensure converged is JAX bool)
                        converged_bool = jnp.asarray(converged, dtype=jnp.bool_)
                        return x.astype(jnp.float32), converged_bool
                    except Exception:
                        # If qpax fails, return failure with JAX bool
                        return x0, jnp.array(False, dtype=jnp.bool_)
                # If qpax not available, return failure with JAX bool
                return x0, jnp.array(False, dtype=jnp.bool_)
            
            def skip_qpax():
                return x0, jnp.array(False, dtype=jnp.bool_)
            
            # Try qpax only if we have valid constraints
            qpax_result, qpax_converged = jax.lax.cond(
                num_valid > 0,
                try_qpax,
                skip_qpax
            )
            
            # Use qpax result if converged, otherwise use enumeration
            def use_qpax():
                return qpax_result
            
            def use_enumeration():
                return _solve_projection_qp_enumeration_jax(x0, A_masked, b_masked, valid_mask)
            
            result = jax.lax.cond(qpax_converged, use_qpax, use_enumeration)
            return result
        
        def return_original():
            return x0
        
        return jax.lax.cond(feasible, return_original, solve_qp)
    
    
    def _solve_projection_qp_enumeration_jax(
        x0: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray, valid_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Optimized JAX enumeration method for QP projection.
        
        This is a JAX-compatible version of the NumPy enumeration method,
        optimized for small problems (2D, few constraints).
        
        Args:
            x0: Reference point, shape (dim,)
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            valid_mask: Boolean mask for valid constraints, shape (m,)
            
        Returns:
            Projected point, shape (dim,)
        """
        m, dim = A.shape
        
        # Active set size 1: projection onto single hyperplane
        def project_single(i):
            """Project onto constraint i."""
            a = A[i]
            denom = jnp.dot(a, a)
            denom = jnp.where(denom < 1e-12, 1e-12, denom)
            alpha = (b[i] - jnp.dot(a, x0)) / denom
            x = x0 + alpha * a
            
            # Check feasibility
            lhs_check = A @ x
            is_feas = jnp.all(lhs_check + 1e-7 >= b)
            obj = jnp.sum((x - x0) ** 2)
            # Return large objective if infeasible
            obj = jnp.where(is_feas, obj, jnp.inf)
            return obj, x
        
        # Vectorize over all constraints
        objs, xs = jax.vmap(project_single)(jnp.arange(m))
        best_idx = jnp.argmin(objs)
        best_x = xs[best_idx]
        best_obj = objs[best_idx]
        
        # For 2D/3D problems, enumerate all pairs (and triples for 3D) of constraints
        # This matches NumPy version's full enumeration (combinations(range(m), k))
        # Since max_constraints_per_point is typically small (<=8), this is affordable
        max_k = min(dim, m)
        
        # Enumerate pairs (k=2) - full enumeration matching NumPy version
        # Since m is typically small (<=8 from max_constraints_per_point), we use a fixed-size approach
        # Generate all pairs (i, j) where i < j using JAX-compatible method
        if max_k >= 2 and m >= 2:
            # Use a fixed maximum number of pairs (for m=8, we have 28 pairs; use 50 as safe upper bound)
            max_pairs = 50
            max_m = 10  # Maximum expected m
            
            def check_pair(i, j):
                """Check a pair of constraints (i, j)."""
                # Skip if not a valid pair (i < j) or out of bounds
                valid = jnp.logical_and(i < m, jnp.logical_and(j < m, i < j))
                valid = jnp.logical_and(valid, valid_mask[i] & valid_mask[j])
                
                AI = A[jnp.array([i, j])]
                bI = b[jnp.array([i, j])]
                rhs = bI - (AI @ x0)
                M = AI @ AI.T
                det = jnp.linalg.det(M)
                
                def solve_pair():
                    lam = jnp.linalg.solve(M, rhs)
                    x = x0 + (AI.T @ lam)
                    lhs_check = A @ x
                    is_feas = jnp.all(lhs_check + 1e-7 >= b)
                    obj = jnp.sum((x - x0) ** 2)
                    return jnp.where(is_feas, obj, jnp.inf), x
                
                def return_inf():
                    return jnp.inf, x0
                
                obj, x = jax.lax.cond(
                    jnp.logical_and(valid, jnp.abs(det) > 1e-10),
                    solve_pair,
                    return_inf
                )
                return obj, x
            
            # Generate all pairs using nested vmap with fixed shape
            # For each i, check all j (j > i will be valid, others will be masked)
            def check_pairs_for_i(i):
                """Check all pairs starting with i."""
                # Use fixed-size array: check j from 0 to max_m-1, but only j > i is valid
                js = jnp.arange(max_m)
                objs, xs = jax.vmap(lambda j: check_pair(i, j))(js)
                return objs, xs
            
            # Check all pairs for all i
            i_indices = jnp.arange(max_m)
            all_pair_objs, all_pair_xs = jax.vmap(check_pairs_for_i)(i_indices)
            # Flatten results: shape (max_m, max_m) -> (max_m*max_m,)
            all_pair_objs_flat = all_pair_objs.flatten()
            all_pair_xs_flat = all_pair_xs.reshape(-1, dim)
            
            # Update best solution (invalid pairs will have obj=inf, so they won't be selected)
            best_pair_idx = jnp.argmin(all_pair_objs_flat)
            best_pair_obj = all_pair_objs_flat[best_pair_idx]
            best_pair_x = all_pair_xs_flat[best_pair_idx]
            
            best_x = jnp.where(best_pair_obj < best_obj, best_pair_x, best_x)
            best_obj = jnp.minimum(best_obj, best_pair_obj)
        
        # For 3D problems, also enumerate triples (k=3) - matching NumPy version
        # For 2D (dim=2), pairs are sufficient, but we can still enumerate triples for completeness
        # Since m is typically small (<=8 from max_constraints_per_point), we can enumerate triples
        if max_k >= 3 and m >= 3 and dim >= 3:
            # Enumerate triples (k=3) for 3D problems
            # Use fixed-size approach for JAX compatibility
            max_triples = 120  # C(10,3) = 120, safe upper bound
            max_m_triples = 10
            
            def check_triple(i, j, k_idx):
                """Check a triple of constraints (i, j, k) where i < j < k."""
                # Generate k from k_idx (k_idx represents position in sequence)
                # For each (i,j) pair, k can be j+1, j+2, ..., m-1
                k = j + 1 + k_idx
                valid = jnp.logical_and(
                    i < m,
                    jnp.logical_and(
                        j < m,
                        jnp.logical_and(
                            k < m,
                            jnp.logical_and(
                                i < j,
                                j < k
                            )
                        )
                    )
                )
                valid = jnp.logical_and(valid, valid_mask[i] & valid_mask[j])
                valid = jnp.logical_and(valid, valid_mask[k] if k < m else False)
                
                AI = A[jnp.array([i, j, k])]
                bI = b[jnp.array([i, j, k])]
                rhs = bI - (AI @ x0)
                M = AI @ AI.T
                det = jnp.linalg.det(M)
                
                def solve_triple():
                    lam = jnp.linalg.solve(M, rhs)
                    x = x0 + (AI.T @ lam)
                    lhs_check = A @ x
                    is_feas = jnp.all(lhs_check + 1e-7 >= b)
                    obj = jnp.sum((x - x0) ** 2)
                    return jnp.where(is_feas, obj, jnp.inf), x
                
                def return_inf():
                    return jnp.inf, x0
                
                obj, x = jax.lax.cond(
                    jnp.logical_and(valid, jnp.abs(det) > 1e-10),
                    solve_triple,
                    return_inf
                )
                return obj, x
            
            # Enumerate all triples: for each (i,j) pair, check all k > j
            def check_triples_for_pair(i, j):
                """Check all triples starting with (i, j)."""
                max_k_for_pair = max_m_triples - j - 1  # k can be j+1 to max_m-1
                k_indices = jnp.arange(max_m_triples)
                objs, xs = jax.vmap(lambda k_idx: check_triple(i, j, k_idx))(k_indices)
                return objs, xs
            
            # For each pair (i, j), check triples
            def check_triples_for_i(i):
                """Check all triples starting with i."""
                js = jnp.arange(max_m_triples)
                all_objs, all_xs = jax.vmap(lambda j: check_triples_for_pair(i, j))(js)
                # Flatten: shape (max_m, max_m) -> (max_m*max_m,)
                return all_objs.flatten(), all_xs.reshape(-1, dim)
            
            i_indices = jnp.arange(max_m_triples)
            all_triple_objs, all_triple_xs = jax.vmap(check_triples_for_i)(i_indices)
            # Flatten further: shape (max_m, max_m*max_m) -> (max_m*max_m*max_m,)
            all_triple_objs_flat = all_triple_objs.flatten()
            all_triple_xs_flat = all_triple_xs.reshape(-1, dim)
            
            # Update best solution
            best_triple_idx = jnp.argmin(all_triple_objs_flat)
            best_triple_obj = all_triple_objs_flat[best_triple_idx]
            best_triple_x = all_triple_xs_flat[best_triple_idx]
            
            best_x = jnp.where(best_triple_obj < best_obj, best_triple_x, best_x)
            best_obj = jnp.minimum(best_obj, best_triple_obj)
        
        # Final fallback: step along most violated constraint
        violations = b - (A @ x0)
        violations_masked = jnp.where(valid_mask, violations, jnp.full((m,), -1e6, dtype=jnp.float32))
        i_max = jnp.argmax(violations_masked)
        a = A[i_max]
        denom = jnp.dot(a, a)
        denom = jnp.where(denom < 1e-12, 1e-12, denom)
        alpha = (b[i_max] - jnp.dot(a, x0)) / denom
        fallback_x = x0 + alpha * a
        
        # Use best solution found
        return jnp.where(best_obj < jnp.inf, best_x, fallback_x).astype(jnp.float32)
    
    
    def _project_cfs_jax(
        positions: jnp.ndarray,
        sdf_fn: Callable,
        grad_fn: Callable,
        clearance: jnp.ndarray,
        max_iterations: int = 5,
        convergence_tol: float = 1e-6,
        max_constraints_per_point: int = 8,
        constraint_margin: float = 0.25,
        use_trajectory_qp: bool = False,
        smoothness_weight: float = 0.0,
    ) -> jnp.ndarray:
        """
        JAX-compatible batch CFS projection.
        
        Projects multiple points onto CFS using linearized constraints and QP.
        This is the JAX version of _project_cfs_batch, designed for GPU acceleration.
        
        Args:
            positions: Points to project, shape (N, dim) as JAX array
            sdf_fn: Function that computes SDF for all obstacles, (points) -> (M, N)
            grad_fn: Function that computes gradients, (point, obs_indices) -> (k, dim)
            clearance: Minimum clearance required (scalar or broadcastable)
            max_iterations: Maximum iterations for iterative projection
            convergence_tol: Convergence tolerance
            max_constraints_per_point: Maximum constraints per point
            constraint_margin: Margin for constraint selection
            use_trajectory_qp: Whether to use trajectory-level QP
            smoothness_weight: Smoothness regularization weight
            
        Returns:
            Projected positions, shape (N, dim) as JAX array
        """
        positions = jnp.asarray(positions, dtype=jnp.float32)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        N, dim = positions.shape
        
        if N == 0:
            return positions
        
        clearance = jnp.asarray(clearance, dtype=jnp.float32)
        if clearance.ndim > 0:
            clearance = clearance[0]  # Use first element if array
        
        current = positions
        
        # Project a single point (used in loop)
        def project_single_point(x_ref: jnp.ndarray, sdf_vals: jnp.ndarray, obs_list_len: int) -> jnp.ndarray:
            """
            Project a single point using CFS.
            
            This matches the NumPy version logic:
            1. First check for candidates within (clearance + margin)
            2. If candidates exist, select k closest from candidates
            3. If no candidates, fallback to k closest from all obstacles
            """
            threshold = clearance + constraint_margin
            cand_mask = sdf_vals < threshold
            num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
            
            # Align with NumPy version: if candidates exist, use them; otherwise fallback to all
            sorted_indices = jnp.argsort(sdf_vals)
            
            def use_candidates():
                """Select k closest from candidates."""
                # Step 1: Collect all candidate indices (those within threshold)
                cand_all = jnp.zeros(obs_list_len, dtype=jnp.int32)
                cand_count = 0
                
                def collect_cand(carry, i):
                    """Collect candidate index i if it's in cand_mask."""
                    cand_all, cand_count = carry
                    is_candidate = cand_mask[i]
                    cand_all = jnp.where(
                        jnp.arange(obs_list_len) == cand_count,
                        jnp.where(is_candidate, i, cand_all[cand_count]),
                        cand_all
                    )
                    cand_count = jnp.where(is_candidate, cand_count + 1, cand_count)
                    return (cand_all, cand_count), None
                
                # Collect all candidates
                initial_state = (cand_all, 0)
                (cand_all, total_candidates), _ = jax.lax.scan(
                    collect_cand,
                    initial_state,
                    jnp.arange(obs_list_len)
                )
                
                # Step 2: Sort candidates by their SDF values and take k closest
                def get_cand_sdf(i):
                    """Get SDF value for candidate at position i."""
                    cand_idx = cand_all[i]
                    return jnp.where(i < total_candidates, sdf_vals[cand_idx], jnp.inf)
                
                cand_sdf_vals = jax.vmap(get_cand_sdf)(jnp.arange(obs_list_len))
                cand_sorted_by_sdf = jnp.argsort(cand_sdf_vals)
                k_val = jnp.minimum(max_constraints_per_point, total_candidates)
                
                def get_sorted_cand(i):
                    """Get sorted candidate index at position i."""
                    sorted_pos = cand_sorted_by_sdf[i]
                    cand_idx = cand_all[sorted_pos]
                    return jnp.where(i < k_val, cand_idx, 0)
                
                cand_fixed = jax.vmap(get_sorted_cand)(jnp.arange(max_constraints_per_point))
                
                # Pad with last valid index if needed
                last_val = jnp.where(k_val > 0, cand_fixed[k_val - 1], 0)
                cand_fixed = jnp.where(
                    jnp.arange(max_constraints_per_point) >= k_val,
                    last_val,
                    cand_fixed
                )
                return cand_fixed, k_val
            
            def use_all():
                """Fallback: take k closest from all obstacles."""
                k_val = jnp.minimum(max_constraints_per_point, obs_list_len)
                cand_fixed = jnp.zeros(max_constraints_per_point, dtype=jnp.int32)
                n_fill = jnp.minimum(k_val, max_constraints_per_point)
                
                def fill_pos(i):
                    """Fill position i with sorted_indices[i] if i < n_fill and i < obs_list_len, else 0."""
                    idx = jnp.where(i < obs_list_len, sorted_indices[i], 0)
                    return jnp.where(i < n_fill, idx, 0)
                
                cand_fixed = jax.vmap(fill_pos)(jnp.arange(max_constraints_per_point))
                last_idx = jnp.maximum(0, jnp.minimum(n_fill - 1, obs_list_len - 1))
                last_val = jnp.where(n_fill > 0, sorted_indices[last_idx], 0)
                cand_fixed = jnp.where(
                    jnp.arange(max_constraints_per_point) >= n_fill,
                    last_val,
                    cand_fixed
                )
                return cand_fixed, k_val
            
            # Use candidates if available, otherwise use all
            cand_indices, k = jax.lax.cond(
                num_candidates > 0,
                use_candidates,
                use_all
            )
            
            # Compute gradients for candidate obstacles
            grads = grad_fn(x_ref, cand_indices)  # (max_constraints_per_point, dim)
            
            # Build linearized constraints A x >= b
            max_k = max_constraints_per_point
            
            def build_constraint(i):
                """Build constraint for obstacle at index i."""
                j = cand_indices[i]
                grad = grads[i]
                d0 = sdf_vals[j]
                
                # Check if gradient is valid
                gnorm = jnp.linalg.norm(grad)
                is_valid = jnp.logical_and(jnp.isfinite(gnorm), gnorm >= 1e-8)
                
                # Normalize gradient
                gnorm_safe = jnp.where(gnorm < 1e-8, 1e-8, gnorm)
                g = grad / gnorm_safe
                b_val = (clearance - d0) / gnorm_safe + jnp.dot(g, x_ref)
                
                # Return constraint and validity flag
                return g, b_val, is_valid
            
            # Build all constraints using vmap
            gs, bs, constraint_valid = jax.vmap(build_constraint)(jnp.arange(max_k))
            
            # Select only the first k constraints that are valid
            candidate_mask = jnp.arange(max_k) < k
            valid_mask = jnp.logical_and(candidate_mask, constraint_valid)
            num_valid_constraints = jnp.sum(valid_mask.astype(jnp.int32))
            
            # Mask out invalid constraints
            A_masked = jnp.where(valid_mask[:, None], gs, jnp.zeros((max_k, dim), dtype=jnp.float32))
            b_masked = jnp.where(valid_mask, bs, jnp.full((max_k,), -1e6, dtype=jnp.float32))
            
            # Check if we have any valid constraints
            has_constraints = num_valid_constraints > 0
            
            def solve_with_constraints():
                # Use all constraints (invalid ones are automatically satisfied due to large negative b)
                x_proj = _solve_projection_qp_identity_jax(x_ref, A_masked, b_masked)
                
                # Post-process: ensure feasibility
                lhs = A_masked @ x_proj
                violations = b_masked - lhs
                max_violation = jnp.max(violations)
                
                def fix_violation():
                    # Find most violated constraint
                    i_max = jnp.argmax(violations)
                    a = A_masked[i_max]
                    b_val = b_masked[i_max]
                    
                    # Project onto this constraint
                    denom = jnp.dot(a, a)
                    denom = jnp.where(denom < 1e-12, 1e-12, denom)
                    alpha = (b_val - jnp.dot(a, x_proj)) / denom
                    x_fixed = x_proj + alpha * a
                    
                    # Iterate to fix violations
                    def iterate_fix(x, _):
                        lhs_check = A_masked @ x
                        violations_check = b_masked - lhs_check
                        i_max_check = jnp.argmax(violations_check)
                        a_check = A_masked[i_max_check]
                        b_check = b_masked[i_max_check]
                        denom_check = jnp.dot(a_check, a_check)
                        denom_check = jnp.where(denom_check < 1e-12, 1e-12, denom_check)
                        alpha_check = (b_check - jnp.dot(a_check, x)) / denom_check
                        return x + alpha_check * a_check, None
                    
                    x_final, _ = jax.lax.scan(iterate_fix, x_fixed, None, length=3)
                    return x_final
                
                def return_proj():
                    return x_proj
                
                return jax.lax.cond(
                    max_violation > 1e-6,
                    fix_violation,
                    return_proj
                )
            
            def return_original():
                return x_ref
            
            return jax.lax.cond(
                has_constraints,
                solve_with_constraints,
                return_original
            )
        
        # Iteratively linearize + solve QP (CFS outer loop)
        def iteration_body(i, current_pos):
            """Single iteration of CFS projection."""
            # Compute SDF for all obstacles at all points
            sdf_matrix = sdf_fn(current_pos)  # (M, N)
            
            # Union SDF: closest obstacle
            union_sdf = jnp.min(sdf_matrix, axis=0)  # (N,)
            violating_mask = union_sdf < clearance
            
            # Check if all feasible
            all_feasible = jnp.all(~violating_mask)
            
            prev = current_pos
            
            # Try trajectory QP if enabled and we have multiple points
            def try_trajectory_qp():
                """Try trajectory-level QP for all points."""
                # Find active points (within clearance + margin)
                active_mask = union_sdf < (clearance + constraint_margin)
                num_active = jnp.sum(active_mask.astype(jnp.int32))
                
                def solve_traj_qp():
                    """Solve trajectory QP."""
                    # Build constraints for each point (same logic as pointwise)
                    # We'll build a dense constraint matrix A: (max_cons, n_vars)
                    max_cons = N * max_constraints_per_point
                    
                    def build_point_constraints(point_idx):
                        """Build constraints for point point_idx, returns (max_constraints_per_point, dim+1)."""
                        x_ref = current_pos[point_idx]
                        sdf_vals = sdf_matrix[:, point_idx]
                        
                        # Select candidate obstacles (simplified version)
                        threshold = clearance + constraint_margin
                        sorted_indices = jnp.argsort(sdf_vals)
                        
                        # Get candidate indices (simplified version)
                        k = jnp.minimum(max_constraints_per_point, sdf_vals.shape[0])
                        cand_indices = jnp.where(
                            jnp.arange(max_constraints_per_point) < k,
                            sorted_indices[jnp.arange(max_constraints_per_point)],
                            0
                        )
                        
                        # Get gradients
                        grads = grad_fn(x_ref, cand_indices)  # (max_constraints_per_point, dim)
                        
                        # Build constraints: each row is [g, b] where constraint is g^T x >= b
                        def build_constraint(j):
                            """Build constraint j."""
                            obs_idx = cand_indices[j]
                            grad = grads[j]
                            d0 = sdf_vals[obs_idx]
                            
                            gnorm = jnp.linalg.norm(grad)
                            is_valid = jnp.logical_and(jnp.isfinite(gnorm), gnorm >= 1e-8)
                            
                            g = grad / jnp.maximum(gnorm, 1e-12)
                            b_val = (clearance - d0) / jnp.maximum(gnorm, 1e-12) + jnp.dot(g, x_ref)
                            
                            # Return [g, b] padded to (dim+1,)
                            constraint = jnp.concatenate([g, jnp.array([b_val])])
                            return jnp.where(is_valid, constraint, jnp.zeros(dim + 1, dtype=jnp.float32))
                        
                        constraints = jax.vmap(build_constraint)(jnp.arange(max_constraints_per_point))
                        return constraints  # (max_constraints_per_point, dim+1)
                    
                    # Build constraints for all points
                    all_point_constraints = jax.vmap(build_point_constraints)(jnp.arange(N))
                    # Shape: (N, max_constraints_per_point, dim+1)
                    
                    # Flatten and extract A and b
                    # Reshape to (N * max_constraints_per_point, dim+1)
                    constraints_flat = all_point_constraints.reshape(-1, dim + 1)
                    A_dense = constraints_flat[:, :dim]  # (N * max_constraints_per_point, dim)
                    b_dense = constraints_flat[:, dim]    # (N * max_constraints_per_point,)
                    
                    # Filter out zero rows (invalid constraints)
                    row_norms = jnp.linalg.norm(A_dense, axis=1)
                    valid_rows = row_norms > 1e-8
                    
                    # Build sparse format for trajectory QP
                    # For simplicity, use all rows (invalid ones will have very negative b)
                    A_valid = jnp.where(valid_rows[:, None], A_dense, jnp.zeros((max_cons, dim), dtype=jnp.float32))
                    b_valid = jnp.where(valid_rows, b_dense, jnp.full((max_cons,), -1e6, dtype=jnp.float32))
                    
                    # Convert to sparse COO format
                    def build_sparse_entry(cons_idx):
                        """Build sparse matrix entry for constraint cons_idx."""
                        point_idx = cons_idx // max_constraints_per_point
                        
                        # Get constraint data
                        a_row = A_valid[cons_idx]  # (dim,)
                        b_val = b_valid[cons_idx]
                        
                        # Build sparse entries: for each dimension, we have one entry
                        values = a_row  # (dim,)
                        rows = jnp.full(dim, cons_idx, dtype=jnp.int32)  # (dim,)
                        cols = point_idx * dim + jnp.arange(dim)  # (dim,)
                        
                        # Stack into (dim, 3) array: [value, row, col]
                        entries = jnp.stack([values, rows.astype(jnp.float32), cols.astype(jnp.float32)], axis=1)
                        return entries, b_val
                    
                    # Build all sparse entries
                    all_entries, all_b = jax.vmap(build_sparse_entry)(jnp.arange(max_cons))
                    
                    # Flatten to get A_data, A_row, A_col
                    A_data = all_entries[:, :, 0].flatten()  # (max_cons * dim,)
                    A_row = all_entries[:, :, 1].flatten().astype(jnp.int32)  # (max_cons * dim,)
                    A_col = all_entries[:, :, 2].flatten().astype(jnp.int32)  # (max_cons * dim,)
                    
                    # Filter out zero entries (from invalid constraints)
                    nonzero_mask = jnp.abs(A_data) > 1e-10
                    A_data = jnp.where(nonzero_mask, A_data, 0.0)
                    A_row = jnp.where(nonzero_mask, A_row, 0)
                    A_col = jnp.where(nonzero_mask, A_col, 0)
                    
                    # Solve trajectory QP
                    x_flat = prev.reshape(-1)
                    try:
                        x_new_flat = _solve_trajectory_qp_identity_jax(
                            x0=x_flat,
                            A_data=A_data,
                            A_row=A_row,
                            A_col=A_col,
                            b=all_b,
                            n_vars=N * dim,
                            n_cons=max_cons,
                            dim=dim,
                            T=N,
                            smoothness_weight=smoothness_weight,
                        )
                        return x_new_flat.reshape(N, dim)
                    except Exception:
                        # If trajectory QP fails, fall back to pointwise
                        return prev
                
                def use_pointwise():
                    """Use pointwise projection."""
                    def project_point(idx):
                        """Project a single point if it's violating."""
                        is_violating = violating_mask[idx]
                        sdf_vals = sdf_matrix[:, idx]
                        projected = project_single_point(current_pos[idx], sdf_vals, sdf_matrix.shape[0])
                        return jnp.where(is_violating, projected, current_pos[idx])
                    
                    return jax.vmap(project_point)(jnp.arange(N))
                
                # Use trajectory QP if enabled, we have multiple points, and qpax is available
                can_use_traj_qp = jnp.logical_and(
                    use_trajectory_qp,
                    jnp.logical_and(N > 1, QPAX_AVAILABLE)
                )
                
                return jax.lax.cond(
                    can_use_traj_qp,
                    solve_traj_qp,
                    use_pointwise
                )
            
            # Use trajectory QP or pointwise
            new_pos = try_trajectory_qp()
            
            # Check convergence
            max_step = jnp.max(jnp.linalg.norm(new_pos - prev, axis=1))
            converged = max_step < convergence_tol
            
            # Final feasibility check if converged
            def check_final_feasibility(pos):
                sdf_matrix_final = sdf_fn(pos)
                union_sdf_final = jnp.min(sdf_matrix_final, axis=0)
                return jnp.all(union_sdf_final >= clearance)
            
            final_feasible = jax.lax.cond(
                converged,
                check_final_feasibility,
                lambda _: jnp.array(False),
                new_pos
            )
            
            # Return new position
            return new_pos
        
        # Run iterations using fori_loop
        current = jax.lax.fori_loop(0, max_iterations, iteration_body, current)
        
        return current
    
    
    # ============================================================================
    # CFSProjectionJax Class
    # ============================================================================
    
    @register_projection("cfs", "jax")
    class CFSProjectionJax(CFSProjectionBase):
        """
        JAX backend implementation of CFS projection.
        
        This implementation uses JAX operations and qpax for QP solving,
        optimized for GPU acceleration and JIT compilation.
        """
        
        def __init__(self, obstacles: ObstacleManager, **config):
            """
            Initialize JAX backend CFS projection.
            
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
            
            # Build JAX-compatible obstacle functions
            self._sdf_fn = None
            self._grad_fn = None
            self._build_obstacle_functions()
        
        def _build_obstacle_functions(self):
            """Build JAX-compatible SDF and gradient functions from obstacles."""
            obstacles_list = list(self.obstacles)
            if len(obstacles_list) == 0:
                # No obstacles: create identity functions
                def sdf_batch(points):
                    return jnp.full((1, points.shape[0]), 1e6, dtype=jnp.float32)
                
                def grad_multiple(point, obs_indices):
                    return jnp.zeros((obs_indices.shape[0], point.shape[0]), dtype=jnp.float32)
                
                self._sdf_fn = sdf_batch
                self._grad_fn = grad_multiple
                return
            
            # Check if all obstacles support JAX
            all_support_jax = all(hasattr(obs, "jax_sdf") for obs in obstacles_list)
            
            if all_support_jax:
                # Use per-obstacle SDF and gradients
                def sdf_batch(points: jnp.ndarray) -> jnp.ndarray:
                    """Compute SDF for all obstacles."""
                    sdf_rows = []
                    for obs in obstacles_list:
                        sdf_vals = obs.jax_sdf(points)
                        sdf_vals = jnp.asarray(sdf_vals, dtype=jnp.float32)
                        if sdf_vals.ndim == 0:
                            sdf_vals = jnp.broadcast_to(sdf_vals, (points.shape[0],))
                        elif sdf_vals.ndim > 1:
                            sdf_vals = sdf_vals.flatten()[:points.shape[0]]
                        elif sdf_vals.shape[0] != points.shape[0]:
                            if sdf_vals.shape[0] < points.shape[0]:
                                last_val = sdf_vals[-1] if sdf_vals.shape[0] > 0 else jnp.array(1e6, dtype=jnp.float32)
                                padding = jnp.full((points.shape[0] - sdf_vals.shape[0],), last_val, dtype=jnp.float32)
                                sdf_vals = jnp.concatenate([sdf_vals, padding])
                            else:
                                sdf_vals = sdf_vals[:points.shape[0]]
                        sdf_rows.append(sdf_vals)
                    
                    if not sdf_rows:
                        return jnp.full((1, points.shape[0]), 1e6, dtype=jnp.float32)
                    
                    return jnp.stack(sdf_rows, axis=0)  # (M, N)
                
                def grad_multiple(point: jnp.ndarray, obs_indices: jnp.ndarray) -> jnp.ndarray:
                    """Compute gradients for specific obstacles."""
                    num_obstacles = len(obstacles_list)
                    
                    def make_grad_fn_for_obs(obstacle):
                        """Create a gradient function for a specific obstacle."""
                        if hasattr(obstacle, "jax_gradient"):
                            def grad_fn(p):
                                return obstacle.jax_gradient(p)
                            return grad_fn
                        else:
                            def grad_fn(p):
                                return _finite_difference_gradient_jax(obstacle, p)
                            return grad_fn
                    
                    grad_fns = [make_grad_fn_for_obs(obs) for obs in obstacles_list]
                    
                    def compute_grad_for_idx(idx: jnp.ndarray) -> jnp.ndarray:
                        """Compute gradient for obstacle at index idx."""
                        idx_int = jnp.clip(jnp.asarray(idx, dtype=jnp.int32), 0, num_obstacles - 1)
                        branches = tuple(grad_fns)
                        grad = jax.lax.switch(idx_int, branches, point)
                        return jnp.asarray(grad, dtype=jnp.float32).flatten()
                    
                    grads = jax.vmap(compute_grad_for_idx)(obs_indices)
                    return grads
                
                self._sdf_fn = sdf_batch
                self._grad_fn = grad_multiple
            else:
                raise RuntimeError(
                    "Not all obstacles support JAX. "
                    "All obstacles must have 'jax_sdf' method to use JAX CFS projection."
                )
        
        def project_batch(
            self,
            positions: np.ndarray,
            clearance: float,
            step: Optional[int],
            **kwargs
        ) -> np.ndarray:
            """
            Batch project multiple points onto CFS using JAX.
            
            Args:
                positions: Points to project, shape (N, dim)
                clearance: Minimum clearance required
                step: Current step (optional)
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
            
            # Convert to JAX arrays
            positions_jax = jnp.asarray(positions, dtype=jnp.float32)
            clearance_jax = jnp.asarray(clearance, dtype=jnp.float32)
            
            # Call JAX projection function
            projected_jax = _project_cfs_jax(
                positions_jax,
                self._sdf_fn,
                self._grad_fn,
                clearance_jax,
                max_iterations=max_iterations,
                convergence_tol=convergence_tol,
                max_constraints_per_point=max_constraints_per_point,
                constraint_margin=constraint_margin,
                use_trajectory_qp=use_trajectory_qp,
                smoothness_weight=smoothness_weight,
            )
            
            # Convert back to NumPy
            return np.asarray(projected_jax)
        
        def solve_projection_qp(
            self,
            x0: np.ndarray,
            A: np.ndarray,
            b: np.ndarray,
        ) -> np.ndarray:
            """
            Solve pointwise projection QP using JAX.
            
            Args:
                x0: Reference point, shape (dim,)
                A: Constraint matrix, shape (m, dim)
                b: Constraint vector, shape (m,)
                
            Returns:
                Projected point, shape (dim,)
            """
            # Convert to JAX arrays
            x0_jax = jnp.asarray(x0, dtype=jnp.float32)
            A_jax = jnp.asarray(A, dtype=jnp.float32)
            b_jax = jnp.asarray(b, dtype=jnp.float32)
            
            # Solve using JAX QP solver
            result_jax = _solve_projection_qp_identity_jax(x0_jax, A_jax, b_jax)
            
            # Convert back to NumPy
            return np.asarray(result_jax)

else:
    # JAX not available - don't register the class
    pass

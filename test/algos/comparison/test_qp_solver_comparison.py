"""
Test script to compare cvxopt vs qpax vs JAX enumeration QP solvers.

This test isolates the QP solver differences to understand how much
the solver choice affects CFS projection results.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

try:
    import qpax
    QPAX_AVAILABLE = True
except ImportError:
    QPAX_AVAILABLE = False
    qpax = None


def solve_qp_cvxopt(x0: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve QP using cvxopt (what CFS uses for trajectory-level QP).
    
    Problem: minimize 0.5 ||x - x0||^2 subject to A x >= b
    
    cvxopt format: minimize 0.5 x^T P x + q^T x subject to G x <= h
    So: P = I, q = -x0, G = -A, h = -b
    
    This is the same QP formulation that CFS uses for trajectory-level projection.
    """
    if not CVXOPT_AVAILABLE:
        raise RuntimeError("cvxopt not available")
    
    x0 = np.asarray(x0, dtype=np.float64).flatten()
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).flatten()
    dim = x0.shape[0]
    
    # Identity Hessian: P = I
    P = cvx_matrix(np.eye(dim, dtype=np.float64))
    
    # Linear term: q = -x0 (from 0.5||x-x0||^2 = 0.5||x||^2 - x^T x0 + 0.5||x0||^2)
    q = cvx_matrix((-x0).astype(np.float64))
    
    # Constraints: A x >= b  =>  -A x <= -b
    G = cvx_matrix((-A).astype(np.float64))
    h = cvx_matrix((-b).astype(np.float64))
    
    # Solve QP
    old_show = cvx_solvers.options.get("show_progress", True)
    cvx_solvers.options["show_progress"] = False
    try:
        sol = cvx_solvers.qp(P, q, G, h)
    finally:
        cvx_solvers.options["show_progress"] = old_show
    
    if sol is None or sol.get("status", "") not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"cvxopt qp failed: status={None if sol is None else sol.get('status')}")
    
    x = np.asarray(sol["x"], dtype=np.float32).reshape(-1)
    return x


def solve_qp_numpy_enumeration(x0: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    NumPy enumeration method (from CFSProjection._solve_projection_qp_identity).
    
    This is what CFS currently uses for single-point projection (not trajectory-level).
    """
    from itertools import combinations
    
    x0 = np.asarray(x0, dtype=np.float32).flatten()
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32).flatten()
    m, dim = A.shape
    
    def is_feasible(A, b, x, tol=1e-7):
        if A.size == 0:
            return True
        lhs = A @ x
        return bool(np.all(lhs + tol >= b))
    
    # If already feasible, return x0
    if is_feasible(A, b, x0):
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
        if is_feasible(A, b, x):
            obj = float(np.sum((x - x0) ** 2))
            if obj < best_obj:
                best_obj = obj
                best_x = x
    
    # Active sets of size 2..dim
    max_k = min(dim, m)
    for k in range(2, max_k + 1):
        for idxs in combinations(range(m), k):
            AI = A[list(idxs)]
            bI = b[list(idxs)]
            rhs = bI - (AI @ x0)
            M = AI @ AI.T
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
            if is_feasible(A, b, x):
                obj = float(np.sum((x - x0) ** 2))
                if obj < best_obj:
                    best_obj = obj
                    best_x = x.astype(np.float32)
    
    # Fallback
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


def solve_qp_qpax(x0: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray) -> np.ndarray:
    """
    qpax method - JAX-compatible QP solver.
    
    This is what the JAX version should use when qpax is available.
    qpax supports general inequality constraints G x <= h.
    """
    if not QPAX_AVAILABLE:
        raise RuntimeError("qpax not available")
    
    dim = x0.shape[0]
    
    # Convert to qpax format: min 0.5 x^T Q x + q^T x s.t. G x <= h
    Q = jnp.eye(dim, dtype=jnp.float32)
    q = -x0
    G = -A  # A x >= b  =>  -A x <= -b
    h = -b
    
    # No equality constraints - use empty arrays instead of None
    A_eq = jnp.zeros((0, dim), dtype=jnp.float32)  # Empty matrix for no equality constraints
    b_eq = jnp.zeros((0,), dtype=jnp.float32)  # Empty vector
    
    try:
        x, s, z, y, converged, iters = qpax.solve_qp(
            Q, q, A_eq, b_eq, G, h,
            solver_tol=1e-6
        )
        if not converged:
            raise RuntimeError(f"qpax did not converge after {iters} iterations")
        return np.asarray(x, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"qpax solve failed: {e}")


def solve_qp_jax_simplified(x0: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray) -> np.ndarray:
    """
    JAX simplified method - iterative projection onto violated constraints.
    
    This iteratively projects onto violated constraints until feasible.
    Better than single-constraint projection but still suboptimal.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float32).flatten()
    A = jnp.asarray(A, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32).flatten()
    m, dim = A.shape
    
    def is_feasible(x):
        lhs = A @ x
        return jnp.all(lhs + 1e-7 >= b)
    
    # Check if already feasible
    feasible_init = is_feasible(x0)
    
    def return_original():
        return x0
    
    def project_iteratively():
        # Iterative projection: repeatedly project onto most violated constraint
        current = x0
        max_iter = 20
        
        def project_step(i, x):
            # Find most violated constraint
            violations = b - (A @ x)
            i_max = jnp.argmax(violations)
            a = A[i_max]
            denom = jnp.dot(a, a)
            denom = jnp.where(denom < 1e-12, 1e-12, denom)
            alpha = (b[i_max] - jnp.dot(a, x)) / denom
            x_new = x + alpha * a
            
            # Check if feasible now
            feasible = is_feasible(x_new)
            # If feasible, keep returning this value; otherwise continue
            return jnp.where(feasible, x_new, x_new)
        
        # Run iterations
        result = jax.lax.fori_loop(0, max_iter, project_step, current)
        return result
    
    # Use cond to handle initial feasibility
    result = jax.lax.cond(feasible_init, return_original, project_iteratively)
    
    return np.asarray(result, dtype=np.float32)


def test_qp_solver_comparison():
    """Compare different QP solvers on the same problem."""
    print("=" * 80)
    print("QP Solver Comparison Test: cvxopt vs NumPy Enumeration vs qpax vs JAX Simplified")
    print("=" * 80)
    print("Note: CFS uses cvxopt for trajectory-level QP, enumeration for single-point QP")
    print("=" * 80)
    
    # Test case 1: Simple 2D projection
    print("\n--- Test Case 1: Simple 2D Projection ---")
    x0 = np.array([0.5, 0.5], dtype=np.float32)
    A = np.array([
        [1.0, 0.0],   # x >= 0.8
        [0.0, 1.0],   # y >= 0.8
        [1.0, 1.0],   # x + y >= 1.2
    ], dtype=np.float32)
    b = np.array([0.8, 0.8, 1.2], dtype=np.float32)
    
    print(f"Initial point: {x0}")
    print(f"Constraints: A x >= b")
    print(f"A:\n{A}")
    print(f"b: {b}")
    
    # cvxopt (used by CFS for trajectory-level QP)
    if CVXOPT_AVAILABLE:
        try:
            result_cvxopt = solve_qp_cvxopt(x0, A, b)
            print(f"\ncvxopt: {result_cvxopt}")
        except Exception as e:
            print(f"\ncvxopt failed: {e}")
            result_cvxopt = None
    else:
        print("\ncvxopt not available")
        result_cvxopt = None
    
    # NumPy enumeration (used by CFS for single-point QP)
    try:
        result_enum = solve_qp_numpy_enumeration(x0, A, b)
        print(f"NumPy (enumeration): {result_enum}")
        if result_cvxopt is not None:
            diff = np.linalg.norm(result_cvxopt - result_enum)
            print(f"  Difference from cvxopt: {diff:.6f}")
    except Exception as e:
        print(f"NumPy enumeration failed: {e}")
        result_enum = None
    
    if JAX_AVAILABLE:
        x0_jax = jnp.asarray(x0)
        A_jax = jnp.asarray(A)
        b_jax = jnp.asarray(b)
        
        # qpax (JAX-compatible QP solver)
        if QPAX_AVAILABLE:
            try:
                result_qpax = solve_qp_qpax(x0_jax, A_jax, b_jax)
                print(f"qpax: {result_qpax}")
                if result_cvxopt is not None:
                    diff = np.linalg.norm(result_cvxopt - result_qpax)
                    print(f"  Difference from cvxopt: {diff:.6f}")
                if result_enum is not None:
                    diff = np.linalg.norm(result_enum - result_qpax)
                    print(f"  Difference from NumPy enumeration: {diff:.6f}")
            except Exception as e:
                print(f"qpax failed: {e}")
                result_qpax = None
        else:
            print("qpax not available")
            result_qpax = None
        
        # JAX simplified
        result_simplified = solve_qp_jax_simplified(x0_jax, A_jax, b_jax)
        print(f"JAX (simplified): {result_simplified}")
        if result_cvxopt is not None:
            diff = np.linalg.norm(result_cvxopt - result_simplified)
            print(f"  Difference from cvxopt: {diff:.6f}")
        if result_enum is not None:
            diff = np.linalg.norm(result_enum - result_simplified)
            print(f"  Difference from NumPy enumeration: {diff:.6f}")
    
    # Test case 2: More complex case with multiple constraints
    print("\n--- Test Case 2: Multiple Constraints (4 constraints, 2D) ---")
    x0 = np.array([0.0, 0.0], dtype=np.float32)
    A = np.array([
        [1.0, 0.0],   # x >= 0.5
        [0.0, 1.0],   # y >= 0.5
        [-1.0, 0.0],  # -x >= -1.0  => x <= 1.0
        [0.0, -1.0],  # -y >= -1.0  => y <= 1.0
    ], dtype=np.float32)
    b = np.array([0.5, 0.5, -1.0, -1.0], dtype=np.float32)
    
    print(f"Initial point: {x0}")
    print(f"Constraints: A x >= b")
    
    if CVXOPT_AVAILABLE:
        try:
            result_cvxopt = solve_qp_cvxopt(x0, A, b)
            print(f"\ncvxopt: {result_cvxopt}")
        except Exception as e:
            print(f"\ncvxopt failed: {e}")
            result_cvxopt = None
    else:
        result_cvxopt = None
    
    try:
        result_enum = solve_qp_numpy_enumeration(x0, A, b)
        print(f"NumPy (enumeration): {result_enum}")
        if result_cvxopt is not None:
            diff = np.linalg.norm(result_cvxopt - result_enum)
            print(f"  Difference from cvxopt: {diff:.6f}")
    except Exception as e:
        result_enum = None
    
    if JAX_AVAILABLE:
        x0_jax = jnp.asarray(x0)
        A_jax = jnp.asarray(A)
        b_jax = jnp.asarray(b)
        
        if QPAX_AVAILABLE:
            try:
                result_qpax = solve_qp_qpax(x0_jax, A_jax, b_jax)
                print(f"qpax: {result_qpax}")
                if result_cvxopt is not None:
                    diff = np.linalg.norm(result_cvxopt - result_qpax)
                    print(f"  Difference from cvxopt: {diff:.6f}")
                if result_enum is not None:
                    diff = np.linalg.norm(result_enum - result_qpax)
                    print(f"  Difference from NumPy enumeration: {diff:.6f}")
            except Exception as e:
                print(f"qpax failed: {e}")
        
        result_simplified = solve_qp_jax_simplified(x0_jax, A_jax, b_jax)
        print(f"JAX (simplified): {result_simplified}")
        if result_cvxopt is not None:
            diff = np.linalg.norm(result_cvxopt - result_simplified)
            print(f"  Difference from cvxopt: {diff:.6f}")
        if result_enum is not None:
            diff = np.linalg.norm(result_enum - result_simplified)
            print(f"  Difference from NumPy enumeration: {diff:.6f}")
    
    # Test case 3: Near-obstacle case (typical CFS scenario)
    print("\n--- Test Case 3: Near-Obstacle Case (CFS-like) ---")
    x0 = np.array([0.2, 0.2], dtype=np.float32)  # Point near obstacle
    # Constraints representing linearized obstacle boundaries
    A = np.array([
        [0.707, 0.707],    # Normal pointing away from obstacle
        [0.866, 0.5],      # Another constraint
        [0.5, 0.866],      # Another constraint
    ], dtype=np.float32)
    b = np.array([0.6, 0.5, 0.5], dtype=np.float32)
    
    print(f"Initial point: {x0}")
    
    if CVXOPT_AVAILABLE:
        try:
            result_cvxopt = solve_qp_cvxopt(x0, A, b)
            print(f"\ncvxopt: {result_cvxopt}")
        except Exception as e:
            print(f"\ncvxopt failed: {e}")
            result_cvxopt = None
    else:
        result_cvxopt = None
    
    try:
        result_enum = solve_qp_numpy_enumeration(x0, A, b)
        print(f"NumPy (enumeration): {result_enum}")
        if result_cvxopt is not None:
            diff = np.linalg.norm(result_cvxopt - result_enum)
            print(f"  Difference from cvxopt: {diff:.6f}")
    except Exception as e:
        result_enum = None
    
    if JAX_AVAILABLE:
        x0_jax = jnp.asarray(x0)
        A_jax = jnp.asarray(A)
        b_jax = jnp.asarray(b)
        
        if QPAX_AVAILABLE:
            try:
                result_qpax = solve_qp_qpax(x0_jax, A_jax, b_jax)
                print(f"qpax: {result_qpax}")
                if result_cvxopt is not None:
                    diff = np.linalg.norm(result_cvxopt - result_qpax)
                    print(f"  Difference from cvxopt: {diff:.6f}")
                if result_enum is not None:
                    diff = np.linalg.norm(result_enum - result_qpax)
                    print(f"  Difference from NumPy enumeration: {diff:.6f}")
            except Exception as e:
                print(f"qpax failed: {e}")
        
        result_simplified = solve_qp_jax_simplified(x0_jax, A_jax, b_jax)
        print(f"JAX (simplified): {result_simplified}")
        if result_cvxopt is not None:
            diff = np.linalg.norm(result_cvxopt - result_simplified)
            print(f"  Difference from cvxopt: {diff:.6f}")
        if result_enum is not None:
            diff = np.linalg.norm(result_enum - result_simplified)
            print(f"  Difference from NumPy enumeration: {diff:.6f}")
    
    # Test case 4: Infeasible case (point inside all constraints)
    print("\n--- Test Case 4: Point Inside All Constraints ---")
    x0 = np.array([0.0, 0.0], dtype=np.float32)
    A = np.array([
        [1.0, 0.0],   # x >= 0.5
        [0.0, 1.0],   # y >= 0.5
    ], dtype=np.float32)
    b = np.array([0.5, 0.5], dtype=np.float32)
    
    print(f"Initial point: {x0} (inside constraints)")
    
    if CVXOPT_AVAILABLE:
        try:
            result_cvxopt = solve_qp_cvxopt(x0, A, b)
            print(f"\ncvxopt: {result_cvxopt}")
        except Exception as e:
            print(f"\ncvxopt failed: {e}")
            result_cvxopt = None
    else:
        result_cvxopt = None
    
    try:
        result_enum = solve_qp_numpy_enumeration(x0, A, b)
        print(f"NumPy (enumeration): {result_enum}")
        if result_cvxopt is not None:
            diff = np.linalg.norm(result_cvxopt - result_enum)
            print(f"  Difference from cvxopt: {diff:.6f}")
    except Exception as e:
        result_enum = None
    
    if JAX_AVAILABLE:
        x0_jax = jnp.asarray(x0)
        A_jax = jnp.asarray(A)
        b_jax = jnp.asarray(b)
        
        if QPAX_AVAILABLE:
            try:
                result_qpax = solve_qp_qpax(x0_jax, A_jax, b_jax)
                print(f"qpax: {result_qpax}")
                if result_cvxopt is not None:
                    diff = np.linalg.norm(result_cvxopt - result_qpax)
                    print(f"  Difference from cvxopt: {diff:.6f}")
                if result_enum is not None:
                    diff = np.linalg.norm(result_enum - result_qpax)
                    print(f"  Difference from NumPy enumeration: {diff:.6f}")
            except Exception as e:
                print(f"qpax failed: {e}")
        
        result_simplified = solve_qp_jax_simplified(x0_jax, A_jax, b_jax)
        print(f"JAX (simplified): {result_simplified}")
        if result_cvxopt is not None:
            diff = np.linalg.norm(result_cvxopt - result_simplified)
            print(f"  Difference from cvxopt: {diff:.6f}")
        if result_enum is not None:
            diff = np.linalg.norm(result_enum - result_simplified)
            print(f"  Difference from NumPy enumeration: {diff:.6f}")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("  - cvxopt: General QP solver (used by CFS for trajectory-level QP)")
    print("  - NumPy enumeration: Active set enumeration (used by CFS for single-point QP)")
    print("  - qpax: JAX-compatible QP solver (supports general inequalities)")
    print("  - JAX simplified: Only projects to most violated constraint (suboptimal fallback)")
    print("=" * 80)


if __name__ == "__main__":
    test_qp_solver_comparison()

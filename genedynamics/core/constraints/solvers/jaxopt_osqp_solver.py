"""
jaxopt.OSQP QP solver backend.

This solver uses jaxopt.OSQP for high-performance QP solving
with JIT compilation and GPU acceleration.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

try:
    from jaxopt import OSQP
    JAXOPT_AVAILABLE = True
except ImportError:
    OSQP = None
    JAXOPT_AVAILABLE = False

from .base import QPSolver
from genedynamics.core.constraints.core.registry import register


def solve_slack_qp_jax(
    u_nom: "jnp.ndarray",
    A: "jnp.ndarray",
    b: "jnp.ndarray",
    rho: "jnp.ndarray | float",
    *,
    control_limit: Optional[float] = None,
    tol: float = 1e-6,
    maxiter: int = 10,
) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """
    JAX-native slack-QP solve (JIT-safe; no NumPy conversions).

    Solve slack-QP:
        min ||u - u_nom||^2 + ρ||ξ||^2
        s.t. A u >= b - ξ, ξ >= 0
             -L <= u <= L   (optional; applied via clipping)

    Notes:
    - Constraints with non-finite b (e.g. -inf) are treated as inactive.
    - Returns (u_star, max_violation) where max_violation ignores inactive constraints.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available for solve_slack_qp_jax")

    u_nom = jnp.asarray(u_nom, dtype=jnp.float32)
    A = jnp.asarray(A, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)

    # Treat non-finite b as inactive constraints.
    valid = jnp.isfinite(b)

    if control_limit is None:
        u0 = u_nom
        L_lo = -jnp.inf
        L_hi = jnp.inf
    else:
        L = jnp.asarray(float(control_limit), dtype=jnp.float32)
        L_lo = -L
        L_hi = L
        u0 = jnp.clip(u_nom, L_lo, L_hi)

    rho_eff = jnp.asarray(rho, dtype=jnp.float32)
    rho_eff = jnp.maximum(rho_eff, 1e-9)
    tol_j = jnp.asarray(float(tol), dtype=jnp.float32)
    maxiter_j = jnp.asarray(int(maxiter), dtype=jnp.int32)
    maxiter_j = jnp.clip(maxiter_j, 0, 10_000)

    def max_violation(u: jnp.ndarray) -> jnp.ndarray:
        # Single matvec per evaluation; treat inactive constraints as -inf.
        viol = jnp.maximum(0.0, jnp.where(valid, b - A @ u, -jnp.inf))
        return jnp.max(viol)

    # Performance note:
    # The previous implementation used a while_loop with two A@u matvecs per iteration
    # (one for selecting the worst constraint, one for recomputing max violation).
    # Here we use a fixed-iteration fori_loop and structure the update so each
    # iteration does only ONE A@u matvec, plus we precompute row norms.
    #
    # This is intentionally "projection style" (POCS-like), not a full QP solve.
    # It is meant to be extremely fast in per-step filters.

    # Fast path uses fori_loop (best performance) but requires static bounds.
    # If maxiter is traced (cannot be converted to Python int), fall back to while_loop.
    try:
        maxiter_i = int(maxiter)
        maxiter_i = max(0, min(maxiter_i, 10_000))
        use_fori = True
    except Exception:
        use_fori = False
        maxiter_i = 0  # unused in this branch

    # Precompute squared row norms (safe even if some rows are dummy/zero).
    row_norm_sq = jnp.sum(A * A, axis=1) + 1e-9

    def body_fn(_i, u):
        lhs = A @ u  # one matvec per iteration
        viol = jnp.where(valid, b - lhs, -jnp.inf)
        viol = jnp.maximum(0.0, viol)
        idx = jnp.argmax(viol)
        v = viol[idx]
        A_row = A[idx]
        # Slack-like damping: larger rho => closer to hard projection.
        den = row_norm_sq[idx] + 1.0 / rho_eff
        lam = v / den
        u_next = jnp.clip(u + lam * A_row, L_lo, L_hi)
        # If already feasible within tol, do a no-op (keeps loop JIT-friendly).
        return jax.lax.cond(v > tol_j, lambda _: u_next, lambda _: u, operand=None)

    def run_fori(_):
        u_star = jax.lax.fori_loop(0, maxiter_i, body_fn, u0)
        v_star = max_violation(u_star)
        return u_star, v_star

    def run_while(_):
        # Dynamic maxiter support. We keep one matvec per iteration, and
        # compute the final violation once at the end.
        def cond_fn(carry):
            i, u = carry
            # Early stop check uses current u.
            v = max_violation(u)
            return jnp.logical_and(i < maxiter_j, v > tol_j)

        def body_while(carry):
            i, u = carry
            return (i + 1, body_fn(i, u))

        _, u_star = jax.lax.while_loop(
            cond_fn, body_while, (jnp.asarray(0, dtype=jnp.int32), u0)
        )
        v_star = max_violation(u_star)
        return u_star, v_star

    return jax.lax.cond(use_fori, run_fori, run_while, operand=None)


def solve_hard_qp_jax(
    u_nom: "jnp.ndarray",
    A: "jnp.ndarray",
    b: "jnp.ndarray",
    *,
    control_limit: Optional[float] = None,
    tol: float = 1e-6,
    maxiter: int = 10,
) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """
    JAX-native hard-QP solve (JIT-safe; no NumPy conversions).

        min ||u - u_nom||^2
        s.t. A u >= b
             -L <= u <= L   (optional; applied via clipping)
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available for solve_hard_qp_jax")

    u_nom = jnp.asarray(u_nom, dtype=jnp.float32)
    A = jnp.asarray(A, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)

    valid = jnp.isfinite(b)

    if control_limit is None:
        u0 = u_nom
        L_lo = -jnp.inf
        L_hi = jnp.inf
    else:
        L = jnp.asarray(float(control_limit), dtype=jnp.float32)
        L_lo = -L
        L_hi = L
        u0 = jnp.clip(u_nom, L_lo, L_hi)

    tol_j = jnp.asarray(float(tol), dtype=jnp.float32)
    maxiter_j = jnp.asarray(int(maxiter), dtype=jnp.int32)
    maxiter_j = jnp.clip(maxiter_j, 0, 10_000)

    def max_violation(u: jnp.ndarray) -> jnp.ndarray:
        viol = jnp.maximum(0.0, jnp.where(valid, b - A @ u, -jnp.inf))
        return jnp.max(viol)

    v0 = max_violation(u0)

    def cond_fn(carry):
        i, _u, v = carry
        return jnp.logical_and(i < maxiter_j, v > tol_j)

    def body_fn(carry):
        i, u, _v = carry
        viol = jnp.maximum(0.0, jnp.where(valid, b - A @ u, -jnp.inf))
        idx = jnp.argmax(viol)
        A_row = A[idx]
        den = jnp.dot(A_row, A_row) + 1e-9
        lam = viol[idx] / den
        u_next = jnp.clip(u + lam * A_row, L_lo, L_hi)
        v_next = max_violation(u_next)
        return (i + 1, u_next, v_next)

    _, u_star, v_star = jax.lax.while_loop(
        cond_fn, body_fn, (jnp.asarray(0, dtype=jnp.int32), u0, v0)
    )
    return u_star, v_star


def solve_slack_qp_prefixsum_jax(
    u_nom: "jnp.ndarray",  # (H, act_dim)
    A_per_step: "jnp.ndarray",  # (H, K, act_dim) constraint normal in action-space
    b_per_step: "jnp.ndarray",  # (H, K) rhs
    rho: "jnp.ndarray | float",
    *,
    control_limit: Optional[float] = None,
    tol: float = 1e-6,
    maxiter: int = 40,
) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """
    Structured slack-QP solver for *prefix-sum coupled* constraints (single-integrator style).

    We assume constraints are:
        sum_{i=0}^t <a_{t,k}, u_i> >= b_{t,k} - xi_{t,k},  xi_{t,k} >= 0
    which is equivalent to:
        <a_{t,k}, cumsum(u)[t]> >= b_{t,k} - xi_{t,k}

    This avoids materializing the dense A_full matrix and expensive A_full @ u matvecs.
    It uses the same greedy most-violated projection updates as the generic solver.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available for solve_slack_qp_prefixsum_jax")

    u_nom = jnp.asarray(u_nom, dtype=jnp.float32)
    A_per_step = jnp.asarray(A_per_step, dtype=jnp.float32)
    b_per_step = jnp.asarray(b_per_step, dtype=jnp.float32)

    H, K, act_dim = A_per_step.shape
    assert u_nom.shape == (H, act_dim)
    assert b_per_step.shape == (H, K)

    valid = jnp.isfinite(b_per_step)  # (H, K)

    if control_limit is None:
        u0 = u_nom
        L_lo = -jnp.inf
        L_hi = jnp.inf
    else:
        L = jnp.asarray(float(control_limit), dtype=jnp.float32)
        L_lo = -L
        L_hi = L
        u0 = jnp.clip(u_nom, L_lo, L_hi)

    rho_eff = jnp.asarray(rho, dtype=jnp.float32)
    rho_eff = jnp.maximum(rho_eff, 1e-9)
    tol_j = jnp.asarray(float(tol), dtype=jnp.float32)
    maxiter_j = jnp.asarray(int(maxiter), dtype=jnp.int32)
    maxiter_j = jnp.clip(maxiter_j, 0, 10_000)

    def max_violation(u: jnp.ndarray) -> jnp.ndarray:
        # prefix sum over time: (H, act_dim)
        u_prefix = jnp.cumsum(u, axis=0)
        # lhs: (H, K)
        lhs = jnp.einsum("hkd,hd->hk", A_per_step, u_prefix)
        viol = jnp.maximum(0.0, jnp.where(valid, b_per_step - lhs, -jnp.inf))
        return jnp.max(viol)

    v0 = max_violation(u0)

    # Fast path: use fori_loop when maxiter is a Python int (best XLA performance).
    # If maxiter is traced (e.g. from adaptive scheduler), fall back to while_loop.
    try:
        maxiter_i = int(maxiter)
        maxiter_i = max(0, min(maxiter_i, 10_000))
        use_fori = True
    except Exception:
        use_fori = False
        maxiter_i = 0

    def body_fn(carry):
        i, u, _v = carry
        u_prefix = jnp.cumsum(u, axis=0)  # (H, act_dim)
        lhs = jnp.einsum("hkd,hd->hk", A_per_step, u_prefix)  # (H, K)
        viol_hk = jnp.maximum(0.0, jnp.where(valid, b_per_step - lhs, -jnp.inf))  # (H, K)

        flat_idx = jnp.argmax(viol_hk.reshape(-1))
        t_idx = flat_idx // jnp.asarray(K, dtype=jnp.int32)
        k_idx = flat_idx - t_idx * jnp.asarray(K, dtype=jnp.int32)

        a = A_per_step[t_idx, k_idx]  # (act_dim,)
        v = viol_hk[t_idx, k_idx]

        # Denominator in flattened space: repeated a across (t_idx+1) action blocks
        den = (jnp.asarray(t_idx + 1, dtype=jnp.float32) * jnp.dot(a, a)) + 1e-9
        lam = v / (den + 1.0 / rho_eff)

        prefix_mask = (jnp.arange(H, dtype=jnp.int32) <= t_idx).astype(jnp.float32)  # (H,)
        u_next = u + (prefix_mask[:, None] * (lam * a[None, :]))
        u_next = jnp.clip(u_next, L_lo, L_hi)
        v_next = max_violation(u_next)
        return (i + 1, u_next, v_next)

    def cond_fn(carry):
        i, _u, v = carry
        return jnp.logical_and(i < maxiter_j, v > tol_j)

    def run_fori(_):
        # fori_loop body: (i, u) -> u, one cumsum+einsum per iteration, early-stop via cond
        def body_fori(_i, u):
            u_prefix = jnp.cumsum(u, axis=0)
            lhs = jnp.einsum("hkd,hd->hk", A_per_step, u_prefix)
            viol_hk = jnp.maximum(0.0, jnp.where(valid, b_per_step - lhs, -jnp.inf))
            flat_idx = jnp.argmax(viol_hk.reshape(-1))
            t_idx = flat_idx // jnp.asarray(K, dtype=jnp.int32)
            k_idx = flat_idx - t_idx * jnp.asarray(K, dtype=jnp.int32)
            a = A_per_step[t_idx, k_idx]
            v = viol_hk[t_idx, k_idx]
            den = (jnp.asarray(t_idx + 1, dtype=jnp.float32) * jnp.dot(a, a)) + 1e-9
            lam = v / (den + 1.0 / rho_eff)
            prefix_mask = (jnp.arange(H, dtype=jnp.int32) <= t_idx).astype(jnp.float32)
            u_next = jnp.clip(
                u + (prefix_mask[:, None] * (lam * a[None, :])), L_lo, L_hi
            )
            do_update = jnp.logical_and(_i < maxiter_i, v > tol_j)
            return jax.lax.cond(do_update, lambda _: u_next, lambda _: u, operand=None)
        u_star = jax.lax.fori_loop(0, maxiter_i, body_fori, u0)
        v_star = max_violation(u_star)
        return u_star, v_star

    def run_while(_):
        _, u_star, v_star = jax.lax.while_loop(
            cond_fn, body_fn, (jnp.asarray(0, dtype=jnp.int32), u0, v0)
        )
        return u_star, v_star

    u_star, v_star = jax.lax.cond(use_fori, run_fori, run_while, operand=None)
    return u_star, v_star


try:
    from genedynamics.solvers.single.cfsmbd.backends._fast_prefixsum_qp import (
        solve_slack_qp_prefixsum_jax as _fast_solve_slack_qp_prefixsum_jax,
    )
    solve_slack_qp_prefixsum_jax = _fast_solve_slack_qp_prefixsum_jax
except ImportError:
    pass  # keep the implementation above


@register("solver", "jaxopt_osqp", "jax")
class JAXOPTOsqpSolver(QPSolver):
    """
    QP solver using jaxopt.OSQP (JAX-based).
    
    This solver provides:
    - JIT compilation for fast repeated solves
    - GPU acceleration
    - Automatic differentiation support
    """
    
    def __init__(
        self,
        use_jit: bool = True,
        **kwargs
    ):
        """
        Initialize jaxopt.OSQP solver.
        
        Args:
            use_jit: Whether to JIT compile solve function
            **kwargs: Additional jaxopt.OSQP options
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available. Install JAX to use JAXOPTOsqpSolver.")
        if not JAXOPT_AVAILABLE:
            raise RuntimeError("jaxopt.OSQP not available. Install jaxopt to use JAXOPTOsqpSolver.")
        
        self.use_jit = use_jit
        self.kwargs = kwargs
        
        # JIT compile solve function if requested
        if use_jit:
            self._solve_fn = jax.jit(self._solve_qp_jax)
        else:
            self._solve_fn = self._solve_qp_jax
    
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
        Solve QP using jaxopt.OSQP.
        
        Args:
            P: Quadratic cost matrix (n, n)
            q: Linear cost vector (n,)
            G: Inequality constraint matrix (m, n)
            h: Inequality constraint RHS (m,)
            A_eq: Equality constraint matrix (p, n)
            b_eq: Equality constraint RHS (p,)
            lb: Lower bounds (n,)
            ub: Upper bounds (n,)
            **kwargs: Additional jaxopt.OSQP options
            
        Returns:
            Tuple of (solution, info)
        """
        # Convert to JAX arrays
        P_jax = jnp.asarray(P, dtype=jnp.float32)
        q_jax = jnp.asarray(q, dtype=jnp.float32)
        
        G_jax = jnp.asarray(G, dtype=jnp.float32) if G is not None else None
        h_jax = jnp.asarray(h, dtype=jnp.float32) if h is not None else None
        A_eq_jax = jnp.asarray(A_eq, dtype=jnp.float32) if A_eq is not None else None
        b_eq_jax = jnp.asarray(b_eq, dtype=jnp.float32) if b_eq is not None else None
        lb_jax = jnp.asarray(lb, dtype=jnp.float32) if lb is not None else None
        ub_jax = jnp.asarray(ub, dtype=jnp.float32) if ub is not None else None
        
        # Merge kwargs
        solver_kwargs = {**self.kwargs, **kwargs}
        
        # Solve
        solution_jax, info_jax = self._solve_fn(
            P_jax, q_jax, G_jax, h_jax, A_eq_jax, b_eq_jax, lb_jax, ub_jax, **solver_kwargs
        )
        
        # Convert back to numpy
        solution = np.asarray(solution_jax)
        info = {
            'status': 'optimal' if info_jax.get('status', 0) == 0 else 'unknown',
            'iterations': info_jax.get('iterations', 0),
            **{k: np.asarray(v) for k, v in info_jax.items() if k not in ['status', 'iterations']}
        }
        
        return solution, info
    
    def _solve_qp_jax(
        self,
        P: jnp.ndarray,
        q: jnp.ndarray,
        G: Optional[jnp.ndarray],
        h: Optional[jnp.ndarray],
        A_eq: Optional[jnp.ndarray],
        b_eq: Optional[jnp.ndarray],
        lb: Optional[jnp.ndarray],
        ub: Optional[jnp.ndarray],
        **kwargs
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Internal JAX-based solve function.
        
        This is the function that gets JIT compiled.
        Uses jaxopt.OSQP for solving QP problems.
        """
        if not JAXOPT_AVAILABLE or OSQP is None:
            raise RuntimeError("jaxopt.OSQP is required but not available")
        
        n = len(q)
        
        # jaxopt.OSQP expects: min (1/2) x^T P x + q^T x
        #                       s.t. G x <= h (inequality)
        #                            A_eq x = b_eq (equality)
        # Note: jaxopt.OSQP doesn't directly support bounds, so we need to add them as constraints
        
        # Handle bounds by adding them as inequality constraints
        if lb is not None or ub is not None:
            if lb is None:
                lb = jnp.full(n, -jnp.inf)
            if ub is None:
                ub = jnp.full(n, jnp.inf)
            
            # Add bounds as: -I x <= -lb  and  I x <= ub
            # This gives: lb <= x <= ub
            I = jnp.eye(n)
            G_lb = -I  # -x <= -lb  =>  x >= lb
            h_lb = -lb
            G_ub = I   # x <= ub
            h_ub = ub
            
            # Combine with existing inequality constraints
            if G is not None and G.size > 0:
                G_combined = jnp.vstack([G, G_lb, G_ub])
                h_combined = jnp.concatenate([h, h_lb, h_ub])
            else:
                G_combined = jnp.vstack([G_lb, G_ub])
                h_combined = jnp.concatenate([h_lb, h_ub])
            
            G = G_combined
            h = h_combined
        
        # Merge kwargs with defaults
        solver_kwargs = {
            'tol': 1e-6,
            'maxiter': 30,
            'check_primal_dual_infeasability': True,
            **self.kwargs,
            **kwargs
        }
        
        # Filter out parameters that OSQP doesn't accept
        # These are typically passed from operator but not needed by OSQP
        # jaxopt.OSQP only accepts: tol, maxiter, check_primal_dual_infeasability, etc.
        # Remove all custom parameters that might be passed from operator
        invalid_params = [
            'use_jit', 'project_states', 'project_actions', 'fix_initial_state',
            'initial_state', 'smoothness_weight', 'use_slack', 'rho',
            'convergence_tol', 'max_iterations', 'solver_backend'
        ]
        for param in invalid_params:
            solver_kwargs.pop(param, None)
        
        # Solve using jaxopt.OSQP
        try:
            solver = OSQP(**solver_kwargs)
            solution_obj, state = solver.run(
                params_obj=(P, q),
                params_ineq=(G, h) if G is not None and G.size > 0 else None,
                params_eq=(A_eq, b_eq) if A_eq is not None and A_eq.size > 0 else None,
            )
            
            # Extract solution
            if hasattr(solution_obj, 'primal'):
                solution = solution_obj.primal
            else:
                solution = solution_obj
            
            # Check if solution is valid
            solution_valid = jnp.all(jnp.isfinite(solution))
            if not solution_valid:
                raise RuntimeError("jaxopt.OSQP produced invalid solution (non-finite values)")
            
            info = {
                'status': 0 if hasattr(state, 'converged') and state.converged else 1,
                'iterations': state.iter_num if hasattr(state, 'iter_num') else 0,
            }
            
            return solution, info
        except Exception as e:
            raise RuntimeError(f"jaxopt.OSQP failed: {e}")
    
    def solve_slack_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        rho: float
    ) -> Tuple[np.ndarray, float]:
        """
        Solve slack-QP: min ||u - u_nom||^2 + ρ||ξ||^2 s.t. A u >= b - ξ, ξ >= 0.
        
        Args:
            u_nom: Nominal action (n,)
            A: Constraint matrix (m, n)
            b: Constraint vector (m,)
            rho: Slack penalty weight
            
        Returns:
            Tuple of (optimal action, violation)
        """
        if A.size == 0:
            return u_nom, 0.0
        
        # Convert to JAX arrays
        u_nom_jax = jnp.asarray(u_nom, dtype=jnp.float32)
        A_jax = jnp.asarray(A, dtype=jnp.float32)
        b_jax = jnp.asarray(b, dtype=jnp.float32)
        
        dim = u_nom.shape[0]
        m = A.shape[0]
        
        # Extended problem: [u; ξ]
        # Q = [I, 0; 0, ρ*I]
        Q = jnp.block([
            [jnp.eye(dim, dtype=jnp.float32), jnp.zeros((dim, m), dtype=jnp.float32)],
            [jnp.zeros((m, dim), dtype=jnp.float32), rho * jnp.eye(m, dtype=jnp.float32)]
        ])
        
        # q = [-u_nom; 0]
        q = jnp.concatenate([-u_nom_jax, jnp.zeros(m, dtype=jnp.float32)])
        
        # G = [-A, I; 0, -I] for A u >= b - ξ, ξ >= 0
        # => -A u - ξ <= -b, -ξ <= 0
        G = jnp.block([
            [-A_jax, jnp.eye(m, dtype=jnp.float32)],
            [jnp.zeros((m, dim), dtype=jnp.float32), -jnp.eye(m, dtype=jnp.float32)]
        ])
        
        # h = [-b; 0]
        h = jnp.concatenate([-b_jax, jnp.zeros(m, dtype=jnp.float32)])
        
        # Solve using jaxopt.OSQP
        if JAXOPT_AVAILABLE and OSQP is not None:
            try:
                solver = OSQP(
                    tol=1e-6,
                    maxiter=30,    
                    check_primal_dual_infeasability=True,
                )
                
                solution, state = solver.run(
                    params_obj=(Q, q),
                    params_ineq=(G, h),
                    params_eq=None,
                )
                
                # Extract primal solution
                if hasattr(solution, 'primal'):
                    x = solution.primal
                else:
                    x = solution
                
                # Check if solution is valid
                if not jnp.all(jnp.isfinite(x)):
                    raise RuntimeError("jaxopt.OSQP slack-QP produced invalid solution")
                
                u_star = np.asarray(x[:dim])
                violation = float(np.maximum(0, b - A @ u_star).max())
                return u_star, violation
            except Exception as e:
                raise RuntimeError(f"jaxopt.OSQP slack-QP failed: {e}")
        else:
            # Final fallback: return nominal with violation
            violation = float(np.maximum(0, b - A @ u_nom).max())
            return u_nom, violation
    
    def solve_hard_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Solve hard-QP: min ||u - u_nom||^2 s.t. A u >= b.
        
        Args:
            u_nom: Nominal action (n,)
            A: Constraint matrix (m, n)
            b: Constraint vector (m,)
            
        Returns:
            Tuple of (optimal action, violation)
        """
        if A.size == 0:
            return u_nom, 0.0
        
        # Convert to JAX arrays
        u_nom_jax = jnp.asarray(u_nom, dtype=jnp.float32)
        A_jax = jnp.asarray(A, dtype=jnp.float32)
        b_jax = jnp.asarray(b, dtype=jnp.float32)
        
        dim = u_nom.shape[0]
        
        # QP formulation: min 0.5||u - u_nom||^2 = 0.5 u^T I u - u_nom^T u
        Q = jnp.eye(dim, dtype=jnp.float32)
        q = -u_nom_jax
        
        # Convert A u >= b to -A u <= -b
        G = -A_jax
        h = -b_jax
        
        # Filter out zero rows (invalid constraints)
        row_norms = jnp.linalg.norm(A_jax, axis=1)
        valid_mask = row_norms > 1e-8
        
        if jnp.sum(valid_mask.astype(jnp.int32)) == 0:
            # No valid constraints
            return u_nom, 0.0
        
        # Use masked constraints
        G_masked = jnp.where(valid_mask[:, None], G, jnp.zeros_like(G))
        h_masked = jnp.where(valid_mask, h, jnp.full_like(h, -1e6))
        
        # Check if already feasible
        lhs = A_jax @ u_nom_jax
        feasible = jnp.all(lhs + 1e-7 >= b_jax)
        
        if feasible:
            return u_nom, 0.0
        
        # Solve using jaxopt.OSQP
        if JAXOPT_AVAILABLE and OSQP is not None:
            try:
                solver = OSQP(
                    tol=1e-6,
                    maxiter=30,
                    check_primal_dual_infeasability=True,
                )
                
                solution, state = solver.run(
                    params_obj=(Q, q),
                    params_ineq=(G_masked, h_masked),
                    params_eq=None,
                )
                
                # Extract primal solution
                if hasattr(solution, 'primal'):
                    u_star_jax = solution.primal
                else:
                    u_star_jax = solution
                
                # Check if solution is valid
                if not jnp.all(jnp.isfinite(u_star_jax)):
                    # Try with looser tolerance
                    solver_retry = OSQP(
                        tol=1e-5,
                        maxiter=30,
                        check_primal_dual_infeasability=True,
                    )
                    solution, state = solver_retry.run(
                        params_obj=(Q, q),
                        params_ineq=(G_masked, h_masked),
                        params_eq=None,
                    )
                    if hasattr(solution, 'primal'):
                        u_star_jax = solution.primal
                    else:
                        u_star_jax = solution
                    
                    if not jnp.all(jnp.isfinite(u_star_jax)):
                        raise RuntimeError("jaxopt.OSQP hard-QP produced invalid solution")
                
                u_star = np.asarray(u_star_jax)
                violation = float(np.maximum(0, b - A @ u_star).max())
                return u_star, violation
            except Exception as e:
                raise RuntimeError(f"jaxopt.OSQP hard-QP failed: {e}")
        else:
            # Fallback: simple projection
            u_star = u_nom.copy()
            violation = float(np.maximum(0, b - A @ u_star).max())
            
            # Simple iterative projection
            for _ in range(10):
                violations = np.maximum(0, b - A @ u_star)
                if violations.max() < 1e-7:
                    break
                
                # Push in direction of violated constraints
                for i in range(A.shape[0]):
                    if violations[i] > 0:
                        n = A[i] / (np.linalg.norm(A[i]) + 1e-8)
                        u_star += violations[i] * n
            
            violation = float(np.maximum(0, b - A @ u_star).max())
            return u_star, violation
    
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
        Solve trajectory QP with smoothness term using JAX native operations.
        
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
            **kwargs: Additional solver options
            
        Returns:
            Tuple of (optimized positions, info)
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available for solve_traj_qp_with_smoothness")
        
        n = len(x_nom)
        w = float(max(0.0, smoothness_weight))
        
        # Convert to JAX arrays
        x_nom_jax = jnp.asarray(x_nom, dtype=jnp.float32)
        A_jax = jnp.asarray(A, dtype=jnp.float32) if A.size > 0 else None
        b_jax = jnp.asarray(b, dtype=jnp.float32) if b.size > 0 else None
        
        # Build Hessian: P = I + w * (D2^T D2) kron I_dim
        # Use JAX native operations
        if w > 0.0 and T >= 3:
            # Build D2^T D2 directly using its known structure
            # More efficient: build using array operations instead of loops
            K = jnp.zeros((T, T), dtype=jnp.float32)
            
            # Main diagonal: [1, 5, 6, ..., 6, 5, 1]
            diag_vals = jnp.ones(T, dtype=jnp.float32) * 6.0
            diag_vals = diag_vals.at[0].set(1.0)
            diag_vals = diag_vals.at[1].set(5.0)
            if T > 2:
                diag_vals = diag_vals.at[T-2].set(5.0)
                diag_vals = diag_vals.at[T-1].set(1.0)
            K = K.at[jnp.arange(T), jnp.arange(T)].set(diag_vals)
            
            # First off-diagonal: [-2, -4, ..., -4, -2]
            off1_vals = jnp.ones(T-1, dtype=jnp.float32) * -4.0
            off1_vals = off1_vals.at[0].set(-2.0)
            if T > 2:
                off1_vals = off1_vals.at[-1].set(-2.0)
            K = K.at[jnp.arange(T-1), jnp.arange(1, T)].set(off1_vals)
            K = K.at[jnp.arange(1, T), jnp.arange(T-1)].set(off1_vals)
            
            # Second off-diagonal: [1, 1, ..., 1]
            off2_vals = jnp.ones(T-2, dtype=jnp.float32)
            K = K.at[jnp.arange(T-2), jnp.arange(2, T)].set(off2_vals)
            K = K.at[jnp.arange(2, T), jnp.arange(T-2)].set(off2_vals)
            
            # P_time = I + w * K
            P_time = jnp.eye(T, dtype=jnp.float32) + w * K
            
            # P = P_time kron I_dim
            # Use jnp.kron for efficiency (JAX-optimized)
            I_dim = jnp.eye(dim, dtype=jnp.float32)
            P = jnp.kron(P_time, I_dim)
        else:
            # No smoothness term
            P = jnp.eye(n, dtype=jnp.float32)
        
        # Linear term: q = -x_nom
        q = -x_nom_jax
        
        # Build equality constraints for fixing initial state (if enabled)
        # Use equality constraint: A_eq x = b_eq where A_eq = [I_dim, 0, 0, ...]
        A_eq = None
        b_eq = None
        if fix_initial_state and initial_state is not None:
            initial_state_jax = jnp.asarray(initial_state, dtype=jnp.float32).flatten()
            if initial_state_jax.shape[0] != dim:
                raise ValueError(
                    f"initial_state has shape {initial_state_jax.shape}, expected ({dim},)"
                )
            
            # Equality constraint: x[0:dim] = initial_state
            # A_eq: [I_dim, 0, 0, ...] where I_dim is identity matrix for first dim variables
            A_eq = jnp.zeros((dim, n), dtype=jnp.float32)
            A_eq = A_eq.at[:, :dim].set(jnp.eye(dim, dtype=jnp.float32))
            b_eq = initial_state_jax
        
        # Solve QP with smoothness
        if use_slack and rho is not None:
            # Slack-QP: min 0.5 x^T P x + q^T x + ρ||ξ||^2 s.t. A x >= b - ξ, ξ >= 0
            m = len(b) if b.size > 0 else 0
            n_total = n + m
            
            # Extended P matrix
            P_ext = jnp.zeros((n_total, n_total), dtype=jnp.float32)
            P_ext = P_ext.at[:n, :n].set(P)
            P_ext = P_ext.at[n:, n:].set(rho * jnp.eye(m, dtype=jnp.float32))
            
            # Extended q vector
            q_ext = jnp.zeros(n_total, dtype=jnp.float32)
            q_ext = q_ext.at[:n].set(q)
            
            # Extended constraints: A x + ξ >= b, ξ >= 0
            # => -A x - ξ <= -b, -ξ <= 0
            G_ext = jnp.zeros((2 * m, n_total), dtype=jnp.float32)
            h_ext = jnp.zeros(2 * m, dtype=jnp.float32)
            
            if A_jax is not None and A_jax.size > 0:
                # -A x - ξ <= -b
                G_ext = G_ext.at[:m, :n].set(-A_jax)
                G_ext = G_ext.at[:m, n:].set(-jnp.eye(m, dtype=jnp.float32))
                h_ext = h_ext.at[:m].set(-b_jax)
            
            # -ξ <= 0 (i.e., ξ >= 0)
            G_ext = G_ext.at[m:, n:].set(-jnp.eye(m, dtype=jnp.float32))
            h_ext = h_ext.at[m:].set(0.0)
            
            # Extended equality constraints for initial state (if enabled)
            # For slack-QP: [A_eq, 0] [x; ξ] = b_eq
            A_eq_ext = None
            b_eq_ext = None
            if A_eq is not None and A_eq.size > 0:
                # A_eq is (dim, n), extend to (dim, n_total) with zeros for slack variables
                A_eq_ext = jnp.zeros((A_eq.shape[0], n_total), dtype=jnp.float32)
                A_eq_ext = A_eq_ext.at[:, :n].set(A_eq)
                b_eq_ext = b_eq
            
            # Solve using jaxopt.OSQP
            if JAXOPT_AVAILABLE and OSQP is not None:
                # Filter out parameters that OSQP doesn't accept
                # Use same defaults as _solve_qp_jax for consistency
                osqp_kwargs = {
                    'tol': 1e-6,
                    'maxiter': 30,
                    'check_primal_dual_infeasability': True,
                    **self.kwargs,
                    **kwargs
                }
                # Remove parameters that OSQP doesn't accept
                # These are typically passed from operator but not needed by OSQP
                osqp_kwargs.pop('use_jit', None)
                osqp_kwargs.pop('project_states', None)
                osqp_kwargs.pop('project_actions', None)
                osqp_kwargs.pop('fix_initial_state', None)
                osqp_kwargs.pop('initial_state', None)
                osqp_kwargs.pop('smoothness_weight', None)
                osqp_kwargs.pop('use_slack', None)
                
                solver = OSQP(**osqp_kwargs)
                
                solution_obj, state = solver.run(
                    params_obj=(P_ext, q_ext),
                    params_ineq=(G_ext, h_ext) if G_ext.size > 0 else None,
                    params_eq=(A_eq_ext, b_eq_ext) if A_eq_ext is not None and A_eq_ext.size > 0 else None,
                )
                
                solution_ext = solution_obj.primal if hasattr(solution_obj, 'primal') else solution_obj
                solution = solution_ext[:n]
                info = {
                    'status': 'optimal' if hasattr(state, 'converged') and state.converged else 'unknown',
                    'iterations': state.iter_num if hasattr(state, 'iter_num') else 0,
                    'slack': np.asarray(solution_ext[n:])
                }
            else:
                raise RuntimeError("jaxopt.OSQP not available for solve_traj_qp_with_smoothness")
        else:
            # Hard-QP: min 0.5 x^T P x + q^T x s.t. A x >= b
            # Convert A x >= b to -A x <= -b
            G = -A_jax if A_jax is not None and A_jax.size > 0 else None
            h = -b_jax if b_jax is not None and b_jax.size > 0 else None
            
            if JAXOPT_AVAILABLE and OSQP is not None:
                # Filter out parameters that OSQP doesn't accept
                # Use same defaults as _solve_qp_jax for consistency
                osqp_kwargs = {
                    'tol': 1e-6,
                    'maxiter': 30,
                    'check_primal_dual_infeasability': True,
                    **self.kwargs,
                    **kwargs
                }
                # Remove parameters that OSQP doesn't accept
                # These are typically passed from operator but not needed by OSQP
                osqp_kwargs.pop('use_jit', None)
                osqp_kwargs.pop('project_states', None)
                osqp_kwargs.pop('project_actions', None)
                osqp_kwargs.pop('fix_initial_state', None)
                osqp_kwargs.pop('initial_state', None)
                osqp_kwargs.pop('smoothness_weight', None)
                osqp_kwargs.pop('use_slack', None)
                
                solver = OSQP(**osqp_kwargs)
                
                solution_obj, state = solver.run(
                    params_obj=(P, q),
                    params_ineq=(G, h) if G is not None and G.size > 0 else None,
                    params_eq=(A_eq, b_eq) if A_eq is not None and A_eq.size > 0 else None,
                )
                
                solution = solution_obj.primal if hasattr(solution_obj, 'primal') else solution_obj
                info = {
                    'status': 'optimal' if hasattr(state, 'converged') and state.converged else 'unknown',
                    'iterations': state.iter_num if hasattr(state, 'iter_num') else 0,
                }
            else:
                raise RuntimeError("jaxopt.OSQP not available for solve_traj_qp_with_smoothness")
        
        # Convert back to numpy
        return np.asarray(solution), info


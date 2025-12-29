"""
qpax / jaxopt QP solver backend.

This solver uses qpax (JAX-based QP solver) for high-performance QP solving
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
    import qpax
    QPAX_AVAILABLE = True
except ImportError:
    QPAX_AVAILABLE = False
    qpax = None

from .base import QPSolver


class QPAXSolver(QPSolver):
    """
    QP solver using qpax (JAX-based).
    
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
        Initialize qpax solver.
        
        Args:
            use_jit: Whether to JIT compile solve function
            **kwargs: Additional qpax options
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available. Install JAX to use QPAXSolver.")
        if not QPAX_AVAILABLE:
            raise RuntimeError("qpax not available. Install qpax to use QPAXSolver.")
        
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
        Solve QP using qpax.
        
        Args:
            P: Quadratic cost matrix (n, n)
            q: Linear cost vector (n,)
            G: Inequality constraint matrix (m, n)
            h: Inequality constraint RHS (m,)
            A_eq: Equality constraint matrix (p, n)
            b_eq: Equality constraint RHS (p,)
            lb: Lower bounds (n,)
            ub: Upper bounds (n,)
            **kwargs: Additional qpax options
            
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
        """
        # Use qpax to solve
        # Note: qpax API may vary, adjust as needed
        try:
            # Try qpax API
            result = qpax.solve_qp(
                P=P,
                q=q,
                G=G,
                h=h,
                A=A_eq,
                b=b_eq,
                lb=lb,
                ub=ub,
                **kwargs
            )
            
            solution = result.primal
            info = {
                'status': result.status,
                'iterations': getattr(result, 'iterations', 0),
            }
            
            return solution, info
        except Exception as e:
            # Fallback: use jaxopt
            try:
                from jaxopt import BoxOSQP
                
                # Convert to box constraints if needed
                if lb is not None or ub is not None:
                    # Combine bounds with inequality constraints
                    n = len(q)
                    if lb is None:
                        lb = jnp.full(n, -jnp.inf)
                    if ub is None:
                        ub = jnp.full(n, jnp.inf)
                    
                    # BoxOSQP expects: min (1/2) x^T P x + q^T x s.t. lb <= x <= ub, G x <= h
                    solver = BoxOSQP()
                    solution, state = solver.run(
                        params_obj=(P, q),
                        params_ineq=(G, h) if G is not None else (None, None),
                        params_eq=(A_eq, b_eq) if A_eq is not None else (None, None),
                        init_params=jnp.zeros(n),
                    )
                    
                    info = {
                        'status': state.state if hasattr(state, 'state') else 0,
                        'iterations': state.iter_num if hasattr(state, 'iter_num') else 0,
                    }
                    
                    return solution, info
            except ImportError:
                raise RuntimeError(f"Failed to solve QP: {e}")



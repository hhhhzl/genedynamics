"""
Per-step QP filter operator.

Solves a QP for each time step independently to enforce per-step constraints
(e.g., CBF constraints that depend on current state).
"""

import time
from typing import Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.core.constraints.operators.base import Operator
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    OperatorInfo,
)
from enerdynamics.core.constraints.core.registry import get_registry, register
from enerdynamics.core.types import Trajectory

try:
    import qpax
    QPAX_AVAILABLE = True
except ImportError:
    QPAX_AVAILABLE = False


@register("operator", "per_step_qp", "numpy")
class PerStepQPFilter(Operator):
    """
    Per-step QP filter: Enforces constraints by solving QP at each time step.
    
    Supports:
    - Hard mode: Strictly satisfy A u >= b
    - Slack mode: min ||u - u_nom||^2 + ρ||ξ||^2 s.t. A u >= b - ξ
    
    Designed for per-step constraints like CBF where each time step
    has independent constraints.
    """
    
    def __init__(
        self,
        use_slack: bool = True,
        solver_backend: str = "numpy",
        **kwargs
    ):
        """
        Initialize per-step QP filter.
        
        Args:
            use_slack: If True, use slack-QP; if False, use hard-QP
            solver_backend: QP solver backend ("numpy", "jax", "qpax")
            **kwargs: Additional arguments
        """
        self.use_slack = use_slack
        self.solver_backend = solver_backend
        
        # Get solver from registry
        registry = get_registry()
        solver_class = registry.get("solver", "qp", solver_backend)
        if solver_class is not None:
            self.solver = solver_class(**kwargs)
        else:
            self.solver = None
    
    def apply(
        self,
        nominal: Trajectory,
        constraints: ConvexConstraint,
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[Trajectory, OperatorInfo]:
        """
        Apply per-step QP filter.
        
        Args:
            nominal: Nominal trajectory
            constraints: Convex constraints (should be per-step)
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            Tuple of (repaired trajectory, operator info)
        """
        start_time = time.time()
        
        # Check if constraints are per-step
        if not constraints.is_per_step():
            # If not per-step, treat as single constraint for all steps
            return self._apply_single_constraint(nominal, constraints, params, state, start_time)
        
        # Per-step constraints: solve QP for each time step
        repaired_actions = []
        violation_before = 0.0
        violation_after = 0.0
        qp_count = 0
        
        # Get constraints - handle BackendArray wrapper
        from enerdynamics.core.constraints.core.array_interface import BackendArray
        
        # Use BackendArray.to_numpy() method which handles all conversions properly
        if isinstance(constraints.A, BackendArray):
            A = constraints.A.to_numpy()
        else:
            if JAX_AVAILABLE:
                try:
                    if hasattr(constraints.A, 'block_until_ready'):
                        constraints.A.block_until_ready()
                    A = np.asarray(jax.device_get(constraints.A), dtype=np.float32)
                except (TypeError, AttributeError, ValueError):
                    A = np.asarray(constraints.A, dtype=np.float32)
            else:
                A = np.asarray(constraints.A, dtype=np.float32)
        
        if isinstance(constraints.b, BackendArray):
            b = constraints.b.to_numpy()
        else:
            if JAX_AVAILABLE:
                try:
                    if hasattr(constraints.b, 'block_until_ready'):
                        constraints.b.block_until_ready()
                    b = np.asarray(jax.device_get(constraints.b), dtype=np.float32)
                except (TypeError, AttributeError, ValueError):
                    b = np.asarray(constraints.b, dtype=np.float32)
            else:
                b = np.asarray(constraints.b, dtype=np.float32)
        
        # Handle batched per-step constraints: A shape (H, m, dim) or (m, dim)
        if A.ndim == 3:
            # Batched: (H, m, dim)
            H = A.shape[0]
            for t in range(min(H, len(nominal.actions))):
                A_t = A[t]  # (m, dim)
                b_t = b[t] if b.ndim > 1 else b  # (m,)
                u_nom = np.asarray(nominal.actions[t], dtype=np.float32)
                
                # Solve QP
                u_star, violation = self._solve_per_step_qp(u_nom, A_t, b_t, params)
                repaired_actions.append(u_star)
                
                violation_before += np.maximum(0, b_t - A_t @ u_nom).max() if A_t.size > 0 else 0.0
                violation_after += violation
                qp_count += 1
        else:
            # Single constraint matrix: apply to all steps
            for t, u_nom in enumerate(nominal.actions):
                u_nom = np.asarray(u_nom, dtype=np.float32)
                
                # Solve QP
                u_star, violation = self._solve_per_step_qp(u_nom, A, b, params)
                repaired_actions.append(u_star)
                
                violation_before += np.maximum(0, b - A @ u_nom).max() if A.size > 0 else 0.0
                violation_after += violation
                qp_count += 1
        
        # Create repaired trajectory
        repaired = Trajectory(
            states=nominal.states,
            actions=repaired_actions,
            info=nominal.info
        )
        
        elapsed_time = time.time() - start_time
        
        info = OperatorInfo(
            success=True,
            violation_before=violation_before / max(len(nominal.actions), 1),
            violation_after=violation_after / max(len(nominal.actions), 1),
            iterations=qp_count,
            time=elapsed_time,
            extra={"qp_count": qp_count}
        )
        
        return repaired, info
    
    def _apply_single_constraint(
        self,
        nominal: Trajectory,
        constraints: ConvexConstraint,
        params: ScheduleParams,
        state: ScheduleState,
        start_time: float
    ) -> Tuple[Trajectory, OperatorInfo]:
        """Apply single constraint to all steps."""
        A = np.asarray(constraints.A, dtype=np.float32)
        b = np.asarray(constraints.b, dtype=np.float32)
        
        repaired_actions = []
        violation_before = 0.0
        violation_after = 0.0
        
        for u_nom in nominal.actions:
            u_nom = np.asarray(u_nom, dtype=np.float32)
            u_star, violation = self._solve_per_step_qp(u_nom, A, b, params)
            repaired_actions.append(u_star)
            violation_before += np.maximum(0, b - A @ u_nom).max() if A.size > 0 else 0.0
            violation_after += violation
        
        repaired = Trajectory(
            states=nominal.states,
            actions=repaired_actions,
            info=nominal.info
        )
        
        elapsed_time = time.time() - start_time
        
        info = OperatorInfo(
            success=True,
            violation_before=violation_before / len(nominal.actions),
            violation_after=violation_after / len(nominal.actions),
            iterations=len(nominal.actions),
            time=elapsed_time
        )
        
        return repaired, info
    
    def _solve_per_step_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        params: ScheduleParams
    ) -> Tuple[np.ndarray, float]:
        """
        Solve per-step QP.
        
        Args:
            u_nom: Nominal action
            A: Constraint matrix (m, dim)
            b: Constraint vector (m,)
            params: Schedule parameters
            
        Returns:
            Tuple of (optimal action, violation)
        """
        if A.size == 0:
            # No constraints
            return u_nom, 0.0
        
        # Check feasibility
        violation = np.maximum(0, b - A @ u_nom).max()
        if violation < 1e-7:
            # Already feasible
            return u_nom, 0.0
        
        if self.use_slack:
            # Slack-QP: min ||u - u_nom||^2 + ρ||ξ||^2 s.t. A u >= b - ξ, ξ >= 0
            return self._solve_slack_qp(u_nom, A, b, params)
        else:
            # Hard-QP: min ||u - u_nom||^2 s.t. A u >= b
            return self._solve_hard_qp(u_nom, A, b)
    
    def _solve_slack_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        params: ScheduleParams
    ) -> Tuple[np.ndarray, float]:
        """Solve slack-QP."""
        if self.solver is not None:
            # Check if solver has solve_slack_qp method
            if hasattr(self.solver, 'solve_slack_qp'):
                return self.solver.solve_slack_qp(u_nom, A, b, params.rho)
            # Otherwise use solve_least_squares_with_constraints
            elif hasattr(self.solver, 'solve_least_squares_with_constraints'):
                solution, info = self.solver.solve_least_squares_with_constraints(
                    u_nom, A, b, rho=params.rho
                )
                violation = float(np.maximum(0, b - A @ solution).max())
                return solution, violation
        
        # Fallback: use qpax if available
        if QPAX_AVAILABLE:
            try:
                import jax.numpy as jnp
                u_nom_jax = jnp.asarray(u_nom)
                A_jax = jnp.asarray(A)
                b_jax = jnp.asarray(b)
                
                # Extended problem: [u; ξ]
                dim = u_nom.shape[0]
                m = A.shape[0]
                
                # Q = [I, 0; 0, ρ*I]
                Q = jnp.block([
                    [jnp.eye(dim), jnp.zeros((dim, m))],
                    [jnp.zeros((m, dim)), params.rho * jnp.eye(m)]
                ])
                
                # q = [-u_nom; 0]
                q = jnp.concatenate([-u_nom_jax, jnp.zeros(m)])
                
                # G = [-A, I; 0, -I] for A u >= b - ξ, ξ >= 0
                G = jnp.block([
                    [-A_jax, jnp.eye(m)],
                    [jnp.zeros((m, dim)), -jnp.eye(m)]
                ])
                
                # h = [-b; 0]
                h = jnp.concatenate([-b_jax, jnp.zeros(m)])
                
                # Solve
                A_eq = jnp.zeros((0, dim + m))
                b_eq = jnp.zeros(0)
                
                x, _, _, _, converged, _ = qpax.solve_qp(Q, q, A_eq, b_eq, G, h)
                
                if converged:
                    u_star = np.asarray(x[:dim])
                    violation = np.maximum(0, b - A @ u_star).max()
                    return u_star, violation
            except:
                pass
        
        # Final fallback: simple projection
        return self._solve_hard_qp(u_nom, A, b)
    
    def _solve_hard_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Solve hard-QP (projection onto feasible set)."""
        if self.solver is not None:
            # Check if solver has solve_hard_qp method
            if hasattr(self.solver, 'solve_hard_qp'):
                return self.solver.solve_hard_qp(u_nom, A, b)
            # Otherwise use solve_least_squares_with_constraints
            elif hasattr(self.solver, 'solve_least_squares_with_constraints'):
                solution, info = self.solver.solve_least_squares_with_constraints(
                    u_nom, A, b, rho=None
                )
                violation = float(np.maximum(0, b - A @ solution).max())
                return solution, violation
        
        # Fallback: simple clipping (not optimal but works)
        u_star = u_nom.copy()
        violation = np.maximum(0, b - A @ u_star).max()
        
        # Simple iterative projection (can be improved)
        for _ in range(10):
            violations = np.maximum(0, b - A @ u_star)
            if violations.max() < 1e-7:
                break
            
            # Push in direction of violated constraints
            for i in range(A.shape[0]):
                if violations[i] > 0:
                    n = A[i] / (np.linalg.norm(A[i]) + 1e-8)
                    u_star += violations[i] * n
        
        violation = np.maximum(0, b - A @ u_star).max()
        return u_star, violation


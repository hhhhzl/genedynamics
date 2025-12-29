"""
Full-horizon trajectory QP filter.

This operator solves a single QP over the entire trajectory horizon,
enforcing trajectory-level constraints (e.g., CFS constraints).
"""

import time
from typing import Tuple, Optional, Dict, Any
import numpy as np

from enerdynamics.core.constraints.operators.base import Operator
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    OperatorInfo,
)
from enerdynamics.core.constraints.core.registry import get_registry, register
from enerdynamics.core.types import Trajectory


@register("operator", "traj_qp", "numpy")
class TrajQPFilter(Operator):
    """
    Full-horizon trajectory QP filter.
    
    Solves a single QP over the entire trajectory:
        min ||u - u_nom||^2 + ρ||ξ||^2
        s.t. A u >= b - ξ, ξ >= 0
    
    This is used for trajectory-level constraints (e.g., CFS) where
    constraints are defined over the entire trajectory, not per-step.
    """
    
    def __init__(
        self,
        use_slack: bool = True,
        solver_backend: str = "numpy",
        **kwargs
    ):
        """
        Initialize trajectory QP filter.
        
        Args:
            use_slack: If True, use slack-QP; if False, use hard-QP
            solver_backend: Solver backend ("numpy", "jax", "osqp")
            **kwargs: Additional arguments
        """
        self.use_slack = use_slack
        self.solver_backend = solver_backend
        self.kwargs = kwargs
        
        # Get solver from registry
        registry = get_registry()
        if solver_backend == "jax":
            solver_class = registry.get("solver", "qpax", "jax")
        elif solver_backend == "osqp":
            solver_class = registry.get("solver", "osqp", "numpy")
        else:
            solver_class = registry.get("solver", "closed_form", "numpy")
        
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
        Apply full-horizon QP filter to trajectory.
        
        Args:
            nominal: Nominal trajectory to repair
            constraints: Convex constraints (A, b) defined over trajectory
            params: Schedule parameters (rho, etc.)
            state: Schedule state
            
        Returns:
            Tuple of (repaired trajectory, operator info)
        """
        start_time = time.time()
        
        # Extract trajectory actions
        H = len(nominal.actions)
        u_nom = np.stack([np.asarray(a, dtype=np.float32) for a in nominal.actions])  # (H, dim)
        u_nom_flat = u_nom.flatten()  # (H*dim,)
        
        # Get constraints
        A = np.asarray(constraints.A, dtype=np.float32)  # (m, H*dim) or (m, dim)
        b = np.asarray(constraints.b, dtype=np.float32)  # (m,)
        
        # Handle different constraint shapes
        if A.ndim == 2:
            if A.shape[1] == u_nom_flat.shape[0]:
                # A is (m, H*dim) - full trajectory constraints
                A_full = A
            elif A.shape[1] == u_nom.shape[1]:
                # A is (m, dim) - per-step constraints, apply to all steps
                # Expand to full trajectory: [A, 0, ...; 0, A, ...; ...]
                m, dim = A.shape
                A_full = np.zeros((m, H * dim), dtype=np.float32)
                for t in range(H):
                    A_full[:, t*dim:(t+1)*dim] = A
            else:
                raise ValueError(f"Constraint matrix A shape {A.shape} incompatible with trajectory")
        else:
            raise ValueError(f"Constraint matrix A must be 2D, got {A.ndim}D")
        
        # Solve QP
        if self.use_slack:
            u_star_flat, info = self._solve_slack_traj_qp(
                u_nom_flat, A_full, b, params
            )
        else:
            u_star_flat, info = self._solve_hard_traj_qp(
                u_nom_flat, A_full, b
            )
        
        # Reshape back to trajectory
        u_star = u_star_flat.reshape(H, -1)
        repaired_actions = [u_star[t] for t in range(H)]
        
        # Create repaired trajectory
        repaired = Trajectory(
            states=nominal.states,
            actions=repaired_actions,
            info=nominal.info
        )
        
        elapsed_time = time.time() - start_time
        
        # Compute violations
        violation_before = np.maximum(0, b - A_full @ u_nom_flat).max() if A_full.size > 0 else 0.0
        violation_after = np.maximum(0, b - A_full @ u_star_flat).max() if A_full.size > 0 else 0.0
        
        operator_info = OperatorInfo(
            success=info.get('status', 'optimal') == 'optimal',
            violation_before=violation_before,
            violation_after=violation_after,
            iterations=info.get('iterations', 1),
            time=elapsed_time,
            extra={
                'qp_count': 1,
                'solver_method': info.get('method', 'traj_qp'),
                **info
            }
        )
        
        return repaired, operator_info
    
    def _solve_slack_traj_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        params: ScheduleParams
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve slack-QP for full trajectory.
        
        min ||u - u_nom||^2 + ρ||ξ||^2
        s.t. A u >= b - ξ, ξ >= 0
        """
        if self.solver is not None:
            return self.solver.solve_least_squares_with_constraints(
                u_nom, A, b, rho=params.rho
            )
        
        # Fallback: use solver base class method
        from enerdynamics.core.constraints.solvers.base import QPSolver
        solver = QPSolver()
        return solver.solve_least_squares_with_constraints(
            u_nom, A, b, rho=params.rho
        )
    
    def _solve_hard_traj_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve hard-QP for full trajectory.
        
        min ||u - u_nom||^2
        s.t. A u >= b
        """
        if self.solver is not None:
            return self.solver.solve_least_squares_with_constraints(
                u_nom, A, b, rho=None
            )
        
        # Fallback: use solver base class method
        from enerdynamics.core.constraints.solvers.base import QPSolver
        solver = QPSolver()
        return solver.solve_least_squares_with_constraints(
            u_nom, A, b, rho=None
        )



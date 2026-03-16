"""
Trajectory-level QP filter in control space.

Solves one QP over the entire action sequence:
    min ||u - u_nom||^2 [+ rho ||ξ||^2]
    s.t. A u >= b (- ξ)
where u is flattened actions of length H*act_dim.
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

from genedynamics.core.constraints.operators.base import Operator
from genedynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    OperatorInfo,
)
from genedynamics.core.constraints.core.registry import get_registry, register
from genedynamics.core.constraints.core.array_interface import BackendArray
from genedynamics.core.types import Trajectory


@register("operator", "traj_qp_actions", "numpy")
@register("operator", "traj_qp_actions", "jax")
class ActionTrajQPFilter(Operator):
    """
    Trajectory-level QP filter for action-space constraints.
    """

    def __init__(
        self,
        use_slack: bool = True,
        solver_backend: str = "numpy",
        **kwargs
    ):
        self.use_slack = use_slack
        self.solver_backend = solver_backend

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
        Apply trajectory-level action QP.
        Expects constraints over flattened actions (per_step=False).
        """
        start_time = time.time()

        if constraints.is_per_step():
            # This operator is for trajectory-level; bail out.
            return nominal, OperatorInfo(success=False, violation_before=0.0, violation_after=0.0, iterations=0, time=0.0, extra={"reason": "per_step_constraint"})

        # Convert A, b to numpy (handle BackendArray or JAX)
        A = self._to_numpy(constraints.A)
        b = self._to_numpy(constraints.b)

        # Flatten nominal actions
        u_nom_flat = np.concatenate([np.asarray(u, dtype=np.float32).reshape(-1) for u in nominal.actions], axis=0)

        # Compute violation before
        violation_before = float(np.maximum(0, b - A @ u_nom_flat).max()) if A.size > 0 else 0.0

        if A.size == 0:
            repaired = Trajectory(states=nominal.states, actions=nominal.actions, info=nominal.info)
            info = OperatorInfo(success=True, violation_before=0.0, violation_after=0.0, iterations=0, time=time.time() - start_time)
            return repaired, info

        # Solve
        if self.use_slack:
            u_star, violation_after = self._solve_slack_qp(u_nom_flat, A, b, params)
        else:
            u_star, violation_after = self._solve_hard_qp(u_nom_flat, A, b)

        # Reshape back to action sequence
        H = len(nominal.actions)
        act_dim = u_star.shape[0] // H if H > 0 else 0
        if act_dim > 0:
            u_seq = u_star.reshape(H, act_dim)
            repaired = Trajectory(states=nominal.states, actions=[u_seq[i] for i in range(H)], info=nominal.info)
        else:
            repaired = nominal

        info = OperatorInfo(
            success=True,
            violation_before=violation_before,
            violation_after=violation_after,
            iterations=1,
            time=time.time() - start_time,
        )
        return repaired, info

    def _to_numpy(self, arr):
        if isinstance(arr, BackendArray):
            return arr.to_numpy()
        if JAX_AVAILABLE:
            try:
                if hasattr(arr, "block_until_ready"):
                    arr.block_until_ready()
                return np.asarray(jax.device_get(arr), dtype=np.float32)
            except Exception:
                pass
        return np.asarray(arr, dtype=np.float32)

    def _solve_slack_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        params: ScheduleParams
    ) -> Tuple[np.ndarray, float]:
        if self.solver is not None:
            if hasattr(self.solver, "solve_slack_qp"):
                return self.solver.solve_slack_qp(u_nom, A, b, params.rho)
            elif hasattr(self.solver, "solve_least_squares_with_constraints"):
                solution, info = self.solver.solve_least_squares_with_constraints(u_nom, A, b, rho=params.rho)
                violation = float(np.maximum(0, b - A @ solution).max())
                return solution, violation

        # Fallback: use hard projection
        return self._solve_hard_qp(u_nom, A, b)

    def _solve_hard_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        if self.solver is not None:
            if hasattr(self.solver, "solve_hard_qp"):
                return self.solver.solve_hard_qp(u_nom, A, b)
            elif hasattr(self.solver, "solve_least_squares_with_constraints"):
                solution, info = self.solver.solve_least_squares_with_constraints(u_nom, A, b, rho=None)
                violation = float(np.maximum(0, b - A @ solution).max())
                return solution, violation

        # Final fallback: simple iterative projection
        u_star = u_nom.copy()
        for _ in range(20):
            violations = np.maximum(0, b - A @ u_star)
            if violations.max() < 1e-7:
                break
            for i in range(A.shape[0]):
                if violations[i] > 0:
                    n = A[i]
                    norm = np.linalg.norm(n) + 1e-8
                    u_star += (violations[i] / norm) * (n / norm)
        violation = float(np.maximum(0, b - A @ u_star).max())
        return u_star, violation


"""Control-Barrier-Function (CBF) safety filter — thin QP projection.

The filter solves a small per-step QP that minimizes the deviation from the
nominal joint command subject to a linear barrier constraint
``A u >= b`` produced by a user-supplied ``barrier_fn``::

    minimize_u   ||u - u_nom||^2
    subject to   A(state) u >= b(state)
                 lb <= u <= ub

This is the same shape solved by
:mod:`genedynamics.core.constraints.operators.qp.per_step_filter` —
:class:`CBFFilter` is a deliberate, self-contained mirror that imports a
solver directly so it stays usable in the deploy loop without dragging in
the schedule / convexify abstractions. When the constraint stack hooks land
on the dispatch path, this class can switch to ``PerStepQPFilter`` by
swapping its ``_solve`` method.

The filter prefers ``osqp`` (fast, sparse, warm-started); falls back to
``scipy.optimize.minimize`` with the SLSQP method when ``osqp`` is not
installed; finally falls back to a soft penalty projection that returns
the unmodified command and flags an infeasibility violation. This keeps
the import side-effect-free so safety stays optional in dev envs.

Constructing :class:`CBFFilter` without a ``barrier_fn`` raises immediately
— the filter is useless without one.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.interfaces.safety import SafetyResult
from genedynamics.deploy.safety.base import BaseSafetyFilter

__all__ = ["CBFFilter", "BarrierFn"]


#: Signature for a per-step linear barrier. Returns ``(A, b)`` such that the
#: safe set on the actuated joint vector ``u`` is ``A u >= b``. ``A`` has
#: shape ``(num_constraints, num_actuated)``, ``b`` shape ``(num_constraints,)``.
BarrierFn = Callable[[RobotState, ControlCommand], Tuple[np.ndarray, np.ndarray]]


class CBFFilter(BaseSafetyFilter):
    """Project a candidate joint-pos command onto a CBF-defined safe set.

    Args:
        spec: Robot spec for joint indexing.
        barrier_fn: Callable returning ``(A, b)`` for the current state.
        joint_lower: Optional override for the lower bounds (defaults to
            ``spec.joint_range``). Length ``num_actuated``.
        joint_upper: Optional override for the upper bounds. Length
            ``num_actuated``.
        slack_weight: Penalty on the L2 norm of the constraint slack
            variable. Set to ``None`` to disable slack and require strict
            feasibility (will fail noisily on infeasible barriers).
    """

    runtime: str = "numpy"

    def __init__(
        self,
        spec: Any,
        *,
        barrier_fn: BarrierFn,
        joint_lower: Optional[np.ndarray] = None,
        joint_upper: Optional[np.ndarray] = None,
        slack_weight: Optional[float] = 1e3,
    ) -> None:
        super().__init__(spec)
        if barrier_fn is None:
            raise ValueError("CBFFilter requires a barrier_fn")
        self.barrier_fn = barrier_fn
        self.slack_weight = float(slack_weight) if slack_weight is not None else None

        n = len(spec.actuated_joints)
        lb = np.full(n, -np.inf, dtype=np.float64)
        ub = np.full(n, +np.inf, dtype=np.float64)
        for i, name in enumerate(spec.actuated_joints):
            jr = spec.joint_range.get(name)
            if jr is None:
                continue
            lb[i] = float(jr[0])
            ub[i] = float(jr[1])
        if joint_lower is not None:
            lb = np.asarray(joint_lower, dtype=np.float64).reshape(n)
        if joint_upper is not None:
            ub = np.asarray(joint_upper, dtype=np.float64).reshape(n)
        self._lb = lb
        self._ub = ub

        # Pick a solver backend lazily so importing the module is cheap.
        self._osqp = None
        try:
            import osqp  # type: ignore

            self._osqp = osqp
        except ImportError:  # pragma: no cover
            pass

    def filter(self, state: RobotState, cmd: ControlCommand) -> SafetyResult:
        if cmd.kind not in ("joint_pos", "mixed") or cmd.joint_pos is None:
            return self._passthrough(cmd)
        u_nom = np.asarray(cmd.joint_pos, dtype=np.float64).reshape(-1)
        n = u_nom.size
        try:
            A, b = self.barrier_fn(state, cmd)
            A = np.asarray(A, dtype=np.float64).reshape(-1, n)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
        except Exception as exc:
            return SafetyResult(
                command=cmd,
                intervened=False,
                violations={"cbf_barrier_error": 1.0},
                extras={"barrier_error": str(exc)},
            )

        # Already-feasible shortcut: skip the QP entirely.
        residual = A @ u_nom - b
        if A.shape[0] == 0 or float(np.min(residual)) >= 0.0:
            return self._passthrough(cmd)

        u_safe, info = self._solve(u_nom, A, b)
        if u_safe is None:
            return SafetyResult(
                command=cmd,
                intervened=False,
                violations={"cbf_infeasible": 1.0},
                extras=info,
            )
        diff = float(np.linalg.norm(u_safe - u_nom))
        worst = float(-np.min(residual))
        new_cmd = ControlCommand(
            kind=cmd.kind,
            joint_pos=u_safe,
            joint_vel=cmd.joint_vel,
            joint_torque=cmd.joint_torque,
            kp=cmd.kp,
            kd=cmd.kd,
            loco_cmd=cmd.loco_cmd,
            extras=cmd.extras,
        )
        return SafetyResult(
            command=new_cmd,
            intervened=True,
            violations={
                "cbf_barrier_violation": worst,
                "cbf_projection_norm": diff,
            },
            extras=info,
        )

    # ------------------------------------------------------------------
    # QP solve
    # ------------------------------------------------------------------

    def _solve(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Solve ``min ||u - u_nom||^2 s.t. A u >= b, lb <= u <= ub``."""
        if self._osqp is not None:
            return self._solve_osqp(u_nom, A, b)
        return self._solve_scipy(u_nom, A, b)

    def _solve_osqp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], dict]:
        import scipy.sparse as sp  # local — only needed when osqp is present

        n = u_nom.size
        m = A.shape[0]
        if self.slack_weight is not None:
            # Augmented variable z = [u; s] with slack s >= 0.
            P = sp.block_diag(
                [sp.eye(n, format="csc"), self.slack_weight * sp.eye(m, format="csc")],
                format="csc",
            )
            q = np.concatenate([-u_nom, np.zeros(m)])
            # Constraint: A u + s >= b   →   [A I] z >= b
            top = sp.hstack([sp.csc_matrix(A), sp.eye(m, format="csc")], format="csc")
            #              lb_u <= u <= ub_u
            mid = sp.hstack([sp.eye(n, format="csc"), sp.csc_matrix((n, m))], format="csc")
            #              0 <= s
            bot = sp.hstack([sp.csc_matrix((m, n)), sp.eye(m, format="csc")], format="csc")
            A_full = sp.vstack([top, mid, bot], format="csc")
            l_full = np.concatenate([b, self._lb, np.zeros(m)])
            u_full = np.concatenate([np.full(m, np.inf), self._ub, np.full(m, np.inf)])
        else:
            P = sp.eye(n, format="csc")
            q = -u_nom
            A_full = sp.vstack([sp.csc_matrix(A), sp.eye(n, format="csc")], format="csc")
            l_full = np.concatenate([b, self._lb])
            u_full = np.concatenate([np.full(m, np.inf), self._ub])

        prob = self._osqp.OSQP()
        prob.setup(P, q, A_full, l_full, u_full, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
        result = prob.solve()
        if result.info.status_val != 1:  # 1 == solved
            return None, {"solver": "osqp", "status": result.info.status}
        u_safe = np.asarray(result.x[:n], dtype=np.float64)
        return u_safe, {"solver": "osqp", "status": "solved"}

    def _solve_scipy(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], dict]:
        try:
            from scipy.optimize import minimize
        except ImportError:  # pragma: no cover
            return None, {"solver": "none"}

        def obj(u: np.ndarray) -> float:
            d = u - u_nom
            return 0.5 * float(d @ d)

        def grad(u: np.ndarray) -> np.ndarray:
            return u - u_nom

        constraints = [
            {"type": "ineq", "fun": lambda u, Ai=A[i], bi=float(b[i]): float(Ai @ u - bi)}
            for i in range(A.shape[0])
        ]
        bounds = list(zip(self._lb.tolist(), self._ub.tolist()))
        result = minimize(
            obj,
            u_nom,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-6},
        )
        if not result.success:
            return None, {"solver": "scipy", "status": result.message}
        return np.asarray(result.x, dtype=np.float64), {
            "solver": "scipy",
            "status": "solved",
        }

"""
9D planning model for D3IL avoiding: state [tcp_xy, q], action qdot.

Dynamics: q' = q + dt*qdot, tcp_xy' = tcp_xy + dt*(J_xy @ qdot).
Optional frozen Jacobian J_xy (2x7) for linearization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import numpy as np


@dataclass(frozen=True)
class AvoidingPlanSpec9D:
    """
    Planning model for D3IL avoiding in 9D (tcp_xy + joint positions).

    State:  [tcp_x, tcp_y, q1..q7]  (9D)
    Action: [qdot1..qdot7]          (7D)

    Dynamics:
      q_next = q + dt * qdot
      tcp_xy_next = tcp_xy + dt * (J_xy @ qdot)
    """

    dt: float = 0.035
    horizon: int = 20
    qdot_limit: float = 1.5  # rad/s per joint
    target_xy: tuple[float, float] = (0.4, 0.35)


class AvoidingPlanEnv9D:
    """
    EDOC/MBD-compatible planning env for 9D D3IL avoiding.

    Provides JAX-friendly transition; execution happens in D3IL MuJoCo env.
    Optional set_linearization(J_xy) to use frozen 2x7 Jacobian for tcp_xy.
    """

    def __init__(self, spec: Optional[AvoidingPlanSpec9D] = None):
        self.spec = spec or AvoidingPlanSpec9D()
        self.dt = float(self.spec.dt)
        self.horizon = int(self.spec.horizon)
        self.state_dim = 9
        self.act_dim = 7
        self.control_limit = float(self.spec.qdot_limit)
        self.target = np.asarray(self.spec.target_xy, dtype=np.float32)
        self._x0: Optional[np.ndarray] = None
        self._J_xy: Optional[np.ndarray] = None  # 2x7, optional frozen Jacobian

    def set_linearization(self, J_xy: np.ndarray) -> None:
        """Set frozen Jacobian for tcp_xy (2x7). If not set, tcp_xy is updated as tcp_xy + 0 (no drift in plan model)."""
        J_xy = np.asarray(J_xy, dtype=np.float32)
        if J_xy.shape != (2, 7):
            raise ValueError(f"J_xy must be (2, 7), got {J_xy.shape}")
        self._J_xy = J_xy.copy()

    def clear_linearization(self) -> None:
        self._J_xy = None

    def get_jacobian_xy(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Return (2, 7) Jacobian for tcp_xy w.r.t. qdot. Used by CFS/MDOC joint-lift filters.
        Returns the frozen _J_xy if set (e.g. via set_linearization from exec_env); else None."""
        _ = state
        if self._J_xy is None:
            return None
        return np.asarray(self._J_xy, dtype=np.float32)

    def set_initial_state(self, x0: np.ndarray) -> None:
        x0 = np.asarray(x0, dtype=np.float32).reshape(-1)
        if x0.size != 9:
            raise ValueError(f"Expected x0 shape (9,), got {x0.shape}")
        self._x0 = x0.copy()

    def reset(self, rng: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        _ = (rng, kwargs)
        if self._x0 is None:
            self._x0 = np.zeros(9, dtype=np.float32)
            self._x0[0], self._x0[1] = 0.4, -0.1
        return self._x0.copy(), {}

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        u = np.asarray(action, dtype=np.float32).reshape(-1)
        if s.size != 9 or u.size != 7:
            raise ValueError("Expected state=(9,), action=(7,)")
        u = np.clip(u, -self.control_limit, self.control_limit)
        tcp_xy = s[:2]
        q = s[2:9]
        q_next = q + self.dt * u
        if self._J_xy is not None:
            tcp_xy_next = tcp_xy + self.dt * (self._J_xy @ u)
        else:
            tcp_xy_next = tcp_xy
        return np.concatenate([tcp_xy_next, q_next], axis=0).astype(np.float32)

    def jax_transition(self, state: Any, action: Any) -> Any:
        import jax.numpy as jnp

        s = state
        u = action
        u = jnp.clip(u, -self.control_limit, self.control_limit)
        tcp_xy = s[:2]
        q = s[2:9]
        q_next = q + self.dt * u
        if self._J_xy is not None:
            J = jnp.asarray(self._J_xy)
            tcp_xy_next = tcp_xy + self.dt * (J @ u)
        else:
            tcp_xy_next = tcp_xy
        return jnp.concatenate([tcp_xy_next, q_next], axis=0)

    def cost(self, state: np.ndarray) -> float:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        pos = s[:2]
        return float(np.sum((pos - self.target) ** 2))

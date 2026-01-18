from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import numpy as np


@dataclass(frozen=True)
class AvoidingPlanSpec:
    """
    Simple planning model for D3IL avoiding.

    State:  [x_des, y_des, x, y]
    Action: [dx, dy]  (delta to desired XY)

    Dynamics (v0, perfect tracking):
      x_des' = x_des + dx
      y_des' = y_des + dy
      x'     = x_des'
      y'     = y_des'
    """

    dt: float = 0.035
    horizon: int = 20
    control_limit: float = 0.05  # conservative per-step delta (meters)
    target_xy: tuple[float, float] = (0.4, 0.35)


class AvoidingPlanEnv:
    """
    EDOC/MBD-compatible planning env for MPC-on-D3IL.

    This is *not* the real MuJoCo dynamics. It exists to provide a JAX-friendly
    transition function for planners. Execution happens in D3IL MuJoCo env.
    """

    def __init__(self, spec: Optional[AvoidingPlanSpec] = None):
        self.spec = spec or AvoidingPlanSpec()
        self.dt = float(self.spec.dt)
        self.horizon = int(self.spec.horizon)
        self.state_dim = 4
        self.act_dim = 2
        self.control_limit = float(self.spec.control_limit)
        self.target = np.asarray(self.spec.target_xy, dtype=np.float32)
        self._x0: Optional[np.ndarray] = None

    def set_initial_state(self, x0: np.ndarray) -> None:
        x0 = np.asarray(x0, dtype=np.float32).reshape(-1)
        if x0.size != 4:
            raise ValueError(f"Expected x0 shape (4,), got {x0.shape}")
        self._x0 = x0.copy()

    def reset(self, rng: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        _ = (rng, kwargs)
        if self._x0 is None:
            # Default: start at origin-ish
            self._x0 = np.array([0.4, -0.1, 0.4, -0.1], dtype=np.float32)
        return self._x0.copy(), {}

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        u = np.asarray(action, dtype=np.float32).reshape(-1)
        if s.size != 4 or u.size != 2:
            raise ValueError("Expected state=(4,), action=(2,)")
        u = np.clip(u, -self.control_limit, self.control_limit)
        des_next = s[:2] + u
        xy_next = des_next  # perfect tracking
        return np.concatenate([des_next, xy_next], axis=0).astype(np.float32)

    def jax_transition(self, state: Any, action: Any) -> Any:
        # Import lazily to keep numpy-only environments usable.
        import jax.numpy as jnp

        s = state
        u = action
        u = jnp.clip(u, -self.control_limit, self.control_limit)
        des_next = s[:2] + u
        xy_next = des_next
        return jnp.concatenate([des_next, xy_next], axis=0)

    def cost(self, state: np.ndarray) -> float:
        # Simple quadratic distance-to-goal on measured xy (dims 2:4)
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        pos = s[2:4]
        return float(np.sum((pos - self.target) ** 2))



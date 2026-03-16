"""
CFS action-space convexifier.

This wraps the standard CFS convexifier (which produces state/position halfspaces)
and converts them into *action-space* linear constraints for simple integrator dynamics.

Supported modes:
- action_mode="u_traj": trajectory-level action QP constraints over flattened u_0..u_{H-1}
    Uses p_t = p0 + dt * sum_{k < t} u_k (single integrator assumption).
    Converts g^T p_t >= b into linear constraints in the full action vector.

- action_mode="u_perstep": per-step action constraints
    Uses p_{t+1} = p_t + dt * u_t.
    Converts g^T p_{t+1} >= b into (dt*g)^T u_t >= b - g^T p_t.

Notes:
- This is intended primarily for 2D single-integrator-like environments where action is velocity.
- For other dynamics, you'll want to extend this module with proper linearization/Jacobians.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from genedynamics.core.constraints.convexify.base import Convexifier
from genedynamics.core.constraints.convexify.cfs.cfs import CFSConvexifier
from genedynamics.core.constraints.core.types import ScheduleParams, ScheduleState, ConvexConstraint
from genedynamics.core.constraints.core.registry import register
from genedynamics.core.constraints.core.array_interface import BackendArray
from genedynamics.core.types import Trajectory, State
from genedynamics.envs.obstacles.base import ObstacleManager


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, BackendArray):
        return x.to_numpy()
    return np.asarray(x, dtype=np.float32)


from genedynamics.core.task_spec import legacy_extract_position


def _group_state_halfspaces_by_t(A: np.ndarray, b: np.ndarray, H: int, state_dim: int, pos_dim: int = 2) -> List[Tuple[int, np.ndarray, float]]:
    """
    From trajectory-flattened state constraints (A x >= b), recover (t, g, b).
    Each row is expected to be non-zero only at one timestep block.
    """
    m = int(A.shape[0])
    if m == 0:
        return []
    A3 = A.reshape(m, H, state_dim)[:, :, :pos_dim]  # (m, H, pos_dim)
    norms = np.linalg.norm(A3, axis=-1)  # (m, H)
    t_idx = np.argmax(norms, axis=1)
    out: List[Tuple[int, np.ndarray, float]] = []
    for i in range(m):
        t = int(t_idx[i])
        g = A3[i, t].astype(np.float32, copy=True)
        if float(np.linalg.norm(g)) < 1e-10:
            continue
        out.append((t, g, float(b[i])))
    return out


@register("convexifier", "cfs_action", "numpy")
@register("convexifier", "cfs_action", "jax")
class CFSActionConvexifier(Convexifier):
    """
    Convert CFS position halfspaces into action-space constraints.
    """

    def __init__(
        self,
        *,
        obstacles: ObstacleManager,
        env: Any,
        action_mode: str = "u_traj",  # "u_traj" | "u_perstep"
        position_extractor: Optional[Callable[[State], np.ndarray]] = None,
        position_dim: Optional[int] = None,
        max_constraints_per_point: int = 8,
        constraint_margin: float = 0.25,
        backend: str = "numpy",
        **kwargs
    ):
        self.obstacles = obstacles
        self.env = env
        self.action_mode = str(action_mode)
        self.position_extractor = position_extractor or legacy_extract_position
        self.position_dim = position_dim  # None => infer from position_extractor output
        self.max_constraints_per_point = int(max_constraints_per_point)
        self.constraint_margin = float(constraint_margin)
        self.backend = str(backend)

        # Internal CFS convexifier (state-space)
        self._cfs = CFSConvexifier(
            obstacles=obstacles,
            position_extractor=self.position_extractor,
            max_constraints_per_point=self.max_constraints_per_point,
            constraint_margin=self.constraint_margin,
            backend=self.backend,
        )

    def build_constraints(self, ref: Trajectory, params: ScheduleParams, state: ScheduleState) -> ConvexConstraint:
        # Build CFS constraints in state/position space
        c_state = self._cfs.build_constraints(ref, params, state)
        A_state = _to_numpy(c_state.A)
        b_state = _to_numpy(c_state.b)

        H_states = len(ref.states)
        if H_states == 0:
            return ConvexConstraint(A=np.zeros((0, 0), dtype=np.float32), b=np.zeros((0,), dtype=np.float32), meta={"type": "cfs_action", "per_step": False})
        state_dim = len(np.asarray(ref.states[0], dtype=np.float32).reshape(-1))
        pos_dim = self.position_dim
        if pos_dim is None:
            pos_vec = np.asarray(self.position_extractor(ref.states[0]), dtype=np.float32).reshape(-1)
            pos_dim = int(len(pos_vec))

        # If no constraints, return empty in appropriate shape
        if A_state.size == 0:
            act_dim = len(np.asarray(ref.actions[0], dtype=np.float32).reshape(-1)) if ref.actions else pos_dim
            H_u = len(ref.actions)
            if self.action_mode == "u_perstep":
                A_empty = np.zeros((H_u, 0, act_dim), dtype=np.float32)
                b_empty = np.zeros((H_u, 0), dtype=np.float32)
                return ConvexConstraint(A=A_empty, b=b_empty, meta={"type": "cfs_action", "per_step": True, "constrains": "actions"})
            else:
                A_empty = np.zeros((0, H_u * act_dim), dtype=np.float32)
                b_empty = np.zeros((0,), dtype=np.float32)
                return ConvexConstraint(A=A_empty, b=b_empty, meta={"type": "cfs_action", "per_step": False, "constrains": "actions"})

        entries = _group_state_halfspaces_by_t(A_state, b_state, H=H_states, state_dim=state_dim, pos_dim=pos_dim)

        dt = float(getattr(self.env, "dt", 1.0))
        H_u = len(ref.actions)
        act_dim = len(np.asarray(ref.actions[0], dtype=np.float32).reshape(-1)) if ref.actions else pos_dim

        # Extract reference positions
        ref_pos = [np.asarray(self.position_extractor(s), dtype=np.float32).reshape(-1)[:pos_dim] for s in ref.states]
        p0 = ref_pos[0]

        if self.action_mode == "u_traj":
            # Trajectory-level action constraints: A_u u_flat >= b_u
            rows: List[np.ndarray] = []
            rhs: List[float] = []
            for (t_s, g, b_val) in entries:
                if t_s <= 0:
                    # depends only on p0; cannot be fixed by actions; skip
                    continue
                # p_t = p0 + dt * sum_{k=0..t_s-1} u_k
                row = np.zeros((H_u * act_dim,), dtype=np.float32)
                t_max = min(t_s, H_u)
                for k in range(t_max):
                    row[k * act_dim : k * act_dim + pos_dim] = dt * g
                rows.append(row)
                rhs.append(float(b_val - float(np.dot(g, p0))))

            if not rows:
                A_u = np.zeros((0, H_u * act_dim), dtype=np.float32)
                b_u = np.zeros((0,), dtype=np.float32)
            else:
                A_u = np.stack(rows, axis=0)
                b_u = np.asarray(rhs, dtype=np.float32)

            return ConvexConstraint(
                A=A_u,
                b=b_u,
                meta={"type": "cfs_action", "per_step": False, "constrains": "actions", "action_mode": "u_traj"},
            )

        if self.action_mode == "u_perstep":
            # Per-step action constraints: for each action u_t, constraints derived from state at t+1
            max_k = self.max_constraints_per_point
            A_ps = np.zeros((H_u, max_k, act_dim), dtype=np.float32)
            b_ps = np.full((H_u, max_k), -1e9, dtype=np.float32)
            counts = np.zeros((H_u,), dtype=np.int32)

            for (t_s, g, b_val) in entries:
                if t_s <= 0:
                    continue
                t_a = t_s - 1
                if t_a < 0 or t_a >= H_u:
                    continue
                idx = int(counts[t_a])
                if idx >= max_k:
                    continue
                # g^T p_{t+1} >= b  and p_{t+1} = p_t + dt*u_t  => (dt*g)^T u_t >= b - g^T p_t
                A_ps[t_a, idx, :pos_dim] = dt * g
                b_ps[t_a, idx] = float(b_val - float(np.dot(g, ref_pos[t_a])))
                counts[t_a] += 1

            return ConvexConstraint(
                A=A_ps,
                b=b_ps,
                meta={"type": "cfs_action", "per_step": True, "constrains": "actions", "action_mode": "u_perstep"},
            )

        raise ValueError(f"Unknown action_mode={self.action_mode}. Expected 'u_traj' or 'u_perstep'.")



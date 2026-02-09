"""
CFS action-space convexifier lifted to joint space (e.g. 7D qdot).

Position halfspace g_xy^T p >= b is mapped to action space via A = J_xy(q)^T g_xy:
  (dt * A)^T u_t >= b - g_xy^T p_t  (per-step)
  or trajectory-level row blocks dt * (J_xy_k^T g) for each step k.

Batch-friendly: builds constraints from ref trajectory in one pass; J_xy computed
per ref state via env.get_jacobian_xy (no inner loops over batch in convexifier).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple
import numpy as np

from enerdynamics.core.constraints.convexify.base import Convexifier
from enerdynamics.core.constraints.convexify.cfs.cfs import CFSConvexifier
from enerdynamics.core.constraints.convexify.cfs.action import (
    _to_numpy,
    _default_extract_position,
    _group_state_halfspaces_by_t,
)
from enerdynamics.core.constraints.core.types import ScheduleParams, ScheduleState, ConvexConstraint
from enerdynamics.core.constraints.core.registry import register
from enerdynamics.core.types import Trajectory, State
from enerdynamics.envs.obstacles.base import ObstacleManager


def _get_jacobian_xy(env: Any, state: np.ndarray) -> Optional[np.ndarray]:
    """Return (2, 7) J_xy at state, or None if not available."""
    get_j = getattr(env, "get_jacobian_xy", None)
    if get_j is None:
        return None
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.size < 9:
        return None
    jxy = get_j(s)
    if jxy is None:
        return None
    jxy = np.asarray(jxy, dtype=np.float32)
    return jxy if jxy.shape == (2, 7) else None


@register("convexifier", "cfs_action_joint", "numpy")
@register("convexifier", "cfs_action_joint", "jax")
class CFSActionJointLiftConvexifier(Convexifier):
    """
    CFS position halfspaces -> action-space constraints for joint dynamics.

    A = J_xy(q)^T g_xy (7D). Same per-step / traj QP structure as cfs_action;
    only the mapping from g_xy to action row uses Jacobian.
    """

    def __init__(
        self,
        *,
        obstacles: ObstacleManager,
        env: Any,
        action_mode: str = "u_traj",
        position_extractor: Optional[Callable[[State], np.ndarray]] = None,
        max_constraints_per_point: int = 8,
        constraint_margin: float = 0.25,
        backend: str = "numpy",
        **kwargs
    ):
        self.obstacles = obstacles
        self.env = env
        self.action_mode = str(action_mode)
        self.position_extractor = position_extractor or _default_extract_position
        self.max_constraints_per_point = int(max_constraints_per_point)
        self.constraint_margin = float(constraint_margin)
        self.backend = str(backend)

        self._cfs = CFSConvexifier(
            obstacles=obstacles,
            position_extractor=self.position_extractor,
            max_constraints_per_point=self.max_constraints_per_point,
            constraint_margin=self.constraint_margin,
            backend=self.backend,
        )

    def build_constraints(self, ref: Trajectory, params: ScheduleParams, state: ScheduleState) -> ConvexConstraint:
        c_state = self._cfs.build_constraints(ref, params, state)
        A_state = _to_numpy(c_state.A)
        b_state = _to_numpy(c_state.b)

        H_states = len(ref.states)
        if H_states == 0:
            return ConvexConstraint(
                A=np.zeros((0, 0), dtype=np.float32),
                b=np.zeros((0,), dtype=np.float32),
                meta={"type": "cfs_action_joint", "per_step": False},
            )
        state_dim = len(np.asarray(ref.states[0], dtype=np.float32).reshape(-1))
        pos_dim = 2

        if A_state.size == 0:
            act_dim = len(np.asarray(ref.actions[0], dtype=np.float32).reshape(-1)) if ref.actions else 7
            H_u = len(ref.actions)
            if self.action_mode == "u_perstep":
                A_empty = np.zeros((H_u, 0, act_dim), dtype=np.float32)
                b_empty = np.zeros((H_u, 0), dtype=np.float32)
                return ConvexConstraint(
                    A=A_empty, b=b_empty,
                    meta={"type": "cfs_action_joint", "per_step": True, "constrains": "actions"},
                )
            A_empty = np.zeros((0, H_u * act_dim), dtype=np.float32)
            b_empty = np.zeros((0,), dtype=np.float32)
            return ConvexConstraint(
                A=A_empty, b=b_empty,
                meta={"type": "cfs_action_joint", "per_step": False, "constrains": "actions"},
            )

        entries = _group_state_halfspaces_by_t(A_state, b_state, H=H_states, state_dim=state_dim, pos_dim=pos_dim)
        dt = float(getattr(self.env, "dt", 1.0))
        H_u = len(ref.actions)
        act_dim = len(np.asarray(ref.actions[0], dtype=np.float32).reshape(-1)) if ref.actions else 7
        ref_pos = [np.asarray(self.position_extractor(s), dtype=np.float32).reshape(-1)[:pos_dim] for s in ref.states]
        ref_states_np = [np.asarray(s, dtype=np.float32).reshape(-1) for s in ref.states]

        if self.action_mode == "u_traj":
            rows: List[np.ndarray] = []
            rhs: List[float] = []
            p0 = ref_pos[0]
            for (t_s, g, b_val) in entries:
                if t_s <= 0:
                    continue
                row = np.zeros((H_u * act_dim,), dtype=np.float32)
                t_max = min(t_s, H_u)
                for k in range(t_max):
                    J_xy = _get_jacobian_xy(self.env, ref_states_np[k])
                    if J_xy is None:
                        row[k * act_dim : k * act_dim + pos_dim] = dt * g
                    else:
                        A_k = np.dot(J_xy.T, g)
                        n = min(act_dim, A_k.size)
                        row[k * act_dim : k * act_dim + n] = dt * A_k[:n]
                if np.linalg.norm(row) < 1e-10:
                    continue
                rows.append(row)
                rhs.append(float(b_val - float(np.dot(g, p0))))
            if not rows:
                A_u = np.zeros((0, H_u * act_dim), dtype=np.float32)
                b_u = np.zeros((0,), dtype=np.float32)
            else:
                A_u = np.stack(rows, axis=0)
                b_u = np.asarray(rhs, dtype=np.float32)
            return ConvexConstraint(
                A=A_u, b=b_u,
                meta={"type": "cfs_action_joint", "per_step": False, "constrains": "actions", "action_mode": "u_traj"},
            )

        if self.action_mode == "u_perstep":
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
                J_xy = _get_jacobian_xy(self.env, ref_states_np[t_a])
                if J_xy is None:
                    A_ps[t_a, idx, :pos_dim] = dt * g
                else:
                    A_action = np.dot(J_xy.T, g)
                    n = min(act_dim, A_action.size)
                    A_ps[t_a, idx, :n] = dt * A_action[:n]
                b_ps[t_a, idx] = float(b_val - float(np.dot(g, ref_pos[t_a])))
                counts[t_a] += 1
            return ConvexConstraint(
                A=A_ps, b=b_ps,
                meta={"type": "cfs_action_joint", "per_step": True, "constrains": "actions", "action_mode": "u_perstep"},
            )

        raise ValueError(f"Unknown action_mode={self.action_mode}. Expected 'u_traj' or 'u_perstep'.")

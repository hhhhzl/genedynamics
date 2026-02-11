from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Avoiding9DAdapter:
    """
    Adapter utilities for using 4D SafeDiffuser plans in 9D D3IL avoiding envs.

    - obs9d_to_obs4d: [x, y, q1..q7] -> [x_des, y_des, x, y]
    - delta_xy_to_qdot7: map XY delta to joint velocities via damped least squares
    - step_9d: execute 7D qdot action in env and parse common step outputs
    """

    env: Any
    dt: float = 0.035
    qdot_limit: float = 1.5
    target_xy: Optional[np.ndarray] = None

    def _get_target_xy(self) -> np.ndarray:
        if self.target_xy is not None:
            t = np.asarray(self.target_xy, dtype=np.float32).reshape(-1)
            if t.size >= 2:
                return t[:2]
        env_t = np.asarray(getattr(self.env, "target"), dtype=np.float32).reshape(-1)
        return env_t[:2]

    def obs9d_to_obs4d(self, obs9d: np.ndarray, target_xy: Optional[np.ndarray] = None) -> np.ndarray:
        obs = np.asarray(obs9d, dtype=np.float32).reshape(-1)
        if obs.size < 2:
            raise ValueError(f"obs9d must have at least 2 dims, got shape={obs.shape}")
        tgt = np.asarray(target_xy, dtype=np.float32).reshape(-1)[:2] if target_xy is not None else self._get_target_xy()
        return np.array([float(tgt[0]), float(tgt[1]), float(obs[0]), float(obs[1])], dtype=np.float32)

    def _get_robot(self) -> Any:
        inner = getattr(getattr(self.env, "_task_env", None), "_env", None)
        return getattr(inner, "robot", None) if inner is not None else None

    def get_current_q(self, fallback_obs9d: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        robot = self._get_robot()
        if robot is not None:
            try:
                robot.receiveState()
                q = np.asarray(robot.current_j_pos, dtype=np.float32).reshape(-1)[:7]
                if q.size == 7:
                    return q.copy()
            except Exception:
                pass
        if fallback_obs9d is not None:
            obs = np.asarray(fallback_obs9d, dtype=np.float32).reshape(-1)
            if obs.size >= 9:
                return obs[2:9].copy()
        return None

    def delta_xy_to_qdot7(self, q: np.ndarray, delta_xy: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32).reshape(-1)[:7]
        dxy = np.asarray(delta_xy, dtype=np.float32).reshape(-1)[:2]
        if q.size != 7:
            raise ValueError(f"expected q shape (7,), got {q.shape}")
        dt_eff = max(1e-6, float(self.dt if dt is None else dt))
        v_xy = dxy / dt_eff

        robot = self._get_robot()
        if robot is None or not hasattr(robot, "getJacobian"):
            return np.zeros(7, dtype=np.float32)

        try:
            J = np.asarray(robot.getJacobian(q), dtype=np.float32)
            J_xy = J[:2, :7]
            lam = 1e-3
            JJt = J_xy @ J_xy.T
            step_xy = np.linalg.solve(JJt + lam * np.eye(2, dtype=np.float32), v_xy)
            qdot = J_xy.T @ step_xy
        except Exception:
            qdot = np.zeros(7, dtype=np.float32)

        qdot = np.asarray(qdot, dtype=np.float32).reshape(-1)[:7]
        qdot = np.clip(qdot, -float(self.qdot_limit), float(self.qdot_limit))
        return qdot.astype(np.float32)

    def solve_ik_xy(
        self,
        q_init: np.ndarray,
        xy_target: np.ndarray,
        *,
        max_iters: int = 20,
        tol: float = 2e-3,
    ) -> np.ndarray:
        q = np.asarray(q_init, dtype=np.float32).reshape(-1)[:7].copy()
        target = np.asarray(xy_target, dtype=np.float32).reshape(-1)[:2]
        robot = self._get_robot()
        if robot is None or not hasattr(robot, "getForwardKinematics") or not hasattr(robot, "getJacobian"):
            return q

        lam = 1e-3
        for _ in range(max_iters):
            pos, _quat = robot.getForwardKinematics(q)
            ee_xy = np.asarray(pos, dtype=np.float32).reshape(-1)[:2]
            err = target - ee_xy
            if float(np.linalg.norm(err)) <= tol:
                break

            J = np.asarray(robot.getJacobian(q), dtype=np.float32)
            J_xy = J[:2, :7]
            JJt = J_xy @ J_xy.T
            step_xy = np.linalg.solve(JJt + lam * np.eye(2, dtype=np.float32), err)
            dq = J_xy.T @ step_xy
            q = q + dq.astype(np.float32)
            q = self._clip_q_joint_limits(q)
        return q.astype(np.float32)

    def _clip_q_joint_limits(self, q: np.ndarray) -> np.ndarray:
        robot = self._get_robot()
        q_out = np.asarray(q, dtype=np.float32).reshape(-1)[:7].copy()
        if robot is None:
            return q_out
        if hasattr(robot, "joint_pos_min") and hasattr(robot, "joint_pos_max"):
            qmin = np.asarray(robot.joint_pos_min, dtype=np.float32).reshape(-1)[:7]
            qmax = np.asarray(robot.joint_pos_max, dtype=np.float32).reshape(-1)[:7]
            if qmin.size == 7 and qmax.size == 7:
                q_out = np.clip(q_out, qmin, qmax)
        return q_out.astype(np.float32)

    def lift_4d_to_9d_trajectory(
        self,
        states_4d: List[np.ndarray],
        *,
        initial_q: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        xy_indices: Tuple[int, int] = (2, 3),
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """
        Convert 4D avoiding trajectory [x_des, y_des, x, y] to 9D [x, y, q1..q7].
        """
        if len(states_4d) < 2:
            return None, None

        q_cur = None
        if initial_q is not None:
            qq = np.asarray(initial_q, dtype=np.float32).reshape(-1)[:7]
            if qq.size == 7:
                q_cur = qq.copy()
        if q_cur is None:
            q_cur = self.get_current_q()
        if q_cur is None or q_cur.size != 7:
            return None, None

        dt_eff = max(1e-6, float(self.dt if dt is None else dt))
        ix, iy = int(xy_indices[0]), int(xy_indices[1])

        states_9d: List[np.ndarray] = []
        actions_9d: List[np.ndarray] = []
        for t in range(len(states_4d)):
            s_t = np.asarray(states_4d[t], dtype=np.float32).reshape(-1)
            if s_t.size <= max(ix, iy):
                return None, None
            xy_t = np.array([s_t[ix], s_t[iy]], dtype=np.float32)
            states_9d.append(np.concatenate([xy_t, q_cur.astype(np.float32)], axis=0))

            if t >= len(states_4d) - 1:
                break

            s_next = np.asarray(states_4d[t + 1], dtype=np.float32).reshape(-1)
            if s_next.size <= max(ix, iy):
                return None, None
            xy_next = np.array([s_next[ix], s_next[iy]], dtype=np.float32)
            q_prev = q_cur.copy()
            try:
                q_solved = self.solve_ik_xy(q_prev, xy_next, max_iters=20, tol=2e-3)
                qdot = (q_solved - q_prev) / dt_eff
                qdot = np.asarray(qdot, dtype=np.float32).reshape(-1)[:7]
                qdot = np.clip(qdot, -float(self.qdot_limit), float(self.qdot_limit))
                q_cur = self._clip_q_joint_limits(q_prev + dt_eff * qdot)
            except Exception:
                qdot = np.zeros(7, dtype=np.float32)
                q_cur = q_prev
            actions_9d.append(np.asarray(qdot, dtype=np.float32))

        if len(actions_9d) > max(0, len(states_9d) - 1):
            actions_9d = actions_9d[: len(states_9d) - 1]
        return states_9d, actions_9d

    def step_9d(
        self,
        qdot7: np.ndarray,
        obs9d: Optional[np.ndarray] = None,
        *,
        t: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool, bool, Dict[str, Any]]:
        u = np.asarray(qdot7, dtype=np.float32).reshape(-1)[:7]
        info: Dict[str, Any] = {}

        try:
            out = self.env.step(obs9d, u, t=t, info={})
        except TypeError:
            try:
                out = self.env.step(None, u)
            except TypeError:
                out = self.env.step(u)

        success = False
        done = False
        if isinstance(out, (tuple, list)):
            if len(out) == 4:
                next_obs, r2, r3, info = out
                info = info or {}
                if isinstance(r2, (bool, np.bool_)) and isinstance(r3, (bool, np.bool_)):
                    success = bool(r2)
                    done = bool(r3)
                else:
                    done = bool(r3)
                    success = bool(info.get("success", info.get("is_success", False)))
            elif len(out) == 5:
                next_obs, _rew, terminated, truncated, info = out
                info = info or {}
                done = bool(terminated) or bool(truncated)
                success = bool(info.get("success", info.get("is_success", False)))
            elif len(out) == 2:
                next_obs, done = out
                done = bool(done)
                info = {}
            else:
                next_obs = out[0]
                info = out[-1] if isinstance(out[-1], dict) else {}
                success = bool(info.get("success", info.get("is_success", False)))
                done = bool(info.get("done", False))
        else:
            next_obs = out

        return np.asarray(next_obs, dtype=np.float32), success, done, info

"""
D3IL avoiding execution env with 7D velocity (qdot) action and 9D state [tcp_xy, q].

Wraps the same ObstacleAvoidanceEnv; converts qdot -> integrated q -> FK -> cartesian
setpoint for the existing cartesian controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from enerdynamics.envs.external.d3il.task_env import D3ILTaskEnv
from enerdynamics.envs.external.d3il.specs.base import D3ILTaskSpec, Context, D3ILTaskConfig


@dataclass(frozen=True)
class D3ILAvoiding7dVelSpecConfig(D3ILTaskConfig):
    quat_wxyz: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)


class D3ILAvoiding7dVelSpec(D3ILTaskSpec):
    """
    Spec for D3IL Avoiding with 7D velocity action and 9D state.

    State:  [tcp_x, tcp_y, q1..q7]
    Action: [qdot1..qdot7]
    """

    dt: float = 0.035
    horizon: int = 150
    state_dim: int = 9
    act_dim: int = 7
    control_limit: float = 1.5
    target: np.ndarray = np.array([0.4, 0.35], dtype=np.float32)

    def __init__(self, config: Optional[D3ILAvoiding7dVelSpecConfig] = None):
        self.config = config or D3ILAvoiding7dVelSpecConfig()

    def make_env(self) -> Any:
        from enerdynamics.envs.external.d3il.bootstrap import ensure_d3il_on_path

        ensure_d3il_on_path()
        try:
            from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import (
                ObstacleAvoidanceEnv,
            )
        except ModuleNotFoundError:
            from environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import (
                ObstacleAvoidanceEnv,
            )
        return ObstacleAvoidanceEnv(render=bool(self.config.render))

    def start_env(self, env: Any) -> None:
        env.start()

    def _robot_state_9d(self, env: Any) -> np.ndarray:
        """Return [tcp_x, tcp_y, q1..q7] from current robot state."""
        robot = env.robot
        robot.receiveState()
        tcp_pos = np.asarray(robot.current_c_pos, dtype=np.float32)
        q = np.asarray(robot.current_j_pos, dtype=np.float32)
        return np.concatenate([tcp_pos[:2], q], axis=0)

    def reset_to_state(
        self, env: Any, rng: Optional[Any] = None, **kwargs: Any
    ) -> Tuple[np.ndarray, Context, Dict[str, Any]]:
        _ = (rng, kwargs)
        obs = env.reset()
        state_9d = self._robot_state_9d(env)
        tcp_z = float(env.robot.current_c_pos[2]) if hasattr(env.robot, "current_c_pos") else 0.12
        ctx: Context = {"q": state_9d[2:9].copy(), "fixed_z": tcp_z}
        info: Dict[str, Any] = {"obs_xy": state_9d[:2]}
        return state_9d, ctx, info

    def action_to_env_action(self, env: Any, action: np.ndarray, ctx: Context) -> Tuple[np.ndarray, Context]:
        robot = env.robot
        u = np.asarray(action, dtype=np.float32).reshape(-1)
        if u.size != 7:
            raise ValueError(f"Expected action shape (7,), got {u.shape}")
        q = np.asarray(ctx["q"], dtype=np.float32)
        q_next = q + self.dt * u
        ctx = {**ctx, "q": q_next}
        pos, quat = robot.getForwardKinematics(q_next)
        pos = np.asarray(pos, dtype=np.float32)
        quat = np.asarray(quat, dtype=np.float32)
        fixed_z = float(ctx["fixed_z"])
        env_action = np.concatenate(
            [pos[:2], np.array([fixed_z], dtype=np.float32), quat],
            axis=0,
        ).astype(np.float32)
        return env_action, ctx

    def step_to_state(
        self, env: Any, env_action: np.ndarray, ctx: Context
    ) -> Tuple[np.ndarray, float, bool, Context, Dict[str, Any]]:
        obs, reward, done, d3il_info = env.step(env_action)
        next_state = self._robot_state_9d(env)
        ctx = {**ctx, "q": next_state[2:9].copy()}
        cost = float(reward) if reward is not None else 0.0
        extra: Dict[str, Any] = {}
        try:
            if isinstance(d3il_info, tuple) and len(d3il_info) >= 2:
                extra["mode_encoding"] = np.asarray(d3il_info[0]).tolist()
                extra["success"] = bool(d3il_info[1])
        except Exception:
            pass
        try:
            if hasattr(env, "check_failure"):
                extra["collision"] = bool(env.check_failure())
        except Exception:
            pass
        info = {"obs_xy": next_state[:2], **extra}
        return next_state, cost, bool(done), ctx, info

    def approx_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        u = np.asarray(action, dtype=np.float32).reshape(-1)
        if s.size != 9 or u.size != 7:
            raise ValueError("Expected state=(9,), action=(7,)")
        q_next = s[2:9] + self.dt * u
        # No J_xy in spec; tcp_xy unchanged in approx
        return np.concatenate([s[:2], q_next], axis=0).astype(np.float32)


@dataclass(frozen=True)
class D3ILAvoiding7dVelConfig:
    render: bool = False
    quat_wxyz: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)


class D3ILAvoiding7dVelEnv:
    """
    Execution env: 7D qdot action, 9D state [tcp_xy, q].
    Wraps D3IL ObstacleAvoidanceEnv via D3ILAvoiding7dVelSpec.
    """

    def __init__(self, config: Optional[D3ILAvoiding7dVelConfig] = None):
        self.config = config or D3ILAvoiding7dVelConfig()
        spec_cfg = D3ILAvoiding7dVelSpecConfig(
            render=bool(self.config.render),
            quat_wxyz=self.config.quat_wxyz,
        )
        self._task_env = D3ILTaskEnv(D3ILAvoiding7dVelSpec(spec_cfg))
        self.dt = self._task_env.dt
        self.horizon = self._task_env.horizon
        self.state_dim = self._task_env.state_dim
        self.act_dim = self._task_env.act_dim
        self.control_limit = self._task_env.control_limit
        self.target = self._task_env.target

    def _lazy_init(self) -> None:
        self._task_env._lazy_init()

    def start(self) -> None:
        self._task_env.start()

    def close(self) -> None:
        self._task_env.close()

    def reset(self, rng: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._task_env.reset(rng=rng, **kwargs)

    def step(
        self,
        state: Optional[np.ndarray],
        action: np.ndarray,
        t: Optional[int] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self._task_env.step(state, action, t=t, info=info)

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self._task_env.transition(state, action)

    def rollout_actions(
        self, state: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """Roll out a sequence of actions from initial state using approx transition (no sim)."""
        x = np.asarray(state, dtype=np.float32).reshape(-1)
        traj = [x.copy()]
        for act in np.asarray(actions, dtype=np.float32):
            x = self._task_env.transition(x, act.reshape(-1))
            traj.append(np.asarray(x, dtype=np.float32))
        return np.stack(traj, axis=0)

    def cost(self, state: np.ndarray) -> float:
        return 0.0

    def robot_state(self) -> np.ndarray:
        self._task_env._lazy_init()
        inner = getattr(self._task_env, "_env", None)
        if inner is None or not hasattr(inner, "robot_state"):
            raise AttributeError("Underlying D3IL env has no robot_state")
        return inner.robot_state()

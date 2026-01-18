from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from enerdynamics.envs.external.d3il.bootstrap import ensure_d3il_on_path
from .base import D3ILTaskSpec, Context, D3ILTaskConfig


@dataclass(frozen=True)
class D3ILAvoidingSpecConfig(D3ILTaskConfig):
    # Controller expects 7D: [x_des, y_des, z_fixed, quat_wxyz(4)]
    quat_wxyz: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)


class D3ILAvoidingSpec(D3ILTaskSpec):
    """
    Spec for D3IL Avoiding task.

    State:  [x_des, y_des, x, y]
    Action: [dx, dy]  (delta in desired XY)
    """

    dt: float = 0.035
    horizon: int = 150
    state_dim: int = 4
    act_dim: int = 2
    control_limit: float = 0.05
    target: np.ndarray = np.array([0.4, 0.35], dtype=np.float32)

    def __init__(self, config: Optional[D3ILAvoidingSpecConfig] = None):
        self.config = config or D3ILAvoidingSpecConfig()

    def make_env(self) -> Any:
        ensure_d3il_on_path()
        from environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import (
            ObstacleAvoidanceEnv,
        )

        return ObstacleAvoidanceEnv(render=bool(self.config.render))

    def start_env(self, env: Any) -> None:
        env.start()

    def reset_to_state(
        self, env: Any, rng: Optional[Any] = None, **kwargs: Any
    ) -> Tuple[np.ndarray, Context, Dict[str, Any]]:
        _ = (rng, kwargs)
        obs = env.reset()
        obs_xy = np.asarray(obs, dtype=np.float32).reshape(-1)[:2]

        tcp_pos = np.asarray(env.robot_state(), dtype=np.float32).reshape(-1)
        fixed_z = float(tcp_pos[2]) if tcp_pos.size >= 3 else 0.12

        ctx: Context = {"fixed_z": fixed_z, "desired_xy": obs_xy.copy()}
        state = np.concatenate([ctx["desired_xy"], obs_xy], axis=0).astype(np.float32)
        info: Dict[str, Any] = {"obs_xy": obs_xy}
        return state, ctx, info

    def action_to_env_action(self, env: Any, action: np.ndarray, ctx: Context) -> Tuple[np.ndarray, Context]:
        _ = env
        u = np.asarray(action, dtype=np.float32).reshape(-1)
        if u.size != 2:
            raise ValueError(f"Expected action shape (2,), got {u.shape}")

        desired_xy = np.asarray(ctx["desired_xy"], dtype=np.float32).reshape(2)
        next_des = (desired_xy + u).astype(np.float32)
        ctx = {**ctx, "desired_xy": next_des}

        fixed_z = float(ctx["fixed_z"])
        env_action = np.concatenate(
            [
                next_des,
                np.array([fixed_z], dtype=np.float32),
                np.array(self.config.quat_wxyz, dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        return env_action, ctx

    def step_to_state(
        self, env: Any, env_action: np.ndarray, ctx: Context
    ) -> Tuple[np.ndarray, float, bool, Context, Dict[str, Any]]:
        obs, reward, done, d3il_info = env.step(env_action)
        obs_xy = np.asarray(obs, dtype=np.float32).reshape(-1)[:2]

        next_des = np.asarray(ctx["desired_xy"], dtype=np.float32).reshape(2)
        next_state = np.concatenate([next_des, obs_xy], axis=0).astype(np.float32)
        cost = -float(reward)

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

        info = {"obs_xy": obs_xy, "x_des_y_des": next_des, **extra}
        return next_state, cost, bool(done), ctx, info

    def approx_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        u = np.asarray(action, dtype=np.float32).reshape(-1)
        if s.size != 4 or u.size != 2:
            raise ValueError("Expected state=(4,), action=(2,)")
        next_des = s[:2] + u
        next_xy = next_des
        return np.concatenate([next_des, next_xy], axis=0).astype(np.float32)



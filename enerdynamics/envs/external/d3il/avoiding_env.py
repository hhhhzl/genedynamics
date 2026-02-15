from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import numpy as np

from enerdynamics.envs.external.d3il.task_env import D3ILTaskEnv
from enerdynamics.envs.external.d3il.specs.avoiding import D3ILAvoidingSpec, D3ILAvoidingSpecConfig


@dataclass(frozen=True)
class D3ILAvoidingConfig:
    """
    Config for D3IL Avoiding (MuJoCo) environment adapter.

    Notes:
    - D3IL uses classic gym (not gymnasium) and requires `env.start()`.
    - We expose a simple 2D action: u = [dx, dy] (delta in desired XY).
    - We expose a 4D state: s = [x_des, y_des, x, y]
        - (x_des, y_des): last commanded desired position (planner-side)
        - (x, y): measured end-effector position (env observation)
    """

    render: bool = False
    # The D3IL avoiding controller expects a 7D action:
    # [x_des, y_des, z_fixed, quat_wxyz(4)]
    quat_wxyz: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    obstacle_level: Optional[int] = None
    obstacle_radius_by_level: Optional[Dict[int, list]] = None
    obstacles: Any = None


class D3ILAvoidingEnv:
    """
    EnerDynamics-compatible wrapper for D3IL's ObstacleAvoidanceEnv.

    This wrapper is intentionally thin and stateful (like Gym):
    - `state` passed to step() is ignored; D3IL maintains internal simulator state.
    - We maintain (x_des, y_des) internally so the caller can operate in delta-action space.
    """

    # Minimal interface expected by EnerDynamics ecosystem
    dt: float = 0.035  # heuristic; D3IL internally uses dt=0.001 and n_substeps=35
    horizon: int = 150
    state_dim: int = 4
    act_dim: int = 2
    # Planner-facing hints (set per-instance in __init__)
    control_limit: float = 0.05

    def __init__(self, config: Optional[D3ILAvoidingConfig] = None):
        self.config = config or D3ILAvoidingConfig()

        spec_cfg = D3ILAvoidingSpecConfig(
            render=bool(self.config.render),
            quat_wxyz=self.config.quat_wxyz,
            obstacle_level=self.config.obstacle_level,
            obstacle_radius_by_level=self.config.obstacle_radius_by_level,
            obstacles=self.config.obstacles,
        )
        self._task_env = D3ILTaskEnv(D3ILAvoidingSpec(spec_cfg))

        # Mirror attributes for compatibility
        self.dt = self._task_env.dt
        self.horizon = self._task_env.horizon
        self.state_dim = self._task_env.state_dim
        self.act_dim = self._task_env.act_dim
        self.control_limit = self._task_env.control_limit
        self.target = self._task_env.target

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def _lazy_init(self) -> None:
        # Backward compatibility: keep method but delegate.
        self._task_env._lazy_init()

    def start(self) -> None:
        self._task_env.start()

    def close(self) -> None:
        self._task_env.close()

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Optional API for compatibility
    # ---------------------------------------------------------------------
    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self._task_env.transition(state, action)

    def cost(self, state: np.ndarray) -> float:
        # Placeholder: no intrinsic cost; use task-specific metrics instead.
        _ = state
        return 0.0

    # ---------------------------------------------------------------------
    # D3IL-specific passthroughs
    # ---------------------------------------------------------------------
    def robot_state(self) -> np.ndarray:
        """
        Expose underlying D3IL env.robot_state() for downstream adapters (e.g., DPCC).
        """
        # Ensure the inner env is initialized
        self._task_env._lazy_init()
        inner_env = getattr(self._task_env, "_env", None)
        if inner_env is None or not hasattr(inner_env, "robot_state"):
            raise AttributeError("Underlying D3IL env has no robot_state")
        return inner_env.robot_state()

    def robot_state_9d(self) -> Optional[np.ndarray]:
        """
        Return 9D state [x, y, q1..q7] for 3D visualization (trajectory_best_exec_3d.gif).
        Uses current_c_pos[:2] and current_j_pos from the underlying sim robot.
        """
        self._task_env._lazy_init()
        inner_env = getattr(self._task_env, "_env", None)
        if inner_env is None or not hasattr(inner_env, "robot"):
            return None
        robot = getattr(inner_env, "robot", None)
        if robot is None:
            return None
        c_pos = getattr(robot, "current_c_pos", None)
        j_pos = getattr(robot, "current_j_pos", None)
        if c_pos is None or j_pos is None:
            return None
        c_pos = np.asarray(c_pos, dtype=np.float32).reshape(-1)
        j_pos = np.asarray(j_pos, dtype=np.float32).reshape(-1)
        if c_pos.size < 2 or j_pos.size < 7:
            return None
        return np.concatenate([c_pos[:2], j_pos[:7]], axis=0).astype(np.float32)

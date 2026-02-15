"""
D3IL avoiding execution env with 7D velocity (qdot) action and 9D state [tcp_xy, q].

Wraps the same ObstacleAvoidanceEnv; converts qdot -> integrated q -> FK -> cartesian
setpoint for the existing cartesian controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import importlib

import numpy as np

from enerdynamics.envs.external.d3il.task_env import D3ILTaskEnv
from enerdynamics.envs.external.d3il.specs.base import D3ILTaskSpec, Context, D3ILTaskConfig


@dataclass(frozen=True)
class D3ILAvoiding7dVelSpecConfig(D3ILTaskConfig):
    quat_wxyz: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    obstacle_level: Optional[int] = None
    obstacle_radius_by_level: Optional[Dict[int, list]] = None
    obstacles: Any = None  # ObstacleManager for EE-only collision
    robot_radius: Optional[float] = None
    collision_ee_only: bool = False


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
    target: np.ndarray = np.array([0.5, 0.35], dtype=np.float32)  # center x of last obstacle row
    success_distance_threshold: float = 0.02  # only report success/done when within this distance

    def __init__(self, config: Optional[D3ILAvoiding7dVelSpecConfig] = None):
        self.config = config or D3ILAvoiding7dVelSpecConfig()

    def make_env(self) -> Any:
        from enerdynamics.envs.external.d3il.bootstrap import ensure_d3il_on_path

        ensure_d3il_on_path()

        # Priority 1: Use framework-generated obstacles directly (positions + sizes).
        has_runtime_obstacles = getattr(self.config, "obstacles", None) is not None
        runtime_circles = self._extract_xy_circles(getattr(self.config, "obstacles", None))
        if has_runtime_obstacles:
            # Always sync MuJoCo scene to framework obstacles, including empty set.
            _patched_obj_list = self._build_obj_list_from_circles(runtime_circles)
            _did_patch = True
        else:
            _patched_obj_list = None
            _did_patch = False

        # Priority 2 (fallback): monkey-patch fixed D3IL layout with custom radii.
        obstacle_level = getattr(self.config, "obstacle_level", None)
        radius_by_level = getattr(self.config, "obstacle_radius_by_level", None)
        if (
            not _did_patch
            and
            obstacle_level is not None
            and radius_by_level is not None
            and isinstance(radius_by_level, dict)
        ):
            radii = radius_by_level.get(obstacle_level)
            if radii is not None and len(radii) >= 2:
                r_first, r_rest = float(radii[0]), float(radii[1])

                def _make_patched_get_obj_list(r1: float, r2: float):
                    def _get_obj_list():
                        try:
                            from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import (
                                Box,
                                Cylinder,
                            )
                        except ModuleNotFoundError:
                            from d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import (
                                Box,
                                Cylinder,
                            )
                        mid_pos = 0.5
                        offset = 0.075
                        first_level_y = -0.1
                        level_distance = 0.18
                        return [
                            Cylinder(
                                name="l1_obs",
                                init_pos=[mid_pos, first_level_y, 0],
                                init_quat=[1, 0, 0, 0],
                                size=[r1, 0.07],
                                rgba=[1, 0, 0, 1],
                                static=True,
                            ),
                            Cylinder(
                                name="l2_top_obs",
                                init_pos=[mid_pos - offset, first_level_y + level_distance, 0],
                                init_quat=[1, 0, 0, 0],
                                size=[r2, 0.1],
                                rgba=[1, 0, 0, 1],
                                static=True,
                            ),
                            Cylinder(
                                name="l2_bottom_obs",
                                init_pos=[mid_pos + offset, first_level_y + level_distance, 0],
                                init_quat=[1, 0, 0, 0],
                                size=[r2, 0.1],
                                rgba=[1, 0, 0, 1],
                                static=True,
                            ),
                            Cylinder(
                                name="l3_top_obs",
                                init_pos=[
                                    mid_pos - 2 * offset,
                                    first_level_y + 2 * level_distance,
                                    0,
                                ],
                                init_quat=[1, 0, 0, 0],
                                size=[r2, 0.1],
                                rgba=[1, 0, 0, 1],
                                static=True,
                            ),
                            Cylinder(
                                name="l3_mid_obs",
                                init_pos=[mid_pos, first_level_y + 2 * level_distance, 0],
                                init_quat=[1, 0, 0, 0],
                                size=[r2, 0.1],
                                rgba=[1, 0, 0, 1],
                                static=True,
                            ),
                            Cylinder(
                                name="l3_bottom_obs",
                                init_pos=[
                                    mid_pos + 2 * offset,
                                    first_level_y + 2 * level_distance,
                                    0,
                                ],
                                init_quat=[1, 0, 0, 0],
                                size=[r2, 0.1],
                                rgba=[1, 0, 0, 1],
                                static=True,
                            ),
                            Box(
                                name="finish_line",
                                init_pos=[0.4, first_level_y + 2.5 * level_distance, 0],
                                init_quat=[1, 0, 0, 0],
                                size=[0.5, 0.01, 0.005],
                                rgba=[0.0, 1.0, 0.0, 0.3],
                                visual_only=True,
                                static=True,
                            ),
                        ]

                    return _get_obj_list

                try:
                    import environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.objects.avoiding_objects as ao
                except ModuleNotFoundError:
                    import d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.objects.avoiding_objects as ao
                ao.get_obj_list = _make_patched_get_obj_list(r_first, r_rest)
                _patched_obj_list = ao.get_obj_list()
                _did_patch = True
            else:
                _did_patch = False
                _patched_obj_list = None
        
        try:
            from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs import avoiding as av_mod
        except ModuleNotFoundError:
            from environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs import avoiding as av_mod
        if _did_patch and _patched_obj_list is not None:
            importlib.reload(av_mod)
            av_mod.obj_list = _patched_obj_list
        return av_mod.ObstacleAvoidanceEnv(render=bool(self.config.render))

    def _extract_xy_circles(self, obstacles: Any) -> list:
        """
        Extract 2D circle obstacles as (x, y, radius) from framework obstacles.
        """
        if obstacles is None:
            return []
        if hasattr(obstacles, "obstacles"):
            obs_iter = getattr(obstacles, "obstacles") or []
        elif isinstance(obstacles, (list, tuple)):
            obs_iter = obstacles
        else:
            return []

        circles = []
        for obs in obs_iter:
            center = getattr(obs, "center", None)
            if center is None:
                continue
            c = np.asarray(center, dtype=np.float32).reshape(-1)
            if c.size < 2:
                continue
            radius = getattr(obs, "radius", None)
            if radius is None and hasattr(obs, "half_extents"):
                he = np.asarray(getattr(obs, "half_extents"), dtype=np.float32).reshape(-1)
                if he.size >= 2:
                    radius = float(max(he[0], he[1]))
            if radius is None:
                continue
            circles.append((float(c[0]), float(c[1]), float(radius)))
        return circles

    def _build_obj_list_from_circles(self, circles: list) -> list:
        """
        Build D3IL obstacle objects from (x, y, radius) circles.
        """
        try:
            from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import (
                Box,
                Cylinder,
            )
        except ModuleNotFoundError:
            from d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import (
                Box,
                Cylinder,
            )

        obj_list = []
        for i, (x, y, r) in enumerate(circles):
            obj_list.append(
                Cylinder(
                    name=f"obs_{i}",
                    init_pos=[x, y, 0.0],
                    init_quat=[1, 0, 0, 0],
                    size=[max(1e-4, float(r)), 0.1],
                    rgba=[1, 0, 0, 1],
                    static=True,
                )
            )

        obj_list.append(
            Box(
                name="finish_line",
                init_pos=[0.4, 0.35, 0],
                init_quat=[1, 0, 0, 0],
                size=[0.5, 0.01, 0.005],
                rgba=[0.0, 1.0, 0.0, 0.3],
                visual_only=True,
                static=True,
            )
        )
        return obj_list

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
            if getattr(self.config, "collision_ee_only", False) and getattr(self.config, "obstacles", None) is not None and getattr(self.config, "robot_radius", None) is not None:
                ee_xy = np.asarray(next_state[:2], dtype=np.float64).reshape(-1)
                obstacles = self.config.obstacles
                robot_radius = float(self.config.robot_radius)
                sdf = obstacles.sdf(ee_xy)
                sdf_val = float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf)
                in_obs = bool(obstacles.contains(ee_xy)) if hasattr(obstacles, "contains") else False
                # Small tolerance 1e-5 so boundary/numerical noise doesn't over-count collision
                margin = max(0.0, robot_radius - 1e-5)
                extra["collision"] = bool(sdf_val < margin or in_obs)
            elif hasattr(env, "check_failure"):
                extra["collision"] = bool(env.check_failure())
        except Exception:
            pass
        if "collision" not in extra:
            try:
                extra["collision"] = bool(env.check_failure()) if hasattr(env, "check_failure") else False
            except Exception:
                extra["collision"] = False
        # Success: (1) within point threshold, or (2) D3IL line task: y >= target_y - margin (over the line)
        dist_to_target = float(np.linalg.norm(np.asarray(next_state[:2], dtype=np.float64) - np.asarray(self.target, dtype=np.float64)))
        target_y = float(np.asarray(self.target, dtype=np.float64).reshape(-1)[1])
        line_margin = 0.02
        over_line = float(next_state[1]) >= target_y - line_margin
        if dist_to_target <= self.success_distance_threshold or over_line:
            done = True
            extra["success"] = True
        else:
            done = False
            extra["success"] = False
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
    obstacle_level: Optional[int] = None
    obstacle_radius_by_level: Optional[Dict[int, list]] = None
    obstacles: Any = None
    robot_radius: Optional[float] = None
    collision_ee_only: bool = False


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
            obstacle_level=getattr(self.config, "obstacle_level", None),
            obstacle_radius_by_level=getattr(self.config, "obstacle_radius_by_level", None),
            obstacles=getattr(self.config, "obstacles", None),
            robot_radius=getattr(self.config, "robot_radius", None),
            collision_ee_only=bool(getattr(self.config, "collision_ee_only", False)),
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
        """Roll out a sequence of actions from initial state using approx transition (no sim).
        When get_jacobian_xy is available (9D), uses J_xy at initial state so tcp_xy evolves
        for diffusion/trajectory visualization; otherwise uses spec approx (tcp_xy fixed)."""
        x = np.asarray(state, dtype=np.float32).reshape(-1)
        traj = [x.copy()]
        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr.reshape(1, -1)
        J_xy = None
        if x.size == 9 and hasattr(self, "get_jacobian_xy"):
            J_xy = self.get_jacobian_xy(x)
        dt = float(self.dt)
        for act in actions_arr:
            u = act.reshape(-1)
            if J_xy is not None and J_xy.shape == (2, 7) and u.size == 7:
                tcp_xy = x[:2] + dt * (J_xy @ u)
                q_next = x[2:9] + dt * u
                x = np.concatenate([tcp_xy, q_next], axis=0).astype(np.float32)
            else:
                x = self._task_env.transition(x, u)
            traj.append(np.asarray(x, dtype=np.float32))
        return np.stack(traj, axis=0)

    def get_jacobian_xy(self, state: np.ndarray) -> Optional[np.ndarray]:
        """
        Return (2, 7) Jacobian for tcp x,y w.r.t. joint velocities at the given 9D state.
        Used to set plan_env linearization so MBD can couple tcp_xy to 7D actions.
        Returns None if the inner env/robot is not available.
        """
        self._task_env._lazy_init()
        inner = getattr(self._task_env, "_env", None)
        if inner is None or not hasattr(inner, "robot"):
            return None
        robot = inner.robot
        if not hasattr(robot, "getJacobian"):
            return None
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size != 9:
            return None
        q = state[2:9]
        try:
            J = robot.getJacobian(q)
            J = np.asarray(J, dtype=np.float32)
            if J.shape[0] >= 6 and J.shape[1] >= 7:
                J_xy = J[0:2, :7].copy()
                return J_xy
            return None
        except Exception:
            return None

    def cost(self, state: np.ndarray) -> float:
        return 0.0

    def robot_state(self) -> np.ndarray:
        self._task_env._lazy_init()
        inner = getattr(self._task_env, "_env", None)
        if inner is None or not hasattr(inner, "robot_state"):
            raise AttributeError("Underlying D3IL env has no robot_state")
        return inner.robot_state()

"""Adapter for policies trained with `unitree_rl_gym` (legged_gym fork).

Implements §14 of the deploy refactor plan: parses an upstream legged-gym
training config (YAML), builds the obs builder + action mapper that match
the trained policy's exact format, and constructs an :class:`RLController`
that the deploy pipeline can run as-is.

This module is the **reference template** for plugging in any third-party
locomotion policy. The piece you customize per upstream repo is
:meth:`UnitreeRLGymG1Controller._parse_train_cfg` — everything else (obs
ordering, action scaling, decimation, last-action feedback, dof
permutation) is handled by :class:`RLController` and the helper builders.

Tests inject ``inference_fn`` directly so the adapter is exercisable
without a real .pt checkpoint; production usage either passes a real
:class:`PolicyArtifact` or also injects a loaded torch JIT module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from genedynamics.deploy.controllers.rl.base import (
    ActionMapper,
    InferenceFn,
    ObsBuilder,
    PolicyArtifact,
    RLController,
)
from genedynamics.deploy.controllers.rl.passthrough import _project_world_gravity
from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
)
from genedynamics.robots.g1 import G1RobotSpec

__all__ = [
    "UnitreeRLGymObsCfg",
    "UnitreeRLGymObsBuilder",
    "UnitreeRLGymActionMapper",
    "UnitreeRLGymG1Controller",
]


# ---------------------------------------------------------------------------
# Per-policy config (mirrors unitree_rl_gym/legged_gym envs/g1 configs)
# ---------------------------------------------------------------------------


@dataclass
class UnitreeRLGymObsCfg:
    """Hyperparameters frozen at training time, parsed from the upstream cfg.

    Attributes:
        history_len: Number of past obs frames the policy was trained on.
        lin_vel_scale / ang_vel_scale / dof_pos_scale / dof_vel_scale:
            Per-feature normalization factors.
        action_scale: ``q_target = default_pose + action * action_scale``.
        default_dof_pos: Default joint angles in **policy DoF order**.
        policy_dof_names: Joint names in policy DoF order. Used to build the
            permutation against the robot spec.
        kp / kd: Per-DoF PD gains in policy DoF order. Held high enough to
            track ``q_target`` against the inertial coupling of the legs.
    """

    history_len: int
    lin_vel_scale: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    default_dof_pos: np.ndarray
    policy_dof_names: Tuple[str, ...]
    kp: np.ndarray
    kd: np.ndarray


# ---------------------------------------------------------------------------
# Obs builder
# ---------------------------------------------------------------------------


class UnitreeRLGymObsBuilder(ObsBuilder):
    """Build obs in the layout `unitree_rl_gym` policies were trained on.

    Single-frame layout, all in **policy DoF order**::

        [ base_lin_vel(3) * lin_vel_scale,
          base_ang_vel(3) * ang_vel_scale,
          projected_gravity(3),
          cmd_vx, cmd_vy, cmd_yaw_rate,
          (q − default)(N) * dof_pos_scale,
          qd(N)            * dof_vel_scale,
          last_action(N) ]

    Then concatenated ``history_len`` times.
    """

    def __init__(
        self,
        spec: G1RobotSpec,
        cfg: UnitreeRLGymObsCfg,
        perm_policy_to_robot: np.ndarray,
    ) -> None:
        self.spec = spec
        self.cfg = cfg
        self.perm_policy_to_robot = np.asarray(perm_policy_to_robot, dtype=np.int64)
        # Inverse: robot_idx → policy_idx (for converting robot-order arrays
        # to policy-order before assembling the obs)
        self.perm_robot_to_policy = np.argsort(self.perm_policy_to_robot)

        n = spec.num_actuated
        self._single_dim = 3 + 3 + 3 + 3 + n + n + n
        self.obs_dim = self._single_dim * cfg.history_len
        self._history: deque[np.ndarray] = deque(maxlen=cfg.history_len)
        self._last_action = np.zeros(n, dtype=np.float32)
        self._actuated_qpos_idx = spec.actuated_qpos_indices
        self._actuated_dof_idx = spec.actuated_dof_indices

    def reset(self) -> None:
        self._history.clear()
        self._last_action[:] = 0.0

    def remember_last_action(self, action: np.ndarray) -> None:
        self._last_action[:] = np.asarray(action, dtype=np.float32).reshape(-1)[: self._last_action.size]

    def build(self, state: RobotState, intent: Intent) -> np.ndarray:
        cfg = self.cfg

        # Joint pos/vel: extract from state (robot order), permute to policy order
        qpos = np.asarray(state.qpos, dtype=np.float32)
        qvel = np.asarray(state.qvel, dtype=np.float32)
        q_robot = qpos[self._actuated_qpos_idx]
        qd_robot = qvel[self._actuated_dof_idx]
        q_policy = q_robot[self.perm_robot_to_policy]
        qd_policy = qd_robot[self.perm_robot_to_policy]

        # Base velocities (LeggedGym uses world frame; some forks use base
        # frame — match the upstream's code path here if your policy diverges)
        if state.base_twist is not None and state.base_twist.shape[0] >= 6:
            base_lin = np.asarray(state.base_twist[:3], dtype=np.float32)
            base_ang = np.asarray(state.base_twist[3:6], dtype=np.float32)
        else:
            base_lin = np.zeros(3, dtype=np.float32)
            base_ang = np.zeros(3, dtype=np.float32)

        # Projected gravity in base frame
        if state.base_pose is not None and state.base_pose.shape[0] >= 7:
            proj_g = _project_world_gravity(np.asarray(state.base_pose[3:7], dtype=np.float32))
        else:
            proj_g = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Command vector
        cmd = np.array(
            [
                float(intent.base_lin_vel[0]) if intent.base_lin_vel is not None else 0.0,
                float(intent.base_lin_vel[1]) if intent.base_lin_vel is not None else 0.0,
                float(intent.base_yaw_rate),
            ],
            dtype=np.float32,
        )

        single = np.concatenate(
            [
                base_lin * cfg.lin_vel_scale,
                base_ang * cfg.ang_vel_scale,
                proj_g,
                cmd,
                (q_policy - cfg.default_dof_pos) * cfg.dof_pos_scale,
                qd_policy * cfg.dof_vel_scale,
                self._last_action,
            ]
        ).astype(np.float32)

        if not self._history:
            for _ in range(cfg.history_len):
                self._history.append(single.copy())
        else:
            self._history.append(single)

        return np.concatenate(list(self._history), axis=0)


# ---------------------------------------------------------------------------
# Action mapper
# ---------------------------------------------------------------------------


class UnitreeRLGymActionMapper(ActionMapper):
    """``action`` (policy order) → joint position targets (robot order).

    Applies the training-time formula
    ``q_target = default_pose + action * action_scale`` and permutes the
    result back into the spec's actuated joint order before emitting a
    :class:`ControlCommand`.
    """

    def __init__(
        self,
        spec: G1RobotSpec,
        cfg: UnitreeRLGymObsCfg,
        perm_policy_to_robot: np.ndarray,
    ) -> None:
        self.spec = spec
        self.cfg = cfg
        self.perm_policy_to_robot = np.asarray(perm_policy_to_robot, dtype=np.int64)
        self.action_dim = spec.num_actuated
        # Pre-build the per-DoF Kp/Kd vectors in robot order.
        self._kp_robot = np.zeros(spec.num_actuated, dtype=np.float64)
        self._kd_robot = np.zeros(spec.num_actuated, dtype=np.float64)
        for policy_idx, robot_idx in enumerate(self.perm_policy_to_robot):
            self._kp_robot[robot_idx] = float(cfg.kp[policy_idx])
            self._kd_robot[robot_idx] = float(cfg.kd[policy_idx])

    def map(self, action: np.ndarray) -> ControlCommand:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != self.action_dim:
            raise ValueError(f"action shape {a.shape} != action_dim {self.action_dim}")
        q_target_policy = self.cfg.default_dof_pos + a * self.cfg.action_scale
        q_target_robot = np.zeros_like(q_target_policy)
        for policy_idx, robot_idx in enumerate(self.perm_policy_to_robot):
            q_target_robot[robot_idx] = q_target_policy[policy_idx]
        q_target_robot = self.spec.clip_to_joint_limits(q_target_robot)
        return ControlCommand(
            kind="joint_pos",
            joint_pos=q_target_robot,
            kp=self._kp_robot.copy(),
            kd=self._kd_robot.copy(),
        )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class UnitreeRLGymG1Controller(RLController):
    """RLController bound to a `unitree_rl_gym` style G1 policy.

    Construct either with a :class:`PolicyArtifact` (loads the .pt and the
    cfg yaml from disk) **or** with explicit ``cfg`` + ``inference_fn``
    (used by tests).

    Example::

        ctrl = UnitreeRLGymG1Controller(
            spec=io.spec,
            artifact=PolicyArtifact(
                ckpt_path=Path("third_party/unitree_rl_gym/.../policy.pt"),
                train_cfg_path=Path("third_party/unitree_rl_gym/.../cfg.yaml"),
                framework="torch",
            ),
            policy_hz=50.0,
            control_hz=200.0,
        )
    """

    def __init__(
        self,
        spec: G1RobotSpec,
        *,
        artifact: Optional[PolicyArtifact] = None,
        cfg: Optional[UnitreeRLGymObsCfg] = None,
        inference_fn: Optional[InferenceFn] = None,
        policy_hz: float = 50.0,
        control_hz: float = 200.0,
    ) -> None:
        if cfg is None:
            if artifact is None or artifact.train_cfg_path is None:
                raise ValueError(
                    "UnitreeRLGymG1Controller requires either an explicit "
                    "`cfg` or an `artifact` with a `train_cfg_path`"
                )
            cfg = self._parse_train_cfg(artifact.train_cfg_path)

        perm = self._build_dof_permutation(spec, cfg.policy_dof_names)
        obs_builder = UnitreeRLGymObsBuilder(spec, cfg, perm)
        action_mapper = UnitreeRLGymActionMapper(spec, cfg, perm)

        super().__init__(
            spec=spec,
            obs_builder=obs_builder,
            action_mapper=action_mapper,
            policy_hz=policy_hz,
            control_hz=control_hz,
            runtime="torch",
            artifact=artifact,
            inference_fn=inference_fn,
        )

    # ------------------------------------------------------------------
    # Cfg parsing — customize per upstream training repo
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_train_cfg(path: Path) -> UnitreeRLGymObsCfg:
        """Parse a `unitree_rl_gym` training cfg yaml.

        The exact key layout depends on the upstream fork. The version
        below targets the public `unitree-rl-gym` G1 walking config; if
        your fork lays out keys differently, subclass and override.
        """
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to parse Unitree RL Gym configs. "
                "Install with `pip install pyyaml`."
            ) from e

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # The expected key paths — adjust per upstream layout.
        obs_scales = raw.get("normalization", {}).get("obs_scales", {})
        control = raw.get("control", {})
        init_state = raw.get("init_state", {})

        policy_dof_names = tuple(raw["asset"]["dof_names"])
        default_pose_map = init_state.get("default_joint_angles", {})
        default_pose = np.asarray(
            [float(default_pose_map[n]) for n in policy_dof_names],
            dtype=np.float32,
        )

        # Kp/Kd: Legged Gym stores these as a {joint_pattern: value} map; we
        # resolve via prefix match against the policy DoF names. Override in
        # a subclass if your fork uses a different layout.
        kp_map = control.get("stiffness", {})
        kd_map = control.get("damping", {})

        def _lookup(d: Mapping[str, float], joint: str, default: float) -> float:
            for pattern, value in d.items():
                if pattern in joint:
                    return float(value)
            return float(default)

        kp = np.asarray(
            [_lookup(kp_map, n, 80.0) for n in policy_dof_names], dtype=np.float64
        )
        kd = np.asarray(
            [_lookup(kd_map, n, 2.0) for n in policy_dof_names], dtype=np.float64
        )

        return UnitreeRLGymObsCfg(
            history_len=int(raw.get("history_len", 1)),
            lin_vel_scale=float(obs_scales.get("lin_vel", 2.0)),
            ang_vel_scale=float(obs_scales.get("ang_vel", 0.25)),
            dof_pos_scale=float(obs_scales.get("dof_pos", 1.0)),
            dof_vel_scale=float(obs_scales.get("dof_vel", 0.05)),
            action_scale=float(control.get("action_scale", 0.25)),
            default_dof_pos=default_pose,
            policy_dof_names=policy_dof_names,
            kp=kp,
            kd=kd,
        )

    @staticmethod
    def _build_dof_permutation(
        spec: G1RobotSpec,
        policy_dof_names: Sequence[str],
    ) -> np.ndarray:
        """Return ``perm[policy_idx] = robot_idx``.

        Raises a clear error if any policy DoF name is missing from the
        robot spec — this is the #1 source of integration bugs and we want
        it to fail loudly at construction, not silently at the first act().
        """
        robot_names = list(spec.actuated_joints)
        missing = [n for n in policy_dof_names if n not in robot_names]
        if missing:
            raise ValueError(
                f"Policy DoF names not found in robot spec: {missing}\n"
                f"Robot has: {robot_names}"
            )
        return np.array(
            [robot_names.index(n) for n in policy_dof_names], dtype=np.int64
        )

"""Callable-based passthrough RL controller.

This module provides three things:

* :class:`PassthroughObsBuilder`   — concatenates a few standard fields
  from :class:`RobotState` / :class:`Intent` into a flat vector. The exact
  layout is configurable but defaults to the bare minimum a "useful" RL
  policy would consume: ``[base_lin_vel(3), base_ang_vel(3), proj_gravity(3),
  cmd(3), q_actuated(N), qd_actuated(N), last_action(N)]``.

* :class:`JointPosActionMapper`    — interprets the policy output as
  ``q_target = default_pose + action * action_scale`` and emits a
  :class:`ControlCommand` of kind ``"joint_pos"``.

* :class:`PassthroughRLController` — convenience subclass of
  :class:`RLController` that takes an inline ``inference_fn`` callable.
  Used by the unit tests and as a reference for the minimal RL adapter
  shape.

Together they let you wrap any Python callable
``f(np.ndarray) -> np.ndarray`` as a Controller without needing torch /
onnx / jax. The most common usage is testing the framework with a hand-
written "policy" that always emits zero actions (i.e. holds the default
pose):

.. code-block:: python

    spec = io.spec
    obs_builder = PassthroughObsBuilder(spec)
    action_mapper = JointPosActionMapper(spec)
    controller = PassthroughRLController(
        spec=spec,
        obs_builder=obs_builder,
        action_mapper=action_mapper,
        inference_fn=lambda obs: np.zeros(action_mapper.action_dim, dtype=np.float32),
        policy_hz=50.0,
        control_hz=200.0,
    )
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from genedynamics.deploy.controllers.rl.base import (
    InferenceFn,
    ObsBuilder,
    ActionMapper,
    RLController,
)
from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
)
from genedynamics.robots.g1 import G1RobotSpec

__all__ = [
    "PassthroughObsBuilder",
    "JointPosActionMapper",
    "PassthroughRLController",
]


# ---------------------------------------------------------------------------
# Obs builder
# ---------------------------------------------------------------------------


@dataclass
class _PassthroughObsCfg:
    """Internal scale factors. Match the most common Legged Gym defaults."""

    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    history_len: int = 1


class PassthroughObsBuilder(ObsBuilder):
    """Generic obs builder that concatenates the standard locomotion features.

    Layout (one frame, length ``3 + 3 + 3 + 3 + N + N + N`` where ``N`` is
    the number of actuated joints)::

        [ base_lin_vel(3) * lin_vel_scale,
          base_ang_vel(3) * ang_vel_scale,
          projected_gravity(3),
          cmd_vx, cmd_vy, cmd_yaw_rate,
          (q − default_pose)(N) * dof_pos_scale,
          qd(N) * dof_vel_scale,
          last_action(N) ]

    The whole frame is then stacked ``history_len`` times.
    """

    def __init__(
        self,
        spec: G1RobotSpec,
        *,
        default_pose: Optional[np.ndarray] = None,
        history_len: int = 1,
        lin_vel_scale: float = 2.0,
        ang_vel_scale: float = 0.25,
        dof_pos_scale: float = 1.0,
        dof_vel_scale: float = 0.05,
    ) -> None:
        self.spec = spec
        self.cfg = _PassthroughObsCfg(
            lin_vel_scale=lin_vel_scale,
            ang_vel_scale=ang_vel_scale,
            dof_pos_scale=dof_pos_scale,
            dof_vel_scale=dof_vel_scale,
            history_len=history_len,
        )
        self.default_pose = (
            np.asarray(default_pose, dtype=np.float32).reshape(-1)
            if default_pose is not None
            else np.asarray(spec.stand_ctrl, dtype=np.float32)
        )
        if self.default_pose.size != spec.num_actuated:
            raise ValueError(
                f"default_pose has {self.default_pose.size} entries, expected {spec.num_actuated}"
            )

        n = spec.num_actuated
        self._single_dim = 3 + 3 + 3 + 3 + n + n + n
        self.obs_dim = self._single_dim * self.cfg.history_len
        self._history: deque[np.ndarray] = deque(maxlen=self.cfg.history_len)
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
        # Base velocities (world frame; the LeggedGym baseline rotates these
        # into the base frame, but for the passthrough we keep it simple).
        if state.base_twist is not None and state.base_twist.shape[0] >= 6:
            base_lin = np.asarray(state.base_twist[:3], dtype=np.float32)
            base_ang = np.asarray(state.base_twist[3:6], dtype=np.float32)
        else:
            base_lin = np.zeros(3, dtype=np.float32)
            base_ang = np.zeros(3, dtype=np.float32)

        # Projected gravity in the base frame (constant world gravity rotated
        # by the inverse pelvis quaternion).
        if state.base_pose is not None and state.base_pose.shape[0] >= 7:
            quat = np.asarray(state.base_pose[3:7], dtype=np.float32)
            proj_g = _project_world_gravity(quat)
        else:
            proj_g = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Joint pos / vel
        qpos = np.asarray(state.qpos, dtype=np.float32)
        qvel = np.asarray(state.qvel, dtype=np.float32)
        q_act = qpos[self._actuated_qpos_idx]
        qd_act = qvel[self._actuated_dof_idx]

        # Cmd vector
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
                (q_act - self.default_pose) * cfg.dof_pos_scale,
                qd_act * cfg.dof_vel_scale,
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


class JointPosActionMapper(ActionMapper):
    """Map ``action ∈ R^N`` to ``q_target = default_pose + action * scale``.

    The output is a :class:`ControlCommand` of kind ``"joint_pos"`` with
    pre-built Kp / Kd vectors. Joint order is the spec's actuated order.
    """

    def __init__(
        self,
        spec: G1RobotSpec,
        *,
        default_pose: Optional[np.ndarray] = None,
        action_scale: float = 0.25,
        kp: float = 80.0,
        kd: float = 2.0,
    ) -> None:
        self.spec = spec
        self.action_dim = spec.num_actuated
        self.action_scale = float(action_scale)
        self.default_pose = (
            np.asarray(default_pose, dtype=np.float64).reshape(-1)
            if default_pose is not None
            else np.asarray(spec.stand_ctrl, dtype=np.float64)
        )
        if self.default_pose.size != spec.num_actuated:
            raise ValueError(
                f"default_pose has {self.default_pose.size} entries, expected {spec.num_actuated}"
            )
        self._kp = np.full(spec.num_actuated, float(kp), dtype=np.float64)
        self._kd = np.full(spec.num_actuated, float(kd), dtype=np.float64)

    def map(self, action: np.ndarray) -> ControlCommand:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != self.action_dim:
            raise ValueError(f"action shape {a.shape} != action_dim {self.action_dim}")
        q_target = self.default_pose + a * self.action_scale
        q_target = self.spec.clip_to_joint_limits(q_target)
        return ControlCommand(
            kind="joint_pos",
            joint_pos=q_target,
            kp=self._kp.copy(),
            kd=self._kd.copy(),
        )


# ---------------------------------------------------------------------------
# Convenience controller
# ---------------------------------------------------------------------------


class PassthroughRLController(RLController):
    """:class:`RLController` constructed directly from a Python callable.

    Args:
        spec: Robot spec.
        obs_builder: Optional :class:`PassthroughObsBuilder`. Defaults to a
            single-frame builder against ``spec.stand_ctrl``.
        action_mapper: Optional :class:`JointPosActionMapper`. Defaults to
            ``action_scale=0.25``, ``Kp=80``, ``Kd=2``.
        inference_fn: ``f(obs) -> action`` callable. Required.
        policy_hz / control_hz / runtime: Forwarded to :class:`RLController`.
    """

    def __init__(
        self,
        spec: G1RobotSpec,
        *,
        inference_fn: InferenceFn,
        obs_builder: Optional[PassthroughObsBuilder] = None,
        action_mapper: Optional[JointPosActionMapper] = None,
        policy_hz: float = 50.0,
        control_hz: float = 200.0,
        runtime: str = "numpy",
    ) -> None:
        super().__init__(
            spec=spec,
            obs_builder=obs_builder or PassthroughObsBuilder(spec),
            action_mapper=action_mapper or JointPosActionMapper(spec),
            policy_hz=policy_hz,
            control_hz=control_hz,
            runtime=runtime,
            inference_fn=inference_fn,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_world_gravity(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Rotate world gravity ``[0, 0, -1]`` into the base frame.

    The base orientation is given as a unit quaternion in (w, x, y, z) order.
    Returns a 3-vector in the base frame, which is what most legged-gym
    style policies expect as their "projected gravity" feature.
    """
    w, x, y, z = (float(q) for q in quat_wxyz)
    # Inverse rotation: gravity is a fixed world vector, expressed in the
    # body frame as R(q)^T · [0, 0, -1].
    # The third column of R(q)^T (= the third row of R(q)) gives that result
    # directly with z component negated.
    g_b_x = -2.0 * (x * z - w * y)
    g_b_y = -2.0 * (y * z + w * x)
    g_b_z = -(1.0 - 2.0 * (x * x + y * y))
    return np.array([g_b_x, g_b_y, g_b_z], dtype=np.float32)

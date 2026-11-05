"""Public CPU adapter for the explicitly versioned passive-socket extension."""
from __future__ import annotations

import numpy as np

from genedynamics.tasks.transfer.compliant_socket import (
    CompliantSocketMechanismSpec, FIXTURE_FIELDS, FIXTURE_REST_POSITION,
    OBSERVATION_VERSION,
)
from .transfer_task import CpuTaskAdapter, FLAT_FIELDS


class CompliantSocketTaskAdapter(CpuTaskAdapter):
    """Expose robot encoders and measured fixture pose as distinct sensors."""
    state_layout = "transfer_public_fixture_flat_v1"

    def __init__(self, spec, *, mechanism_spec=None, **kwargs):
        super().__init__(spec, mechanism_spec=mechanism_spec or CompliantSocketMechanismSpec(), **kwargs)
        self._robot_qpos_indices = self._provider.robot_qpos_indices.copy()
        self._robot_dof_indices = self._provider.robot_dof_indices.copy()

    @property
    def observation_size(self):
        if self._state is None:
            return super().observation_size - 2 + 18
        return self.flat_observation().shape[-1]

    def describe(self):
        return {**super().describe(), "flat_fields": (*FLAT_FIELDS, *FIXTURE_FIELDS),
                "observation_version": OBSERVATION_VERSION,
                "robot_joint_names": [self._model.joint(int(i)).name
                                      for i in self._provider.robot_joint_ids]}

    def observe(self):
        public = super().observe()
        public["joint_position"] = public["joint_position"][:, self._robot_qpos_indices]
        public["joint_velocity"] = public["joint_velocity"][:, self._robot_dof_indices]
        fixtures = [self._provider.fixture_observation(data) for data in self._state.physics]
        for name in FIXTURE_FIELDS:
            public[name] = np.asarray([row[name] for row in fixtures], dtype=np.float32)
        displacement = public["fixture_position"] - FIXTURE_REST_POSITION.astype(np.float32)
        public["goal_position"] += displacement
        return public

    def flat_observation(self, observation=None):
        observation = self.observe() if observation is None else observation
        return np.concatenate([np.asarray(observation[key]).reshape(self.num_envs, -1)
                               for key in (*FLAT_FIELDS, *FIXTURE_FIELDS)], axis=-1)

    def public_state_fields(self):
        public, fields, offset = self.observe(), [], 0
        for name in (*FLAT_FIELDS, *FIXTURE_FIELDS):
            value = np.asarray(public[name]); size = int(np.prod(value.shape[1:]))
            fields.append(dict(name=name, shape=list(value.shape[1:]), start=offset,
                               stop=offset + size, dtype=str(value.dtype)))
            offset += size
        return fields

    def _row(self, trace, index, requested):
        row = super()._row(trace, index, requested)
        data, model = self._state.physics[index], self._model
        joints = self._provider.robot_joint_ids
        q = data.qpos[self._robot_qpos_indices]
        v = data.qvel[self._robot_dof_indices]
        margins = np.minimum(q - model.jnt_range[joints, 0], model.jnt_range[joints, 1] - q)
        limited = model.jnt_limited[joints].astype(bool)
        limiting = int(np.argmin(np.where(limited, margins, np.inf)))
        row.update(
            joint_position=q.copy(), joint_velocity=v.copy(),
            joint_speed_peak=float(np.max(np.abs(v))),
            joint_limit=bool(np.any(margins[limited] < -1e-4)),
            joint_margin_rad=margins.copy(),
            minimum_joint_margin_rad=float(margins[limiting]),
            limiting_joint=model.joint(int(joints[limiting])).name,
            **self._provider.fixture_observation(data),
        )
        return row

    def _finish_step(self, *args, **kwargs):
        step = super()._finish_step(*args, **kwargs)
        distance = np.linalg.norm(step.observation["tcp_position"] - step.observation["goal_position"], axis=-1)
        rewards = -distance - .001 + np.where(step.status == 1, 1., np.where(step.status != 0, -1., 0.))
        return type(step)(step.observation, rewards.astype(np.float32), step.done,
                          step.status, step.executed_action, step.trace)

    def step_acp(self, command):
        raise NotImplementedError("ACP passive-fixture torque execution needs an explicit qualified binding")

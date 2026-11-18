"""Real CPU task adapter: public sensors, histories, replay and charged resets.

The physics objects and construction mechanism never enter observation dicts.
All four routes use existing MuJoCo scenes; T4 Go2+Z1 retains its actual stance
QP. Native robot/vision/controller reproduction remains a separate obligation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import numpy as np

from genedynamics.tasks.transfer.episode import (
    CONTRACT_VERSION,
    EpisodeRecord,
    TaskStep,
    numpy_action,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.specs import task_spec

FLAT_FIELDS = (
    "joint_position", "joint_velocity", "tcp_position", "tcp_rotation", "tcp_twist",
    "measured_wrench", "goal_position", "goal_rotation", "previous_action", "previous_request",
    "reference_position", "reference_rotation", "elapsed_seconds", "body_position", "body_twist",
    "controller_feedback",
)


def _take(value, order):
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_take(item, order) for item in value))
    if isinstance(value, tuple):
        return tuple(value[int(index)] for index in order)
    return np.asarray(value)[order].copy()


class CpuTaskAdapter:
    action_size = 13

    def __init__(self, spec, *, mechanism_spec=None, seed=0, history_len=4,
                 shared_execution=None):
        import mujoco
        from genedynamics.tasks.transfer.runtime import ExecutionSpec, MechanismSpec

        if isinstance(history_len, bool) or int(history_len) != history_len or history_len < 1:
            raise ValueError("history_len must be a positive integer")
        self.spec, self.config = spec, spec.config
        self.seed, self.history_len = int(seed), int(history_len)
        if mechanism_spec is not None and not isinstance(mechanism_spec, MechanismSpec):
            raise TypeError("one construction-only mechanism per adapter; heterogeneous batches need separate adapters")
        from genedynamics.tasks.transfer.specs import validate_shared_execution
        self.shared_execution = validate_shared_execution(
            spec, mechanism_spec or MechanismSpec(), shared_execution,
        )
        if spec.family == "obstacle":
            from genedynamics.envs.domains.manipulation.transfer_workcell.runtime import TransferWorkcell
            self.runner = TransferWorkcell(execution=ExecutionSpec(backend="numpy", device="cpu", physics_backend="mujoco"),
                                         task=self.config, mechanism=mechanism_spec)
        else:
            from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
            self.runner = ContactRunner(config=self.config, mechanism=mechanism_spec)
        self._provider = self.runner._impl
        # CPU contact tasks retain exact physics-rate contact evidence for
        # independent evaluation.  The provider keeps it on a diagnostic side
        # channel; it never enters observe(), histories, or policy inputs.
        if spec.family != "obstacle" or spec.name in {
                "aperture_brake", "sequential_aperture_reorientation"}:
            self._provider.record_events = True
        self._model, self._sid = self._provider.model, self._provider.sid
        self._state = None
        self._episodes = []
        self._current = []
        self._mode = None
        self._body_start_planar = self._body_target = None
        self._previous_request = None
        self._previous_action = None
        self._public_reference_position = self._public_reference_rotation = None
        self._history_obs = self._history_act = self._history_valid = None
        self._acp_executor = None
        self.accounting = dict(reset_calls=0, reset_worlds=0, preparation_control_steps=0,
                               control_steps=0, physics_steps=0, command_requests=0)
        # Irreversible real executions on this adapter instance. Logical
        # episode state may be rewound; already incurred calls never are.
        self.lifetime_accounting = dict(self.accounting)
        self._integration = mujoco.mjtState.mjSTATE_INTEGRATION

    @property
    def state(self):
        """Evaluator/checkpoint access only; this is never a public policy view."""
        return self._state

    @property
    def num_envs(self):
        return 0 if self._state is None else len(self._state.sample_id)

    @property
    def observation_size(self):
        if self._state is None:
            # Dimension is model/schema metadata, no target reset is needed.
            return self._model.nq + self._model.nv + 3 + 9 + 6 + 6 + 3 + 9 + 13 + 13 + 3 + 9 + 1 + 3 + 6 + 2
        return self.flat_observation().shape[-1]

    @property
    def command_size(self):
        return 14

    @property
    def records(self):
        return deepcopy(self._episodes)

    def describe(self):
        return {**self.spec.describe(), "scene": self.runner.describe(),
                "history_len": self.history_len, "observation_size": self.observation_size,
                "flat_fields": FLAT_FIELDS, "command_size": self.command_size,
                "action_size": self.action_size, "accounting": dict(self.accounting),
                "lifetime_accounting": dict(self.lifetime_accounting),
                "shared_execution": deepcopy(self.shared_execution),
                "scope": "adapted task interface, not native baseline or GPU physical equivalence"}

    def public_body_intent(self, public):
        """Return the task-owned body request shared by every T4 method row."""
        if self.spec.name == "reposition_then_dock":
            actual = np.asarray(public["body_planar_pose"], dtype=np.float32)
            target = np.asarray(public["body_target"], dtype=np.float32)
            if actual.shape != target.shape or actual.ndim != 2 or actual.shape[1] != 3:
                raise ValueError("public planar body pose/target shape differs")
            return np.clip(
                2. * (target - actual),
                np.asarray([-.010, -.010, -.025], dtype=np.float32),
                np.asarray([.010, .010, .025], dtype=np.float32),
            ).astype(np.float32)
        if self.shared_execution.get("body_intent_profile") != "public_planar_v1":
            return None
        elapsed = np.asarray(public["elapsed_seconds"], dtype=float).reshape(-1)
        start = float(self.shared_execution["body_intent_start_seconds"])
        stop = start + float(self.shared_execution["body_intent_duration_seconds"])
        active = (elapsed >= start) & (elapsed < stop)
        velocity = np.asarray(
            self.shared_execution["body_velocity_world"], dtype=np.float32,
        )
        return np.where(active[:, None], velocity[None], 0.).astype(np.float32)

    def _require_reset(self):
        if self._state is None:
            raise RuntimeError("reset the task before observing or stepping")

    def _charge(self, key, value):
        self.accounting[key] += int(value)
        self.lifetime_accounting[key] += int(value)

    def reset(self, sample_ids=(0,), *, seed=None, mask=None, initial=None, modes=None):
        from genedynamics.tasks.transfer.runtime import RandomRequest

        sample_ids = tuple(int(value) for value in sample_ids)
        used_seed = self.seed if seed is None else int(seed)
        request = RandomRequest(used_seed, sample_ids)
        initial = initial or ("jammed" if self.spec.alias in {"T3", "T4"} else "clear")
        if self.spec.family == "obstacle" and initial != "clear":
            raise ValueError("obstacle transfer supports the declared clear reset only")
        old = self._state
        if mask is not None:
            if old is None or tuple(old.sample_id) != sample_ids:
                raise ValueError("partial reset keeps slot identities; reorder explicitly before resetting")
            reset_mask = np.asarray(mask, dtype=bool)
            if reset_mask.shape != (len(sample_ids),):
                raise ValueError("partial reset mask must have one flag per sample")
        else:
            reset_mask = np.ones(len(sample_ids), dtype=bool)
        reset_modes = np.asarray(
            (self._mode if mask is not None else np.zeros(len(sample_ids), dtype=int))
            if modes is None else modes,
        )
        if reset_modes.shape != (len(sample_ids),) or not np.isin(reset_modes, (0, 1)).all():
            raise ValueError("reset modes must contain one 0/1 request per world")
        # Preparation is part of a target interaction. Go2 matched-keyway
        # reset performs 50 genuine 20 ms stance transitions before placement.
        self._charge("reset_calls", 1)
        self._charge("reset_worlds", int(reset_mask.sum()))
        if mask is None:
            self._current = [-1] * len(sample_ids)
        # Register requested resets before physics; an unsuccessful preparation
        # remains a charged, incomplete row instead of disappearing.
        for slot in np.flatnonzero(reset_mask):
            identity = sample_ids[slot]
            generation = sum(record.sample_id == identity for record in self._episodes)
            self._current[slot] = len(self._episodes)
            self._episodes.append(EpisodeRecord(identity, used_seed, generation, initial,
                                                requested_mode=int(reset_modes[slot])))
        kwargs = {} if mask is None else dict(previous=old, mask=reset_mask)
        if self.spec.family != "obstacle":
            kwargs["initial"] = initial
        if self.config.robot == "go2_z1":
            # Count real stance preparation even if it fails before step 50.
            # The wrapper is local to this provider instance and always removed.
            original_step = self._provider._step
            def prepare_step(state, *args, **step_kwargs):
                result, trace = original_step(state, *args, **step_kwargs)
                if trace.valid:
                    self._charge("preparation_control_steps", 1)
                    slot = sample_ids.index(int(state.sample_id))
                    record = self._episodes[self._current[slot]]
                    record.preparation_control_steps += 1
                    record.preparation_traces.append({key: np.asarray(getattr(trace, key)).copy() for key in trace._fields})
                return result, trace
            self._provider._step = prepare_step
            try:
                self._state = self.runner.reset_batch(request, **kwargs)
            finally:
                self._provider._step = original_step
        else:
            self._state = self.runner.reset_batch(request, **kwargs)
        for index in np.flatnonzero(reset_mask):
            data = self._state.physics[index]
            physical = {
                key: np.asarray(getattr(data, key)).tolist()
                for key in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat")
            }
            self._episodes[self._current[index]].reset_evidence = {
                "physical_state": physical,
                "physical_state_sha256": hashlib.sha256(
                    json.dumps(physical, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                "identity_excluded_from_physical_hash": True,
                "tcp_position": data.site_xpos[self._sid].copy(),
                "tcp_rotation": data.site_xmat[self._sid].reshape(3, 3).copy(),
                "reset_profile": getattr(self.config, "reset_profile", "legacy"),
                "reset_joint_noise": getattr(self.config, "reset_joint_noise", None),
                "reset_task_noise": getattr(self.config, "reset_task_noise", None),
            }
        self.seed = used_seed
        if mask is None:
            self._mode = reset_modes.astype(np.int32, copy=True)
            self._previous_request = np.zeros((len(sample_ids), 13), dtype=np.float32)
            self._previous_action = np.zeros_like(self._previous_request)
            self._public_reference_position = np.stack([data.site_xpos[self._sid].copy() for data in self._state.physics])
            self._public_reference_rotation = np.stack([data.site_xmat[self._sid].reshape(3, 3).copy() for data in self._state.physics])
            self._history_obs = np.zeros((len(sample_ids), self.history_len, self.observation_size), dtype=np.float32)
            self._history_act = np.zeros((len(sample_ids), self.history_len, 13), dtype=np.float32)
            self._history_valid = np.zeros((len(sample_ids), self.history_len), dtype=bool)
            if self.spec.name == "reposition_then_dock":
                from genedynamics.tasks.transfer.reposition_dock import target_from_reset
                starts = []
                for data in self._state.physics:
                    rotation = data.xmat[self._model.body("base").id].reshape(3, 3)
                    starts.append([
                        data.xpos[self._model.body("base").id, 0],
                        data.xpos[self._model.body("base").id, 1],
                        np.arctan2(rotation[1, 0], rotation[0, 0]),
                    ])
                self._body_start_planar = np.asarray(starts, dtype=np.float32)
                self._body_target = np.stack([
                    target_from_reset(start, mode)
                    for start, mode in zip(self._body_start_planar, self._mode)
                ])
        else:
            self._mode[reset_mask] = reset_modes[reset_mask]
            self._previous_request[reset_mask] = 0
            self._previous_action[reset_mask] = 0
            self._history_obs[reset_mask] = 0
            self._history_act[reset_mask] = 0
            self._history_valid[reset_mask] = False
            for index in np.flatnonzero(reset_mask):
                self._public_reference_position[index] = self._state.physics[index].site_xpos[self._sid]
                self._public_reference_rotation[index] = self._state.physics[index].site_xmat[self._sid].reshape(3, 3)
            if self.spec.name == "reposition_then_dock":
                from genedynamics.tasks.transfer.reposition_dock import target_from_reset
                for index in np.flatnonzero(reset_mask):
                    data = self._state.physics[index]
                    rotation = data.xmat[self._model.body("base").id].reshape(3, 3)
                    self._body_start_planar[index] = [
                        data.xpos[self._model.body("base").id, 0],
                        data.xpos[self._model.body("base").id, 1],
                        np.arctan2(rotation[1, 0], rotation[0, 0]),
                    ]
                    self._body_target[index] = target_from_reset(
                        self._body_start_planar[index], self._mode[index],
                    )
        if self.spec.name == "reposition_then_dock":
            for index in np.flatnonzero(reset_mask):
                record = self._episodes[self._current[index]]
                record.reset_evidence.update(
                    body_start_planar=self._body_start_planar[index].copy(),
                    body_target=self._body_target[index].copy(),
                )
        if self._acp_executor is not None:
            self._acp_executor.reset(reset_mask, self._state)
        self._push_history(reset_mask)
        return self.observe()

    def set_modes(self, modes):
        self._require_reset()
        modes = np.asarray(modes, dtype=int)
        if modes.ndim == 0:
            modes = np.full(self.num_envs, modes)
        if modes.shape != (self.num_envs,) or not np.isin(modes, (0, 1)).all():
            raise ValueError("one declared source-mode request (0/1) per world is required")
        self._mode = modes.copy()
        if self.spec.name == "reposition_then_dock":
            from genedynamics.tasks.transfer.reposition_dock import target_from_reset
            self._body_target = np.stack([
                target_from_reset(start, mode)
                for start, mode in zip(self._body_start_planar, self._mode)
            ])
        for slot, mode in enumerate(modes):
            current = self._current[slot]
            if current >= 0:
                self._episodes[current].requested_mode = int(mode)

    def observe(self):
        import mujoco

        self._require_reset()
        state, model, sid = self._state, self._model, self._sid
        joint_q, joint_v, positions, rotations, twists, wrenches = [], [], [], [], [], []
        body_positions, body_twists, feedback = [], [], []
        for index, data in enumerate(state.physics):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            joint_q.append(data.qpos.copy())
            joint_v.append(data.qvel.copy())
            positions.append(data.site_xpos[sid].copy())
            rotations.append(data.site_xmat[sid].reshape(3, 3).copy())
            twists.append(np.r_[jp @ data.qvel, jr @ data.qvel])
            if self.spec.family == "obstacle":
                # T1 only permits collision-free motion. Contact force truth
                # is not introduced as an additional privileged sensor.
                wrenches.append(np.zeros(6))
            else:
                wrenches.append(self._provider._measure(data)[0].copy())
            body_position, body_twist, controller_feedback = np.zeros(3), np.zeros(6), np.zeros(2)
            if self.config.robot == "go2_z1":
                body = model.body("base").id
                body_position = data.xpos[body].copy()
                mujoco.mj_jacBody(model, data, jp, jr, body)
                body_twist = np.r_[jp @ data.qvel, jr @ data.qvel]
                rotation = data.xmat[body].reshape(3, 3)
                tilt = np.arctan2(np.linalg.norm(rotation[:2, 2]), rotation[2, 2])
                controller_feedback = np.array([state.wbc_residual[index], tilt])
            elif getattr(self.config, "mobile", False):
                body_position = np.r_[data.qpos[:2], 0.]
                body_twist = np.r_[data.qvel[:2], 0., 0., 0., data.qvel[2]]
            body_positions.append(body_position)
            body_twists.append(body_twist)
            feedback.append(controller_feedback)
        goal = self.config.goal_position if self.spec.family == "obstacle" else self.config.goal
        from genedynamics.envs.domains.manipulation.transfer_workcell.workcell import rotation
        goal_rot = rotation(self.config.goal_yaw if self.spec.family == "obstacle" else 0.)
        data = dict(
            joint_position=joint_q, joint_velocity=joint_v, tcp_position=positions, tcp_rotation=rotations,
            tcp_twist=twists, measured_wrench=wrenches, goal_position=np.tile(goal, (self.num_envs, 1)),
            goal_rotation=np.tile(goal_rot, (self.num_envs, 1, 1)), previous_action=self._previous_action,
            previous_request=self._previous_request, reference_position=self._public_reference_position,
            reference_rotation=self._public_reference_rotation,
            elapsed_seconds=np.asarray(state.steps)[:, None] * self.config.control_dt,
            body_position=body_positions, body_twist=body_twists, controller_feedback=feedback,
        )
        if self.spec.name == "reposition_then_dock":
            from genedynamics.tasks.transfer.reposition_dock import TARGET_BOUNDS
            planar = []
            for value in state.physics:
                rotation = value.xmat[model.body("base").id].reshape(3, 3)
                planar.append([
                    value.xpos[model.body("base").id, 0],
                    value.xpos[model.body("base").id, 1],
                    np.arctan2(rotation[1, 0], rotation[0, 0]),
                ])
            data.update(
                body_planar_pose=planar,
                body_target=self._body_target,
                body_target_bounds=np.tile(TARGET_BOUNDS[None], (self.num_envs, 1, 1)),
            )
        if self.spec.name in {
                "aperture_brake", "sequential_aperture_reorientation"}:
            from genedynamics.tasks.transfer.aperture import object_pose_and_twist
            object_positions, object_rotations, object_twists = [], [], []
            for position, rotation_value, twist in zip(positions, rotations, twists):
                object_position, object_rotation, object_twist = object_pose_and_twist(
                    np.asarray(position), np.asarray(rotation_value), np.asarray(twist),
                )
                object_positions.append(object_position)
                object_rotations.append(object_rotation)
                object_twists.append(object_twist)
            geometry = [
                self.config.aperture_plane_x, self.config.aperture_center_y,
                self.config.aperture_center_z, self.config.aperture_half_width,
                self.config.aperture_half_height, *self.config.aperture_exit_zone,
            ]
            if self.spec.name == "sequential_aperture_reorientation":
                geometry.extend([
                    self.config.second_aperture_plane_y,
                    self.config.second_aperture_center_x,
                    self.config.second_aperture_center_z,
                    self.config.second_aperture_half_width,
                    self.config.second_aperture_half_height,
                    *self.config.reorientation_zone_x,
                    *self.config.reorientation_zone_y,
                    *self.config.sequential_exit_zone_y,
                ])
            data.update(
                carried_object_position=object_positions,
                carried_object_rotation=object_rotations,
                carried_object_twist=object_twists,
                aperture_geometry=np.tile(np.asarray(geometry), (self.num_envs, 1)),
            )
        # Own every array. A policy cannot mutate simulator/controller/history
        # state by modifying an observation it was given.
        return {key: np.array(value, dtype=np.float32, copy=True) for key, value in data.items()}

    def flat_observation(self, observation=None):
        observation = self.observe() if observation is None else observation
        return np.concatenate([np.asarray(observation[key]).reshape(self.num_envs, -1) for key in FLAT_FIELDS], axis=-1)

    def public_state_fields(self):
        """Describe the serialized public vector without private state fields."""
        public, fields, offset = self.observe(), [], 0
        for name in FLAT_FIELDS:
            value = np.asarray(public[name])
            size = int(np.prod(value.shape[1:]))
            fields.append(dict(name=name, shape=list(value.shape[1:]),
                               start=offset, stop=offset + size, dtype=str(value.dtype)))
            offset += size
        return fields

    def _push_history(self, mask):
        indices = np.flatnonzero(mask)
        if not len(indices):
            return
        values = self.flat_observation()
        self._history_obs[indices, :-1] = self._history_obs[indices, 1:]
        self._history_act[indices, :-1] = self._history_act[indices, 1:]
        self._history_valid[indices, :-1] = self._history_valid[indices, 1:]
        self._history_obs[indices, -1] = values[indices]
        self._history_act[indices, -1] = self._previous_action[indices]
        self._history_valid[indices, -1] = True

    def fada_observation(self):
        observation = self.observe()
        command = np.concatenate((observation["goal_position"], observation["goal_rotation"].reshape(self.num_envs, 9),
                                  np.eye(2, dtype=np.float32)[self._mode]), axis=-1)
        return {"history_obs": self._history_obs.copy(), "history_act": self._history_act.copy(),
                "history_valid_mask": self._history_valid.copy(), "current_command": command}

    def reorder(self, order):
        self._require_reset()
        order = np.asarray(order, dtype=int)
        if order.shape != (self.num_envs,) or sorted(order.tolist()) != list(range(self.num_envs)):
            raise ValueError("order must be a permutation of the current world slots")
        self._state = _take(self._state, order)
        self._current = [self._current[int(index)] for index in order]
        names = [
            "_mode", "_previous_request", "_previous_action",
            "_public_reference_position", "_public_reference_rotation",
            "_history_obs", "_history_act", "_history_valid",
        ]
        if self.spec.name == "reposition_then_dock":
            names.extend(("_body_start_planar", "_body_target"))
        for name in names:
            setattr(self, name, getattr(self, name)[order].copy())
        if self._acp_executor is not None:
            self._acp_executor.reorder(order)
        return self.observe()

    def _row(self, trace, index, requested):
        data = self._state.physics[index]
        model = self._model
        scalar = lambda key: np.asarray(getattr(trace, key))[index, 0].item()
        row = {"tcp_position": np.asarray(trace.tcp_position[index, 0]).copy(),
               "tcp_rotation": np.asarray(trace.tcp_rotation[index, 0]).copy(),
               "valid": bool(scalar("valid")), "backend_status": int(scalar("status")),
               "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(requested).all())}
        if self.spec.family == "obstacle":
            row.update(clearance=float(scalar("clearance_lower_bound")), joint_speed_peak=float(scalar("max_joint_speed")))
            if self.spec.name in {
                    "aperture_brake", "sequential_aperture_reorientation"}:
                events = getattr(self._provider, "last_event_trace", None)
                required = (
                    "tcp_position_substeps", "tcp_rotation_substeps",
                    "tcp_twist_substeps", "frame_contact_substeps",
                    "frame_contact_mask_substeps",
                )
                if events is None or any(name not in events for name in required):
                    raise RuntimeError("aperture step omitted physics-rate evaluator evidence")
                row.update(**{
                    name: np.asarray(events[name][index, 0]).copy()
                    for name in required
                })
            joint_pos, joint_ids = data.qpos, np.arange(model.njnt)
        elif self.config.robot == "go2_z1":
            joint_pos = data.qpos[self._provider.qindices]
            joint_ids = model.dof_jntid[self._provider.dofs]
            row.update(joint_speed_peak=float(np.max(np.abs(data.qvel[self._provider.dofs[12:]]))),
                       body_height=float(trace.base_position[index, 0, 2]),
                       base_tilt=float(scalar("base_tilt")), support_count=int(scalar("support_count")),
                       wbc_residual=float(scalar("wbc_residual")))
            if self.spec.name == "reposition_then_dock":
                rotation = data.xmat[model.body("base").id].reshape(3, 3)
                row.update(
                    body_planar_pose=np.asarray([
                        data.xpos[model.body("base").id, 0],
                        data.xpos[model.body("base").id, 1],
                        np.arctan2(rotation[1, 0], rotation[0, 0]),
                    ]),
                    body_target=self._body_target[index].copy(),
                    body_twist=np.r_[data.qvel[:3], data.qvel[3:6]].copy(),
                    foot_positions=data.geom_xpos[self._provider.feet].copy(),
                    foot_anchor_positions=np.asarray(
                        self._state.foot_anchors[index], dtype=float,
                    ).copy(),
                )
        else:
            joint_pos, joint_ids = data.qpos, np.arange(model.njnt)
            row["joint_speed_peak"] = float(np.max(np.abs(data.qvel)))
        tolerance = 1e-5 if self.spec.family == "obstacle" else 1e-4
        row["joint_limit"] = bool(np.any(joint_pos < model.jnt_range[joint_ids, 0] - tolerance)
                                  or np.any(joint_pos > model.jnt_range[joint_ids, 1] + tolerance))
        # Evaluator-only evidence. None of these plant-applied fields is added
        # to observe()/flat_observation(), so the public actor contract stays fixed.
        limited = np.asarray(model.jnt_limited[joint_ids], dtype=bool)
        margins = np.minimum(joint_pos - model.jnt_range[joint_ids, 0],
                             model.jnt_range[joint_ids, 1] - joint_pos)
        limiting = int(np.argmin(np.where(limited, margins, np.inf)))
        row.update(
            joint_position=np.asarray(data.qpos).copy(),
            joint_velocity=np.asarray(data.qvel).copy(),
            joint_margin_rad=margins.copy(),
            minimum_joint_margin_rad=float(margins[limiting]) if limited.any() else None,
            limiting_joint=str(model.joint(int(joint_ids[limiting])).name) if limited.any() else None,
            public_reference_position=self._public_reference_position[index].copy(),
            public_reference_rotation=self._public_reference_rotation[index].copy(),
            public_reference_lead=(self._public_reference_position[index] - row["tcp_position"]).copy(),
            issued_action=self._previous_action[index].copy(),
            applied_action=np.asarray(trace.command[index, 0]).copy(),
        )
        for name in ("force_integral", "approach_integral", "contact_seen", "loss_count", "hold_count"):
            value = getattr(self._state, name, None)
            if value is not None:
                row[f"execution_{name}"] = np.asarray(value)[index].copy()
        for name in ("reference_position", "reference_rotation"):
            value = getattr(self._state, name, None)
            if value is not None:
                row[f"applied_{name}"] = np.asarray(value)[index].copy()
        if self.spec.family != "obstacle":
            row.update(normal_force=float(scalar("normal_force")), force_peak=float(scalar("force_peak")),
                       lateral_force_peak=float(scalar("lateral_force_peak")),
                       forbidden_collision=bool(scalar("forbidden_collision")),
                       axial_support=bool(self._provider._axial_support(data)),
                       loading=bool(requested[12] > .1 and trace.command[index, 0, 12] > .1))
            exact_evidence_task = (
                self.config.execution_profile == "substep_wrench_governor"
                or getattr(self.config,"guide_profile",None)=="planar_corner_v1"
                or hasattr(self._provider,"fixture_observation")
            )
            if (self._episodes[self._current[index]].controller == "shared_skill"
                    and exact_evidence_task):
                events = getattr(self._provider, "last_event_trace", None)
                required = (
                    "axial_support_substeps", "normal_force_substeps",
                    "tcp_position_substeps", "joint_speed_substeps",
                    "force_peak_substeps", "lateral_force_substeps",
                )
                if events is None or any(name not in events for name in required):
                    raise RuntimeError("shared contact step omitted physics-rate evaluator evidence")
                row.update(
                    axial_support_substeps=np.asarray(
                        events["axial_support_substeps"][index, 0], dtype=bool,
                    ).copy(),
                    normal_force_substeps=np.asarray(
                        events["normal_force_substeps"][index, 0], dtype=float,
                    ).copy(),
                    tcp_position_substeps=np.asarray(
                        events["tcp_position_substeps"][index, 0], dtype=float,
                    ).copy(),
                    joint_speed_substeps=np.asarray(
                        events["joint_speed_substeps"][index, 0], dtype=float,
                    ).copy(),
                    force_peak_substeps=np.asarray(
                        events["force_peak_substeps"][index, 0], dtype=float,
                    ).copy(),
                    lateral_force_substeps=np.asarray(
                        events["lateral_force_substeps"][index, 0], dtype=float,
                    ).copy(),
                    contact_region_substeps=np.asarray(
                        events["contact_region_substeps"][index, 0], dtype=np.uint8,
                    ).copy(),
                )
                if hasattr(self._provider,"fixture_observation"):
                    fixture_required=("fixture_position_substeps","fixture_twist_substeps","fixture_stop_substeps")
                    if any(name not in events for name in fixture_required):
                        raise RuntimeError("compliant step omitted physics-rate fixture evidence")
                    row.update(
                        fixture_position_substeps=np.asarray(
                            events["fixture_position_substeps"][index,0],dtype=float,
                        ).copy(),
                        fixture_twist_substeps=np.asarray(
                            events["fixture_twist_substeps"][index,0],dtype=float,
                        ).copy(),
                        fixture_stop_substeps=np.asarray(
                            events["fixture_stop_substeps"][index,0],dtype=bool,
                        ).copy(),
                    )
        return row

    def record_control_diagnostics(self, diagnostics):
        """Attach pre-transition public controller memory to the latest step."""
        if self.num_envs != 1:
            raise ValueError("controller diagnostics require an explicit per-world binding")
        record = self._episodes[self._current[0]]
        if (not record.traces or len(record.traces) != len(record.control_diagnostics)
                or record.control_diagnostics[-1] != {"recording": "unavailable"}):
            raise ValueError("one diagnostic snapshot must follow each physical step")
        record.control_diagnostics[-1] = deepcopy(diagnostics)

    def step(self, action):
        self._require_reset()
        action = numpy_action(action).astype(np.float32, copy=False)
        if action.shape == (13,) and self.num_envs == 1:
            action = action[None]
        if action.shape != (self.num_envs, 13):
            raise ValueError("shared skill action must have shape [B,13]")
        active = np.asarray(self._state.status) == 0
        if not active.any():
            raise RuntimeError("all episodes ended; reset before requesting more target execution")
        if any(self._episodes[self._current[i]].controller != "shared_skill" for i in np.flatnonzero(active)):
            raise ValueError("controller family cannot change inside an episode")
        self._charge("command_requests", int(active.sum()))
        result = self.runner.collect_chunk(self._state, action[:, None, :])
        self._state = result.state
        return self._finish_step(action, result.trace, active)

    def step_contact(self, action, *, body_twist):
        """Execute one shared skill tick with an explicit public body intent.

        The body command is an ordinary low-level input carried by
        ``ContactCommands``.  It is recorded with the requested skill and does
        not expose plant carry, mechanism parameters, or evaluator state.
        """
        from genedynamics.tasks.transfer.contact import ContactCommands

        self._require_reset()
        if self.spec.family == "obstacle" or not getattr(self.config, "mobile", False):
            raise ValueError("body intent requires a mobile contact task")
        action = numpy_action(action).astype(np.float32, copy=False)
        if action.shape == (13,) and self.num_envs == 1:
            action = action[None]
        body_twist = numpy_action(body_twist).astype(np.float32, copy=False)
        if body_twist.shape == (3,) and self.num_envs == 1:
            body_twist = body_twist[None]
        if action.shape != (self.num_envs, 13):
            raise ValueError("shared skill action must have shape [B,13]")
        if body_twist.shape != (self.num_envs, 3):
            raise ValueError("public planar body twist must have shape [B,3]")
        if not np.isfinite(body_twist).all():
            raise ValueError("public planar body twist must be finite")
        active = np.asarray(self._state.status) == 0
        if not active.any():
            raise RuntimeError("all episodes ended; reset before requesting more target execution")
        if any(self._episodes[self._current[i]].controller != "shared_skill" for i in np.flatnonzero(active)):
            raise ValueError("controller family cannot change inside an episode")
        self._charge("command_requests", int(active.sum()))
        result = self.runner.collect_chunk(
            self._state,
            ContactCommands(action[:, None, :], body_twist[:, None, :]),
        )
        self._state = result.state
        requests = [
            {"skill": action[index].copy(), "body_twist": body_twist[index].copy()}
            for index in range(self.num_envs)
        ]
        step = self._finish_step(
            action, result.trace, active, replay_request=requests,
        )
        executed = {
            "skill": step.executed_action,
            "body_twist": body_twist.copy(),
        }
        return type(step)(step.observation, step.reward, step.done, step.status,
                          executed, step.trace)

    def _finish_step(self, action, trace, active, *, applied_action=None, replay_request=None):
        # The transition already happened. Charge before recording/observation,
        # so a serialization or sensor error cannot erase incurred physics.
        steps = int(np.count_nonzero(np.asarray(trace.valid)[:, 0]))
        self._charge("control_steps", steps)
        self._charge("physics_steps", steps * self.config.substeps)
        applied = np.asarray(trace.command[:, 0], dtype=np.float32) if applied_action is None else applied_action
        self._previous_request[active] = action[active]
        if applied_action is None:
            issued = self._state.controller.last_command if self.spec.family == "obstacle" else self._state.last_command
            self._previous_action[active] = np.asarray(issued)[active]
            from genedynamics.core.control.transfer import integrate_rotation
            for index in np.flatnonzero(active):
                self._public_reference_position[index] += issued[index, :3] * self.config.control_dt
                if self.spec.family != "obstacle" and issued[index, 12] > .01:
                    self._public_reference_position[index, 2] = trace.tcp_position[index, 0, 2]
                if (self.spec.family != "obstacle"
                        and self.config.execution_profile == "substep_wrench_governor"):
                    self._public_reference_position[index, :2] = np.asarray(
                        self._state.reference_position,
                    )[index, :2]
                self._public_reference_rotation[index] = integrate_rotation(
                    np, self._public_reference_rotation[index], issued[index, 3:6], self.config.control_dt)
        else:
            # ACP uses structured targets rather than shared skill channels.
            # No synthetic delayed 13D command is disclosed as a public input.
            self._previous_action[active] = 0.
        trace_dict = {key: np.asarray(getattr(trace, key))[:, 0].copy() for key in trace._fields}
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        recording_errors = []
        for index in np.flatnonzero(active):
            record = self._episodes[self._current[index]]
            request = action[index].copy() if replay_request is None else deepcopy(replay_request[index])
            try:
                row = self._row(trace, index, action[index])
            except Exception as error:
                record.state = "aborted"
                record.incomplete_transitions.append({
                    "requested_action": request,
                    "physical_trace": {key: value[index].copy() for key, value in trace_dict.items()},
                    "error_type": type(error).__name__, "reason": str(error),
                })
                recording_errors.append(error)
                continue
            if isinstance(request, dict) and "body_twist" in request:
                row["issued_body_twist"] = np.asarray(
                    request["body_twist"], dtype=float,
                ).copy()
            record.requested_actions.append(request)
            record.traces.append(row)
            record.control_diagnostics.append({"recording": "unavailable"})
            if row["backend_status"]:
                record.state = "ended"
            goal = self.config.goal_position if self.spec.family == "obstacle" else self.config.goal
            distance = np.linalg.norm(row["tcp_position"] - goal)
            rewards[index] = -float(distance) - .001 + (1. if row["backend_status"] == 1 else -1. if row["backend_status"] else 0.)
        if recording_errors:
            raise recording_errors[0]
        self._push_history(active)
        return TaskStep(self.observe(), rewards, np.asarray(self._state.status) != 0,
                        np.asarray(self._state.status).copy(), applied.copy(), trace_dict)

    def reject(self, reason, *, mask=None):
        self._require_reset()
        active = np.asarray(self._state.status) == 0
        chosen = active if mask is None else active & np.asarray(mask, dtype=bool)
        if chosen.shape != (self.num_envs,):
            raise ValueError("rejection mask must contain one flag per episode")
        status = np.asarray(self._state.status).copy()
        status[chosen] = 7
        self._state = self._state._replace(status=status)
        for index in np.flatnonzero(chosen):
            record = self._episodes[self._current[index]]
            record.state, record.rejection_reason = "rejected", str(reason)

    def evaluate(self):
        return {**evaluate_records(self._episodes, self.spec), "accounting": dict(self.accounting),
                "lifetime_accounting": dict(self.lifetime_accounting)}

    def _contract_hash(self):
        value = dict(version=CONTRACT_VERSION, task=self.spec.name, config=asdict(self.config),
                     history_len=self.history_len, scene=self.runner.describe(),
                     shared_execution=self.shared_execution)
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

    def _runtime_profile_neutral_hash(self):
        """Hash physical/reset identity while excluding the runtime profile.

        This receipt is intentionally narrower than a normal checkpoint
        contract.  It permits one T4 post-preparation integration state to be
        cloned across predeclared runtime-controller profiles, while still
        binding the MuJoCo model, construction mechanism, reset contract and
        public history schema.  It must never be used to move a continuation
        between mechanisms, reset families or task geometries.
        """
        config = asdict(self.config)
        config.pop("execution_profile", None)
        mechanism = asdict(self._provider.mechanism)
        model = self._model
        digest = hashlib.sha256(json.dumps(dict(
            version=CONTRACT_VERSION,
            task=self.spec.name,
            config=config,
            mechanism=mechanism,
            history_len=self.history_len,
            provider=f"{type(self._provider).__module__}.{type(self._provider).__qualname__}",
            model=dict(nq=model.nq, nv=model.nv, nu=model.nu, nbody=model.nbody,
                       njnt=model.njnt, ngeom=model.ngeom),
        ), sort_keys=True, default=str).encode())
        for value in (
            model.body_mass, model.body_inertia, model.body_ipos,
            model.jnt_type, model.jnt_qposadr, model.jnt_dofadr,
            model.jnt_range, model.dof_damping, model.actuator_ctrlrange,
            model.geom_bodyid, model.geom_size, model.geom_pos,
            model.geom_quat, model.geom_friction, model.geom_solref,
            model.geom_solimp,
        ):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def get_checkpoint(self):
        """Portable full integration/controller/history state; no target replay."""
        import mujoco

        self._require_reset()
        integration = []
        for data in self._state.physics:
            value = np.zeros(mujoco.mj_stateSize(self._model, self._integration))
            mujoco.mj_getState(self._model, data, value, self._integration)
            integration.append(value)
        def encode(value):
            if isinstance(value, tuple) and hasattr(value, "_fields"):
                return {"namedtuple": type(value).__name__, "fields": {key: encode(getattr(value, key)) for key in value._fields if key != "physics"}}
            return np.asarray(value).copy()
        return dict(schema_version=1, contract_sha256=self._contract_hash(),
                    runtime_profile_neutral_sha256=self._runtime_profile_neutral_hash(),
                    integration=np.stack(integration),
                    state=encode(self._state), seed=self.seed, episodes=[asdict(row) for row in self._episodes],
                    current=list(self._current), accounting=dict(self.accounting),
                    lifetime_accounting_at_export=dict(self.lifetime_accounting),
                    buffers={name: deepcopy(getattr(self, name)) for name in (
                        "_mode", "_previous_request", "_previous_action",
                        "_public_reference_position", "_public_reference_rotation",
                        "_history_obs", "_history_act", "_history_valid",
                        *(("_body_start_planar", "_body_target")
                          if self.spec.name == "reposition_then_dock" else ()),
                    )},
                    acp=None if self._acp_executor is None else self._acp_executor.get_checkpoint())

    def restore_checkpoint(self, payload):
        import mujoco
        from genedynamics.tasks.transfer.runtime import ControllerState, ExecutionState
        from genedynamics.tasks.transfer.contact import ContactState

        if payload.get("schema_version") != 1 or payload.get("contract_sha256") != self._contract_hash():
            raise ValueError("task checkpoint schema/physical contract differs")
        data_states = []
        for integration in payload["integration"]:
            data = mujoco.MjData(self._model)
            mujoco.mj_setState(self._model, data, np.asarray(integration), self._integration)
            mujoco.mj_forward(self._model, data)
            # Forward may modify integration warm starts; retain the exact
            # saved integration variables after reconstructing sensor kinematics.
            mujoco.mj_setState(self._model, data, np.asarray(integration), self._integration)
            data_states.append(data)
        constructors = {"ControllerState": ControllerState, "ExecutionState": ExecutionState, "ContactState": ContactState}
        if self.config.robot == "go2_z1":
            from genedynamics.envs.domains.manipulation.transfer_workcell.backends.floating_contact import FloatingState
            constructors["FloatingState"] = FloatingState
        def decode(value):
            if isinstance(value, dict) and "namedtuple" in value:
                constructor = constructors[value["namedtuple"]]
                fields = {key: decode(item) for key, item in value["fields"].items()}
                if "physics" in constructor._fields:
                    fields["physics"] = tuple(data_states)
                return constructor(**fields)
            return np.asarray(value).copy()
        self._state = decode(payload["state"])
        self.seed, self._current = int(payload["seed"]), list(payload["current"])
        self._episodes = [EpisodeRecord(**deepcopy(row)) for row in payload["episodes"]]
        self.accounting = dict(payload["accounting"])
        for name, value in payload["buffers"].items():
            setattr(self, name, np.asarray(value).copy())
        if payload["acp"] is not None:
            from .acp_transfer import ACPContactExecutor
            self._acp_executor = ACPContactExecutor(self)
            self._acp_executor.restore_checkpoint(payload["acp"])
        elif self._acp_executor is not None:
            self._acp_executor = None
        return self.observe()

    def restore_runtime_profile_checkpoint(self, payload):
        """Clone an exact post-preparation T4 state across runtime profiles."""
        allowed = {
            "legacy", "joint_margin_bounds", "joint_margin_public_reference",
        }
        if self.spec.alias != "T4" or self.config.execution_profile not in allowed:
            raise ValueError("runtime-profile continuation cloning is restricted to T4 isolation profiles")
        if payload.get("runtime_profile_neutral_sha256") != self._runtime_profile_neutral_hash():
            raise ValueError("runtime-profile continuation physical/reset contract differs")
        value = deepcopy(payload)
        value["contract_sha256"] = self._contract_hash()
        return self.restore_checkpoint(value)

    def replay_episode(self, record):
        """Fresh CPU execution of one frozen command stream, explicitly charged."""
        from genedynamics.tasks.transfer.evaluation import evaluate_episode

        if not isinstance(record, EpisodeRecord):
            record = EpisodeRecord(**deepcopy(record))
        mechanism = self._provider.mechanism
        replay = type(self)(self.spec, mechanism_spec=mechanism, seed=record.seed, history_len=self.history_len)
        replay.reset((record.sample_id,), seed=record.seed, initial=record.initial)
        replay.set_modes([record.requested_mode])
        for command in record.requested_actions:
            if record.controller == "shared_skill":
                if isinstance(command, dict) and set(command) == {"skill", "body_twist"}:
                    replay.step_contact(
                        np.asarray(command["skill"])[None],
                        body_twist=np.asarray(command["body_twist"])[None],
                    )
                else:
                    replay.step(np.asarray(command)[None])
            else:
                replay.step_acp(command)
        if record.state == "rejected":
            replay.reject(record.rejection_reason)
        reproduced = replay.records[0]
        # It is a distinct charged audit replay, not a new independent sample.
        original = evaluate_records([record], self.spec)["episodes"][0]
        actual = evaluate_records([reproduced], self.spec)["episodes"][0]
        errors = []
        mismatches = set()
        if len(record.traces) != len(reproduced.traces):
            raise ValueError("CPU replay changed executed episode length")
        for left, right in zip(record.traces, reproduced.traces):
            errors.extend(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key]))) for key in ("tcp_position", "tcp_rotation"))
            for key in set(left) | set(right):
                if key not in left or key not in right or not np.array_equal(np.asarray(left[key]), np.asarray(right[key])):
                    mismatches.add(key)
        return dict(report=actual, matches_status=original["status"] == actual["status"],
                    max_pose_error=float(max(errors, default=0.)), trace_exact=not mismatches,
                    mismatched_fields=sorted(mismatches), accounting=dict(replay.accounting),
                    scope="charged development CPU replay; not an independent statistical repeat")

    def step_acp(self, command):
        self._require_reset()
        if self.spec.family != "contact" or getattr(self.config, "mobile", False):
            raise NotImplementedError("ACP anisotropic torque adapter currently supports fixed contact arms only; mobile needs WBC integration")
        from .acp_transfer import ACPContactExecutor
        if self._acp_executor is None:
            self._acp_executor = ACPContactExecutor(self)
            self._acp_executor.reset(np.ones(self.num_envs, dtype=bool), self._state)
        return self._acp_executor.step(command)

    def execute_acp_chunk(self, prediction):
        """Execute the declared waypoint prefix at its physical sample clock.

        Each interpolation advances actual 20 ms CPU dynamics, is charged and
        remains a separate replay record. The return is the last TaskStep plus
        all executed TaskSteps; this is a synchronous adapted controller, not
        the author's asynchronous hardware deployment.
        """
        from scipy.spatial.transform import Rotation, Slerp

        reference = numpy_action(prediction["reference_pose_world"])
        virtual = numpy_action(prediction["virtual_target_pose_world"])
        stiffness = numpy_action(prediction["stiffness_tool"])
        if reference.ndim != 5 or reference.shape[0] != self.num_envs or reference.shape[2:] != (1, 4, 4):
            raise ValueError("fixed-arm ACP chunk must contain [B,H,1,4,4] poses")
        horizon = int(prediction["execution_horizon"])
        times = np.asarray(prediction["waypoint_times_s"], dtype=float)
        if (not 1 <= horizon <= reference.shape[1] or times.shape != (reference.shape[1],)
                or times[0] != 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError("ACP chunk has invalid execution horizon/waypoint timing")
        interval = float(times[1] - times[0]) if len(times) > 1 else self.config.control_dt
        duration = float(times[horizon - 1] + interval)
        controls = int(np.ceil(duration / self.config.control_dt))
        steps = []
        for control in range(controls):
            if np.all(np.asarray(self._state.status) != 0):
                break
            time = control * self.config.control_dt
            left = min(int(np.searchsorted(times[:horizon], time, side="right") - 1), horizon - 1)
            right = min(left + 1, horizon - 1)
            fraction = 0. if left == right else (time - times[left]) / (times[right] - times[left])
            def interpolate(poses):
                value = poses[:, left, 0].copy()
                value[:, :3, 3] = ((1 - fraction) * poses[:, left, 0, :3, 3]
                                     + fraction * poses[:, right, 0, :3, 3])
                if left != right:
                    for world in range(self.num_envs):
                        value[world, :3, :3] = Slerp([0., 1.], Rotation.from_matrix(
                            poses[world, [left, right], 0, :3, :3]))([fraction]).as_matrix()[0]
                return value
            command = dict(reference_pose_world=interpolate(reference), virtual_target_pose_world=interpolate(virtual),
                           stiffness_tool=(1 - fraction) * stiffness[:, left, 0] + fraction * stiffness[:, right, 0])
            steps.append(self.step_acp(command))
        return {"last_step": steps[-1] if steps else None, "steps": steps,
                "requested_control_steps": controls, "executed_control_steps": len(steps)}


def instantiate_task(task, robot=None, mechanism_spec=None, seed=0, *, history_len=4,
                     config_overrides=None, shared_execution=None):
    from genedynamics.tasks.transfer.compliant_socket import ENVIRONMENT_NAME
    adapter = CpuTaskAdapter
    if task in {ENVIRONMENT_NAME,"compliant_fixture_insertion"}:
        from .compliant_socket import CompliantSocketTaskAdapter
        adapter = CompliantSocketTaskAdapter
    return adapter(task_spec(task, robot=robot, config_overrides=config_overrides),
                          mechanism_spec=mechanism_spec, seed=seed, history_len=history_len,
                          shared_execution=shared_execution)

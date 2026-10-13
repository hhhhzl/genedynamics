"""Serial float64 MuJoCo diagnostic backend; never the canonical GPU collector."""
import copy
import numpy as np
import mujoco

from genedynamics.learning.cross_embodiment.contracts import (
    ControllerState, ExecutionState, StepTrace, ChunkResult, MechanismSpec, mechanism_specs,
)
from genedynamics.learning.cross_embodiment.core.capabilities import Capabilities
from genedynamics.learning.cross_embodiment.core.control_math import (
    bounded_command, impedance, integrate_error, integrate_rotation, orientation_distance, terminal_status,
)
from ..workcell import build_workcell, rotation


def validate_execution(execution, backend, physics):
    if execution.physics_backend != physics or execution.controller_backend != "builtin":
        raise NotImplementedError(f"{backend} supports physics_backend={physics}, controller_backend=builtin")
    if execution.execution_mode not in {"collect", "evaluate"}:
        raise NotImplementedError(f"{backend} execution supports collect/evaluate only")


def validate_reset(request, previous, mask, noise_batch, nq):
    batch = len(request.sample_ids)
    if (previous is None) != (mask is None):
        raise ValueError("partial reset requires both previous state and mask")
    if previous is not None and (len(previous.sample_id) != batch or np.shape(mask) != (batch,)):
        raise ValueError("partial reset batch/mask mismatch")
    if noise_batch is not None and np.shape(noise_batch) != (batch, nq):
        raise ValueError(f"reset noise must have shape {(batch, nq)}")


class MujocoExecutionBackend:
    def __init__(self, *, execution, task, skill, mechanism):
        validate_execution(execution, "numpy", "mujoco")
        if execution.device != "cpu":
            raise NotImplementedError("NumPy/MuJoCo reference supports only cpu")
        self.execution, self.task, self.skill, self.mechanism = execution, task, skill, mechanism
        self.mechanisms = mechanism_specs(mechanism)
        self.heterogeneous = not isinstance(mechanism, MechanismSpec)
        self.scenes = tuple(build_workcell(task, value) for value in self.mechanisms)
        self.scene = self.scenes[0]
        self.model = self.scene.model
        self.models = tuple(scene.model for scene in self.scenes)
        self.sid = self.scene.binding.element_ids["tool_mount"]
        self.limits = np.asarray(self.scene.binding.profile.controller_defaults["torque_limits"])

    def capabilities(self):
        return Capabilities("numpy", "cpu", "mujoco", frozenset({"collect", "evaluate"}))

    def describe(self):
        return {**self.scene.manifest, "backend": "numpy", "device": "cpu",
                "physics_dtype": "float64", "command_dtype": "float32",
                "mechanism_batch": [scene.manifest["mechanism"] for scene in self.scenes],
                "mechanism_model_sha256s": [scene.manifest["model_sha256"] for scene in self.scenes],
                "mechanism_batch_size": len(self.mechanisms) if self.heterogeneous else None,
                "microbatch_size": self.execution.microbatch_size,
                "compiled_chunk": False, "gpu_qualification": "not_run"}

    def synchronize(self, value):
        return None

    def _validate_batch(self, batch):
        if self.heterogeneous and batch != len(self.mechanisms):
            raise ValueError(f"mechanism batch has {len(self.mechanisms)} worlds; requested batch has {batch}")

    def _world(self, index):
        index = index if self.heterogeneous else 0
        return self.scenes[index], self.models[index], self.mechanisms[index]

    def reset_batch(self, request, *, previous=None, mask=None, noise_batch=None):
        validate_reset(request, previous, mask, noise_batch, self.model.nq)
        self._validate_batch(len(request.sample_ids))
        states = []
        for i, identity in enumerate(request.sample_ids):
            if previous is not None and not mask[i]:
                states.append(self._slice(previous, i))
                continue
            scene, model, _ = self._world(i)
            noise = (np.random.default_rng(np.random.SeedSequence([request.seed, identity])).normal(size=self.model.nq)
                     if noise_batch is None else np.asarray(noise_batch[i]))
            if not np.all(np.isfinite(noise)):
                raise ValueError("reset noise must be finite")
            data = mujoco.MjData(model)
            data.qpos[:] = np.clip(scene.start_qpos + self.task.reset_joint_noise * noise,
                                   model.jnt_range[:, 0] + .002, model.jnt_range[:, 1] - .002)
            mujoco.mj_forward(model, data)
            controller = ControllerState(data.site_xpos[self.sid].copy(), data.site_xmat[self.sid].reshape(3, 3).copy(),
                                         np.zeros((self.task.max_delay_steps+1, 13)), np.zeros(13), np.zeros(6))
            states.append(ExecutionState(data, controller, identity, 0, 0, 0))
        return self._stack(states)

    @staticmethod
    def _slice(state, i):
        return ExecutionState(state.physics[i], ControllerState(*(x[i] for x in state.controller)),
                              state.sample_id[i], state.steps[i], state.hold_count[i], state.status[i])

    @staticmethod
    def _stack(states):
        return ExecutionState(tuple(s.physics for s in states),
                              ControllerState(*(np.stack([s.controller[j] for s in states]) for j in range(len(ControllerState._fields)))),
                              *(np.asarray([s[j] for s in states]) for j in range(2, 6)))

    def _gravity(self, data, model=None):
        model = self.model if model is None else model
        tau = np.zeros(model.nv)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for body in range(1, model.nbody):
            point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ self.scene.nominal_ipos[body]
            mujoco.mj_jac(model, data, jp, jr, point, body)
            tau += jp.T @ (-self.scene.nominal_masses[body] * model.opt.gravity)
        return tau

    def _step(self, state, command, world=0):
        scene, model, mechanism = self._world(world)
        task, data = self.task, copy.copy(state.physics)
        active = state.status == 0
        controller = state.controller
        input_finite = bool(np.all(np.isfinite(command)))
        command = np.nan_to_num(command, nan=0., posinf=0., neginf=0.)
        bounded = bounded_command(np, command, controller.last_command, self.skill, task.control_dt)
        buffer = np.concatenate((bounded[None], controller.command_buffer[:-1]))
        applied = buffer[mechanism.delay_steps]
        refpos = controller.reference_position + applied[:3] * task.control_dt
        refrot = integrate_rotation(np, controller.reference_rotation, applied[3:6], task.control_dt)
        integral = integrate_error(np, controller.integral_error, data.site_xpos[self.sid],
                                   data.site_xmat[self.sid].reshape(3,3), refpos, refrot, task.control_dt)
        clearance, speed, saturation = task.contact_margin, 0., 0.
        joint_limit, finite = False, input_finite
        tau = np.zeros(model.nu)
        if active:
            for _ in range(task.substeps):
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, self.sid)
                tau, ratio = impedance(np, data.site_xpos[self.sid], data.site_xmat[self.sid].reshape(3,3),
                                       data.qvel, jp, jr, self._gravity(data, model), refpos, refrot,
                                       applied, integral, self.skill, mechanism.servo_gain, self.limits,
                                       data.qpos, scene.start_qpos)
                data.ctrl[:] = tau
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                clearance = min(clearance, min((c.dist for c in data.contact), default=task.contact_margin))
                speed = max(speed, np.max(np.abs(data.qvel)))
                saturation = max(saturation, ratio)
                joint_limit |= bool(np.any(data.qpos < model.jnt_range[:,0] - 1e-5)
                                    or np.any(data.qpos > model.jnt_range[:,1] + 1e-5))
                finite &= bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        pos, rot = data.site_xpos[self.sid].copy(), data.site_xmat[self.sid].reshape(3,3).copy()
        pe = np.linalg.norm(pos - task.goal_position)
        ae = orientation_distance(np, rot, rotation(task.goal_yaw))
        status, hold = terminal_status(np, state.status, state.steps + int(active), state.hold_count,
                                       clearance < -task.collision_tolerance, joint_limit,
                                       speed > task.joint_speed_limit, np.asarray(finite), pe, ae, task)
        new = (ExecutionState(data, ControllerState(refpos, refrot, buffer, bounded, integral), state.sample_id,
                              state.steps+1, int(hold), int(status)) if active else state)
        trace = StepTrace(pos, rot, pe, ae, clearance, speed, saturation, applied, tau,
                          new.status, active, active and new.status != 0)
        return new, trace

    def collect_chunk(self, state, commands):
        commands = np.asarray(commands, dtype=np.float32)
        if commands.ndim != 3 or commands.shape[0] != len(state.sample_id) or commands.shape[2] != 13 or commands.shape[1] < 1:
            raise ValueError("commands must have shape [B,L,13], L>=1")
        self._validate_batch(len(state.sample_id))
        states, traces = [], []
        for b in range(len(state.sample_id)):
            current, trace = self._slice(state, b), []
            for command in commands[b]:
                current, step = self._step(current, command, b)
                trace.append(step)
            states.append(current)
            traces.append(StepTrace(*(np.stack([t[i] for t in trace]) for i in range(len(StepTrace._fields)))))
        return ChunkResult(self._stack(states), StepTrace(*(np.stack([t[i] for t in traces]) for i in range(len(StepTrace._fields)))))

    evaluate_chunk = collect_chunk

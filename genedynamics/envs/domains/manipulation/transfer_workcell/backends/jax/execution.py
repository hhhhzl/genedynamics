"""Canonical JAX collector: environment vmap, control/physics scan, whole-chunk jit.

Device selection is explicit. Host work ends at construction/reset/input transfer;
the compiled execution kernel contains no Python per-world simulation loop, host
callback or device-to-host conversion. Torch/native providers can implement the
same public protocol without changing the frontend.
"""
from functools import partial, wraps
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import support
from genedynamics.core.backends.runtime.device import select_device

from genedynamics.tasks.transfer.runtime import (
    ControllerState, ExecutionState, StepTrace, ChunkResult, MechanismSpec,
    mechanism_specs,
)
from genedynamics.core.backends.capabilities import Capabilities
from genedynamics.core.control.transfer import (
    bounded_command, impedance, integrate_error, integrate_rotation, orientation_distance, terminal_status,
)


def _fp32_scope(fn):
    """Isolate MJX's internal array defaults without changing caller settings."""
    @wraps(fn)
    def call(*args, **kwargs):
        with jax.experimental.disable_x64():
            return fn(*args, **kwargs)
    return call


class JaxExecutionBackend:
    @_fp32_scope
    def __init__(self, *, execution, task, skill, mechanism):
        if execution.physics_backend != "mjx" or execution.controller_backend != "builtin":
            raise NotImplementedError("JAX execution requires physics_backend=mjx, controller_backend=builtin")
        if execution.execution_mode not in {"collect", "evaluate"}:
            raise NotImplementedError("JAX execution currently supports collect/evaluate only")
        self.device = select_device(execution.device)
        from genedynamics.envs.domains.manipulation.transfer_workcell.workcell import build_workcell, rotation
        self.execution, self.task, self.skill, self.mechanism = execution, task, skill, mechanism
        self.mechanisms = mechanism_specs(mechanism)
        self.heterogeneous = not isinstance(mechanism, MechanismSpec)
        # Compile each complete host model at construction. MuJoCo derives
        # inertia, subtree mass, dof_M0 and inverse weights from payload mass;
        # changing body_mass alone would create an inconsistent physics model.
        self.scenes = tuple(build_workcell(task, value) for value in self.mechanisms)
        self.scene = self.scenes[0]
        host = self.scene.model
        self.sid = self.scene.binding.element_ids["tool_mount"]
        self.body = int(host.site_bodyid[self.sid])
        self.model = mjx.put_model(host, device=self.device, impl="jax")
        self.initial = mjx.make_data(host, device=self.device, impl="jax")
        # Honor the local FP32 contract even if another solver enabled x64.
        cast = lambda x: x.astype(jp.float32) if hasattr(x, "dtype") and x.dtype.kind == "f" else x
        self.model = jax.tree.map(cast, self.model)
        self.initial = jax.tree.map(cast, self.initial)
        self.models = None
        if self.heterogeneous:
            models = [self.model] + [jax.tree.map(cast, mjx.put_model(scene.model, device=self.device, impl="jax"))
                                    for scene in self.scenes[1:]]
            try:
                # MJX's non-array topology metadata must match. All numeric
                # leaves, including derived physics quantities, are batched.
                self.models = jax.tree.map(lambda *values: jp.stack(values), *models)
            except (TypeError, ValueError) as error:
                raise ValueError("mechanism batch must have identical MJX topology and static metadata") from error
        with jax.default_device(self.device):
            self.start_qpos = jp.asarray(self.scene.start_qpos, dtype=jp.float32)
            self.ranges = jp.asarray(host.jnt_range, dtype=jp.float32)
            self.limits = jp.asarray(self.scene.binding.profile.controller_defaults["torque_limits"], dtype=jp.float32)
            self.goal = jp.asarray(task.goal_position, dtype=jp.float32)
            self.goal_rotation = jp.asarray(rotation(task.goal_yaw), dtype=jp.float32)
            self.masses = jp.asarray(self.scene.nominal_masses, dtype=jp.float32)
            self.ipos = jp.asarray(self.scene.nominal_ipos, dtype=jp.float32)
            # Kinematic ancestry, known topology; independent of target mechanism.
            mask = np.zeros((host.nbody, host.nv), dtype=np.float32)
            for bid in range(1, host.nbody):
                ancestor = bid
                while ancestor:
                    mask[bid, host.dof_bodyid == ancestor] = 1
                    ancestor = host.body_parentid[ancestor]
            self.body_dof_mask = jp.asarray(mask)
            self.root_ids = jp.asarray(host.body_rootid)
        self._reset = jax.jit(jax.vmap(self._reset_one, in_axes=(None, 0, 0)), device=self.device)
        self._collect = jax.jit(jax.vmap(self._chunk_one, in_axes=(None, None, None, 0, 0)), device=self.device)
        self._reset_batched = jax.jit(jax.vmap(self._reset_one, in_axes=(0, 0, 0)), device=self.device)
        self._collect_batched = jax.jit(jax.vmap(self._chunk_one, in_axes=(0, 0, 0, 0, 0)), device=self.device)
        self.gain = jax.device_put(np.asarray([value.servo_gain for value in self.mechanisms], dtype=np.float32)
                                   if self.heterogeneous else np.float32(self.mechanisms[0].servo_gain), self.device)
        self.delay = jax.device_put(np.asarray([value.delay_steps for value in self.mechanisms], dtype=np.int32)
                                    if self.heterogeneous else np.int32(self.mechanisms[0].delay_steps), self.device)

    def capabilities(self):
        return Capabilities("jax", str(self.device), "mjx", frozenset({"collect", "evaluate"}),
                            compiled_chunk=True, device_resident=True)

    def describe(self):
        return {**self.scene.manifest, "backend": "jax", "device": str(self.device),
                "physics_dtype": "float32", "command_dtype": "float32", "compiled_chunk": True,
                "mechanism_batch": [scene.manifest["mechanism"] for scene in self.scenes],
                "mechanism_model_sha256s": [scene.manifest["model_sha256"] for scene in self.scenes],
                "mechanism_batch_size": len(self.mechanisms) if self.heterogeneous else None,
                "microbatch_size": self.execution.microbatch_size,
                "jax_version": jax.__version__, "gpu_qualification": "not_run"}

    @staticmethod
    def synchronize(value):
        jax.block_until_ready(value)

    def _validate_batch(self, batch):
        if self.heterogeneous and batch != len(self.mechanisms):
            raise ValueError(f"mechanism batch has {len(self.mechanisms)} worlds; requested batch has {batch}")

    def _slices(self, batch):
        size = self.execution.microbatch_size or batch
        return (slice(start, min(start+size, batch)) for start in range(0, batch, size))

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def _slice_tree(value, start, stop):
        # Python indices become static compiler metadata, rather than eager
        # int32 arrays implicitly copied from host during dynamic slicing.
        return jax.tree.map(lambda leaf: jax.lax.slice_in_dim(leaf, start, stop, axis=0), value)

    @staticmethod
    @jax.jit
    def _concatenate(values):
        return jax.tree.map(lambda *leaves: jp.concatenate(leaves, axis=0), *values)

    @staticmethod
    def _join(values):
        return values[0] if len(values) == 1 else JaxExecutionBackend._concatenate(values)

    def _reset_one(self, model, sample_id, noise):
        data = mjx.forward(model, self.initial.replace(
            qpos=jp.clip(self.start_qpos + self.task.reset_joint_noise * noise,
                         self.ranges[:, 0] + .002, self.ranges[:, 1] - .002)))
        controller = ControllerState(data.site_xpos[self.sid], data.site_xmat[self.sid],
                                     jp.zeros((self.task.max_delay_steps+1, 13), dtype=jp.float32),
                                     jp.zeros(13, dtype=jp.float32), jp.zeros(6, dtype=jp.float32))
        return ExecutionState(data, controller, sample_id, jp.int32(0), jp.int32(0), jp.int32(0))

    @_fp32_scope
    def reset_batch(self, request, *, previous=None, mask=None, noise_batch=None):
        batch = len(request.sample_ids)
        self._validate_batch(batch)
        if (previous is None) != (mask is None):
            raise ValueError("partial reset requires both previous state and mask")
        if previous is not None and (previous.sample_id.shape != (batch,) or np.shape(mask) != (batch,)):
            raise ValueError("partial reset batch/mask mismatch")
        if noise_batch is not None and np.shape(noise_batch) != (batch, self.scene.model.nq):
            raise ValueError("reset noise must be [B,nq]")
        ids = jax.device_put(np.asarray(request.sample_ids, dtype=np.uint32), self.device)
        if noise_batch is None:
            key = jax.device_put(jax.random.PRNGKey(request.seed), self.device)
            keys = jax.vmap(lambda identity: jax.random.fold_in(key, identity))(ids)
            noise = jax.vmap(lambda k: jax.random.normal(k, (self.scene.model.nq,), dtype=jp.float32))(keys)
        else:
            noise = jax.device_put(jp.asarray(noise_batch, dtype=jp.float32), self.device)
        if self.execution.microbatch_size is None or self.execution.microbatch_size >= batch:
            state = (self._reset_batched(self.models, ids, noise) if self.heterogeneous
                     else self._reset(self.model, ids, noise))
        else:
            states = []
            for index in self._slices(batch):
                ids_part = self._slice_tree(ids, index.start, index.stop)
                noise_part = self._slice_tree(noise, index.start, index.stop)
                if self.heterogeneous:
                    states.append(self._reset_batched(self._slice_tree(self.models, index.start, index.stop),
                                                      ids_part, noise_part))
                else:
                    states.append(self._reset(self.model, ids_part, noise_part))
            state = self._join(states)
        if previous is not None:
            mask = jax.device_put(jp.asarray(mask, dtype=jp.bool_), self.device)
            state = jax.tree.map(lambda new, old: jp.where(mask.reshape((batch,) + (1,)*(new.ndim-1)), new, old),
                                 state, previous)
        return state

    def _gravity(self, model, data):
        points = data.xpos + jp.einsum("bij,bj->bi", data.xmat, self.ipos)
        offset = points - data.subtree_com[self.root_ids]
        jacp = (data.cdof[None, :, 3:] + jp.cross(data.cdof[None, :, :3], offset[:, None, :]))
        jacp = jacp * self.body_dof_mask[:, :, None]
        return jp.einsum("bnk,b,k->n", jacp, -self.masses, model.opt.gravity)

    def _control_step(self, model, gain, delay, state, command):
        task, ctrl = self.task, state.controller
        active = state.status == 0
        input_finite = jp.all(jp.isfinite(command))
        command = jp.nan_to_num(command, nan=0., posinf=0., neginf=0.)
        bounded = bounded_command(jp, command, ctrl.last_command, self.skill, task.control_dt)
        buffer = jp.concatenate((bounded[None], ctrl.command_buffer[:-1]))
        applied = buffer[delay]
        refpos = ctrl.reference_position + applied[:3] * task.control_dt
        refrot = integrate_rotation(jp, ctrl.reference_rotation, applied[3:6], task.control_dt)
        integral = integrate_error(jp, ctrl.integral_error, state.physics.site_xpos[self.sid],
                                   state.physics.site_xmat[self.sid], refpos, refrot, task.control_dt)

        def physics_step(carry, unused):
            data, clearance, speed, saturation, joint_limit, finite, tau = carry
            jp_t, jr_t = support.jac(model, data, data.site_xpos[self.sid], self.body)
            tau, ratio = impedance(jp, data.site_xpos[self.sid], data.site_xmat[self.sid], data.qvel,
                                   jp_t.T, jr_t.T, self._gravity(model, data), refpos, refrot, applied,
                                   integral, self.skill, gain, self.limits, data.qpos, self.start_qpos)
            data = mjx.step(model, data.replace(ctrl=tau))
            # step returns kinematic/contact fields from BEFORE integration.
            # Refresh them without solving identical constraints a second time.
            data = mjx.kinematics(model, data)
            data = mjx.com_pos(model, data)
            data = mjx.com_vel(model, data)
            data = mjx.collision(model, data)
            contact_dist = jp.where(data.contact.geom[:, 0] >= 0, data.contact.dist, task.contact_margin)
            clearance = jp.minimum(clearance, jp.min(contact_dist, initial=task.contact_margin))
            speed = jp.maximum(speed, jp.max(jp.abs(data.qvel)))
            joint_limit |= jp.any(data.qpos < self.ranges[:, 0] - 1e-5) | jp.any(data.qpos > self.ranges[:, 1] + 1e-5)
            finite &= jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
            return (data, clearance, speed, jp.maximum(saturation, ratio), joint_limit, finite, tau), None

        initial = (state.physics, jp.float32(task.contact_margin), jp.float32(0), jp.float32(0),
                   jp.bool_(False), input_finite, jp.zeros(self.model.nu, dtype=jp.float32))
        result, _ = jax.lax.scan(physics_step, initial, None, length=task.substeps)
        data, clearance, speed, saturation, joint_limit, finite, tau = result
        pos, rot = data.site_xpos[self.sid], data.site_xmat[self.sid]
        pe, ae = jp.linalg.norm(pos-self.goal), orientation_distance(jp, rot, self.goal_rotation)
        status, hold = terminal_status(jp, state.status, state.steps+1, state.hold_count,
                                       clearance < -task.collision_tolerance, joint_limit,
                                       speed > task.joint_speed_limit, finite, pe, ae, task)
        candidate = ExecutionState(data, ControllerState(refpos, refrot, buffer, bounded, integral),
                                   state.sample_id, state.steps+1, hold, status)
        new = jax.tree.map(lambda n, o: jp.where(active, n, o), candidate, state)
        # Done environments retain their physics and pose; validity excludes them
        # from every rollout denominator. No automatic reset inside a chunk.
        pos, rot = new.physics.site_xpos[self.sid], new.physics.site_xmat[self.sid]
        trace = StepTrace(pos, rot, jp.linalg.norm(pos-self.goal), orientation_distance(jp, rot, self.goal_rotation),
                          clearance, speed, saturation, applied, tau, new.status, active, active & (new.status != 0))
        return new, trace

    def _chunk_one(self, model, gain, delay, state, commands):
        state, trace = jax.lax.scan(lambda s, c: self._control_step(model, gain, delay, s, c), state, commands)
        return ChunkResult(state, trace)

    @_fp32_scope
    def collect_chunk(self, state, commands):
        if len(np.shape(commands)) != 3 or np.shape(commands)[0] != state.sample_id.shape[0] or np.shape(commands)[2] != 13 or np.shape(commands)[1] < 1:
            raise ValueError("commands must have shape [B,L,13], L>=1")
        batch = state.sample_id.shape[0]
        self._validate_batch(batch)
        commands = jax.device_put(jp.asarray(commands, dtype=jp.float32), self.device)
        if self.execution.microbatch_size is None or self.execution.microbatch_size >= batch:
            return (self._collect_batched(self.models, self.gain, self.delay, state, commands) if self.heterogeneous
                    else self._collect(self.model, self.gain, self.delay, state, commands))
        results = []
        # Python dispatch is per memory-bounded whole chunk, never per world
        # or physics step. Device slicing/concatenation do not materialize host
        # tensors. A smaller final slice has its own JIT shape signature.
        for index in self._slices(batch):
            current = self._slice_tree(state, index.start, index.stop)
            command_part = self._slice_tree(commands, index.start, index.stop)
            if self.heterogeneous:
                results.append(self._collect_batched(self._slice_tree(self.models, index.start, index.stop),
                                                     self._slice_tree(self.gain, index.start, index.stop),
                                                     self._slice_tree(self.delay, index.start, index.stop),
                                                     current, command_part))
            else:
                results.append(self._collect(self.model, self.gain, self.delay, current, command_part))
        return self._join(results)

    evaluate_chunk = collect_chunk

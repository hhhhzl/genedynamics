"""Fixed-iteration, batched, device-resident Panda semantic state projection."""
from dataclasses import asdict
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import support

from ...contracts import ControllerState, ExecutionState, Status
from ...source_query import TaskSemantics, SourceQueryResult
from ...core.control_math import orientation_error
from ...core.source_support import semantic_finite_and_rotation, support_result
from .execution import JaxExecutionBackend, _fp32_scope


class JaxSourceQueryBackend:
    def __init__(self, *, execution, task, skill, query, mechanism):
        self.runner = JaxExecutionBackend(execution=execution, task=task, skill=skill, mechanism=mechanism)
        self.query = query
        self._build = jax.jit(jax.vmap(self._one, in_axes=(None, 0, 0, 0)), device=self.runner.device)
        self._observers = {}

    def synchronize(self, value):
        self.runner.synchronize(value)

    def describe(self):
        return {"environment": "obstacle_transfer", "source_robot": "panda",
                "backend": "jax", "query": asdict(self.query), "compiled_batch": True,
                "supported_phase": 0, "controller_initialization": "fresh_zero_history",
                "ik_seed_rule": "source_start_or_previous_source_qpos",
                "model_sha256": self.runner.scene.manifest["model_sha256"]}

    def _kinematics(self, model, qpos):
        r = self.runner
        data = mjx.kinematics(model, r.initial.replace(qpos=qpos))
        return mjx.com_pos(model, data)

    def _jacobian(self, model, data):
        r = self.runner
        jacp, jacr = support.jac(model, data, data.site_xpos[r.sid], r.body)
        return jp.concatenate((jacp.T, jacr.T))

    def _one(self, model, semantic, identity, q0):
        r, spec = self.runner, self.query
        pos, rot, twist, phase = semantic
        valid = semantic_finite_and_rotation(jp, pos, rot, twist, phase)
        # Invalid requests stay finite for failure recording; never feed NaNs to IK.
        pos = jp.where(valid, pos, jp.asarray(r.task.start_position, dtype=jp.float32))
        rot = jp.where(valid, rot, jp.diag(jp.asarray([1., -1., -1.], dtype=jp.float32)))
        twist = jp.where(valid, twist, jp.zeros(6, dtype=jp.float32))

        def iterate(_, q):
            data = self._kinematics(model, q)
            error = jp.concatenate((pos-data.site_xpos[r.sid], orientation_error(jp, data.site_xmat[r.sid], rot)))
            jac = self._jacobian(model, data)
            dq = jac.T @ jp.linalg.solve(jac @ jac.T + spec.damping**2 * jp.eye(6, dtype=jp.float32), error)
            candidate = jp.clip(q + jp.clip(dq, -spec.joint_step_limit, spec.joint_step_limit),
                                r.ranges[:, 0] + .002, r.ranges[:, 1] - .002)
            converged = (jp.linalg.norm(error[:3]) < spec.position_tolerance * .1) & (jp.linalg.norm(error[3:]) < spec.orientation_tolerance * .1)
            return jp.where(converged, q, candidate)

        q = jax.lax.fori_loop(0, spec.iterations, iterate, q0)
        data = self._kinematics(model, q)
        jac = self._jacobian(model, data)
        qvel = jac.T @ jp.linalg.solve(jac @ jac.T + spec.damping**2 * jp.eye(6, dtype=jp.float32), twist)
        data = mjx.forward(model, r.initial.replace(qpos=q, qvel=qvel))
        contact_dist = jp.where(data.contact.geom[:, 0] >= 0, data.contact.dist, r.task.contact_margin)
        clearance = jp.min(contact_dist, initial=r.task.contact_margin)
        ok, reason, pe, ae, ve = support_result(jp, valid, phase, pos, rot, twist, data.site_xpos[r.sid],
                                               data.site_xmat[r.sid], jac @ qvel, q, qvel, r.ranges,
                                               clearance, r.task, spec)
        controller = ControllerState(data.site_xpos[r.sid], data.site_xmat[r.sid],
                                     jp.zeros((r.task.max_delay_steps+1, 13), dtype=jp.float32),
                                     jp.zeros(13, dtype=jp.float32), jp.zeros(6, dtype=jp.float32))
        # Unsupported virtual states freeze if accidentally sent to a collector.
        # Query failures are counted from support/reason, once, by the caller.
        status = jp.where(ok, jp.int32(0), jp.int32(Status.SOURCE_SUPPORT_FAILURE))
        state = ExecutionState(data, controller, identity, jp.int32(0), jp.int32(0), status)
        return SourceQueryResult(semantic, state, ok, reason, pe, ae, ve, clearance)

    @_fp32_scope
    def build(self, semantics, request, *, previous=None):
        r, batch = self.runner, len(request.sample_ids)
        if previous is not None and previous.sample_id.shape != (batch,):
            raise ValueError("source warm-start batch mismatch")
        semantics = TaskSemantics(*(jax.device_put(jp.asarray(x, dtype=jp.float32), r.device) for x in semantics))
        ids = jax.device_put(np.asarray(request.sample_ids, dtype=np.uint32), r.device)
        q0 = jp.broadcast_to(r.start_qpos, (batch, r.scene.model.nq))
        if previous is not None:
            if previous.physics.qpos.shape != q0.shape:
                raise ValueError("source warm-start topology mismatch")
            reusable = (previous.sample_id == ids) & jp.all(jp.isfinite(previous.physics.qpos), axis=1)
            q0 = jp.where(reusable[:, None], previous.physics.qpos, q0)
        if r.execution.microbatch_size is None or r.execution.microbatch_size >= batch:
            return self._build(r.model, semantics, ids, q0)
        results = [self._build(r.model, r._slice_tree(semantics, index.start, index.stop),
                               r._slice_tree(ids, index.start, index.stop),
                               r._slice_tree(q0, index.start, index.stop)) for index in r._slices(batch)]
        return r._join(results)

    @_fp32_scope
    def observe(self, target, state):
        # Compile once for this topology. This includes actual velocity, not a
        # finite-difference estimate or commanded twist.
        key = id(target)
        if key not in self._observers:
            def one(model, data):
                jp_t, jr_t = support.jac(model, data, data.site_xpos[target.sid], target.body)
                twist = jp.concatenate((jp_t.T @ data.qvel, jr_t.T @ data.qvel))
                return TaskSemantics(data.site_xpos[target.sid], data.site_xmat[target.sid], twist, jp.int32(0))
            self._observers[key] = jax.jit(jax.vmap(one, in_axes=(0 if target.heterogeneous else None, 0)), device=target.device)
        return self._observers[key](target.models if target.heterogeneous else target.model, state.physics)

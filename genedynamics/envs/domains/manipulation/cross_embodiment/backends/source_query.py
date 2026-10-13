"""Serial MuJoCo reference for semantic source queries (diagnostic only)."""
from dataclasses import asdict
import numpy as np
import mujoco

from genedynamics.learning.cross_embodiment.contracts import ControllerState, ExecutionState, Status
from genedynamics.learning.cross_embodiment.source_query import TaskSemantics, SourceQueryResult
from genedynamics.learning.cross_embodiment.core.control_math import orientation_error
from genedynamics.learning.cross_embodiment.core.source_support import semantic_finite_and_rotation, support_result
from .mujoco import MujocoExecutionBackend


class MujocoSourceQueryBackend:
    def __init__(self, *, execution, task, skill, query, mechanism):
        self.runner = MujocoExecutionBackend(execution=execution, task=task, skill=skill, mechanism=mechanism)
        self.query = query

    def synchronize(self, value):
        self.runner.synchronize(value)

    def describe(self):
        return {"environment": "obstacle_transfer", "source_robot": "panda", "backend": "numpy",
                "query": asdict(self.query), "compiled_batch": False, "supported_phase": 0,
                "controller_initialization": "fresh_zero_history",
                "ik_seed_rule": "source_start_or_previous_source_qpos",
                "model_sha256": self.runner.scene.manifest["model_sha256"]}

    @staticmethod
    def _jacobian(model, data, sid):
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        return np.vstack((jp, jr))

    def _one(self, semantic, identity, q0):
        r, spec = self.runner, self.query
        pos, rot, twist, phase = semantic
        valid = semantic_finite_and_rotation(np, pos, rot, twist, phase)
        pos = np.asarray(pos) if valid else np.asarray(r.task.start_position)
        rot = np.asarray(rot) if valid else np.diag([1., -1., -1.])
        twist = np.asarray(twist) if valid else np.zeros(6)
        data = mujoco.MjData(r.model)
        data.qpos[:] = q0
        for _ in range(spec.iterations):
            mujoco.mj_kinematics(r.model, data)
            mujoco.mj_comPos(r.model, data)
            error = np.r_[pos-data.site_xpos[r.sid], orientation_error(np, data.site_xmat[r.sid].reshape(3, 3), rot)]
            if np.linalg.norm(error[:3]) < spec.position_tolerance*.1 and np.linalg.norm(error[3:]) < spec.orientation_tolerance*.1:
                continue  # fixed iteration semantics, matching device provider
            jac = self._jacobian(r.model, data, r.sid)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + spec.damping**2*np.eye(6), error)
            data.qpos[:] = np.clip(data.qpos + np.clip(dq, -spec.joint_step_limit, spec.joint_step_limit),
                                   r.model.jnt_range[:, 0]+.002, r.model.jnt_range[:, 1]-.002)
        mujoco.mj_kinematics(r.model, data)
        mujoco.mj_comPos(r.model, data)
        jac = self._jacobian(r.model, data, r.sid)
        data.qvel[:] = jac.T @ np.linalg.solve(jac @ jac.T + spec.damping**2*np.eye(6), twist)
        mujoco.mj_forward(r.model, data)
        clearance = min(r.task.contact_margin, min((c.dist for c in data.contact), default=r.task.contact_margin))
        ok, reason, pe, ae, ve = support_result(np, valid, phase, pos, rot, twist, data.site_xpos[r.sid],
                                               data.site_xmat[r.sid].reshape(3, 3), jac @ data.qvel,
                                               data.qpos, data.qvel, r.model.jnt_range, clearance, r.task, spec)
        ctrl = ControllerState(data.site_xpos[r.sid].copy(), data.site_xmat[r.sid].reshape(3, 3).copy(),
                               np.zeros((r.task.max_delay_steps+1, 13)), np.zeros(13), np.zeros(6))
        return SourceQueryResult(semantic, ExecutionState(data, ctrl, identity, 0, 0,
                                                         0 if ok else int(Status.SOURCE_SUPPORT_FAILURE)),
                                 ok, reason, pe, ae, ve, clearance)

    def build(self, semantics, request, *, previous=None):
        batch = len(request.sample_ids)
        if previous is not None and np.shape(previous.sample_id) != (batch,):
            raise ValueError("source warm-start batch mismatch")
        results = []
        for i, identity in enumerate(request.sample_ids):
            q0 = self.runner.scene.start_qpos
            if previous is not None and previous.sample_id[i] == identity and np.all(np.isfinite(previous.physics[i].qpos)):
                q0 = previous.physics[i].qpos
            results.append(self._one(TaskSemantics(*(np.asarray(x[i]) for x in semantics)), identity, q0))
        return SourceQueryResult(semantics, self.runner._stack([x.virtual_state for x in results]),
                                 *(np.asarray([x[j] for x in results]) for j in range(2, len(SourceQueryResult._fields))))

    def observe(self, target, state):
        records = []
        for i, data in enumerate(state.physics):
            model = target.models[i] if target.heterogeneous else target.model
            jac = self._jacobian(model, data, target.sid)
            records.append(TaskSemantics(data.site_xpos[target.sid].copy(), data.site_xmat[target.sid].reshape(3, 3).copy(),
                                         jac @ data.qvel, np.int32(0)))
        return TaskSemantics(*(np.stack([x[j] for x in records]) for j in range(4)))

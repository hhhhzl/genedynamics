"""Explicit ACP-style anisotropic torque control in the existing CPU workcell.

This is an adapted simulator controller. It preserves world virtual targets
and a complete symmetric tool-frame stiffness matrix; it does not identify
the author's proprietary hardware servo, asynchronous timing or damping.
The physical scene, MuJoCo timestep, mechanism and success predicates remain
the task's. No 19D ACP action is flattened into the shared 13D hybrid policy.
"""
from __future__ import annotations

from copy import copy, deepcopy

import numpy as np

from genedynamics.tasks.transfer.episode import TaskStep, numpy_action


class ACPContactExecutor:
    def __init__(self, task):
        if task.spec.family != "contact" or task.config.mobile:
            raise NotImplementedError("anisotropic ACP control currently supports fixed contact arms only")
        self.task = task
        self.queues = None

    def reset(self, mask, state):
        if self.queues is None or len(self.queues) != len(state.sample_id):
            self.queues = [[] for _ in state.sample_id]
        for index in np.flatnonzero(mask):
            self.queues[index] = []

    def reorder(self, order):
        self.queues = [self.queues[int(index)] for index in order]

    def get_checkpoint(self):
        return {"queues": deepcopy(self.queues)}

    def restore_checkpoint(self, payload):
        self.queues = deepcopy(payload["queues"])

    def _commands(self, command):
        fields = ("reference_pose_world", "virtual_target_pose_world", "stiffness_tool")
        result = {}
        for key, shape in zip(fields, ((4, 4), (4, 4), (6, 6))):
            value = numpy_action(command[key]).astype(float, copy=True)
            if value.shape == shape and self.task.num_envs == 1:
                value = value[None]
            if value.shape != (self.task.num_envs, *shape) or not np.isfinite(value).all():
                raise ValueError(f"ACP {key} must be finite [B,{shape}]")
            result[key] = value
        for key in fields[:2]:
            rotation = result[key][:, :3, :3]
            if (not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-5)
                    or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5)
                    or not np.allclose(result[key][:, 3], [0., 0., 0., 1.])):
                raise ValueError("ACP targets must be proper world-from-tool SE(3) poses")
        stiffness = result["stiffness_tool"]
        if not np.allclose(stiffness, np.swapaxes(stiffness, -1, -2), atol=1e-6):
            raise ValueError("ACP stiffness matrix must be symmetric")
        if np.any(np.linalg.eigvalsh(stiffness) <= 0):
            raise ValueError("ACP stiffness must be positive definite; negative diffusion stiffness is rejected")
        return result

    def _step_one(self, state, command, world):
        import mujoco
        from genedynamics.tasks.transfer.contact import ContactTrace, contact_termination
        from genedynamics.core.control.transfer import orientation_error

        task, provider, cfg = self.task, self.task._provider, self.task.config
        if state.status != 0:
            return state, None, None
        # Known command delay applies to the complete ACP target and matrix.
        # Its history is private execution state, never a public sensor.
        queue = self.queues[world]
        queue.insert(0, deepcopy(command))
        del queue[5:]
        delay = provider.mechanism.delay_steps
        if len(queue) <= delay:
            pose = np.eye(4)
            pose[:3, :3] = state.reference_rotation
            pose[:3, 3] = state.reference_position
            applied = dict(reference_pose_world=pose, virtual_target_pose_world=pose,
                           stiffness_tool=np.diag([800.] * 3 + [60.] * 3))
        else:
            applied = queue[delay]
        data, model, sid = copy(state.physics), provider.model, provider.sid
        reference, virtual = applied["reference_pose_world"], applied["virtual_target_pose_world"]
        # The matrix is expressed in the measured tool frame at the command
        # boundary. Keep that frame fixed during the 2 ms physical substeps.
        rotation = data.site_xmat[sid].reshape(3, 3)
        transform = np.zeros((6, 6)); transform[:3, :3] = rotation; transform[3:, 3:] = rotation
        stiffness = transform @ applied["stiffness_tool"] @ transform.T
        eigenvalues, eigenvectors = np.linalg.eigh(stiffness)
        damping = (eigenvectors * (2. * np.sqrt(eigenvalues))) @ eigenvectors.T
        wrench, _, touch, forbidden = provider._measure(data)
        peak, lateral, saturation = 0., 0., 0.
        for _ in range(cfg.substeps):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            jac = np.r_[jp, jr]
            error = np.r_[virtual[:3, 3] - data.site_xpos[sid],
                          orientation_error(np, data.site_xmat[sid].reshape(3, 3), virtual[:3, :3])]
            desired = stiffness @ error - damping @ (jac @ data.qvel)
            desired[:2] = np.clip(desired[:2], -cfg.tangential_wrench_limit, cfg.tangential_wrench_limit)
            desired[3:] = np.clip(desired[3:], -cfg.orientation_wrench_limit, cfg.orientation_wrench_limit)
            posture = 10. * (provider.scene.start_qpos - data.qpos)
            null = posture - jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), jac @ posture)
            gravity = np.zeros(model.nv)
            for body in range(1, model.nbody):
                point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ provider.scene.nominal_ipos[body]
                mujoco.mj_jac(model, data, jp, jr, point, body)
                gravity += jp.T @ (-provider.scene.nominal_masses[body] * model.opt.gravity)
            tau = (provider.mechanism.servo_gain * (jac.T @ desired + null)
                   + gravity + model.dof_damping * data.qvel - 2. * data.qvel)
            saturation = max(saturation, float(np.max(np.abs(tau) / provider.limits)))
            data.ctrl[:] = np.clip(tau, -provider.limits, provider.limits)[provider.actuator_dofs]
            mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
            wrench, force, contact, bad = provider._measure(data)
            peak = max(peak, force); lateral = max(lateral, float(np.linalg.norm(wrench[:2])))
            touch |= contact; forbidden |= bad
        # This force flag is solely for the common support-loss predicate. It
        # is NOT a force command to an ACP hybrid controller. Actual normal
        # force remains exclusively measured from MuJoCo contacts.
        loading = bool(virtual[2, 3] < reference[2, 3] - 1e-7)
        predicate_command = np.zeros(13); predicate_command[12] = cfg.desired_force if loading else 0.
        predicate_state = state._replace(last_command=predicate_command)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        status, seen, loss, hold, lost = contact_termination(
            np, predicate_state, data.site_xpos[sid], wrench[2], peak, forbidden, finite, saturation > 1,
            cfg, rotation=data.site_xmat[sid].reshape(3, 3), lateral_peak=lateral,
            axial_contact=provider._axial_support(data),
        )
        if np.any(data.qpos < model.jnt_range[:, 0] - 1e-4) or np.any(data.qpos > model.jnt_range[:, 1] + 1e-4):
            status = 3
        if np.max(np.abs(data.qvel)) > 3.:
            status = 4
        new = state._replace(physics=data, reference_position=reference[:3, 3].copy(),
                             reference_rotation=reference[:3, :3].copy(), measured_force=wrench[2],
                             contact_seen=seen, loss_count=loss, hold_count=hold, status=int(status), steps=state.steps + 1)
        stage = 3 if cfg.insertion and data.site_xpos[sid, 2] < cfg.goal[2] + .01 else (1 if wrench[2] > cfg.contact_threshold else 0)
        trace = ContactTrace(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy(), predicate_command,
                             wrench, wrench[2], peak, touch, forbidden, lost, saturation, new.force_integral,
                             stage, new.status, True, bool(new.status), lateral)
        return new, trace, applied

    def step(self, command):
        from genedynamics.tasks.transfer.contact import ContactTrace

        task = self.task
        commands = self._commands(command)
        active = np.asarray(task._state.status) == 0
        if not active.any():
            raise RuntimeError("all episodes ended; reset before ACP execution")
        for world in np.flatnonzero(active):
            record = task._episodes[task._current[world]]
            if record.requested_actions and record.controller != "acp_anisotropic_adapted":
                raise ValueError("controller family cannot change inside an episode")
            record.controller = "acp_anisotropic_adapted"
        states, traces, applied = [], [], []
        for world in range(task.num_envs):
            one = task._provider._slice(task._state, world)
            requested = {key: value[world].copy() for key, value in commands.items()}
            new, trace, actual = self._step_one(one, requested, world)
            if trace is None:
                # Only shapes are padded; no physical transition is requested
                # for an already terminated world in a mixed active batch.
                data = one.physics
                trace = ContactTrace(data.site_xpos[task._sid].copy(), data.site_xmat[task._sid].reshape(3, 3).copy(),
                                     np.zeros(13), np.zeros(6), one.measured_force, 0., False, False, False,
                                     0., one.force_integral, 0, one.status, False, False, 0.)
                actual = requested
            states.append(new); traces.append(trace); applied.append(actual)
        task._state = task._provider._stack(states)
        task._public_reference_position[active] = commands["reference_pose_world"][active, :3, 3]
        task._public_reference_rotation[active] = commands["reference_pose_world"][active, :3, :3]
        trace = ContactTrace(*(np.stack([np.asarray(row[field])[None] for row in traces]) for field in range(len(ContactTrace._fields))))
        # Common 13D history explicitly records zero shared-skill channels;
        # the ACP physical controls and request are retained structurally.
        placeholder = np.zeros((task.num_envs, 13), dtype=np.float32)
        placeholder[:, 12] = trace.command[:, 0, 12]
        requests = [{key: value[world].copy() for key, value in commands.items()} for world in range(task.num_envs)]
        task._charge("command_requests", int(active.sum()))
        result = task._finish_step(placeholder, trace, active, applied_action=np.zeros_like(placeholder), replay_request=requests)
        for world in np.flatnonzero(active):
            task._episodes[task._current[world]].traces[-1]["acp_loading_definition"] = "virtual target below reference in world z"
        physical = {key: np.stack([row[key] for row in applied]) for key in commands}
        physical["actuator_command"] = np.stack([data.ctrl.copy() for data in task._state.physics])
        return TaskStep(result.observation, result.reward, result.done, result.status, physical, result.trace)

"""Explicit four-task adapter for ACP's native 19D diffusion commands.

The diffusion model continues to predict reference pose, virtual target and
scalar stiffness. This adapter maps that structured command to the benchmark's
public 13D task controller/WBC. It is a declared task adaptation, not the
original hardware servo or a native-paper quantitative reproduction.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from genedynamics.tasks.transfer.guide_observation import public_memory
from genedynamics.solvers.single.acp.learning.task_training import build_policy, history_features, pose
from genedynamics.core.control.transfer import integrate_rotation, orientation_error
from genedynamics.solvers.single.acp import ACPSolver
from genedynamics.solvers.single.acp.backends import spatial
from genedynamics.solvers.single.acp.backends.compliance import decode_stiffness_tool


REFERENCE_STIFFNESS = np.asarray([800., 800., 800., 60., 60., 60.])


def _batch_pose(position, rotation):
    value = np.tile(np.eye(4), (len(position), 1, 1))
    value[:, :3, :3] = rotation
    value[:, :3, 3] = position
    return value


class ACPSharedControllerAdapter:
    """Map native ACP targets to the common task controller without private state."""
    def __init__(self, spec):
        self.spec = spec

    def decode(self, prediction, public, *, waypoint=0):
        reference = np.asarray(prediction["reference_pose_world"])[:, waypoint, 0]
        virtual = np.asarray(prediction["virtual_target_pose_world"])[:, waypoint, 0]
        stiffness = np.asarray(prediction["stiffness_tool"])[:, waypoint, 0]
        batch = len(reference)
        if reference.shape != (batch, 4, 4) or virtual.shape != reference.shape or stiffness.shape != (batch, 6, 6):
            raise ValueError("ACP shared adapter requires [B,H,1] structured commands")
        dt = self.spec.config.control_dt
        old_position = np.asarray(public["reference_position"], dtype=float)
        old_rotation = np.asarray(public["reference_rotation"], dtype=float)
        action = np.zeros((batch, 13), dtype=np.float32)
        action[:, :3] = (reference[:, :3, 3] - old_position) / dt
        action[:, 3:6] = np.stack([orientation_error(np, old_rotation[i], reference[i, :3, :3]) for i in range(batch)]) / dt
        diagonal = np.diagonal(stiffness, axis1=-2, axis2=-1)
        if np.any(diagonal <= 0) or not np.isfinite(diagonal).all():
            raise ValueError("ACP shared adapter requires positive finite stiffness")
        action[:, 6:12] = np.log(diagonal / REFERENCE_STIFFNESS)
        if self.spec.family == "contact":
            raw_scalar = prediction.get("stiffness_scalar")
            if raw_scalar is not None:
                scalar = np.asarray(raw_scalar)
                scalar = scalar.reshape(batch, -1)[:, waypoint]
            else:
                scalar = diagonal[:, 2]
            action[:, 12] = np.maximum(0., scalar * (reference[:, 2, 3] - virtual[:, 2, 3]))
        return action

    def encode_teacher_request(self, shared_action, public, *, scalar_stiffness=800.):
        shared = np.asarray(shared_action, dtype=np.float32)
        if shared.ndim == 1:
            shared = shared[None]
        if shared.shape != (len(public["tcp_position"]), 13):
            raise ValueError("teacher shared action must be [B,13]")
        dt = self.spec.config.control_dt
        reference = _batch_pose(np.asarray(public["reference_position"], dtype=float), np.asarray(public["reference_rotation"], dtype=float))
        reference[:, :3, 3] += shared[:, :3] * dt
        for i in range(len(reference)):
            reference[i, :3, :3] = integrate_rotation(np, reference[i, :3, :3], shared[i, 3:6], dt)
        virtual = reference.copy()
        scalar = np.full((len(reference),), float(scalar_stiffness))
        if self.spec.family == "contact":
            virtual[:, 2, 3] -= np.maximum(shared[:, 12], 0.) / scalar
        current = _batch_pose(np.asarray(public["tcp_position"], dtype=float), np.asarray(public["tcp_rotation"], dtype=float))
        stiffness = decode_stiffness_tool(reference[:, None, None], virtual[:, None, None], scalar[:, None, None], current[:, None, None])
        prediction = {
            "reference_pose_world": reference[:, None, None],
            "virtual_target_pose_world": virtual[:, None, None],
            "stiffness_tool": stiffness,
            "stiffness_scalar": scalar[:, None, None],
        }
        request = np.concatenate((spatial.SE3_to_pose9(reference), spatial.SE3_to_pose9(virtual), scalar[:, None]), axis=-1).astype(np.float32)
        return request, prediction


class Native19Demonstrator:
    """Execute a public teacher through the same native19-to-shared adapter."""
    def __init__(self, task, teacher, *, scalar_stiffness=800.):
        self.teacher = teacher
        self.adapter = ACPSharedControllerAdapter(task.spec)
        self.scalar_stiffness = scalar_stiffness
        self.requests = []
        self.adapted_actions = []

    def get_state(self):
        if not hasattr(self.teacher, "get_state"):
            raise AttributeError("wrapped teacher does not expose continuation state")
        return {
            "teacher": self.teacher.get_state(),
            "requests": deepcopy(self.requests),
            "adapted_actions": deepcopy(self.adapted_actions),
        }

    def restore(self, state):
        self.teacher.restore(state["teacher"])
        self.requests = deepcopy(state["requests"])
        self.adapted_actions = deepcopy(state["adapted_actions"])

    def reset(self):
        if hasattr(self.teacher, "reset"):
            self.teacher.reset()
        self.requests.clear()
        self.adapted_actions.clear()

    def act(self, public):
        shared = self.teacher.act(public)
        request, prediction = self.adapter.encode_teacher_request(shared, public, scalar_stiffness=self.scalar_stiffness)
        adapted = self.adapter.decode(prediction, public)
        self.requests.append(request[0].copy())
        self.adapted_actions.append(adapted[0].copy())
        return adapted


class ACPFourTaskActor:
    """Deploy a native PyTorch ACP checkpoint through the declared adapter."""
    def __init__(self, task, payload, mode, seed, device="cpu"):
        from genedynamics.core.random import seed_all
        policy = build_policy(payload["shape_meta"], payload["history_len"], action_chart=payload["action_chart"])
        self.solver = ACPSolver(policy, payload["shape_meta"], device=device, raw_sample_interval_s=.02,
                                slow_down_factor=1., execution_horizon=1)
        self.solver.load_policy_state_dict(payload)
        seed_all(seed)
        self.memory = public_memory(task.spec, [mode])
        self.adapter = ACPSharedControllerAdapter(task.spec)
        self.history = []
        self.length = payload["history_len"]
        self.context = None
        self.inference_seconds = []
        self.irreversible_network_calls = 0

    def observe_tick(self, public):
        self.history.append(public)
        self.context = self.memory.update(public)

    def act(self, public):
        started = time.perf_counter()
        prediction = self.solver.act({
            "sparse": history_features(self.history, self.length, controller_context=self.context),
            "current_tool_pose_world": pose(public["tcp_position"], public["tcp_rotation"])[:, None],
        })
        self.inference_seconds.append(time.perf_counter() - started)
        self.irreversible_network_calls += 1
        return self.adapter.decode(prediction, public)


__all__ = ["ACPSharedControllerAdapter", "Native19Demonstrator", "ACPFourTaskActor"]

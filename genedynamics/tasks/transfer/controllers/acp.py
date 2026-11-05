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
from genedynamics.core.control.transfer import integrate_rotation, orientation_error
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
        coordinate = float(waypoint)
        reference_waypoints = np.asarray(prediction["reference_pose_world"])
        virtual_waypoints = np.asarray(prediction["virtual_target_pose_world"])
        if reference_waypoints.shape != virtual_waypoints.shape or reference_waypoints.ndim != 5:
            raise ValueError("ACP shared adapter requires [B,H,R,4,4] pose commands")
        lower = int(np.floor(coordinate))
        upper = min(lower + 1, reference_waypoints.shape[1] - 1)
        if lower < 0 or lower >= reference_waypoints.shape[1]:
            raise ValueError("ACP waypoint coordinate lies outside the predicted horizon")
        fraction = coordinate - lower
        reference = spatial.interpolate_SE3(
            reference_waypoints[:, lower, 0], reference_waypoints[:, upper, 0], fraction,
        )
        virtual = spatial.interpolate_SE3(
            virtual_waypoints[:, lower, 0], virtual_waypoints[:, upper, 0], fraction,
        )
        raw_scalar = prediction.get("stiffness_scalar")
        current = _batch_pose(
            np.asarray(public["tcp_position"], dtype=float),
            np.asarray(public["tcp_rotation"], dtype=float),
        )
        if raw_scalar is not None:
            scalar_waypoints = np.asarray(raw_scalar).reshape(len(reference), -1)
            scalar = (
                (1. - fraction) * scalar_waypoints[:, lower]
                + fraction * scalar_waypoints[:, upper]
            )
            stiffness = decode_stiffness_tool(reference, virtual, scalar, current)
        else:
            stiffness_waypoints = np.asarray(prediction["stiffness_tool"])
            stiffness = (
                (1. - fraction) * stiffness_waypoints[:, lower, 0]
                + fraction * stiffness_waypoints[:, upper, 0]
            )
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
            if raw_scalar is not None:
                scalar = np.asarray(scalar)
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
    VERSION = "public-acp-native19-sparse-interpolation-v2"

    def __init__(self, task, payload, mode, seed, device="cpu"):
        from genedynamics.core.random import seed_all
        from genedynamics.solvers.single.acp import ACPSolver
        from genedynamics.solvers.single.acp.learning.task_training import (
            build_policy, history_features, pose,
        )

        self._history_features = history_features
        self._pose = pose
        self.spec = task.spec
        self.mode = int(mode)
        self.seed = int(seed)
        policy = build_policy(payload["shape_meta"], payload["history_len"], action_chart=payload["action_chart"])
        # ACP's released runner executes 12 sparse waypoints and continuously
        # interpolates between them.  Here raw_sample_interval_s is the public
        # task period; shape_meta supplies the sparse waypoint downsampling.
        self.solver = ACPSolver(
            policy, payload["shape_meta"], device=device,
            raw_sample_interval_s=float(task.spec.config.control_dt),
            slow_down_factor=1.,
        )
        self.solver.load_policy_state_dict(payload)
        seed_all(seed)
        self.memory = public_memory(task.spec, [mode])
        self.adapter = ACPSharedControllerAdapter(task.spec)
        self.length = payload["history_len"]
        action_meta = payload["shape_meta"]["sample"]["action"]["sparse"]
        self.action_down_sample_steps = int(action_meta["down_sample_steps"])
        if self.action_down_sample_steps < 1:
            raise ValueError("ACP requires positive sparse action downsampling")
        if self.solver.execution_horizon != 12:
            raise ValueError("released ACP execution profile requires 12 waypoints")
        if self.solver.execution_horizon >= int(action_meta["horizon"]):
            raise ValueError("ACP interpolation requires a prediction beyond the execution chunk")
        import torch
        self._rng = (
            torch.cuda.get_rng_state(self.solver.device)
            if self.solver.device.type == "cuda" else torch.get_rng_state()
        )
        self.reset()

    @staticmethod
    def _cache_prediction(prediction):
        """Retain only native structured commands needed by the task adapter."""
        keys = (
            "reference_pose_world", "virtual_target_pose_world",
            "stiffness_tool", "stiffness_scalar", "waypoint_times_s",
        )
        return {
            key: np.asarray(prediction[key]).copy()
            for key in keys if key in prediction
        }

    def reset(self):
        self.solver.reset()
        self.memory.reset([self.mode])
        self.history = []
        self.context = None
        self.inference_seconds = []
        self.irreversible_network_calls = 0
        self.last_tick = -1
        self.cached_action = None
        self.cached_prediction = None
        self.chunk_control_step = 0
        self.chunk_waypoint = 0.
        self.execution_horizon = self.solver.execution_horizon
        self.execution_control_steps = (
            self.execution_horizon * self.action_down_sample_steps
        )

    def get_state(self):
        return deepcopy({
            "version": self.VERSION,
            "task": self.spec.alias,
            "robot": self.spec.config.robot,
            "mode": self.mode,
            "seed": self.seed,
            "memory": self.memory.get_state(),
            "history": self.history,
            "context": self.context,
            "inference_seconds": self.inference_seconds,
            "irreversible_network_calls": self.irreversible_network_calls,
            "last_tick": self.last_tick,
            "cached_action": self.cached_action,
            "cached_prediction": self.cached_prediction,
            "chunk_control_step": self.chunk_control_step,
            "chunk_waypoint": self.chunk_waypoint,
            "execution_horizon": self.execution_horizon,
            "action_down_sample_steps": self.action_down_sample_steps,
            "execution_control_steps": self.execution_control_steps,
            "rng": self._rng,
        })

    def restore(self, state):
        expected = (
            self.VERSION, self.spec.alias, self.spec.config.robot,
            self.mode, self.seed, self.solver.execution_horizon,
            self.action_down_sample_steps, self.execution_control_steps,
        )
        actual = (
            state["version"], state["task"], state["robot"],
            state["mode"], state["seed"], state["execution_horizon"],
            state["action_down_sample_steps"], state["execution_control_steps"],
        )
        if actual != expected:
            raise ValueError("public ACP binding or execution profile differs")
        self.solver.reset()
        self.memory.restore(state["memory"])
        self.history = deepcopy(state["history"])
        self.context = deepcopy(state["context"])
        self.inference_seconds = list(state["inference_seconds"])
        self.irreversible_network_calls = int(state["irreversible_network_calls"])
        self.last_tick = int(state["last_tick"])
        self.cached_action = deepcopy(state["cached_action"])
        self.cached_prediction = deepcopy(state["cached_prediction"])
        self.chunk_control_step = int(state["chunk_control_step"])
        self.chunk_waypoint = float(state["chunk_waypoint"])
        self.execution_horizon = int(state["execution_horizon"])
        self.action_down_sample_steps = int(state["action_down_sample_steps"])
        self.execution_control_steps = int(state["execution_control_steps"])
        self._rng = state["rng"].clone()

    def observe_tick(self, public):
        self.history.append(public)
        self.context = self.memory.update(public)

    def act(self, public):
        # The formal executor calls one public action per observed tick.  Keep
        # ACP's native history and controller context causal inside the actor,
        # as the other learned task actors do, while executing the predicted
        # sparse chunk between diffusion calls as in the source controller.
        tick = int(np.rint(
            np.asarray(public["elapsed_seconds"]).item()
            / self.spec.config.control_dt
        ))
        if tick == self.last_tick:
            return self.cached_action.copy()
        if tick != self.last_tick + 1:
            raise ValueError(
                "ACP must observe every public control tick; reset at episode start"
            )
        self.observe_tick(public)
        if (
            self.cached_prediction is None
            or self.chunk_control_step >= self.execution_control_steps
        ):
            import torch
            devices = (
                [self.solver.device.index or 0]
                if self.solver.device.type == "cuda" else []
            )
            started = time.perf_counter()
            with torch.random.fork_rng(devices=devices):
                if devices:
                    torch.cuda.set_rng_state(self._rng, self.solver.device)
                else:
                    torch.set_rng_state(self._rng)
                prediction = self.solver.act({
                    "sparse": self._history_features(
                        self.history, self.length,
                        controller_context=self.context,
                    ),
                    "current_tool_pose_world": self._pose(
                        public["tcp_position"], public["tcp_rotation"]
                    )[:, None],
                })
                self._rng = (
                    torch.cuda.get_rng_state(self.solver.device)
                    if devices else torch.get_rng_state()
                )
            self.inference_seconds.append(time.perf_counter() - started)
            self.irreversible_network_calls += 1
            self.cached_prediction = self._cache_prediction(prediction)
            self.chunk_control_step = 0
        self.chunk_waypoint = (
            self.chunk_control_step / self.action_down_sample_steps
        )
        self.cached_action = self.adapter.decode(
            self.cached_prediction, public, waypoint=self.chunk_waypoint,
        )
        self.chunk_control_step += 1
        self.last_tick = tick
        return self.cached_action.copy()


__all__ = ["ACPSharedControllerAdapter", "Native19Demonstrator", "ACPFourTaskActor"]

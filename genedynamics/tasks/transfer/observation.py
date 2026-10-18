"""Shared causal controller memory. Inputs are public observations and task geometry.

This is available to every method, including learned baselines. It does not
expose expert actions, plant state, delayed references or future observations.
Call update at every observed control tick, even when an actor executes chunks.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from genedynamics.tasks.transfer.behavior import (
    advance_behavior, behavior_reset, keyway_alignment)
from genedynamics.core.control.transfer import orientation_error

FEATURE_VERSION = "public-controller-memory-v1"
CONTEXT_DIM = 53


def obstacle_routes(goal):
    return np.asarray([[[.38, -.12, .67], [.61, -.12, .67], [.64, .08, .58], goal],
                       [[.38, -.16, .55], [.66, -.16, .55], [.66, .08, .55], goal]], dtype=np.float64)


class PublicControllerMemory:
    """Backend-neutral, checkpointable observer; never receives a task executor."""
    def __init__(self, spec, modes):
        if spec.alias not in {"T1", "T3"}:
            raise ValueError("shared memory is frozen for T1/T3 only")
        self.spec = spec
        self.reset(modes)

    def reset(self, modes):
        self.modes = np.asarray(modes, dtype=np.int32).reshape(-1)
        if np.any((self.modes < 0) | (self.modes > 1)):
            raise ValueError("mode must be 0 or 1")
        self.memory = behavior_reset(len(self.modes), self.modes)
        self.last_tick = np.full(len(self.modes), -1, dtype=np.int64)
        self.start_position = self.start_rotation = None
        self.start_reference_position = self.start_reference_rotation = None

    def get_state(self):
        return deepcopy({key: value for key, value in vars(self).items() if key != "spec"})

    def restore(self, state):
        for key, value in deepcopy(state).items():
            if key not in {"modes", "memory", "last_tick", "start_position", "start_rotation",
                           "start_reference_position", "start_reference_rotation"}:
                raise ValueError("unknown public memory field")
            setattr(self, key, value)

    def reorder(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        self.memory = type(self.memory)(*(np.asarray(x)[indices].copy() for x in self.memory))
        for key in ("modes", "last_tick", "start_position", "start_rotation",
                    "start_reference_position", "start_reference_rotation"):
            if getattr(self, key) is not None:
                setattr(self, key, getattr(self, key)[indices].copy())

    def update(self, public):
        p = np.asarray(public["tcp_position"], dtype=np.float64)
        r = np.asarray(public["tcp_rotation"], dtype=np.float64)
        rp, rr = np.asarray(public["reference_position"]), np.asarray(public["reference_rotation"])
        tick = np.rint(np.asarray(public["elapsed_seconds"]).reshape(-1) / self.spec.config.control_dt).astype(np.int64)
        if len(p) != len(self.modes) or np.any(tick < self.last_tick):
            raise ValueError("reset/reorder public memory explicitly before changing episodes or rows")
        # Skipping observations would change the causal phase/integral clock.
        if np.any((self.last_tick >= 0) & (tick > self.last_tick + 1)):
            raise ValueError("public memory requires every observed control tick")
        fresh = tick != self.last_tick
        if self.start_position is None:
            self.start_position, self.start_rotation = p.copy(), r.copy()
            self.start_reference_position, self.start_reference_rotation = rp.copy(), rr.copy()
        boundary = fresh & ((tick % 10 == 0) if self.spec.alias == "T3" else True)
        if self.spec.alias == "T3":
            updated = advance_behavior(self.memory, p, r, np.asarray(public["measured_wrench"])[:, 2],
                dt=10 * self.spec.config.control_dt, goal_z=self.spec.config.goal[2],
                alignment_mask=keyway_alignment(p, r, self.spec.config.mating_clearance))
            self.memory = type(updated)(*(np.where(boundary.reshape((-1,) + (1,) * (np.ndim(a) - 1)), a, b)
                                         for a, b in zip(updated, self.memory)))
        else:
            routes = obstacle_routes(self.spec.config.goal_position)
            old = self.memory.phase
            target = routes[self.modes, old]
            phase = np.minimum(old + ((np.linalg.norm(p - target, axis=-1) < .012) & fresh), 3)
            target = routes[self.modes, phase]
            angles = np.stack([orientation_error(np, a, b) for a, b in zip(r, public["goal_rotation"])])
            error = np.concatenate((target - p, angles), axis=-1)
            previous = np.where((phase != old)[:, None], 0., self.memory.tracking_integral)
            integral = np.clip(previous + .8 * self.spec.config.control_dt * error, -.04, .04)
            self.memory = self.memory._replace(phase=phase, tracking_integral=np.where(fresh[:, None], integral, previous))
        for key, value in (("start_position", p), ("start_rotation", r),
                           ("start_reference_position", rp), ("start_reference_rotation", rr)):
            mask = boundary.reshape((-1,) + (1,) * (value.ndim - 1))
            setattr(self, key, np.where(mask, value, getattr(self, key)))
        self.last_tick = tick.copy()
        scale = .05 if self.spec.alias == "T3" else max(.05, float(np.linalg.norm(
            np.asarray(self.spec.config.goal_position) - self.spec.config.start_position)))
        def relative(position, rotation):
            translation = np.einsum("bji,bj->bi", r, position - p) / scale
            rotation = np.swapaxes(r, -1, -2) @ rotation
            return np.concatenate((translation, rotation[..., :, :2].swapaxes(-1, -2).reshape(len(p), 6)), axis=-1)
        high = np.asarray(([.12]*3 + [.6]*3 + [1.]*6 + [1.]) if self.spec.alias == "T1"
                          else ([.08]*3 + [.4]*3 + [1.]*6 + [self.spec.config.force_limit]))
        return np.concatenate((relative(rp, rr), np.asarray(public["previous_action"]) / high,
            np.eye(4)[self.memory.phase], np.eye(2)[self.modes],
            self.memory.tracking_integral / (.04 if self.spec.alias == "T1" else .08),
            ((tick % 10) / 10. if self.spec.alias == "T3" else np.zeros_like(tick))[:, None],
            relative(self.start_position, self.start_rotation),
            relative(self.start_reference_position, self.start_reference_rotation)), axis=-1).astype(np.float32)


def context_sequence(public, spec, mode):
    observer = PublicControllerMemory(spec, [mode])
    return np.stack([observer.update({key: np.asarray(value)[i:i+1] for key, value in public.items()})[0]
                     for i in range(len(public["tcp_position"]))])

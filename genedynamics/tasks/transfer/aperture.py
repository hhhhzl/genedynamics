"""Task-owned geometry and public controller for semantic aperture braking."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from genedynamics.core.control.transfer import orientation_error


VERSION = "public-aperture-brake-v3"
SEQUENTIAL_VERSION = "public-sequential-aperture-reorientation-v1"


def aperture_rotation(yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([[c, s, 0.], [s, -c, 0.], [0., 0., -1.]])


def _box_vertices(center, half_size):
    signs = np.asarray([
        [x, y, z]
        for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)
    ])
    return np.asarray(center, dtype=float) + signs * np.asarray(half_size, dtype=float)


# Only the rigidly carried object enters the envelope test. The gripper is
# separately covered by the scene's ordinary collision termination.
OBJECT_VERTICES_TCP = np.concatenate((
    _box_vertices((0., 0., .07), (.06, .01, .01)),
    _box_vertices((-.05, .04, .07), (.01, .03, .01)),
))
OBJECT_CENTER_TCP = (
    OBJECT_VERTICES_TCP.min(axis=0) + OBJECT_VERTICES_TCP.max(axis=0)
) / 2


def object_vertices(tcp_position, tcp_rotation):
    """Return complete payload vertices in world coordinates."""
    position = np.asarray(tcp_position, dtype=float)
    rotation = np.asarray(tcp_rotation, dtype=float)
    return position + OBJECT_VERTICES_TCP @ rotation.T


def object_pose_and_twist(tcp_position, tcp_rotation, tcp_twist):
    """Rigidly transform the public TCP state to the payload-envelope frame."""
    position = np.asarray(tcp_position, dtype=float)
    rotation = np.asarray(tcp_rotation, dtype=float)
    twist = np.asarray(tcp_twist, dtype=float)
    offset = rotation @ OBJECT_CENTER_TCP
    object_position = position + offset
    object_twist = np.r_[twist[:3] + np.cross(twist[3:], offset), twist[3:]]
    return object_position, rotation.copy(), object_twist


def fixed_orientation_fits_both_apertures(config, *, samples=3601):
    """Conservatively test whether one yaw can fit both orthogonal openings."""
    yaws = np.linspace(0., np.pi, int(samples), endpoint=True)
    local = OBJECT_VERTICES_TCP - OBJECT_CENTER_TCP
    for yaw in yaws:
        vertices = local @ aperture_rotation(float(yaw)).T
        first_fits = (
            np.ptp(vertices[:, 1]) <= 2 * config.aperture_half_width
            and np.ptp(vertices[:, 2]) <= 2 * config.aperture_half_height
        )
        second_fits = (
            np.ptp(vertices[:, 0]) <= 2 * config.second_aperture_half_width
            and np.ptp(vertices[:, 2]) <= 2 * config.second_aperture_half_height
        )
        if first_fits and second_fits:
            return True
    return False


class PublicApertureBrakePolicy:
    """Known-geometry source skill: align, cross, brake, and hold."""

    VERSION = VERSION

    def __init__(self, spec, *, mode=0):
        if spec.name != "aperture_brake" or spec.config.scene_profile != "aperture_brake_v1":
            raise ValueError("semantic aperture-brake task required")
        self.spec = spec
        self.mode = int(mode)
        if self.mode not in (0, 1):
            raise ValueError("aperture source mode must be 0 or 1")
        cfg = spec.config
        side = -1. if self.mode == 0 else 1.
        z = cfg.goal_position[2]
        object_offset = np.asarray(cfg.goal_position) - np.asarray([
            cfg.goal_position[0], cfg.aperture_center_y, cfg.aperture_center_z,
        ])
        tcp_center_y = cfg.aperture_center_y + object_offset[1]
        self.waypoints = np.asarray([
            [.425, tcp_center_y + side * .015, z],
            [cfg.aperture_plane_x - .075, tcp_center_y, z],
            [cfg.aperture_exit_zone[0] + .025, tcp_center_y, z],
            [*cfg.goal_position],
        ], dtype=float)
        self.reset()

    def reset(self, *, modes=None):
        if modes is not None:
            values = np.asarray(modes, dtype=int).reshape(-1)
            if values.shape != (1,) or int(values[0]) != self.mode:
                raise ValueError("aperture base is frozen to one requested mode")
        self.modes = np.asarray([self.mode], dtype=np.int64)
        self.phase = np.zeros(1, dtype=np.int64)
        self.path_distance = np.zeros(1, dtype=float)
        self.last_tick = np.full(1, -1, dtype=np.int64)
        self.integral = np.zeros((1, 6), dtype=float)
        self.raw = np.zeros((1, 13), dtype=np.float32)

    def reference(self):
        return self.waypoints[self.phase].copy()

    def get_state(self):
        return deepcopy({
            "version": self.VERSION, "mode": self.mode,
            "phase": self.phase, "path_distance": self.path_distance,
            "last_tick": self.last_tick,
            "integral": self.integral, "raw": self.raw,
        })

    def restore(self, state):
        if state.get("version") != self.VERSION or int(state.get("mode", -1)) != self.mode:
            raise ValueError("aperture public binding differs")
        for name in ("phase", "path_distance", "last_tick", "integral", "raw"):
            setattr(self, name, np.asarray(state[name]).copy())

    def reorder(self, indices):
        indices = np.asarray(indices, dtype=int)
        if indices.shape != (1,) or int(indices[0]) != 0:
            raise ValueError("batch-one aperture actor only supports identity reorder")

    def act(self, public):
        position = np.asarray(public["tcp_position"], dtype=float)
        rotation = np.asarray(public["tcp_rotation"], dtype=float)
        reference_position = np.asarray(public["reference_position"], dtype=float)
        reference_rotation = np.asarray(public["reference_rotation"], dtype=float)
        goal_rotation = np.asarray(public["goal_rotation"], dtype=float)
        elapsed = np.asarray(public["elapsed_seconds"], dtype=float).reshape(-1)
        if position.shape != (1, 3):
            raise ValueError("aperture source policy is qualified batch-one")
        tick = int(np.rint(elapsed[0] / self.spec.config.control_dt))
        previous_tick = int(self.last_tick[0])
        if tick < previous_tick or (previous_tick >= 0 and tick > previous_tick + 1):
            raise ValueError("observe every aperture control tick")
        fresh = tick != previous_tick
        phase = int(self.phase[0])
        target = self.waypoints[phase]
        tolerance = .006 if phase < 2 else .004
        if fresh and np.linalg.norm(position[0] - target) < tolerance and phase < 3:
            self.phase[0] += 1
            self.integral[:] = 0.
            phase = int(self.phase[0])
            target = self.waypoints[phase]
        angular = orientation_error(np, rotation[0], goal_rotation[0])
        issued_angular = orientation_error(np, reference_rotation[0], goal_rotation[0])
        error = np.r_[target - position[0], angular]
        if fresh:
            self.integral = np.clip(
                self.integral + .8 * self.spec.config.control_dt * error,
                -.04, .04,
            )
        speed = (.08, .08, .025, .009)[phase]
        command = np.zeros((1, 13), dtype=np.float32)
        command[0, :3] = np.clip(
            4. * (target - reference_position[0])
            + 3. * (target - position[0]) + self.integral[0, :3],
            -speed, speed,
        )
        command[0, 3:6] = np.clip(
            3. * issued_angular + 3. * angular + self.integral[0, 3:], -.6, .6,
        )
        command[0, 6:12] = 1.
        if fresh:
            self.raw = command
            self.last_tick[0] = tick
        return self.raw.copy()


class PublicSequentialAperturePolicy:
    """Known-geometry source skill for two orthogonal framed apertures."""

    VERSION = SEQUENTIAL_VERSION

    def __init__(self, spec, *, mode=0):
        if (spec.name != "sequential_aperture_reorientation"
                or spec.config.scene_profile
                != "sequential_aperture_reorientation_v1"):
            raise ValueError("sequential aperture task required")
        self.spec = spec
        self.mode = int(mode)
        if self.mode not in (0, 1):
            raise ValueError("sequential aperture source mode must be 0 or 1")
        cfg = spec.config
        first_rotation = aperture_rotation(0.)
        second_rotation = aperture_rotation(np.pi / 2)
        first_offset = first_rotation @ OBJECT_CENTER_TCP
        second_offset = second_rotation @ OBJECT_CENTER_TCP
        side = -1. if self.mode == 0 else 1.

        def tcp(object_center, offset):
            return np.asarray(object_center, dtype=float) - offset

        self.waypoints = np.asarray([
            tcp((.445, cfg.aperture_center_y + side * .006,
                 cfg.aperture_center_z), first_offset),
            tcp((cfg.aperture_plane_x - .07, cfg.aperture_center_y,
                 cfg.aperture_center_z), first_offset),
            tcp((.61, cfg.aperture_center_y,
                 cfg.aperture_center_z), first_offset),
            tcp((.61, -.10, cfg.aperture_center_z), second_offset),
            tcp((cfg.second_aperture_center_x, -.10,
                 cfg.second_aperture_center_z), second_offset),
            tcp((cfg.second_aperture_center_x,
                 cfg.sequential_exit_zone_y[0] + .010,
                 cfg.second_aperture_center_z), second_offset),
            np.asarray(cfg.goal_position, dtype=float),
        ])
        self.rotations = np.asarray(
            [first_rotation] * 3 + [second_rotation] * 4,
            dtype=float,
        )
        self.reset()

    def reset(self, *, modes=None):
        if modes is not None:
            values = np.asarray(modes, dtype=int).reshape(-1)
            if values.shape != (1,) or int(values[0]) != self.mode:
                raise ValueError("sequential aperture base is frozen to one mode")
        self.modes = np.asarray([self.mode], dtype=np.int64)
        self.phase = np.zeros(1, dtype=np.int64)
        self.path_distance = np.zeros(1, dtype=float)
        self.last_tick = np.full(1, -1, dtype=np.int64)
        self.integral = np.zeros((1, 6), dtype=float)
        self.raw = np.zeros((1, 13), dtype=np.float32)

    def reference(self):
        return self.waypoints[self.phase].copy()

    def get_state(self):
        return deepcopy({
            "version": self.VERSION, "mode": self.mode,
            "phase": self.phase, "path_distance": self.path_distance,
            "last_tick": self.last_tick, "integral": self.integral,
            "raw": self.raw,
        })

    def restore(self, state):
        if (state.get("version") != self.VERSION
                or int(state.get("mode", -1)) != self.mode):
            raise ValueError("sequential aperture public binding differs")
        for name in ("phase", "path_distance", "last_tick", "integral", "raw"):
            setattr(self, name, np.asarray(state[name]).copy())

    def reorder(self, indices):
        indices = np.asarray(indices, dtype=int)
        if indices.shape != (1,) or int(indices[0]) != 0:
            raise ValueError("batch-one sequential aperture actor required")

    def act(self, public):
        position = np.asarray(public["tcp_position"], dtype=float)
        rotation = np.asarray(public["tcp_rotation"], dtype=float)
        reference_position = np.asarray(public["reference_position"], dtype=float)
        reference_rotation = np.asarray(public["reference_rotation"], dtype=float)
        elapsed = np.asarray(public["elapsed_seconds"], dtype=float).reshape(-1)
        if position.shape != (1, 3):
            raise ValueError("sequential aperture policy is qualified batch-one")
        tick = int(np.rint(elapsed[0] / self.spec.config.control_dt))
        previous_tick = int(self.last_tick[0])
        if tick < previous_tick or (previous_tick >= 0 and tick > previous_tick + 1):
            raise ValueError("observe every sequential aperture control tick")
        fresh = tick != previous_tick
        phase = int(self.phase[0])
        target, target_rotation = self.waypoints[phase], self.rotations[phase]
        angle = np.linalg.norm(orientation_error(np, rotation[0], target_rotation))
        tolerance = .007 if phase < 5 else (.009 if phase == 5 else .004)
        braking_ready = True
        if phase == 5:
            object_twist = np.asarray(
                public["carried_object_twist"], dtype=float,
            )
            braking_ready = bool(
                np.linalg.norm(object_twist[0, :3])
                <= self.spec.config.aperture_brake_speed
            )
        if (fresh and phase < len(self.waypoints) - 1
                and np.linalg.norm(position[0] - target) < tolerance
                and angle < np.deg2rad(4.) and braking_ready):
            self.phase[0] += 1
            self.integral[:] = 0.
            phase = int(self.phase[0])
            target, target_rotation = self.waypoints[phase], self.rotations[phase]
        angular = orientation_error(np, rotation[0], target_rotation)
        issued_angular = orientation_error(np, reference_rotation[0], target_rotation)
        error = np.r_[target - position[0], angular]
        if fresh:
            self.integral[0] = np.clip(
                self.integral[0] + .8 * self.spec.config.control_dt * error,
                -.04, .04,
            )
        speed = (.075, .065, .045, .025, .035, .022, .015)[phase]
        command = np.zeros((1, 13), dtype=np.float32)
        command[0, :3] = np.clip(
            4. * (target - reference_position[0])
            + 3. * (target - position[0]) + self.integral[0, :3],
            -speed, speed,
        )
        command[0, 3:6] = np.clip(
            3. * issued_angular + 3. * angular + self.integral[0, 3:],
            -.45, .45,
        )
        command[0, 6:12] = 1.
        if fresh:
            self.raw = command
            self.last_tick[0] = tick
        return self.raw.copy()


class PublicSequentialLegacyPhaseAdapter:
    """Expose the seven-stage task planner through a frozen four-phase bank."""

    VERSION = "public-sequential-aperture-four-phase-adapter-v1"
    PHASE_MAP = np.asarray([0, 1, 2, 2, 2, 3, 3], dtype=np.int64)

    def __init__(self, policy):
        if not isinstance(policy, PublicSequentialAperturePolicy):
            raise TypeError("sequential aperture policy required")
        self.policy = policy
        self._task_public = {}
        self._sync()

    def _sync(self):
        self.phase = self.PHASE_MAP[np.asarray(self.policy.phase, dtype=int)]
        self.modes = self.policy.modes
        self.path_distance = self.policy.path_distance
        self.raw = self.policy.raw

    def act(self, public):
        value = self.policy.act({**public, **self._task_public})
        self._sync()
        return value

    def reset(self, **kwargs):
        self._task_public = {}
        value = self.policy.reset(**kwargs)
        self._sync()
        return value

    def set_task_public(self, public):
        self._task_public = {
            "carried_object_twist": np.asarray(
                public["carried_object_twist"], dtype=np.float32,
            ).copy(),
        }

    def reference(self):
        return self.policy.reference()

    def reorder(self, indices):
        value = self.policy.reorder(indices)
        self._sync()
        return value

    def get_state(self):
        return {
            "version": self.VERSION,
            "mapped_phase": self.phase.copy(),
            "policy": self.policy.get_state(),
        }

    def __getattr__(self, name):
        return getattr(self.policy, name)


def bind_aperture_public_planner(actor, spec, *, mode):
    """Install the semantic public planner under an existing T1 method actor.

    Learned response/transport components remain unchanged.  ASID and
    AdaptSim retain their learned public speed scale, while their route points
    are replaced because aperture geometry is a new, equally public task
    input.  This is a declared task adaptation for development pilots, not a
    claim that the original training covered the new geometry.
    """
    if spec.name not in {
            "aperture_brake", "sequential_aperture_reorientation"}:
        raise ValueError("semantic aperture task required")
    if not hasattr(actor, "base"):
        raise ValueError("T1 method actor has no task-owned public planner")
    if spec.name == "aperture_brake":
        actor.base = PublicApertureBrakePolicy(spec, mode=mode)
    else:
        actor.base = PublicSequentialLegacyPhaseAdapter(
            PublicSequentialAperturePolicy(spec, mode=mode)
        )
    return PublicApertureMethodExecution(actor)


class PublicApertureMethodExecution:
    """Expose new task fields while preserving the frozen response chart."""

    VERSION = "public-aperture-method-execution-v1"

    def __init__(self, actor):
        self.actor = actor
        self.base = actor.base

    def act(self, public):
        from genedynamics.solvers.single.cross_ebdm.learning.response import (
            PUBLIC_FIELDS,
        )
        # Keep the response model's exact sensor schema.  The task-owned
        # planner receives its additional public braking signal out of band;
        # it is never added to the frozen response feature vector.
        if hasattr(self.base, "set_task_public"):
            self.base.set_task_public(public)
        return self.actor.act({key: public[key] for key in PUBLIC_FIELDS})

    def reset(self):
        return self.actor.reset()

    def get_state(self):
        return {
            "version": self.VERSION,
            "actor": self.actor.get_state() if hasattr(self.actor, "get_state") else None,
        }


__all__ = [
    "OBJECT_CENTER_TCP", "OBJECT_VERTICES_TCP", "PublicApertureBrakePolicy",
    "PublicSequentialAperturePolicy", "SEQUENTIAL_VERSION", "aperture_rotation",
    "PublicSequentialLegacyPhaseAdapter",
    "fixed_orientation_fits_both_apertures",
    "PublicApertureMethodExecution",
    "VERSION", "bind_aperture_public_planner", "object_pose_and_twist",
    "object_vertices",
]

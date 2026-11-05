"""Task-owned geometry and public controller for semantic aperture braking."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from genedynamics.core.control.transfer import orientation_error


VERSION = "public-aperture-brake-v3"


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

    def reset(self):
        self.phase = 0
        self.last_tick = -1
        self.integral = np.zeros(6, dtype=float)
        self.raw = np.zeros((1, 13), dtype=np.float32)

    def get_state(self):
        return deepcopy({
            "version": self.VERSION, "mode": self.mode,
            "phase": self.phase, "last_tick": self.last_tick,
            "integral": self.integral, "raw": self.raw,
        })

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
        if tick < self.last_tick or (self.last_tick >= 0 and tick > self.last_tick + 1):
            raise ValueError("observe every aperture control tick")
        fresh = tick != self.last_tick
        target = self.waypoints[self.phase]
        tolerance = .006 if self.phase < 2 else .004
        if fresh and np.linalg.norm(position[0] - target) < tolerance and self.phase < 3:
            self.phase += 1
            self.integral[:] = 0.
            target = self.waypoints[self.phase]
        angular = orientation_error(np, rotation[0], goal_rotation[0])
        issued_angular = orientation_error(np, reference_rotation[0], goal_rotation[0])
        error = np.r_[target - position[0], angular]
        if fresh:
            self.integral = np.clip(
                self.integral + .8 * self.spec.config.control_dt * error,
                -.04, .04,
            )
        speed = (.08, .08, .025, .009)[self.phase]
        command = np.zeros((1, 13), dtype=np.float32)
        command[0, :3] = np.clip(
            4. * (target - reference_position[0])
            + 3. * (target - position[0]) + self.integral[:3],
            -speed, speed,
        )
        command[0, 3:6] = np.clip(
            3. * issued_angular + 3. * angular + self.integral[3:], -.6, .6,
        )
        command[0, 6:12] = 1.
        if fresh:
            self.raw = command
            self.last_tick = tick
        return self.raw.copy()


__all__ = [
    "OBJECT_CENTER_TCP", "OBJECT_VERTICES_TCP", "PublicApertureBrakePolicy",
    "VERSION", "object_pose_and_twist", "object_vertices",
]

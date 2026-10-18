"""Task-level demonstrators shared by data collection and baseline training."""

from __future__ import annotations

import numpy as np


class PublicTeacher:
    """Public-observation teacher for the four transfer task interfaces."""

    def __init__(self, task, *, mode=0, recipe="default"):
        self.task, self.mode, self.recipe = task, int(mode), recipe
        self.waypoint, self.integral, self.commands = 0, np.zeros(6), []
        if task.spec.alias == "T1":
            if mode == 1:
                self.route = [
                    np.array([.38, -.16, .55]),
                    np.array([.66, -.16, .55]),
                    np.array([.66, .08, .55]),
                    np.asarray(task.config.goal_position),
                ]
            else:
                self.route = [
                    np.array([.38, -.12, .70]),
                    np.array([.66, -.12, .70]),
                    np.array([.66, .08, .70]),
                    np.asarray(task.config.goal_position),
                ]
            if mode == 0 and recipe in {"upper_compact", "upper_clearance"}:
                height = .65 if recipe == "upper_compact" else .67
                self.route = [
                    np.array([.38, -.12, height]),
                    np.array([.61, -.12, height]),
                    np.array([.64, .08, .58]),
                    np.asarray(task.config.goal_position),
                ]
        else:
            from .behavior import behavior_reset

            self.memory = behavior_reset(1, [mode])

    def act(self, observation):
        from genedynamics.core.control.transfer import orientation_error

        if self.task.spec.alias == "T1":
            p, r = observation["tcp_position"][0], observation["tcp_rotation"][0]
            refp = observation["reference_position"][0]
            refr = observation["reference_rotation"][0]
            target = self.route[self.waypoint]
            if np.linalg.norm(p - target) < .012 and self.waypoint < len(self.route) - 1:
                self.waypoint += 1
                self.integral[:] = 0.
                target = self.route[self.waypoint]
            target_r = observation["goal_rotation"][0]
            error = np.r_[target - p, orientation_error(np, r, target_r)]
            self.integral = np.clip(
                self.integral + .8 * self.task.config.control_dt * error,
                -.04,
                .04,
            )
            action = np.zeros(13, dtype=np.float32)
            action[:3] = np.clip(
                4. * (target - refp) + 3. * (target - p) + self.integral[:3],
                -.12,
                .12,
            )
            action[3:6] = np.clip(
                3. * orientation_error(np, refr, target_r)
                + 3. * orientation_error(np, r, target_r)
                + self.integral[3:],
                -.6,
                .6,
            )
            action[6:12] = .5 if self.recipe == "stiffness05" else 1.
            return action
        if not self.commands:
            from .behavior import (
                advance_behavior,
                keyway_alignment,
                reference_commands,
                teacher_reference,
            )

            cfg = self.task.config
            aligned = keyway_alignment(
                observation["tcp_position"],
                observation["tcp_rotation"],
                cfg.mating_clearance,
            )
            self.memory = advance_behavior(
                self.memory,
                observation["tcp_position"],
                observation["tcp_rotation"],
                observation["measured_wrench"][:, 2],
                dt=10 * cfg.control_dt,
                insertion=True,
                goal_z=cfg.goal[2],
                alignment_mask=aligned,
            )
            reference = teacher_reference(
                self.memory, insertion=True, goal_z=cfg.goal[2]
            )
            commands = reference_commands(
                reference,
                observation["reference_position"],
                observation["reference_rotation"],
                observation["tcp_position"],
                observation["tcp_rotation"],
                integral_bias=self.memory.tracking_integral,
                dt=cfg.control_dt,
            )
            if self.recipe == "recovery_stiffness":
                if self.memory.phase[0] <= 1:
                    commands[..., 6:12] += [.25, .25, 0., .25, .25, .25]
                elif self.memory.phase[0] >= 2:
                    commands[..., 6:12] += [1., 1., 0., 1., 1., 1.]
                    if self.memory.phase[0] == 3:
                        commands[..., 12] *= 1.5
            self.commands = list(commands[0])
        return self.commands.pop(0)


__all__ = ["PublicTeacher"]

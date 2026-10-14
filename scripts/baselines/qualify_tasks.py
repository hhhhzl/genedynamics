"""Qualify source task interfaces with public-feedback teachers and CPU replay.

These are development teachers and successful source demonstrations, not
trained external baselines or final benchmark evidence. Every attempt and its
actual target-free source cost is retained. Default task deadlines are intact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from genedynamics.experiments.baselines import instantiate_task

ROOT = Path(__file__).resolve().parents[2]


class PublicTeacher:
    def __init__(self, task, *, mode=0, recipe="default"):
        self.task, self.mode, self.recipe = task, int(mode), recipe
        self.waypoint, self.integral, self.commands = 0, np.zeros(6), []
        if task.spec.alias == "T1":
            self.route = ([np.array([.38, -.16, .55]), np.array([.66, -.16, .55]),
                           np.array([.66, .08, .55]), np.asarray(task.config.goal_position)]
                          if mode == 1 else [np.array([.38, -.12, .70]), np.array([.66, -.12, .70]),
                                            np.array([.66, .08, .70]), np.asarray(task.config.goal_position)])
            if mode == 0 and recipe == "upper_compact":
                self.route = [np.array([.38, -.12, .65]), np.array([.61, -.12, .65]),
                              np.array([.64, .08, .58]), np.asarray(task.config.goal_position)]
            if mode == 0 and recipe == "upper_clearance":
                self.route = [np.array([.38, -.12, .67]), np.array([.61, -.12, .67]),
                              np.array([.64, .08, .58]), np.asarray(task.config.goal_position)]
        else:
            from genedynamics.learning.cross_embodiment.contact_behavior import behavior_reset
            self.memory = behavior_reset(1, [mode])

    def act(self, observation):
        from genedynamics.learning.cross_embodiment.core.control_math import orientation_error

        if self.task.spec.alias == "T1":
            p, r = observation["tcp_position"][0], observation["tcp_rotation"][0]
            refp, refr = observation["reference_position"][0], observation["reference_rotation"][0]
            target = self.route[self.waypoint]
            if np.linalg.norm(p - target) < .012 and self.waypoint < len(self.route) - 1:
                self.waypoint += 1; self.integral[:] = 0.; target = self.route[self.waypoint]
            target_r = observation["goal_rotation"][0]
            error = np.r_[target - p, orientation_error(np, r, target_r)]
            self.integral = np.clip(self.integral + .8 * self.task.config.control_dt * error, -.04, .04)
            action = np.zeros(13, dtype=np.float32)
            action[:3] = np.clip(4. * (target - refp) + 3. * (target - p) + self.integral[:3], -.12, .12)
            action[3:6] = np.clip(3. * orientation_error(np, refr, target_r)
                                        + 3. * orientation_error(np, r, target_r) + self.integral[3:], -.6, .6)
            action[6:12] = .5 if self.recipe == "stiffness05" else 1.
            return action
        if not self.commands:
            from genedynamics.learning.cross_embodiment.contact_behavior import advance_behavior, keyway_alignment, reference_commands, teacher_reference
            cfg = self.task.config
            aligned = keyway_alignment(observation["tcp_position"], observation["tcp_rotation"], cfg.mating_clearance)
            self.memory = advance_behavior(self.memory, observation["tcp_position"], observation["tcp_rotation"],
                                           observation["measured_wrench"][:, 2], dt=10 * cfg.control_dt,
                                           insertion=True, goal_z=cfg.goal[2], alignment_mask=aligned)
            reference = teacher_reference(self.memory, insertion=True, goal_z=cfg.goal[2])
            commands = reference_commands(reference, observation["reference_position"], observation["reference_rotation"],
                                          observation["tcp_position"], observation["tcp_rotation"],
                                          integral_bias=self.memory.tracking_integral, dt=cfg.control_dt)
            if self.recipe == "recovery_stiffness":
                if self.memory.phase[0] <= 1:
                    commands[..., 6:12] += [.25, .25, 0., .25, .25, .25]
                elif self.memory.phase[0] >= 2:
                    commands[..., 6:12] += [1., 1., 0., 1., 1., 1.]
                    if self.memory.phase[0] == 3:
                        commands[..., 12] *= 1.5
            self.commands = list(commands[0])
        return self.commands.pop(0)


def qualify_source(task_name, *, seed, mode=0, recipe="default", output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    task = instantiate_task(task_name, robot="panda", seed=seed)
    contract = dict(task=task.describe(), seed=int(seed), mode=int(mode), recipe=recipe,
                    role="source interface development; public-feedback teacher; not a trained paper baseline")
    source_paths = [Path(__file__), ROOT / "genedynamics/experiments/baselines/contact_task.py",
                    ROOT / "genedynamics/experiments/baselines/acp_execution.py", ROOT / "genedynamics/experiments/baselines/evaluation.py",
                    ROOT / "genedynamics/experiments/baselines/contracts.py", ROOT / "genedynamics/experiments/baselines/task_specs.py"]
    contract["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    for path in source_paths:
        target = output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (output / "pending.json").write_text(json.dumps(contract, indent=2, default=str) + "\n")
    started = time.monotonic()
    task.reset((seed,), seed=seed)
    teacher = PublicTeacher(task, mode=mode, recipe=recipe)
    observations, actions, next_observations, rewards = [], [], [], []
    public = {key: [value[0].copy()] for key, value in task.observe().items()}
    while task.state.status[0] == 0:
        observation = task.observe()
        action = teacher.act(observation)
        observations.append(task.flat_observation(observation)[0])
        actions.append(action.copy())
        result = task.step(action)
        next_observations.append(task.flat_observation(result.observation)[0])
        rewards.append(result.reward[0])
        for key, value in result.observation.items():
            public[key].append(value[0].copy())
    record = task.records[0]
    report = task.evaluate()
    replay = task.replay_episode(record)
    arrays = dict(observations=np.asarray(observations), actions=np.asarray(actions),
                  next_observations=np.asarray(next_observations), rewards=np.asarray(rewards),
                  **{"public." + key: np.asarray(value) for key, value in public.items()},
                  **{"trace." + key: np.asarray([row[key] for row in record.traces]) for key in record.traces[0]})
    np.savez_compressed(output / "episode.npz", **arrays)
    if task.spec.alias == "T1":
        from genedynamics.learning.cross_embodiment.data import observed_route_modes
        observed_mode = int(observed_route_modes(arrays["public.tcp_position"][1:][None, :, None],
                                                arrays["trace.valid"][None, :, None])[0])
    else:
        from genedynamics.learning.cross_embodiment.contact_reference import classify_contact_modes
        observed_mode = int(classify_contact_modes(arrays["public.tcp_position"][1:][None],
                                                   arrays["public.measured_wrench"][1:, 2][None],
                                                   arrays["trace.valid"][None], insertion=True)[0])
    digest = hashlib.sha256((output / "episode.npz").read_bytes()).hexdigest()
    summary = dict(contract=contract, report=report, replay=replay, episode_sha256=digest,
                   elapsed_wall_seconds=time.monotonic() - started,
                   observed_mode=observed_mode, requested_mode=int(mode), mode_match=observed_mode == mode,
                   valid_source_demo=bool(report["successes"] == 1 and report["status_mismatches"] == 0
                                          and replay["matches_status"] and replay["max_pose_error"] == 0. and replay["trace_exact"]),
                   valid_source_mode_demo=bool(report["successes"] == 1 and observed_mode == mode),
                   source_interaction_accounting={"collection": task.accounting, "audit_replay": replay["accounting"]})
    (output / "report.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("T1", "T3"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--recipe", choices=("default", "stiffness05", "recovery_stiffness", "upper_compact", "upper_clearance"), default="default")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == ROOT / "latex" or ROOT / "latex" in args.output.resolve().parents:
        parser.error("source qualification artifacts belong outside latex/")
    value = qualify_source(args.task, seed=args.seed, mode=args.mode, recipe=args.recipe, output=args.output)
    print(json.dumps({"valid_source_demo": value["valid_source_demo"], "report": value["report"],
                      "replay": value["replay"], "path": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

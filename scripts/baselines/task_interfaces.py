"""Run complete default-deadline physical interface qualification from configs.

Public geometric teachers diagnose task/controller support. They are never
reported as trained paper baselines; every attempt and CPU replay is charged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from genedynamics.experiments.baselines import instantiate_task
from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.experiments.artifacts import atomic_json
from scripts.baselines.qualify_tasks import PublicTeacher


def episode_json(record):
    """Persist all native NumPy trace arrays without dropping audit fields."""
    def numpy_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported episode value: {type(value).__name__}")
    return json.loads(json.dumps(asdict(record), default=numpy_value, allow_nan=False))


class GuideTeacher:
    def __init__(self, task, mode, *, speed_scale=1.):
        from genedynamics.learning.cross_embodiment.contact_behavior import behavior_reset
        self.task, self.memory, self.commands = task, behavior_reset(1, [mode]), []
        if not 0. < speed_scale <= 1.:
            raise ValueError("teacher speed scale must be in (0,1]")
        self.speed_scale = float(speed_scale)

    def act(self, observation):
        from genedynamics.learning.cross_embodiment.contact_behavior import (
            advance_behavior, reference_commands, teacher_reference,
        )
        if not self.commands:
            cfg = self.task.config
            position = observation["tcp_position"]
            # Existing spatial-guide teacher's public outlet event. The y
            # channel must reach its outlet before starting the x motion.
            aligned = np.where(self.memory.mode == 0, position[:, 0] > .455, position[:, 1] > -.181)
            self.memory = advance_behavior(self.memory, observation["tcp_position"], observation["tcp_rotation"],
                    observation["measured_wrench"][:, 2], insertion=False, dt=10 * cfg.control_dt, goal_z=cfg.goal[2],
                    alignment_mask=aligned)
            capture = (self.memory.phase == 3) & (np.max(np.abs(position[:, :2] - np.asarray(cfg.goal)[:2]), axis=-1) < .003)
            integral = self.memory.tracking_integral.copy(); integral[capture, :2] = 0.
            self.memory = self.memory._replace(tracking_integral=integral)
            reference = teacher_reference(self.memory, insertion=False, goal_z=cfg.goal[2])
            commands = reference_commands(reference, observation["reference_position"], observation["reference_rotation"],
                    observation["tcp_position"], observation["tcp_rotation"], integral_bias=self.memory.tracking_integral,
                    dt=cfg.control_dt)
            commands[..., :2] *= self.speed_scale
            self.commands = list(commands[0])
        return self.commands.pop(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, default=Path("results/_development/p3_20261014"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.use_development_output_root(args.development_root)
    robot = cfg.env_params["robot"]
    output = cfg.output_dir / f"level_{robot}" / f"seed_{args.seed}"
    from scripts.baselines import require_non_paper_output
    output = require_non_paper_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "attempt.json").exists():
        raise RuntimeError("charged interface attempt exists; choose a new seed")
    atomic_json(output / "attempt.json", {"status": "running", "scope": "physical task interface development", "seed": args.seed})
    task = instantiate_task(cfg.env_name, robot=robot, seed=args.seed)
    atomic_json(output / "contract.json", task.describe())
    (output / "config_snapshot.yaml").write_text(args.config.read_text())
    start = time.monotonic()
    try:
        task.reset((args.seed,), seed=args.seed, initial=cfg.env_params.get("initial"))
        teacher = GuideTeacher(task, args.mode, speed_scale=cfg.method_params.get("teacher_speed_scale", 1.)) if task.spec.alias == "T2" else PublicTeacher(task, mode=args.mode, recipe="recovery_stiffness")
        while task.state.status[0] == 0:
            task.step(teacher.act(task.observe()))
        evaluation = task.evaluate()
        records = task.records
        row = records[-1]
        atomic_json(output / "episode_record.json", episode_json(row))
        arrays = {key: np.asarray([trace[key] for trace in row.traces]) for key in row.traces[0]}
        arrays["requested_actions"] = np.asarray(row.requested_actions)
        np.savez_compressed(output / "episode.npz", **arrays)
        replay = task.replay_episode(row)
        success = evaluation["successes"] == 1
        report = {"status": "physical_success_and_replay_qualified" if success and replay["trace_exact"] else "capability_not_qualified",
                  "task": task.describe(), "evaluation": evaluation, "replay": replay,
                  "mode_requested": args.mode, "wall_seconds": time.monotonic() - start,
                  "initial": cfg.env_params.get("initial"),
                  "scope": "public teacher/default physical deadline; not native baseline or trained benchmark performance"}
        atomic_json(output / "qualification.json", report)
        atomic_json(output / "attempt.json", {"status": "complete", "qualification": report["status"],
                    "lifetime_accounting": task.lifetime_accounting, "audit_replay_accounting": replay["accounting"]})
        print(json.dumps({"status": report["status"], "evaluation": evaluation}, indent=2))
    except BaseException as error:
        atomic_json(output / "attempt.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                    "lifetime_accounting": task.lifetime_accounting, "evaluation": task.evaluate()})
        raise


if __name__ == "__main__":
    main()

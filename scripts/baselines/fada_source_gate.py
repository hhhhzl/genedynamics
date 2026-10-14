"""Score both modes of a committed FADA task source student in real Panda physics.

This is a source stability gate, with two full episodes and charged exact CPU
replay. It performs no training, target execution or checkpoint selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from genedynamics.experiments.artifacts import atomic_json
from genedynamics.experiments.baselines import instantiate_task
from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.learning.baselines.fada.task_training import PORTABLE_FIELDS, bounded_task_action, portable_history
from genedynamics.learning.baselines.fada.source_training import SourceTaskPolicy
from genedynamics.solvers.single.fada import FADASolver
from scripts.baselines import require_non_paper_output


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Direct source-only stage checkpoint, preserving its stored action decoder")
    parser.add_argument("--task", choices=("T1", "T3"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.checkpoint:
        if not args.task or not args.output: parser.error("direct checkpoint gate requires --task and --output")
        checkpoint_path = args.checkpoint.resolve()
        origin, task_name, output = checkpoint_path, args.task, args.output.resolve()
    else:
        if not args.qualification: parser.error("provide --qualification or --checkpoint")
        origin = args.qualification.resolve()
        output = require_non_paper_output(origin.parent / "source_closed_loop_gate")
        context = json.loads((origin.parent / "source-context.json").read_text())
        cfg = ExperimentConfig.from_yaml(context["argv"]["config"])
        checkpoint_path, task_name = origin.parent / "source_policy.pt", cfg.env_name
    output = require_non_paper_output(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["cfg"]["obs_dim"] != 35 or tuple(checkpoint["genedynamics_task_adapter"]["portable_fields"]) != PORTABLE_FIELDS:
        raise ValueError("source gate requires the audited35D measured physical state chart")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), root / "genedynamics/learning/baselines/fada/task_training.py",
             root / "genedynamics/learning/baselines/fada/source_training.py",
             root / "genedynamics/experiments/baselines/contact_task.py", root / "genedynamics/experiments/baselines/evaluation.py"]
    provenance = {"status": "running", "seed": args.seed, "target_calls": 0,
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "parent_qualification_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        "code_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    atomic_json(output / "provenance.json", provenance)
    atomic_json(output / "attempt.json", {"status": "running", "seed": args.seed, "target_calls": 0})
    task = instantiate_task(task_name, robot="panda", seed=args.seed, history_len=checkpoint["cfg"]["history_len"])
    policy = SourceTaskPolicy.from_checkpoint(checkpoint, task=task, device=args.device)
    public = [task.reset((args.seed, args.seed + 1), seed=args.seed)]
    task.set_modes([0, 1]); policy.reset()
    for _ in range(task.config.timeout_steps):
        result = task.step(policy.act(portable_history(task)))
        public.append(result.observation)
        if result.done.all(): break
    evaluation = task.evaluate()
    replays = [task.replay_episode(record) for record in task.records]
    arrays, observed = {}, []
    for index, record in enumerate(task.records):
        length = len(record.traces)
        arrays[f"episode_{index}.actions"] = np.asarray(record.requested_actions)
        for name in record.traces[0]:
            arrays[f"episode_{index}.trace.{name}"] = np.asarray([row[name] for row in record.traces])
        for name in public[0]:
            arrays[f"episode_{index}.public.{name}"] = np.asarray([row[name][index] for row in public[:length + 1]])
        positions = arrays[f"episode_{index}.public.tcp_position"][1:][None]
        valid = arrays[f"episode_{index}.trace.valid"][None]
        if task.spec.alias == "T1":
            from genedynamics.learning.cross_embodiment.data import observed_route_modes
            mode = int(observed_route_modes(positions[:, :, None], valid[:, :, None])[0])
        else:
            from genedynamics.learning.cross_embodiment.contact_reference import classify_contact_modes
            force = arrays[f"episode_{index}.public.measured_wrench"][1:, 2][None]
            mode = int(classify_contact_modes(positions, force, valid, insertion=True)[0])
        observed.append(mode)
    np.savez_compressed(output / "episodes.npz", **arrays)
    (output / "episode_records.json").write_text(json.dumps([asdict(row) for row in task.records], default=_json, indent=2) + "\n")
    passed = (evaluation["successes"] == 2 and evaluation["status_mismatches"] == 0 and observed == [0, 1]
              and all(row["matches_status"] and row["trace_exact"] for row in replays))
    summary = {"status": "source_stability_passed" if passed else "source_performance_pending",
        "scope": "two full Panda nominal-source episodes; no target queries, no training or checkpoint selection",
        "source_qualification": str(origin), "task_contract": task.describe(), "source_model_config": checkpoint["cfg"],
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "action_decoder": checkpoint["genedynamics_task_adapter"].get("action_decoder", {"kind": "continuous13D"}),
        "evaluation": evaluation, "observed_modes": observed, "requested_modes": [0, 1], "replays": replays,
        "source_interaction_accounting": {"collection": task.lifetime_accounting, "audit_replays": [row["accounting"] for row in replays]},
        "target_calls": 0, "source_stability_qualified": passed,
        "episodes_sha256": hashlib.sha256((output / "episodes.npz").read_bytes()).hexdigest()}
    atomic_json(output / "qualification.json", json.loads(json.dumps(summary, default=_json)))
    atomic_json(output / "attempt.json", {"status": "complete", "seed": args.seed, "source_stability_qualified": passed, "target_calls": 0})
    print(json.dumps({"status": summary["status"], "successes": evaluation["successes"], "observed_modes": observed}))


if __name__ == "__main__":
    main()

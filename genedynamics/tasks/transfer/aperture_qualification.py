"""Task-owned CPU source qualification for semantic aperture braking."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

from genedynamics.envs.adapters.transfer_task import instantiate_task
from .aperture import PublicApertureBrakePolicy
from .episode import audit_episode_evidence, episode_evidence
from .runtime import MechanismSpec


def qualify_aperture(config_path, *, progress=None):
    config_path = Path(config_path)
    plan = json.loads(config_path.read_text())
    if plan.get("device") != "cpu" or plan.get("task") != "aperture_brake":
        raise ValueError("explicit semantic CPU aperture plan required")
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    started = time.monotonic()
    rows = []
    mechanism = MechanismSpec(**plan["mechanism"])
    for request in plan["episodes"]:
        seed, mode = int(request["seed"]), int(request["mode"])
        task = instantiate_task(
            plan["task"], robot=plan["robot"], mechanism_spec=mechanism,
            seed=seed, config_overrides={
                "start_position": tuple(request["start_position"]),
            },
        )
        actor = PublicApertureBrakePolicy(task.spec, mode=mode)
        public = task.reset((seed,), seed=seed, modes=[mode])
        for _ in range(task.spec.config.timeout_steps):
            if task.state.status[0] != 0:
                break
            public = task.step(actor.act(public)).observation
        evaluation = task.evaluate()
        destination = output / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=False)
        evidence_path = destination / "evidence.json.gz"
        evidence_path.write_bytes(gzip.compress(json.dumps(
            episode_evidence(task.records, task.spec, evaluation),
            sort_keys=True, allow_nan=False,
        ).encode(), mtime=0))
        audit = audit_episode_evidence(evidence_path)
        row = {
            **request,
            "evidence_path": str(evidence_path),
            "evidence_audit_passed": audit["passed"],
            "evaluation": evaluation,
            "actor_state": actor.get_state(),
            "scene": task.runner.describe(),
        }
        rows.append(row)
        (output / "progress.json").write_text(json.dumps({
            "completed": len(rows), "episodes": rows,
        }, indent=2, default=str) + "\n")
        if progress is not None:
            progress({"seed": seed, "outcomes": evaluation["outcomes"]})
    report = {
        "version": plan["version"],
        "plan_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(Path(plan["contract"]).read_bytes()).hexdigest(),
        "episodes": rows, "device": "cpu", "gpu_calls": 0,
        "passed": all(
            row["evidence_audit_passed"]
            and row["evaluation"]["status_mismatches"] == 0
            and row["evaluation"]["successes"] == row["evaluation"]["requested_episodes"]
            for row in rows
        ),
        "formal_performance_evidence": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return {"passed": report["passed"], "report": str(output / "report.json")}


__all__ = ["qualify_aperture"]

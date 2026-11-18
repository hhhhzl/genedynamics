"""Fresh CPU source qualification for semantic reposition-then-dock."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

from genedynamics.envs.adapters.transfer_task import instantiate_task
from .episode import audit_episode_evidence, episode_evidence
from .reposition_dock import PublicRepositionThenDockPolicy
from .specs import mechanism_spec


def qualify_reposition_dock(config_path, *, progress=None):
    config_path = Path(config_path)
    plan = json.loads(config_path.read_text())
    if plan.get("device") != "cpu" or plan.get("task") != "reposition_then_dock":
        raise ValueError("explicit semantic CPU reposition plan required")
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    started = time.monotonic()
    rows = []
    mechanism = mechanism_spec(
        plan["task"], plan["mechanism"],
        body_support_parameters=plan.get("body_support_parameters"),
    )
    for request in plan["episodes"]:
        seed, mode = int(request["seed"]), int(request["mode"])
        task = instantiate_task(
            plan["task"], robot=plan["robot"], mechanism_spec=mechanism,
            seed=seed,
        )
        public = task.reset((seed,), seed=seed, modes=[mode], initial="jammed")
        actor = PublicRepositionThenDockPolicy(
            task.spec, mode=mode, seed=seed,
        )
        for _ in range(task.spec.config.timeout_steps):
            if task.state.status[0] != 0:
                break
            action, body = actor.act_with_body(public)
            public = task.step_contact(action, body_twist=body).observation
        evaluation = task.evaluate()
        destination = output / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=False)
        evidence_path = destination / "evidence.json.gz"
        evidence_path.write_bytes(gzip.compress(json.dumps(
            episode_evidence(task.records, task.spec, evaluation),
            sort_keys=True, allow_nan=False,
        ).encode(), mtime=0))
        audit = audit_episode_evidence(evidence_path)
        episode = evaluation["episodes"][-1]
        row = {
            **request,
            "evidence_path": str(evidence_path),
            "evidence_audit_passed": audit["passed"],
            "evaluation": evaluation,
            "actor_state": actor.get_state(),
            "scene": task.runner.describe(),
            "event_order_passed": bool(
                episode["success"]
                and all(value is not None for value in episode["event_control_steps"].values())
            ),
        }
        rows.append(row)
        (output / "progress.json").write_text(json.dumps({
            "completed": len(rows), "episodes": rows,
        }, indent=2, default=str) + "\n")
        if progress is not None:
            progress({"seed": seed, "mode": mode, "outcome": episode["outcome"]})
    report = {
        "version": plan["version"],
        "plan_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(Path(plan["contract"]).read_bytes()).hexdigest(),
        "episodes": rows, "device": "cpu", "gpu_calls": 0,
        "passed": all(
            row["evidence_audit_passed"] and row["event_order_passed"]
            and row["evaluation"]["status_mismatches"] == 0
            and row["evaluation"]["successes"]
            == row["evaluation"]["requested_episodes"]
            for row in rows
        ),
        "formal_performance_evidence": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "report.json").write_text(json.dumps(
        report, indent=2, default=str,
    ) + "\n")
    return {"passed": report["passed"], "report": str(output / "report.json")}


__all__ = ["qualify_reposition_dock"]

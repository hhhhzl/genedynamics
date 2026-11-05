"""Task-owned CPU qualification and public logs for the passive socket."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.compliant_socket import (
    CompliantSocketMechanismSpec, ENVIRONMENT_NAME, FixtureRelativePolicy,
)
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.tasks.transfer.episode import episode_evidence, audit_episode_evidence
from genedynamics.learning.datasets.transfer.target import (
    PublicTargetTrajectory, FIXTURE_VERSION, FIXTURE_SCHEMA,
)


def _public_contact_probe(action, public, tick, state, spec):
    """Apply one declared lateral pulse using only measured public wrench."""
    if spec is None:
        return action
    if (spec.get("version") != "public-contact-x-pulse-v1"
            or set(spec) != {"version", "wrench_trigger_N", "x_delta_mps", "positive_controls", "negative_controls"}
            or not 0.5 <= float(spec["wrench_trigger_N"]) <= 10.
            or not 0. < float(spec["x_delta_mps"]) <= .008
            or not 1 <= int(spec["positive_controls"]) <= 20
            or not 1 <= int(spec["negative_controls"]) <= 20):
        raise ValueError("bounded public contact-x probe contract required")
    if state["trigger_tick"] is None:
        wrench = np.asarray(public["measured_wrench"])
        if wrench.shape != (1, 6):
            raise ValueError("probe requires one measured public wrench")
        if abs(float(wrench[0, 2])) >= float(spec["wrench_trigger_N"]):
            state["trigger_tick"] = tick
    if state["trigger_tick"] is None:
        return action
    relative = tick - state["trigger_tick"]
    positive = int(spec["positive_controls"])
    negative = int(spec["negative_controls"])
    if relative < positive + negative:
        sign = 1. if relative < positive else -1.
        modified = action.copy()
        modified[0, 0] = np.clip(modified[0, 0] + sign * float(spec["x_delta_mps"]), -.02, .02)
        state["applied_controls"] += 1
        return modified
    return action


def qualify_compliant_socket(config_path, *, progress=None):
    """Execute the recipe and retain every requested episode and its evidence.

    Output identity and physical requests belong to the recipe. The optional
    callback exposes progress without selecting or replacing episodes.
    """
    config_path = Path(config_path); plan = json.loads(config_path.read_text())
    if plan["device"] != "cpu" or plan["environment"] != ENVIRONMENT_NAME:
        raise ValueError("explicit CPU passive-socket plan required")
    output = Path(plan["output_dir"]); output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    started = time.monotonic()
    mechanism = CompliantSocketMechanismSpec(**plan["mechanism"])
    ring = plan["ringdown"]
    ring_task = instantiate_task(ENVIRONMENT_NAME, robot=plan["robot"], mechanism_spec=mechanism,
        config_overrides={"fixture_initial_offset_m": ring["initial_offset_m"], "reset_task_noise": 0.})
    ring_task.reset((ring["seed"],), seed=ring["seed"], initial="clear")
    energies = []
    for _ in range(ring["controls"]):
        public = ring_task.observe()
        x = float(public["fixture_position"][0, 0] - .44)
        v = float(public["fixture_twist"][0, 0])
        energies.append(.5 * mechanism.fixture_stiffness_Npm * x*x + .5 * mechanism.fixture_mass_kg * v*v)
        if ring_task.state.status[0] != 0:
            break
        ring_task.step(np.zeros((1, 13), dtype=np.float32))
    ring_report = dict(energy_initial_J=energies[0], energy_final_J=energies[-1],
        final_energy_fraction=energies[-1]/energies[0], controls=ring_task.accounting["control_steps"],
        evaluation=ring_task.evaluate(), passed=bool(energies[-1]/energies[0] <= ring["maximum_final_energy_fraction"]))
    (output / "ringdown.json").write_text(json.dumps(ring_report, indent=2) + "\n")
    rows = []
    probe = plan.get("public_probe")
    for request in plan["episodes"]:
        seed, mode = request["seed"], request["mode"]
        task = instantiate_task(ENVIRONMENT_NAME, robot=plan["robot"], mechanism_spec=mechanism, seed=seed)
        actor = FixtureRelativePolicy(PublicLivenessMatingFeedbackPolicy(task.spec,
            feedback=PublicBiasFeedbackSpec(**plan["feedback"]), modes=[mode], seed=seed))
        public = task.reset((seed,), seed=seed, modes=[mode],
                            initial=request.get("initial","jammed")); observations = [public]; actions = []
        probe_state = {"trigger_tick": None, "applied_controls": 0}
        for tick in range(plan["max_controls_per_episode"]):
            if task.state.status[0] != 0:
                break
            action = actor.act(public).astype(np.float32)
            action = _public_contact_probe(action, public, tick, probe_state, probe)
            step = task.step(action); public = step.observation
            observations.append(public); actions.append(action[0])
        evaluation = task.evaluate(); destination = output / f"seed_{seed}"
        trajectory = PublicTargetTrajectory(dict(version=FIXTURE_VERSION, task="T3", environment=ENVIRONMENT_NAME,
            observation_schema=FIXTURE_SCHEMA, permission=plan["permission"], robot=plan["robot"], mode=mode,
            seed=seed, purpose=request["purpose"], chart="shared13", controls=len(actions),
            recording_end="terminal" if task.state.status[0] else "budget_stop"),
            {key: np.stack([row[key][0] for row in observations]) for key in public}, np.asarray(actions))
        trajectory.save(destination / "public")
        evidence_path = destination / "evidence.json.gz"
        evidence_path.write_bytes(gzip.compress(json.dumps(episode_evidence(task.records, task.spec, evaluation)).encode()))
        audit = audit_episode_evidence(evidence_path)
        row = dict(**request, public_path=str(destination / "public"), public_sha256=trajectory.digest,
                   evidence_path=str(evidence_path), evidence_audit_passed=audit["passed"], evaluation=evaluation)
        if probe is not None:
            row["public_probe"] = dict(version=probe["version"], trigger_tick=probe_state["trigger_tick"],
                                       applied_controls=probe_state["applied_controls"])
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(dict(completed=len(rows), episodes=rows), indent=2) + "\n")
        if progress is not None:
            progress({"seed": seed, "outcomes": evaluation["outcomes"],
                      "fixture_max_displacement_m": evaluation["episodes"][0]["fixture_max_displacement_m"]})
    report = dict(version=plan["version"], plan_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        contract_sha256=hashlib.sha256(Path(plan["contract"]).read_bytes()).hexdigest(),
        ringdown=ring_report, episodes=rows, device="cpu", gpu_calls=0,
        passed=bool(ring_report["passed"] and all(
            row["evidence_audit_passed"] and row["evaluation"]["status_mismatches"] == 0
            and row["evaluation"]["successes"]==row["evaluation"]["requested_episodes"]
            for row in rows)),
        formal_performance_evidence=False, elapsed_seconds=time.monotonic()-started)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"passed": report["passed"], "report": str(output / "report.json")}


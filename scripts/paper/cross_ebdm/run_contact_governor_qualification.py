#!/usr/bin/env python3
"""Run the frozen CPU qualification for the shared T2 contact governor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import yaml

from genedynamics.envs.adapters.transfer_task import CpuTaskAdapter
from genedynamics.tasks.transfer.contact import (
    ContactMechanismSpec, GOVERNOR_NOMINAL,
)
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.episode import controller_diagnostics, episode_evidence
from genedynamics.tasks.transfer.guide_evaluation import evaluate_strict_guide_episode
from genedynamics.tasks.transfer.specs import task_spec


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/arm/guided_contact_alignment/validation/finite_state_contact_qualification.yaml"
SOURCE_PATHS = (
    "genedynamics/tasks/transfer/contact.py",
    "genedynamics/tasks/transfer/controllers/guide_contact.py",
    "genedynamics/tasks/transfer/controllers/bindings.py",
    "genedynamics/envs/adapters/transfer_task.py",
    "genedynamics/envs/domains/manipulation/transfer_workcell/backends/contact.py",
    "genedynamics/envs/domains/manipulation/transfer_workcell/backends/jax/contact.py",
    "genedynamics/tasks/transfer/evaluation.py",
    "genedynamics/tasks/transfer/guide_evaluation.py",
    "scripts/paper/cross_ebdm/run_contact_governor_qualification.py",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"__nonfinite_float__": str(value)}
    return value


def _write_json(path: Path, value) -> str:
    payload = (json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_gzip_json(path: Path, value) -> str:
    payload = (json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
    ) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(rows, failures, expected, *, version, offline_recovery=False):
    contact_losses = [
        row["maximum_contact_loss_seconds"] for row in rows
        if row["maximum_contact_loss_seconds"] is not None
    ]
    return {
        "version": version,
        "status": "complete" if len(rows) == expected and not failures else "incomplete",
        "device": "cpu", "gpu_calls": 0,
        "offline_reporting_recovery": bool(offline_recovery),
        "episodes": len(rows), "expected_episodes": expected,
        "successes": sum(row["success"] for row in rows),
        "constraint_violations": sum(row["constraint_violation"] for row in rows),
        "contact_failures": sum(row["outcome"] in {
            "axial_support_loss", "guide_contact_loss", "guide_contact_coverage",
        } for row in rows),
        "mode_mismatches": sum(not row["mode_match"] for row in rows),
        "finite_phase_exits": sum(
            row["terminal_governor_phase"] == GOVERNOR_NOMINAL
            and row["terminal_governor_phase_age"] == 0
            for row in rows
        ),
        "outcomes": {
            outcome: sum(row["outcome"] == outcome for row in rows)
            for outcome in sorted({row["outcome"] for row in rows})
        },
        "maximum_total_force_N": max(
            (row["force_peak_N"] for row in rows), default=None,
        ),
        "maximum_lateral_force_N": max(
            (row["lateral_force_peak_N"] for row in rows), default=None,
        ),
        "maximum_contact_loss_seconds": max(contact_losses, default=None),
        "gate_passed": bool(
            len(rows) == expected and not failures
            and all(row["success"] for row in rows)
            and not any(row["constraint_violation"] for row in rows)
            and not any(not row["mode_match"] for row in rows)
            and all(row["terminal_governor_phase"] == GOVERNOR_NOMINAL
                    and row["terminal_governor_phase_age"] == 0 for row in rows)
        ),
        "failures": failures, "rows": rows,
        "finished_unix_seconds": time.time(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--finalize-existing", action="store_true",
        help="build report.json from saved progress.json without physics",
    )
    parser.add_argument(
        "--audit-existing", action="store_true",
        help="independently reevaluate saved evidence without physics",
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("T2 qualification is CPU-only and requires CUDA_VISIBLE_DEVICES=''")
    config_path = args.config.resolve()
    design = yaml.safe_load(config_path.read_text())
    output = (args.output or ROOT / design["output_dir"]).resolve()
    protocol = design["metadata"]["protocol"]
    protocols = {
        "transfer_finite_state_contact_qualification": {
            "version": "finite-state-contact-qualification-v1",
            "seeds": (930290, 930291, 930292, 930293),
            "scope": "one-shot CPU development qualification; all failures remain in the denominator",
        },
        "transfer_coupled_reference_admission_regression": {
            "version": "coupled-reference-admission-regression-v1",
            "seeds": (930270, 930271, 930272, 930273, 930290, 930291, 930292, 930293),
            "scope": "accessed CPU mechanism regression; not independent performance evidence",
            "contact_time_constant": .04,
        },
        "transfer_coupled_reference_admission_qualification": {
            "version": "coupled-reference-admission-qualification-v1",
            "seeds": (930294, 930295, 930296, 930297),
            "scope": "one-shot CPU development qualification; all failures remain in the denominator",
            "contact_time_constant": .02,
        },
        "transfer_coupled_reference_admission_accessed_extension": {
            "version": "coupled-reference-admission-accessed-extension-v1",
            "seeds": (930294, 930295, 930296, 930297),
            "scope": "accessed CPU mechanism regression after failed v1 qualification; not independent performance evidence",
            "contact_time_constant": .04,
        },
        "transfer_coupled_reference_admission_qualification_v2": {
            "version": "coupled-reference-admission-qualification-v2",
            "seeds": (930302, 930303, 930304, 930305),
            "scope": "one-shot CPU development qualification after contact-material revision; all failures remain in the denominator",
            "contact_time_constant": .04,
        },
    }
    if protocol not in protocols:
        raise ValueError("unknown T2 contact-governor validation protocol")
    contract = protocols[protocol]
    if args.audit_existing:
        from genedynamics.tasks.transfer.episode import audit_episode_evidence
        evidence_paths = sorted(output.glob("mode_*/seed_*/episode_evidence.json.gz"))
        expected = len(design["seeds"]) * len(design["metadata"]["requested_modes"])
        audits = [audit_episode_evidence(path) for path in evidence_paths]
        report = {
            "version": f"{contract['version']}-evidence-audit-v1",
            "status": "complete" if len(audits) == expected else "incomplete",
            "evidence_files": len(audits), "expected_evidence_files": expected,
            "passed": len(audits) == expected and all(row["passed"] for row in audits),
            "physics_calls": 0, "gpu_calls": 0, "rows": audits,
        }
        _write_json(output / "evidence_audit.json", report)
        print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.finalize_existing:
        progress = json.loads((output / "progress.json").read_text())
        expected = len(design["seeds"]) * len(design["metadata"]["requested_modes"])
        report = _report(
            progress["rows"], progress["failures"], expected,
            version=contract["version"], offline_recovery=True,
        )
        _write_json(output / "report.json", report)
        _write_json(output / "reporting_recovery.json", {
            "reason": "original report aggregation compared a null contact-loss field with floats",
            "physics_calls": 0,
            "source": "saved progress.json containing all eight completed episode summaries",
            "progress_sha256": _sha(output / "progress.json"),
            "report_sha256": _sha(output / "report.json"),
        })
        print(json.dumps(
            {key: value for key, value in report.items() if key != "rows"},
            indent=2, sort_keys=True,
        ))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite accessed qualification output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    seeds = tuple(int(value) for value in design["seeds"])
    modes = tuple(int(value) for value in design["metadata"]["requested_modes"])
    if seeds != contract["seeds"] or modes != (0, 1):
        raise ValueError("T2 validation cohort or mode order differs")
    binding = dict(design["method_params"]["actor_binding"])
    expected_reference = (
        "public_finite_state_guide_contact_v1"
        if protocol == "transfer_finite_state_contact_qualification"
        else "public_finite_state_guide_contact_v2"
    )
    if binding != {
        "version": "transfer-actor-binding-v1",
        "reference_version": expected_reference,
    }:
        raise ValueError("T2 public actor binding differs")
    expected_env = {
        "robot": "xarm7", "execution_profile": "substep_wrench_governor",
    }
    if protocol != "transfer_finite_state_contact_qualification":
        expected_env["physics_dt"] = .001
    if design["env_params"] != expected_env:
        raise ValueError("T2 execution profile differs")
    expected_time_constant = float(contract.get("contact_time_constant", .02))
    mechanism_params = dict(design.get(
        "mechanism_params", {"contact_time_constant": .02},
    ))
    if mechanism_params != {"contact_time_constant": expected_time_constant}:
        raise ValueError("T2 contact mechanism differs")

    snapshot = {
        "version": contract["version"],
        "design_path": str(config_path.relative_to(ROOT)),
        "design_sha256": _sha(config_path),
        "source_sha256": {name: _sha(ROOT / name) for name in SOURCE_PATHS},
        "device": "cpu", "cuda_visible_devices": "",
        "seeds": seeds, "modes": modes,
        "actor_binding": binding,
        "mechanism_params": mechanism_params,
        "started_unix_seconds": time.time(),
        "scope": contract["scope"],
    }
    _write_json(output / "run_manifest.json", snapshot)

    rows, failures = [], []
    for mode in modes:
        for seed in seeds:
            relative = Path(f"mode_{mode}") / f"seed_{seed}"
            episode_dir = output / relative
            try:
                base = task_spec("T2", robot="xarm7")
                from dataclasses import replace
                spec = replace(
                    base,
                    config=replace(
                        base.config,
                        execution_profile="substep_wrench_governor",
                        physics_dt=float(design["env_params"].get(
                            "physics_dt", base.config.physics_dt,
                        )),
                    ),
                )
                task = CpuTaskAdapter(
                    spec,
                    mechanism_spec=ContactMechanismSpec(**mechanism_params),
                    seed=seed,
                )
                observation = task.reset((seed,), seed=seed, modes=(mode,))
                actor = create_transfer_actor(
                    task, "public_geometry",
                    {"mode": mode, "actor_binding": binding},
                )
                for _ in range(spec.config.timeout_steps):
                    step = task.step(actor.act(observation))
                    task.record_control_diagnostics(controller_diagnostics(actor))
                    observation = step.observation
                    if bool(step.done[0]):
                        break
                record = task.records[-1]
                strict = evaluate_strict_guide_episode(record, spec, mode)
                evaluation = task.evaluate()
                traces = record.traces
                row = {
                    "seed": seed,
                    "requested_mode": mode,
                    "actor": type(actor).__name__,
                    "actor_version": actor.get_state()["version"],
                    "execution_profile": spec.config.execution_profile,
                    "success": bool(strict["success"]),
                    "outcome": strict["outcome"],
                    "mode_match": bool(strict["mode_match"]),
                    "constraint_violation": bool(strict["constraint_violation"]),
                    "control_steps": len(traces),
                    "first_contact_step": strict["first_contact_step"],
                    "first_outlet_step": strict["first_outlet_step"],
                    "continuous_guide_contact_ratio": strict["continuous_guide_contact_ratio"],
                    "maximum_contact_loss_seconds": strict["maximum_contact_loss_seconds"],
                    "force_peak_N": strict["force_peak_N"],
                    "lateral_force_peak_N": strict["lateral_force_peak_N"],
                    "terminal_governor_phase": int(np.asarray(task.state.governor_phase)[0]),
                    "terminal_governor_phase_age": int(np.asarray(task.state.governor_phase_age)[0]),
                    "accounting": dict(task.accounting),
                }
                evidence = episode_evidence(task.records, spec, evaluation)
                row["episode_evidence_sha256"] = _write_gzip_json(
                    episode_dir / "episode_evidence.json.gz", evidence,
                )
                _write_json(episode_dir / "summary.json", row)
                rows.append(row)
            except Exception as error:
                failure = {
                    "scope": str(relative), "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                _write_json(episode_dir / "failure.json", failure)
            _write_json(output / "progress.json", {
                "completed_episodes": len(rows), "failures": failures,
                "rows": rows,
            })

    expected = len(seeds) * len(modes)
    report = _report(rows, failures, expected, version=contract["version"])
    _write_json(output / "report.json", report)
    print(json.dumps(
        {key: value for key, value in report.items() if key != "rows"},
        indent=2, sort_keys=True,
    ))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

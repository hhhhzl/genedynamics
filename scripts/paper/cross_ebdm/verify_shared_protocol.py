#!/usr/bin/env python3
"""Zero-physics audit for the four-task common execution protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import (
    validate_transfer_actor_manifest,
)
from genedynamics.tasks.transfer.specs import (
    mechanism_spec,
    task_spec,
    validate_shared_execution,
)


ROOT = Path(__file__).resolve().parents[3]
TASKS = {
    "T1": ROOT / "configs/arm/obstacle_transfer",
    "T2": ROOT / "configs/arm/guided_contact_alignment",
    "T3": ROOT / "configs/arm/keyed_insertion_recovery",
    "T4": ROOT / "configs/quadruped/mobile_keyed_insertion_recovery",
}
CATEGORIES = ("main", "baseline", "ablation")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify_pointer(pointer: dict, label: str) -> dict:
    path = resolve(str(pointer["path"]))
    actual = sha(path)
    if actual != pointer["sha256"]:
        raise ValueError(f"{label} SHA256 mismatch")
    return {"path": str(path.relative_to(ROOT)), "sha256": actual}


def verify_contract(pointer: dict) -> dict:
    checked = verify_pointer(pointer, "shared_task_contract")
    document = json.loads((ROOT / checked["path"]).read_text())
    parents = document.get("extends", [])
    if isinstance(parents, dict):
        parents = [parents]
    checked["version"] = document["version"]
    checked["parents"] = [verify_pointer(parent, "parent contract") for parent in parents]
    return checked


def verify_actor_revision(document: dict) -> dict:
    revision = document.get("development_revision") or {}
    if revision.get("name") == "public_mode0_contact_establishment_transport_admission_v2":
        parent_pointer = revision["supersedes"]
        parent_path = resolve(parent_pointer["path"])
        checked_parent = verify_pointer(
            parent_pointer, "superseded contact-establishment actor manifest",
        )
        parent = json.loads(parent_path.read_text())
        affected = ["main/cross_ebdm", *(
            f"ablation/{key}" for key in sorted(document["ablation"])
        )]
        if revision.get("affected_rows") != affected:
            raise ValueError(
                "mode-conditioned admission affected-row set changed"
            )
        mutable_top = {"main", "ablation", "development_revision", "scope"}
        immutable_keys = (set(document) | set(parent)) - mutable_top
        if any(document.get(key) != parent.get(key) for key in immutable_keys):
            raise ValueError(
                "mode-conditioned admission changed immutable catalog data"
            )
        for family in ("main", "ablation"):
            if set(document[family]) != set(parent[family]):
                raise ValueError(
                    f"mode-conditioned admission changed {family} row inventory"
                )
            for key, row in document[family].items():
                previous = dict(parent[family][key])
                current = dict(row)
                label = f"{family}/{key}"
                if label not in affected:
                    if current != previous:
                        raise ValueError(
                            f"mode-conditioned admission changed unaffected row: {label}"
                        )
                    continue
                if previous.pop("actor_parameters", None) != {
                    "contact_establishment_admission_version":
                    "public_contact_establishment_admission_v1",
                }:
                    raise ValueError(
                        f"{label} did not supersede contact-establishment v1"
                    )
                expected = {
                    **previous,
                    "actor_parameters": {
                        "contact_establishment_admission_version":
                        "public_mode0_contact_establishment_admission_v2",
                    },
                }
                if label == "main/cross_ebdm":
                    if previous.get("method_identity") != (
                        "ours_contact_established_transport_v4"
                    ):
                        raise ValueError(
                            "mode-conditioned admission superseded an unexpected main identity"
                        )
                    expected["method_identity"] = (
                        "ours_mode0_contact_established_transport_v5"
                    )
                if current != expected:
                    raise ValueError(
                        f"{label} changed beyond the mode-conditioned runtime admission"
                    )
        decision_pointer = revision["decision_evidence"]
        decision_path = resolve(decision_pointer["path"])
        checked_decision = verify_pointer(
            decision_pointer, "mode-conditioned admission decision evidence",
        )
        decision = json.loads(decision_path.read_text())
        if not (
            decision.get("status") == "development_diagnosis_complete"
            and decision.get("performance_evidence") is False
            and decision.get("may_be_reused_for_qualification_or_confirmation")
            is False
            and decision.get("formal_outcome_match") is True
            and decision.get("mode_one_action_stream_parity") is True
            and decision.get("mode_zero_failures_repaired") is True
        ):
            raise ValueError(
                "mode-conditioned admission decision evidence changed semantics"
            )
        invalid_pointer = revision["invalid_predecessor"]
        invalid_path = resolve(invalid_pointer["path"])
        checked_invalid = verify_pointer(
            invalid_pointer, "mode-conditioned admission invalid predecessor",
        )
        invalid = json.loads(invalid_path.read_text())
        if not (
            invalid.get("status") == "invalid_incomplete_qualification"
            and invalid.get("performance_evidence_admitted") is False
            and invalid.get("reusable_records") == 0
            and invalid.get("failure_replacement") is False
            and invalid_pointer.get("reusable_records") == 0
        ):
            raise ValueError(
                "mode-conditioned admission predecessor is not zero-reuse invalid"
            )
        return {
            "revision": revision["name"],
            "transport_only_check": None,
            "mode_conditioned_contact_establishment_check": {
                "passed": True,
                "affected_rows": affected,
                "only_runtime_admission_and_main_identity_changed": True,
                "mode_one_action_stream_parity": True,
                "mode_zero_development_failures_repaired": True,
                "development_evidence_excluded_from_formal_performance": True,
                "invalid_predecessor_reusable_records": 0,
                "immutable_artifact_catalog_unchanged": True,
                "decision_evidence": checked_decision,
                "invalid_predecessor": checked_invalid,
                "supersedes": checked_parent,
            },
        }
    if revision.get("name") != "transport_only_without_public_contact_guard_v1":
        return {
            "revision": revision.get("name"),
            "transport_only_check": None,
            "mode_conditioned_contact_establishment_check": None,
        }
    parent_pointer = revision["supersedes"]
    parent_path = resolve(parent_pointer["path"])
    verify_pointer(parent_pointer, "superseded actor manifest")
    parent = json.loads(parent_path.read_text())
    affected = ["main/cross_ebdm", *(
        f"ablation/{key}" for key in sorted(document["ablation"])
    )]
    if revision.get("affected_rows") != affected:
        raise ValueError("transport-only actor revision affected-row set changed")
    unchanged_top = {
        "version", "binding_version", "task", "inventory", "ablation_inventory",
        "rejection_scoring", "artifacts", "formal_performance_evidence",
        "confirmation_states_opened",
    }
    if any(document.get(key) != parent.get(key) for key in unchanged_top):
        raise ValueError("transport-only actor revision changed immutable catalog data")
    for family in ("main", "ablation"):
        for key, row in document[family].items():
            previous = dict(parent[family][key])
            current = dict(row)
            label = f"{family}/{key}"
            if label in affected:
                if previous.pop("actor_parameters", None) != {
                    "contact_guard_version": "public_measured_contact_guard_v1",
                }:
                    raise ValueError(f"{label} did not supersede the expected guard")
                if label == "main/cross_ebdm":
                    previous["method_identity"] = "ours_transport_only_v3"
                if current != previous:
                    raise ValueError(f"{label} changed beyond guard removal")
            elif current != previous:
                raise ValueError(f"unaffected actor row changed: {label}")
    return {
        "revision": revision["name"],
        "mode_conditioned_contact_establishment_check": None,
        "transport_only_check": {
            "passed": True,
            "affected_rows": affected,
            "only_runtime_guard_and_main_identity_changed": True,
            "immutable_artifact_catalog_unchanged": True,
            "decision_evidence": verify_pointer(
                revision["decision_evidence"], "guard-removal decision evidence",
            ),
            "supersedes": verify_pointer(
                parent_pointer, "superseded actor manifest",
            ),
        },
    }


def leaf_paths(root: Path) -> list[Path]:
    return sorted(
        path for category in CATEGORIES for path in (root / category).glob("*.yaml")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/_architecture/cross_ebdm_shared_protocol_20261021/report.json",
    )
    parser.add_argument(
        "--qualification-report",
        type=Path,
        help="Completed behavioral predeployment report used to decide the training gate.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output

    task_reports = {}
    total_configs = 0
    for alias, root in TASKS.items():
        paths = leaf_paths(root)
        if len(paths) != 13:
            raise ValueError(f"{alias} must expose exactly 13 leaf configs, got {len(paths)}")
        configs = [ExperimentConfig.from_yaml(path) for path in paths]
        total_configs += len(configs)
        methods = {config.method for config in configs}
        if len(methods) != 9:
            raise ValueError(f"{alias} does not expose nine main-table methods")

        base = ExperimentConfig.from_yaml(root / "_base.yaml")
        inherited = base.env_params.get("shared_execution")
        for path, config in zip(paths, configs):
            raw = yaml.safe_load(path.read_text()) or {}
            if "shared_execution" in (raw.get("env_params") or {}):
                raise ValueError(f"leaf config overrides shared execution: {path}")
            if config.env_params.get("shared_execution") != inherited:
                raise ValueError(f"shared execution inheritance drift: {path}")

        contract = None
        if alias == "T1":
            if inherited is not None:
                raise ValueError("T1 unexpectedly acquired a contact execution contract")
        else:
            contract = verify_contract(base.metadata["shared_task_contract"])
            env = dict(base.env_params)
            robot = env.pop("robot")
            env.pop("history_len")
            env.pop("mode")
            shared = env.pop("shared_execution")
            mechanism_values = env.pop("mechanism", {})
            spec = task_spec(alias, robot=robot, config_overrides=env or None)
            validate_shared_execution(
                spec, mechanism_spec(alias, mechanism_values), shared,
            )

        manifest = base.method_params["actor_binding"]["manifest"]
        manifest_document = validate_transfer_actor_manifest(manifest)
        statuses = {}
        for family in ("main", "ablation"):
            for row in manifest_document[family].values():
                statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        task_reports[alias] = {
            "leaf_configs": len(paths),
            "methods": sorted(methods),
            "shared_execution": inherited,
            "shared_task_contract": contract,
            "actor_manifest": verify_pointer(manifest, "actor manifest"),
            "actor_revision": verify_actor_revision(manifest_document),
            "actor_status_counts_with_ablation_aliases": statuses,
            "checks": {
                "leaf_override_absent": True,
                "all_leaves_inherit_exact_common_value": True,
                "task_construction_values_match_contract": True,
                "actor_manifest_and_artifacts_hash_valid": True,
            },
        }

    source_paths = [
        ROOT / "genedynamics/tasks/transfer/specs.py",
        ROOT / "genedynamics/tasks/transfer/contact.py",
        ROOT / "genedynamics/envs/adapters/transfer_task.py",
        ROOT / "genedynamics/experiments/plugins/environments/transfer_tasks.py",
        ROOT / "genedynamics/experiments/plugins/methods/transfer/registry.py",
        ROOT / "genedynamics/tasks/transfer/controllers/bindings.py",
        ROOT / "genedynamics/tasks/transfer/controllers/guide_contact.py",
        ROOT / "genedynamics/tasks/transfer/controllers/guide_lift_route.py",
        ROOT / "genedynamics/tasks/transfer/controllers/mating_feedback.py",
        ROOT / "genedynamics/tasks/transfer/controllers/cross_ebdm/arm_dense.py",
        ROOT / "genedynamics/tasks/transfer/controllers/cross_ebdm/dense.py",
        ROOT / "genedynamics/tasks/transfer/controllers/cross_ebdm/obstacle_planned.py",
        ROOT / "genedynamics/tasks/transfer/controllers/cross_ebdm/guide_corridor_v2.py",
        ROOT / "genedynamics/tasks/transfer/controllers/dial.py",
        ROOT / "genedynamics/tasks/transfer/controllers/recovery_execution.py",
        ROOT / "genedynamics/solvers/single/cross_ebdm/learning/task_transport.py",
        ROOT / "genedynamics/solvers/single/response_mpc/arm.py",
        ROOT / "genedynamics/tasks/transfer/evaluation.py",
        ROOT / "genedynamics/tasks/transfer/guide_evaluation.py",
        ROOT / "genedynamics/tasks/transfer/confirmation.py",
        ROOT / "scripts/paper/cross_ebdm/qualify_method_behavior.py",
        ROOT / "scripts/paper/cross_ebdm/audit_behavioral_predeployment.py",
        ROOT / "scripts/paper/cross_ebdm/prepare_confirmation.py",
        ROOT / "scripts/paper/cross_ebdm/run_confirmation.py",
        ROOT / "scripts/paper/cross_ebdm/run_confirmation.sh",
    ]
    protocol_validity = {
        "schema_version": "cross-ebdm-protocol-validity-v3",
        "status": "passed", "passed": True, "physics_steps": 0, "gpu_used": False,
        "configuration_count": total_configs,
        "tasks": {
            alias: {
                "shared_execution": row["shared_execution"],
                "shared_task_contract": row["shared_task_contract"],
                "checks": row["checks"],
            }
            for alias, row in task_reports.items()
        },
    }
    architecture_equivalence = {
        "schema_version": "cross-ebdm-architecture-equivalence-v3",
        "status": "passed", "passed": True, "physics_steps": 0, "gpu_used": False,
        "definition": "Every leaf resolves through the original base-config inheritance and the single task-owned environment/method plugins; method actor identity remains selected only by the binding manifest.",
        "configuration_count": total_configs,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in source_paths
        },
        "task_actor_manifests": {
            alias: row["actor_manifest"] for alias, row in task_reports.items()
        },
    }
    validity_path = output.with_name("protocol_validity.json")
    equivalence_path = output.with_name("architecture_equivalence.json")
    validity_path.parent.mkdir(parents=True, exist_ok=True)
    validity_path.write_text(json.dumps(protocol_validity, indent=2, sort_keys=True) + "\n")
    equivalence_path.write_text(json.dumps(architecture_equivalence, indent=2, sort_keys=True) + "\n")
    qualification = None
    qualification_gate = {
        "status": "blocked_pending_fresh_predeployment_qualification",
        "old_confirmation_reused": False,
        "gpu_training_allowed": False,
    }
    if args.qualification_report is not None:
        qualification_path = (
            args.qualification_report
            if args.qualification_report.is_absolute()
            else ROOT / args.qualification_report
        )
        qualification = json.loads(qualification_path.read_text())
        if qualification.get("schema_version") not in {
            "four-task-behavioral-predeployment-v3",
            "four-task-behavioral-predeployment-v4",
            "four-task-behavioral-predeployment-v5",
            "four-task-behavioral-predeployment-v6",
            "four-task-behavioral-predeployment-v7",
            "four-task-behavioral-predeployment-v8",
            "four-task-behavioral-predeployment-v9",
            "four-task-behavioral-predeployment-v10",
            "four-task-behavioral-predeployment-v11",
            "four-task-behavioral-predeployment-v12",
            "four-task-behavioral-predeployment-v13",
            "four-task-behavioral-predeployment-v14",
            "four-task-behavioral-predeployment-v15",
            "four-task-behavioral-predeployment-v16",
            "four-task-behavioral-predeployment-composition-v1",
        }:
            raise ValueError("qualification report is not a supported behavioral protocol")
        if qualification.get("confirmation_states_opened") != 0:
            raise ValueError("qualification report opened confirmation states")
        if qualification.get("old_invalid_outcomes_used") is not False:
            raise ValueError("qualification report reused invalid outcomes")
        allowed = bool(qualification.get("gpu_training_allowed"))
        qualification_gate = {
            "status": "gpu_training_admitted" if allowed else "qualification_failed",
            "old_confirmation_reused": False,
            "gpu_training_allowed": allowed,
            "behavioral_predeployment": {
                "path": str(qualification_path.relative_to(ROOT)),
                "sha256": sha(qualification_path),
            },
        }
    report = {
        "schema_version": "cross-ebdm-shared-protocol-zero-physics-v3",
        "status": "passed",
        "physics_steps": 0,
        "gpu_used": False,
        "configuration_count": total_configs,
        "tasks": task_reports,
        "protocol_validity": {
            "path": str(validity_path.relative_to(ROOT)), "sha256": sha(validity_path),
        },
        "architecture_equivalence": {
            "path": str(equivalence_path.relative_to(ROOT)), "sha256": sha(equivalence_path),
        },
        "qualification_gate": qualification_gate,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "sha256": sha(output),
        "configuration_count": total_configs,
        "physics_steps": 0,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Task-owned construction of public actors for the four transfer tasks.

Solvers own algorithms and checkpoints.  This module owns the explicit mapping
from a solver's public actor to a task contract.  Experiment plugins call only
``create_transfer_actor`` and do not contain method or task behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


BINDING_VERSION = "transfer-actor-binding-v1"
MANIFEST_VERSION = "transfer-actor-binding-manifest-v1"


def _root() -> Path:
    return Path(__file__).resolve().parents[4]


def _checked(pointer: Any, label: str) -> Path:
    if not isinstance(pointer, dict):
        raise ValueError(f"actor_binding.{label} must contain path and sha256")
    path = Path(str(pointer.get("path", ""))).expanduser()
    if not path.is_absolute():
        path = _root() / path
    expected = str(pointer.get("sha256", ""))
    if path.is_dir():
        expected_files = pointer.get("sha256")
        actual_files = {
            child.name: hashlib.sha256(child.read_bytes()).hexdigest()
            for child in sorted(path.iterdir()) if child.is_file()
        }
        if not isinstance(expected_files, dict) or actual_files != expected_files:
            raise ValueError(f"actor_binding.{label} failed directory SHA256 validation")
        return path
    if not path.is_file():
        raise ValueError(f"actor_binding.{label} is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError(f"actor_binding.{label} failed SHA256 validation")
    return path


def _manifest(pointer: Any) -> dict[str, Any]:
    path = _checked(pointer, "manifest")
    document = json.loads(path.read_text())
    if document.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"actor binding manifest must use {MANIFEST_VERSION!r}"
        )
    if document.get("binding_version") != BINDING_VERSION:
        raise ValueError(
            f"actor binding manifest must resolve {BINDING_VERSION!r}"
        )
    if document.get("confirmation_states_opened") != 0:
        raise ValueError("actor binding manifest must not open confirmation states")
    if document.get("formal_performance_evidence") is not False:
        raise ValueError("migration manifest cannot contain formal performance evidence")
    return document


def _validate_pointer_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            _checked(value, label)
            return
        for key, child in value.items():
            _validate_pointer_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_pointer_tree(child, f"{label}[{index}]")


def validate_transfer_actor_manifest(pointer: Any) -> dict[str, Any]:
    """Validate every immutable artifact and the complete 9-by-4/6-cell schema."""
    document = _manifest(pointer)
    if set(document.get("main", {})) != {
        "cross_ebdm", "task_only", "fada", "acp", "human2sim2robot",
        "dial_transfer", "response_mpc", "public_geometry",
        "public_source_prior",
    }:
        raise ValueError("actor binding manifest must contain exactly nine main rows")
    if set(document.get("ablation", {})) != {
        "frozen_warm", "task_only", "matching_no_band", "band_only",
        "ungated_full", "gated_full",
    }:
        raise ValueError("actor binding manifest must contain exactly six ablation cells")
    _validate_pointer_tree(document.get("inventory"), "inventory")
    _validate_pointer_tree(document.get("ablation_inventory"), "ablation_inventory")
    _validate_pointer_tree(document.get("rejection_scoring"), "rejection_scoring")
    _validate_pointer_tree(document.get("artifacts", {}), "artifacts")
    _validate_pointer_tree(
        document.get("development_revision", {}), "development_revision",
    )
    _validate_pointer_tree(document.get("main", {}), "main")
    _validate_pointer_tree(document.get("ablation", {}), "ablation")
    for family in ("main", "ablation"):
        for key, row in document[family].items():
            actor_parameters = row.get("actor_parameters", {})
            if not isinstance(actor_parameters, dict):
                raise ValueError(f"{family}.{key}.actor_parameters must be a mapping")
            if set(actor_parameters) - {
                "reference_version", "contact_guard_version",
                "contact_establishment_admission_version",
                "completion_admission_version", "task_dominance_version",
            }:
                raise ValueError(f"{family}.{key} has unsupported actor_parameters")
            if ("reference_version" in actor_parameters
                    and row.get("actor") != "public_geometry"):
                raise ValueError(
                    f"{family}.{key}.reference_version requires public geometry"
                )
            if ("contact_guard_version" in actor_parameters
                    and row.get("actor") != "cross_ebdm"):
                raise ValueError(
                    f"{family}.{key}.contact_guard_version requires cross_ebdm"
                )
            if ("contact_establishment_admission_version" in actor_parameters
                    and row.get("actor") != "cross_ebdm"):
                raise ValueError(
                    f"{family}.{key}.contact_establishment_admission_version "
                    "requires cross_ebdm"
                )
            if ({"contact_guard_version", "contact_establishment_admission_version"}
                    <= set(actor_parameters)):
                raise ValueError(
                    f"{family}.{key} declares mutually exclusive T2 selectors"
                )
            if ("completion_admission_version" in actor_parameters
                    and row.get("actor") != "cross_ebdm"):
                raise ValueError(
                    f"{family}.{key}.completion_admission_version requires cross_ebdm"
                )
            if ("task_dominance_version" in actor_parameters
                    and row.get("actor") != "cross_ebdm"):
                raise ValueError(
                    f"{family}.{key}.task_dominance_version requires cross_ebdm"
                )
            status = row.get("status")
            if status == "instances_ready":
                instance_ids: set[str] | None = None
                artifacts = document["artifacts"]
                policy_sets_by_mode = row.get("policy_sets_by_mode")
                if policy_sets_by_mode is not None:
                    if row.get("policy_set") is not None:
                        raise ValueError(
                            f"{family}.{key} cannot mix policy_set and policy_sets_by_mode"
                        )
                    if (
                        not isinstance(policy_sets_by_mode, dict)
                        or set(policy_sets_by_mode) != {"0", "1"}
                        or row.get("actor") != "cross_ebdm"
                    ):
                        raise ValueError(
                            f"{family}.{key}.policy_sets_by_mode must bind both public modes"
                        )
                for set_key, catalog_key in (
                    ("policy_set", "policy_sets"),
                    ("policy_sets_by_mode", "policy_sets"),
                    ("admission_policy_set", "policy_sets"),
                    ("checkpoint_set", "checkpoint_sets"),
                    ("response_set", None),
                    ("joint_admission_set", None),
                ):
                    name = row.get(set_key)
                    if name is None:
                        continue
                    names = name.values() if set_key == "policy_sets_by_mode" else (name,)
                    for selected_name in names:
                        if selected_name == "fixed_response_companion_policy":
                            ids = {f"instance_{index:02d}" for index in range(1, 9)}
                        else:
                            catalog = (
                                artifacts if catalog_key is None
                                else artifacts[catalog_key]
                            )
                            values = catalog.get(selected_name)
                            if not isinstance(values, dict):
                                raise ValueError(
                                    f"{family}.{key}.{set_key} is unresolved"
                                )
                            ids = set(values)
                        if instance_ids is None:
                            instance_ids = ids
                        elif instance_ids != ids:
                            raise ValueError(
                                f"{family}.{key} artifact instance sets differ"
                            )
                if row.get("admission_policy_set") is not None and (
                    row.get("actor") != "cross_ebdm"
                    or actor_parameters.get("task_dominance_version") is None
                ):
                    raise ValueError(
                        f"{family}.{key}.admission_policy_set requires task dominance"
                    )
                if instance_ids != {f"instance_{index:02d}" for index in range(1, 9)}:
                    raise ValueError(f"{family}.{key} must resolve eight instances")
            elif status == "predeployment_rejection":
                if row.get("instance_role") == "deterministic_single_identity":
                    expected = ["fixed"]
                else:
                    expected = [f"instance_{index:02d}" for index in range(1, 9)]
                if row.get("logical_instance_slots") != expected:
                    raise ValueError(
                        f"{family}.{key} rejection slots do not match its identity role"
                    )
                if row.get("target_physical_execution") != 0:
                    raise ValueError(f"{family}.{key} rejection cannot execute physics")
            elif status != "deterministic_single_identity":
                raise ValueError(f"{family}.{key} has unsupported status {status!r}")
    return document


def resolve_transfer_binding(
    config: dict[str, Any], *, task_alias: str | None = None,
    instance_id: str | None = None,
) -> dict[str, Any]:
    """Resolve one compact task manifest selection into an actor binding."""
    selection = config.get("actor_binding")
    if not isinstance(selection, dict) or selection.get("version") != BINDING_VERSION:
        raise ValueError(f"method requires actor_binding.version={BINDING_VERSION!r}")
    if selection.get("manifest") is None:
        return selection
    # Full-catalog validation is a separate zero-physics preflight. Runtime
    # resolution revalidates the manifest SHA and every selected artifact as
    # it is loaded, without hashing unrelated methods for every episode.
    document = _manifest(selection["manifest"])
    if task_alias is not None and document.get("task") != task_alias:
        raise ValueError(
            f"actor manifest task {document.get('task')!r} does not match {task_alias!r}"
        )
    family = str(selection.get("family", "main"))
    key = selection.get("key")
    if family not in {"main", "ablation"} or key not in document[family]:
        raise ValueError(f"unknown actor manifest selection {family}/{key}")
    row = document[family][key]
    status = row["status"]
    resolved = {
        "version": BINDING_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "task": document["task"],
        "family": family,
        "binding_key": key,
        "method_identity": row["method_identity"],
        "actor": row["actor"],
        "status": status,
    }
    actor_parameters = row.get("actor_parameters", {})
    if set(actor_parameters) & set(resolved):
        raise ValueError(f"{family}.{key}.actor_parameters overwrite binding identity")
    resolved.update(actor_parameters)
    if status == "predeployment_rejection":
        resolved.update({
            "reason": row["reason"],
            "evidence": row["evidence"],
            "scoring": row["scoring"],
            "logical_instance_slots": row["logical_instance_slots"],
            "instance_role": row["instance_role"],
            "target_physical_execution": 0,
        })
        return resolved
    if status == "deterministic_single_identity":
        selected_instance = "fixed"
    else:
        selected_instance = instance_id or selection.get("instance")
        if selected_instance not in {
            f"instance_{index:02d}" for index in range(1, 9)
        }:
            raise ValueError(
                f"{family}/{key} requires actor instance_01 through instance_08"
            )
    resolved["instance_id"] = selected_instance
    artifacts = document["artifacts"]
    for name in (
        "source_manifest", "source_prior", "feedback", "feedback_evidence",
        "known_body_path", "maximum_query_distance",
    ):
        if name in artifacts:
            resolved[name] = artifacts[name]
    if row.get("checkpoint_set"):
        resolved["checkpoint"] = artifacts["checkpoint_sets"][
            row["checkpoint_set"]
        ][selected_instance]
    if row.get("policy_set"):
        name = row["policy_set"]
        if name == "fixed_response_companion_policy":
            resolved["policy_checkpoint"] = artifacts[name]
        else:
            resolved["policy_checkpoint"] = artifacts["policy_sets"][name][
                selected_instance
            ]
    if row.get("policy_sets_by_mode"):
        mode = config.get("mode")
        resolved["policy_checkpoints_by_mode"] = {
            key: artifacts["policy_sets"][name][selected_instance]
            for key, name in row["policy_sets_by_mode"].items()
        }
        if mode is not None:
            mode_key = str(int(mode))
            if mode_key not in resolved["policy_checkpoints_by_mode"]:
                raise ValueError(f"unsupported public mode {mode!r}")
            resolved["policy_checkpoint"] = resolved[
                "policy_checkpoints_by_mode"
            ][mode_key]
    if row.get("admission_policy_set"):
        resolved["admission_task_only_checkpoint"] = artifacts["policy_sets"][
            row["admission_policy_set"]
        ][selected_instance]
    if row.get("response_set"):
        resolved["response_checkpoint"] = artifacts[row["response_set"]][
            selected_instance
        ]
    if row.get("joint_admission_set"):
        selected = artifacts[row["joint_admission_set"]][selected_instance]
        resolved["joint_admission_checkpoint"] = selected["checkpoint"]
        resolved["joint_admission_calibration"] = selected["calibration"]
    if row.get("sample_stream_set"):
        resolved["sample_stream_sha256"] = artifacts["sample_streams"][
            row["sample_stream_set"]
        ][selected_instance]
    return resolved


def _binding(config: dict[str, Any]) -> dict[str, Any]:
    return resolve_transfer_binding(config)


def _torch_payload(binding: dict[str, Any], label: str):
    import torch

    return torch.load(
        _checked(binding.get(label), label), map_location="cpu", weights_only=True
    )


def _source_bank(binding: dict[str, Any], response_spec: Any, task: str):
    from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import (
        PublicSkillBank,
    )

    path = _checked(binding.get("source_manifest"), "source_manifest")
    document = json.loads(path.read_text())
    entries = document.get("entries")
    if entries is None:
        entries = (document.get("data_manifest") or {}).get("offline")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source_manifest has no entries/data_manifest.offline")
    resolved = []
    for entry in entries:
        row = dict(entry)
        source = Path(str(row.get("path", ""))).expanduser()
        if not source.is_absolute():
            source = _root() / source
        row["path"] = str(source.resolve())
        resolved.append(row)
    bank = PublicSkillBank(resolved, response_spec, task=task)
    expected = binding.get("source_bank_sha256")
    if expected and bank.digest != expected:
        raise ValueError("actor_binding source bank digest changed")
    return bank


def _cross_components(task: Any, binding: dict[str, Any], device: str):
    alias = task.spec.alias
    if alias in {"T1", "T2"}:
        from genedynamics.solvers.single.cross_ebdm.learning.task_response import (
            PublicTaskIssuedResponseEnsemble,
        )
        from genedynamics.solvers.single.cross_ebdm.learning.task_transport import (
            PublicTaskResidualTransport,
        )
        from genedynamics.solvers.single.cross_ebdm.learning.diffusion_transport import (
            PublicTaskDiffusionTransport, VERSION as DIFFUSION_TRANSPORT_VERSION,
        )

        policy_payload = _torch_payload(binding, "policy_checkpoint")
        policy_type = (PublicTaskDiffusionTransport
                       if policy_payload.get("version") == DIFFUSION_TRANSPORT_VERSION
                       else PublicTaskResidualTransport)
        response_type = PublicTaskIssuedResponseEnsemble
        bank_task = alias
    else:
        from genedynamics.solvers.single.cross_ebdm.learning.issued_response import (
            PublicIssuedResponseEnsemble,
        )
        from genedynamics.solvers.single.cross_ebdm.learning.local_transport import (
            PublicLocalResidualTransport,
        )

        policy_type = PublicLocalResidualTransport
        response_type = PublicIssuedResponseEnsemble
        bank_task = "T3"
        policy_payload = _torch_payload(binding, "policy_checkpoint")
    policy = policy_type.from_checkpoint(
        policy_payload, device
    ).eval()
    response = None
    if binding.get("response_checkpoint") is not None:
        response = response_type.from_checkpoint(
            _torch_payload(binding, "response_checkpoint"), device
        ).eval()
    spec = response.spec if response is not None else policy.response_spec
    return policy, response, _source_bank(binding, spec, bank_task)


def _feedback(binding: dict[str, Any]):
    from genedynamics.tasks.transfer.controllers.cross_ebdm.policies import (
        PublicBiasFeedbackSpec,
    )

    values = binding.get("feedback")
    if not isinstance(values, dict):
        raise ValueError("actor_binding.feedback is required for T3/T4")
    return PublicBiasFeedbackSpec(**values)


def _prior_path(binding: dict[str, Any]) -> Path | None:
    pointer = binding.get("source_prior")
    return None if pointer is None else _checked(pointer, "source_prior")


def _dense_actor(task: Any, binding: dict[str, Any], *, mode: int, device: str):
    policy, response, bank = _cross_components(task, binding, device)
    alias = task.spec.alias
    common = dict(mode=mode, device=device, maximum_query_distance=float(
        binding.get("maximum_query_distance", 5.0)
    ))
    if alias == "T1":
        from .cross_ebdm.obstacle_planned import PublicPlannedObstacleDenseActor

        path = binding.get("known_body_path")
        if path is None:
            raise ValueError("T1 dense binding requires known_body_path")
        return PublicPlannedObstacleDenseActor(
            task.spec, policy, bank, path=path, **common
        )
    if alias == "T2":
        if binding.get("joint_admission_checkpoint") is not None:
            from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
                PublicJointAdmissionCalibration,
                PublicJointAdmissionEnsemble,
            )
            from .cross_ebdm.guide_corridor_v2 import (
                PublicJointRiskAdmittedCorridorGuideDenseActor,
            )

            risk = PublicJointAdmissionEnsemble.from_checkpoint(
                _torch_payload(binding, "joint_admission_checkpoint"), device,
            ).eval()
            calibration_path = _checked(
                binding.get("joint_admission_calibration"),
                "joint_admission_calibration",
            )
            calibration_values = json.loads(calibration_path.read_text())
            for name in (
                "additive_transformed_correction", "risk_limits",
                "interpolation_fractions",
            ):
                calibration_values[name] = tuple(calibration_values[name])
            calibration = PublicJointAdmissionCalibration(**calibration_values)
            return PublicJointRiskAdmittedCorridorGuideDenseActor(
                task.spec, policy, bank, risk_model=risk,
                risk_calibration=calibration, **common,
            )
        admission = binding.get("contact_establishment_admission_version")
        if admission is not None:
            if admission != "public_mode0_contact_establishment_admission_v2":
                raise ValueError("unknown T2 contact-establishment admission")
            from .cross_ebdm.guide_corridor_v2 import (
                PublicModeConditionedContactEstablishedCorridorGuideDenseActor,
            )

            return PublicModeConditionedContactEstablishedCorridorGuideDenseActor(
                task.spec, policy, bank, **common,
            )
        if binding.get("contact_guard_version") is not None:
            if binding["contact_guard_version"] != "public_measured_contact_guard_v1":
                raise ValueError("unknown T2 public contact guard")
            from .cross_ebdm.guide_corridor_v2 import (
                PublicContactGuardedCorridorGuideDenseActor,
            )

            return PublicContactGuardedCorridorGuideDenseActor(
                task.spec, policy, bank, **common,
            )
        from .cross_ebdm.guide_corridor_v2 import PublicCorridorGuideDenseActor

        return PublicCorridorGuideDenseActor(task.spec, policy, bank, **common)
    feedback = _feedback(binding)
    options = dict(
        prior_path=_prior_path(binding), seed=int(binding.get("seed", task.seed)),
        feedback=feedback, **common,
    )
    if alias == "T3":
        dominance = binding.get("task_dominance_version")
        if dominance not in {
            None, "public_transport_over_task_only_dominance_v1",
            "public_full_transport_over_task_only_dominance_v2",
            "public_adaptive_step_transport_dominance_v3",
            "public_expanded_adaptive_step_transport_dominance_v4",
            "public_ensemble_robust_adaptive_step_dominance_v5",
            "public_expanded_ensemble_robust_adaptive_step_dominance_v6",
            "public_bounded_endpoint_ensemble_robust_dominance_v7",
            "public_capped_expansion_ensemble_robust_dominance_v8",
            "public_horizon_consistent_ensemble_robust_dominance_v9",
            "public_validated_bridge_ensemble_robust_dominance_v10",
            "public_completion_value_ensemble_robust_dominance_v11",
            "public_transient_phase_adaptive_step_dominance_v12",
            "public_phase_scheduled_adaptive_step_dominance_v13",
            "public_mode_conditioned_finite_time_dominance_v14",
            "public_mode_specific_recorded_completion_dominance_v15",
            "public_mode_specific_supported_phase_convex_dominance_v16",
            "public_mode_specific_supported_phase_capped_expansion_dominance_v17",
            "public_mode_specific_calibrated_authority_dominance_v18",
        }:
            raise ValueError("unknown T3 task-dominance execution version")
        if dominance is not None:
            if response is None:
                raise ValueError("T3 task dominance requires a public response model")
            from .cross_ebdm.dense import (
                PublicAdaptiveStepTaskDominantDenseTransportActor,
                PublicExpandedAdaptiveStepTaskDominantDenseTransportActor,
                PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor,
                PublicBoundedEndpointEnsembleRobustDenseTransportActor,
                PublicCappedExpansionEnsembleRobustDenseTransportActor,
                PublicHorizonConsistentEnsembleRobustDenseTransportActor,
                PublicValidatedBridgeEnsembleRobustDenseTransportActor,
                PublicCompletionValueEnsembleRobustDenseTransportActor,
                PublicEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor,
                PublicFullTaskDominantDenseTransportActor,
                PublicTaskDominantDenseTransportActor,
                PublicTransientPhaseAdaptiveStepDenseTransportActor,
                PublicPhaseScheduledAdaptiveStepDenseTransportActor,
                PublicModeConditionedAdaptiveStepDenseTransportActor,
                PublicModeConditionedCompletionValueDenseTransportActor,
                PublicModeSpecificRecordedCompletionDenseTransportActor,
                PublicModeSpecificSupportedPhaseConvexDenseTransportActor,
                PublicModeSpecificSupportedPhaseCappedExpansionDenseTransportActor,
                PublicModeSpecificCalibratedAuthorityDenseTransportActor,
            )

            task_only_policy = type(policy).from_checkpoint(
                _torch_payload(binding, "admission_task_only_checkpoint"), device,
            ).eval()
            actor_type = {
                "public_transport_over_task_only_dominance_v1": (
                    PublicTaskDominantDenseTransportActor
                ),
                "public_full_transport_over_task_only_dominance_v2": (
                    PublicFullTaskDominantDenseTransportActor
                ),
                "public_adaptive_step_transport_dominance_v3": (
                    PublicAdaptiveStepTaskDominantDenseTransportActor
                ),
                "public_expanded_adaptive_step_transport_dominance_v4": (
                    PublicExpandedAdaptiveStepTaskDominantDenseTransportActor
                ),
                "public_ensemble_robust_adaptive_step_dominance_v5": (
                    PublicEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor
                ),
                "public_expanded_ensemble_robust_adaptive_step_dominance_v6": (
                    PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor
                ),
                "public_bounded_endpoint_ensemble_robust_dominance_v7": (
                    PublicBoundedEndpointEnsembleRobustDenseTransportActor
                ),
                "public_capped_expansion_ensemble_robust_dominance_v8": (
                    PublicCappedExpansionEnsembleRobustDenseTransportActor
                ),
                "public_horizon_consistent_ensemble_robust_dominance_v9": (
                    PublicHorizonConsistentEnsembleRobustDenseTransportActor
                ),
                "public_validated_bridge_ensemble_robust_dominance_v10": (
                    PublicValidatedBridgeEnsembleRobustDenseTransportActor
                ),
                "public_completion_value_ensemble_robust_dominance_v11": (
                    PublicCompletionValueEnsembleRobustDenseTransportActor
                ),
                "public_transient_phase_adaptive_step_dominance_v12": (
                    PublicTransientPhaseAdaptiveStepDenseTransportActor
                ),
                "public_phase_scheduled_adaptive_step_dominance_v13": (
                    PublicPhaseScheduledAdaptiveStepDenseTransportActor
                ),
                "public_mode_conditioned_finite_time_dominance_v14": (
                    PublicModeConditionedCompletionValueDenseTransportActor
                    if mode == 0
                    else PublicModeConditionedAdaptiveStepDenseTransportActor
                ),
                "public_mode_specific_recorded_completion_dominance_v15": (
                    PublicModeSpecificRecordedCompletionDenseTransportActor
                ),
                "public_mode_specific_supported_phase_convex_dominance_v16": (
                    PublicModeSpecificSupportedPhaseConvexDenseTransportActor
                ),
                "public_mode_specific_supported_phase_capped_expansion_dominance_v17": (
                    PublicModeSpecificSupportedPhaseCappedExpansionDenseTransportActor
                ),
                "public_mode_specific_calibrated_authority_dominance_v18": (
                    PublicModeSpecificCalibratedAuthorityDenseTransportActor
                ),
            }[dominance]
            completion_options = {}
            if dominance == "public_completion_value_ensemble_robust_dominance_v11" or (
                dominance == "public_mode_conditioned_finite_time_dominance_v14"
                and mode == 0
            ):
                from genedynamics.solvers.single.cross_ebdm.learning.completion_value import (
                    PublicCompletionValueEnsemble,
                )

                completion_path = _checked(
                    binding.get("completion_value_calibration"),
                    "completion_value_calibration",
                )
                completion_report = json.loads(completion_path.read_text())
                if (
                    completion_report.get("schema_version")
                    != "t3-public-completion-response-calibration-v1"
                    or completion_report.get("passed") is not True
                    or completion_report.get("new_physics_calls") != 0
                ):
                    raise ValueError("passed no-physics completion calibration required")
                instance = binding.get("instance_id")
                expected_response = completion_report["artifacts"][
                    "response_checkpoints"
                ].get(instance)
                if expected_response != binding.get("response_checkpoint"):
                    raise ValueError("response checkpoint differs from completion calibration")
                expected_value = completion_report["artifacts"][
                    "completion_value_checkpoint"
                ]
                if expected_value != binding.get("completion_value_checkpoint"):
                    raise ValueError("completion-value checkpoint differs from calibration")
                completion_value = PublicCompletionValueEnsemble.from_checkpoint(
                    _torch_payload(binding, "completion_value_checkpoint"), device,
                ).eval()
                completion_options = {
                    "completion_value": completion_value,
                    "minimum_admission_gain_controls": completion_report[
                        "calibration"
                    ]["minimum_admission_gain_controls"],
                }
            authority_options = {}
            if dominance == "public_mode_specific_calibrated_authority_dominance_v18":
                authority_options["mode_one_max_blend"] = binding.get(
                    "mode_one_max_blend"
                )
            return actor_type(
                task.spec, policy, bank, task_only_policy=task_only_policy,
                response=response,
                retreat_release_version=binding.get(
                    "retreat_release_version", "legacy"
                ),
                **completion_options, **authority_options, **options,
            )
        from .cross_ebdm.dense import PublicDenseTransportActor

        return PublicDenseTransportActor(
            task.spec, policy, bank,
            retreat_release_version=binding.get(
                "retreat_release_version", "legacy"
            ),
            **options,
        )
    from .cross_ebdm.mobile_dense import PublicLivenessMobileDenseTransportActor
    from .recovery_execution import (
        PublicCompletionDominantRecoveryExecution,
        PublicLivenessRecoveryExecution,
    )

    candidate = PublicLivenessMobileDenseTransportActor(
        task.spec, policy, bank, **options
    )
    completion = binding.get("completion_admission_version")
    if completion not in {None, "public_completion_dominance_v1"}:
        raise ValueError("unknown T4 completion admission version")
    wrapper = (
        PublicCompletionDominantRecoveryExecution
        if completion == "public_completion_dominance_v1"
        else PublicLivenessRecoveryExecution
    )
    return wrapper(
        candidate, task.spec, feedback, mode=mode, seed=options["seed"]
    )


def _model_actor(
    task: Any, binding: dict[str, Any], *, mode: int, device: str, dial: bool
):
    policy, response, bank = _cross_components(task, binding, device)
    if response is None:
        raise ValueError("model-based actor requires response_checkpoint")
    alias = task.spec.alias
    common = dict(
        response=response, mode=mode,
        device=device, maximum_query_distance=float(
            binding.get("maximum_query_distance", 5.0)
        ),
    )
    if alias in {"T1", "T2"}:
        if dial and alias == "T1":
            from .dial import PublicPlannedObstacleDIALMPCActor as Actor
            common["seed"] = int(binding.get("seed", task.seed))
        elif dial:
            from .dial import PublicArmTaskDIALMPCActor as Actor
            common["seed"] = int(binding.get("seed", task.seed))
        elif alias == "T1":
            from .cross_ebdm.obstacle_planned import PublicPlannedObstacleMPCActor as Actor
        else:
            from .cross_ebdm.guide_corridor_v2 import PublicCorridorGuideMPCActor as Actor
        if alias == "T1":
            path = binding.get("known_body_path")
            if path is None:
                raise ValueError("T1 model binding requires known_body_path")
            common["path"] = path
        return Actor(task.spec, policy, bank, **common)
    feedback = _feedback(binding)
    common.update(
        feedback=feedback, prior_path=_prior_path(binding),
        seed=int(binding.get("seed", task.seed)),
    )
    if alias == "T3":
        common["retreat_release_version"] = binding.get(
            "retreat_release_version", "legacy"
        )
        if dial:
            from .dial import PublicContactDIALMPCActor as Actor
        else:
            from genedynamics.solvers.single.response_mpc.contact import (
                PublicResponseMPCActor as Actor,
            )
        return Actor(task.spec, policy, bank, **common)
    from .recovery_execution import PublicLivenessRecoveryExecution
    if dial:
        from .dial import PublicMobileDIALMPCActor as Actor
    else:
        from .cross_ebdm.mobile_dense import (
            PublicLivenessMobileResponseMPCActor as Actor,
        )
    candidate = Actor(task.spec, policy, bank, **common)
    return PublicLivenessRecoveryExecution(
        candidate, task.spec, feedback, mode=mode, seed=common["seed"]
    )


def _public_geometry(task: Any, binding: dict[str, Any], mode: int):
    alias = task.spec.alias
    if alias == "T1":
        from .obstacle_path import PublicObstaclePathPolicy

        path = binding.get("known_body_path")
        if path is None:
            raise ValueError("T1 public geometry requires known_body_path")
        return PublicObstaclePathPolicy(task.spec, modes=[mode], path=path)
    if alias == "T2":
        version = binding.get("reference_version", "legacy")
        if version == "public_predictive_guide_contact_v1":
            from .guide_contact import PublicPredictiveGuideContactPolicy

            return PublicPredictiveGuideContactPolicy(task.spec, mode=mode)
        if version == "public_progress_qualified_guide_contact_v2":
            from .guide_contact import PublicProgressQualifiedGuideContactPolicy

            return PublicProgressQualifiedGuideContactPolicy(task.spec, mode=mode)
        if version == "public_wrench_governed_guide_contact_v3":
            from .guide_contact import PublicWrenchGovernedGuideContactPolicy

            return PublicWrenchGovernedGuideContactPolicy(task.spec, mode=mode)
        if version == "public_low_load_wrench_governed_guide_contact_v4":
            from .guide_contact import PublicLowLoadWrenchGovernedGuideContactPolicy

            return PublicLowLoadWrenchGovernedGuideContactPolicy(
                task.spec, mode=mode,
            )
        if version == "public_low_load_predictive_guide_contact_v5":
            from .guide_contact import PublicLowLoadPredictiveGuideContactPolicy

            return PublicLowLoadPredictiveGuideContactPolicy(task.spec, mode=mode)
        if version == "public_anisotropic_predictive_guide_contact_v6":
            from .guide_contact import PublicAnisotropicPredictiveGuideContactPolicy

            return PublicAnisotropicPredictiveGuideContactPolicy(
                task.spec, mode=mode,
            )
        if version == "public_surface_consistent_guide_contact_v7":
            from .guide_contact import PublicSurfaceConsistentGuideContactPolicy

            return PublicSurfaceConsistentGuideContactPolicy(task.spec, mode=mode)
        if version == "public_channel_centered_guide_contact_v8":
            from .guide_contact import PublicChannelCenteredGuideContactPolicy

            return PublicChannelCenteredGuideContactPolicy(task.spec, mode=mode)
        if version == "public_settled_contact_ramp_v9":
            from .guide_contact import PublicSettledContactRampGuideContactPolicy

            return PublicSettledContactRampGuideContactPolicy(task.spec, mode=mode)
        if version == "public_low_load_settled_contact_v10":
            from .guide_contact import PublicLowLoadSettledGuideContactPolicy

            return PublicLowLoadSettledGuideContactPolicy(task.spec, mode=mode)
        if version == "public_post_outlet_lift_route_v11":
            from .guide_contact import PublicPostOutletLiftGuideContactPolicy

            return PublicPostOutletLiftGuideContactPolicy(task.spec, mode=mode)
        if version == "public_damped_post_outlet_lift_route_v12":
            from .guide_contact import PublicDampedPostOutletLiftGuideContactPolicy

            return PublicDampedPostOutletLiftGuideContactPolicy(
                task.spec, mode=mode,
            )
        if version == "public_bounded_unloading_post_outlet_route_v13":
            from .guide_contact import PublicBoundedUnloadingPostOutletLiftGuideContactPolicy

            return PublicBoundedUnloadingPostOutletLiftGuideContactPolicy(
                task.spec, mode=mode,
            )
        if version == "public_filtered_normal_post_outlet_route_v14":
            from .guide_contact import PublicFilteredNormalPostOutletLiftGuideContactPolicy

            return PublicFilteredNormalPostOutletLiftGuideContactPolicy(
                task.spec, mode=mode,
            )
        if version == "public_hybrid_normal_post_outlet_route_v15":
            from .guide_contact import PublicHybridNormalPostOutletLiftGuideContactPolicy

            return PublicHybridNormalPostOutletLiftGuideContactPolicy(
                task.spec, mode=mode,
            )
        if version in {
            "public_finite_state_guide_contact_v1",
            "public_finite_state_guide_contact_v2",
        }:
            from .guide_contact import public_finite_state_guide_contact_policy

            return public_finite_state_guide_contact_policy(
                task.spec, mode=mode,
            )
        if version in {
            "public_guide_contact_v1", "public_guide_contact_v2",
            "public_guide_contact_v3", "public_guide_contact_v4",
            "public_guide_contact_v5",
            "public_guide_contact_v6",
            "public_guide_contact_v7",
            "public_guide_contact_v8",
            "public_guide_contact_v9",
            "public_guide_contact_v10",
        }:
            from .guide_contact import PublicGuideContactPolicy

            transit_force = (
                2.0 if version in {
                    "public_guide_contact_v2", "public_guide_contact_v6",
                } else None
            )
            return PublicGuideContactPolicy(
                task.spec, mode=mode, transit_force_N=transit_force,
                wrench_governor=version in {
                    "public_guide_contact_v3", "public_guide_contact_v4",
                    "public_guide_contact_v5",
                    "public_guide_contact_v6",
                    "public_guide_contact_v7",
                    "public_guide_contact_v8",
                    "public_guide_contact_v9",
                    "public_guide_contact_v10",
                },
                filtered_wrench_governor=version != "public_guide_contact_v3",
                tangent_speed_limit_mps=(
                    (.0065, .01) if version == "public_guide_contact_v10"
                    else .0075 if version in {
                        "public_guide_contact_v5", "public_guide_contact_v6",
                        "public_guide_contact_v7",
                        "public_guide_contact_v8",
                        "public_guide_contact_v9",
                    } else None
                ),
                normal_force_decay=(
                    ({"public_guide_contact_v7": .8,
                      "public_guide_contact_v8": .9}.get(version))
                ),
                unilateral_transit=version in {
                    "public_guide_contact_v9", "public_guide_contact_v10",
                },
            )
        if version != "legacy":
            raise ValueError(f"unknown T2 public reference version: {version}")
        from .guide_corridor_lift_route import PublicGuideCorridorLiftRoutePolicy

        return PublicGuideCorridorLiftRoutePolicy(task.spec, mode=mode)
    feedback = _feedback(binding)
    seed = int(binding.get("seed", task.seed))
    if alias == "T3":
        retreat_release = binding.get("retreat_release_version", "legacy")
        if retreat_release == "legacy":
            from .mating_feedback import PublicMatingFeedbackPolicy as Actor
        elif retreat_release == "safe_settled_v1":
            from .mating_feedback import PublicLivenessMatingFeedbackPolicy as Actor
        else:
            raise ValueError(
                f"unknown T3 retreat release version: {retreat_release}"
            )
        return Actor.load(
            task.spec, None, feedback=feedback, modes=[mode], seed=seed
        )
    from .mating_feedback import PublicLivenessMatingFeedbackPolicy

    return PublicLivenessMatingFeedbackPolicy.load(
        task.spec, None, feedback=feedback, modes=[mode], seed=seed
    )


def _create_transfer_actor(task: Any, method: str, config: dict[str, Any]):
    """Create one batch-one public actor from a SHA-bound package artifact."""
    mode = int(config.get("mode", getattr(task, "_experiment_mode", 0)))
    device = str(config.get("device", "cpu"))
    if device == "gpu":
        device = "cuda"
    if method == "task_reference":
        from genedynamics.tasks.transfer.teachers import PublicTeacher

        return PublicTeacher(task, mode=mode, recipe=str(config.get("recipe", "default")))
    task_alias = getattr(getattr(task, "spec", None), "alias", None)
    binding = resolve_transfer_binding(config, task_alias=task_alias)
    if binding.get("status") == "predeployment_rejection":
        raise ValueError(
            "predeployment rejection must be scored before creating a task actor"
        )
    shared = getattr(task, "shared_execution", {})
    if task_alias == "T3" and shared:
        binding = dict(binding)
        feedback = dict(binding.get("feedback", {}))
        feedback["insertion_lead_z_m"] = float(shared["insertion_lead_z_m"])
        binding["feedback"] = feedback
        binding["retreat_release_version"] = shared["retreat_release_version"]
    if method in {"cross_ebdm", "task_only"}:
        return _dense_actor(task, binding, mode=mode, device=device)
    if method == "response_mpc":
        return _model_actor(task, binding, mode=mode, device=device, dial=False)
    if method == "dial_transfer":
        return _model_actor(task, binding, mode=mode, device=device, dial=True)
    if method == "public_geometry":
        return _public_geometry(task, binding, mode)
    if method == "public_source_prior":
        alias = task.spec.alias
        if alias in {"T1", "T2"}:
            policy, _, bank = _cross_components(task, binding, device)
            if alias == "T1":
                from .cross_ebdm.obstacle_planned import (
                    PublicPlannedObstacleSourcePriorActor as Actor,
                )
                path = binding.get("known_body_path")
                if path is None:
                    raise ValueError("T1 source prior requires known_body_path")
                return Actor(task.spec, policy, bank, path=path, mode=mode, device=device)
            from .cross_ebdm.guide_corridor_v2 import (
                PublicCorridorGuideSourcePriorActor as Actor,
            )
            return Actor(task.spec, policy, bank, mode=mode, device=device)
        feedback = _feedback(binding)
        prior = _prior_path(binding)
        if prior is None:
            raise ValueError("T3/T4 source prior requires source_prior")
        seed = int(binding.get("seed", task.seed))
        if alias == "T3":
            retreat_release = binding.get("retreat_release_version", "legacy")
            if retreat_release == "legacy":
                from .mating_feedback import PublicMatingFeedbackPolicy as Actor
            elif retreat_release == "safe_settled_v1":
                from .mating_feedback import PublicLivenessMatingFeedbackPolicy as Actor
            else:
                raise ValueError(
                    f"unknown T3 retreat release version: {retreat_release}"
                )
            return Actor.load(
                task.spec, prior, feedback=feedback, modes=[mode], seed=seed
            )
        from .mating_feedback import PublicLivenessMatingFeedbackPolicy

        return PublicLivenessMatingFeedbackPolicy.load(
            task.spec, prior, feedback=feedback, modes=[mode], seed=seed
        )
    if method == "fada":
        alias = task.spec.alias
        if alias == "T2":
            from .fada.guide_corridor_v2 import PublicCorridorGuideFADAActor as Actor
        else:
            # The frozen final T1/T3/T4 checkpoints all carry the explicit
            # 199D public-current-joint chart. JointTaskPolicy also owns the
            # T1 selected path and T4 body-relative anchor declared in their
            # checkpoint metadata. The older 67D planned/mobile wrappers
            # cannot load these checkpoints and are not the validated actor.
            from .fada.joint import PublicJointFADATaskActor as Actor
        return Actor(
            task.spec, _torch_payload(binding, "checkpoint"), mode=mode, device=device
        )
    if method == "adaptsim":
        from .adaptsim import (
            public_guide_checkpoint_actor, public_mating_actor,
            public_mobile_mating_actor,
            public_obstacle_actor,
        )

        if (binding.get("actor") != "adaptsim"
                or binding.get("task") != task_alias
                or task_alias not in {"T1", "T2", "T3", "T4"}
                or binding.get("status") != "instances_ready"
                or not isinstance(binding.get("instance_id"), str)
                or not binding["instance_id"]):
            raise ValueError("AdaptSim actor binding identity differs")
        recipe = json.loads(_checked(binding.get("training_recipe"), "training_recipe").read_text())
        report_path = _checked(binding.get("training_report"), "training_report")
        report = json.loads(report_path.read_text())
        if task_alias == "T1":
            audit = json.loads(_checked(binding.get("training_audit"), "training_audit").read_text())
            return public_obstacle_actor(
                task, _torch_payload(binding, "checkpoint"), recipe, report, audit,
                checkpoint_sha256=binding["checkpoint"]["sha256"],
                report_sha256=binding["training_report"]["sha256"], mode=mode,
            )
        if task_alias == "T2":
            audit = json.loads(_checked(binding.get("training_audit"), "training_audit").read_text())
            return public_guide_checkpoint_actor(
                task, _torch_payload(binding, "checkpoint"), recipe, report, audit,
                checkpoint_sha256=binding["checkpoint"]["sha256"],
                report_sha256=binding["training_report"]["sha256"], mode=mode,
            )
        if task_alias == "T4":
            audit = json.loads(_checked(binding.get("training_audit"), "training_audit").read_text())
            return public_mobile_mating_actor(
                task, _torch_payload(binding, "checkpoint"), recipe, report, audit,
                checkpoint_sha256=binding["checkpoint"]["sha256"],
                report_sha256=binding["training_report"]["sha256"], mode=mode,
                seed=int(binding.get("seed", task.seed)),
            )
        return public_mating_actor(
            task, _torch_payload(binding, "checkpoint"), recipe, report,
            checkpoint_sha256=binding["checkpoint"]["sha256"], mode=mode,
            seed=int(binding.get("seed", task.seed)),
        )
    if method == "asid":
        from .asid import (
            public_guide_checkpoint_actor, public_mating_checkpoint_actor,
            public_mobile_mating_checkpoint_actor,
            public_obstacle_actor,
        )

        if (binding.get("actor") != "asid"
                or binding.get("task") != task_alias
                or task_alias not in {"T1", "T2", "T3", "T4"}
                or binding.get("status") != "instances_ready"
                or not isinstance(binding.get("instance_id"), str)
                or not binding["instance_id"]):
            raise ValueError("ASID actor binding identity differs")
        checkpoint = json.loads(_checked(binding.get("checkpoint"), "checkpoint").read_text())
        recipe = json.loads(_checked(binding.get("training_recipe"), "training_recipe").read_text())
        report_pointer = binding.get("training_report")
        report = json.loads(_checked(report_pointer, "training_report").read_text())
        if task_alias == "T2":
            audit = json.loads(_checked(binding.get("training_audit"), "training_audit").read_text())
            return public_guide_checkpoint_actor(
                task, checkpoint, recipe, report, audit,
                report_pointer=report_pointer, mode=mode,
            )
        if task_alias == "T3":
            audit = json.loads(_checked(binding.get("training_audit"), "training_audit").read_text())
            return public_mating_checkpoint_actor(
                task, checkpoint, recipe, report, audit,
                report_pointer=report_pointer, mode=mode,
                seed=int(binding.get("seed", task.seed)),
            )
        if task_alias == "T4":
            audit = json.loads(_checked(binding.get("training_audit"), "training_audit").read_text())
            return public_mobile_mating_checkpoint_actor(
                task, checkpoint, recipe, report, audit,
                report_pointer=report_pointer, mode=mode,
                seed=int(binding.get("seed", task.seed)),
            )
        return public_obstacle_actor(
            task, checkpoint, recipe, report, report_pointer=report_pointer, mode=mode,
        )
    if method == "acp":
        from .acp import ACPFourTaskActor

        return ACPFourTaskActor(
            task, _torch_payload(binding, "checkpoint"), mode,
            int(binding.get("seed", task.seed)), device=device,
        )
    if method == "human2sim2robot":
        from .human2sim2robot import PublicObjectTrajectoryPPOActor

        return PublicObjectTrajectoryPPOActor(
            task.spec, _torch_payload(binding, "checkpoint"), mode=mode,
            seed=int(binding.get("seed", task.seed)), device=device,
        )
    raise ValueError(f"unsupported transfer method binding: {method!r}")


def create_transfer_actor(task: Any, method: str, config: dict[str, Any]):
    """Create the solver actor, then apply the task-owned shared interface."""
    if getattr(getattr(task, "spec", None), "name", None) == "compliant_keyed_insertion_recovery":
        from genedynamics.tasks.transfer.compliant_socket import FixtureRelativePolicy
        if method != "public_geometry":
            raise ValueError(
                "passive-socket v1 currently admits only public_geometry; "
                "learned methods require a new fixture-observation checkpoint binding"
            )
        if config.get("task_actor_interface") != FixtureRelativePolicy.VERSION:
            raise ValueError("passive socket requires its explicit shared task-frame interface")
        return FixtureRelativePolicy(_create_transfer_actor(task, method, config))
    actor = _create_transfer_actor(task, method, config)
    task_alias = getattr(getattr(task, "spec", None), "alias", None)
    shared = getattr(task, "shared_execution", {}) or {}
    interface = config.get(
        "task_actor_interface", shared.get("terminal_actor_interface"),
    )
    if interface is None:
        return actor
    from .guide_contact import PublicTerminalOwnedGuideActor

    supported = {
        PublicTerminalOwnedGuideActor.TERMINAL_CAPTURE_VERSION,
        PublicTerminalOwnedGuideActor.POST_OUTLET_COMPLETION_VERSION,
        PublicTerminalOwnedGuideActor.CLEARANCE_PROGRESS_VERSION,
        PublicTerminalOwnedGuideActor.TERMINAL_SET_ADMISSION_VERSION,
        PublicTerminalOwnedGuideActor.TERMINAL_TRANSPORT_VERSION,
        PublicTerminalOwnedGuideActor.TERMINAL_STRICT_PROGRESS_VERSION,
        PublicTerminalOwnedGuideActor.HANDOFF_VIABILITY_VERSION,
        PublicTerminalOwnedGuideActor.STAGE_DEBT_VIABILITY_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_VIABILITY_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_LIVENESS_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAPTURE_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
    }
    if task_alias != "T2" or interface not in supported:
        raise ValueError("unknown shared terminal actor interface")
    # The public geometry row is the unwrapped task sentinel.  The terminal
    # interface and its parameters belong to method actors; inheriting the
    # task's shared execution contract must not wrap or reject the sentinel.
    if method == "public_geometry":
        return actor
    parameters = dict(config.get(
        "task_actor_interface_parameters",
        shared.get("terminal_actor_interface_parameters", {}),
    ))
    if parameters and interface not in {
        PublicTerminalOwnedGuideActor.HANDOFF_VIABILITY_VERSION,
        PublicTerminalOwnedGuideActor.STAGE_DEBT_VIABILITY_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_VIABILITY_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_LIVENESS_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAPTURE_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAPTURE_BAND_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_PLANAR_BRAKING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_PRE_DESCENT_SETTLING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_REFERENCE_UNLOADING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_REFERENCE_DEBT_UNLOADING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAUSAL_REFERENCE_UNLOADING_VERSION,
        PublicTerminalOwnedGuideActor.COMPLETION_BUDGET_CAUSAL_TRAVERSE_VERSION,
    }:
        raise ValueError(
            "terminal actor parameters require a declared viability or "
            "completion-budget interface"
        )
    mode = int(config.get("mode", getattr(task, "_experiment_mode", 0)))
    return PublicTerminalOwnedGuideActor(
        actor, task.spec, mode=mode, version=interface, **parameters,
    )


__all__ = [
    "BINDING_VERSION", "MANIFEST_VERSION", "create_transfer_actor",
    "resolve_transfer_binding", "validate_transfer_actor_manifest",
]

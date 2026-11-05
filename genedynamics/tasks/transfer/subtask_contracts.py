"""Frozen semantic contracts for the cross-embodiment transfer benchmark suites.

The contracts describe task meaning and evaluation evidence.  They deliberately
do not contain experiment budgets, devices, seeds, or method parameters.  A
pending contract can be inspected, but cannot be requested as runnable.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]

CONTRACT_PATHS: Mapping[str, str] = {
    "dual_route_carry": (
        "configs/arm/obstacle_transfer/protocol/subtasks/"
        "dual_route_carry_v1.json"
    ),
    "aperture_brake": (
        "configs/arm/obstacle_transfer/protocol/subtasks/"
        "aperture_brake_v1.json"
    ),
    "dual_channel_contact": (
        "configs/arm/guided_contact_alignment/protocol/subtasks/"
        "dual_channel_contact_v1.json"
    ),
    "corner_guided_seating": (
        "configs/arm/guided_contact_alignment/protocol/subtasks/"
        "corner_guided_seating_v1.json"
    ),
    "jam_recovery": (
        "configs/arm/keyed_insertion_recovery/protocol/subtasks/"
        "jam_recovery_v1.json"
    ),
    "compliant_fixture_insertion": (
        "configs/arm/compliant_keyed_insertion_recovery/protocol/subtasks/"
        "compliant_fixture_insertion_v1.json"
    ),
    "stance_insertion": (
        "configs/quadruped/mobile_keyed_insertion_recovery/protocol/subtasks/"
        "stance_insertion_v1.json"
    ),
    "reposition_then_dock": (
        "configs/quadruped/mobile_keyed_insertion_recovery/protocol/subtasks/"
        "reposition_then_dock_v1.json"
    ),
}

FAMILY_SUITES: Mapping[str, tuple[str, str]] = {
    "constrained_object_transport": ("dual_route_carry", "aperture_brake"),
    "guided_contact_alignment": (
        "dual_channel_contact", "corner_guided_seating",
    ),
    "connector_assembly_recovery": (
        "jam_recovery", "compliant_fixture_insertion",
    ),
    "support_coupled_assembly": (
        "stance_insertion", "reposition_then_dock",
    ),
}

RUNNABLE_STATUS = "implemented_contract_frozen"
PENDING_STATUS = "contract_frozen_implementation_pending"
_ALLOWED_STATUSES = {RUNNABLE_STATUS, PENDING_STATUS}
_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "contract_version",
    "semantic_id",
    "family",
    "embodiment",
    "environment",
    "implementation_status",
    "semantics",
    "event_order",
    "interface",
    "evaluation",
    "mechanism_axes",
    "source_qualification",
    "evaluator_evidence",
    "historical_evidence_policy",
}
_FORBIDDEN_ID = re.compile(r"^T[1-4]$", re.IGNORECASE)


@dataclass(frozen=True)
class SubtaskContract:
    """Validated view of one immutable semantic task contract."""

    path: Path
    payload: Mapping[str, Any]

    @property
    def semantic_id(self) -> str:
        return str(self.payload["semantic_id"])

    @property
    def family(self) -> str:
        return str(self.payload["family"])

    @property
    def implementation_status(self) -> str:
        return str(self.payload["implementation_status"])

    @property
    def runnable(self) -> bool:
        return self.implementation_status == RUNNABLE_STATUS


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _require_nonempty_strings(value: Any, location: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{location} entries must be non-empty strings")


def _walk(value: Any, location: str = "contract"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item, f"{location}.{key}"
            yield from _walk(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{location}[{index}]")


def validate_subtask_contract(payload: Mapping[str, Any], *, path: Path) -> None:
    """Reject incomplete, numbered, budget-bearing, or non-resolving contracts."""
    value = _require_mapping(payload, "contract")
    missing = _REQUIRED_TOP_LEVEL - set(value)
    if missing:
        raise ValueError(f"contract is missing required fields: {sorted(missing)}")

    semantic_id = value["semantic_id"]
    if semantic_id not in CONTRACT_PATHS:
        raise ValueError(f"unknown semantic task id: {semantic_id!r}")
    if _FORBIDDEN_ID.fullmatch(str(semantic_id)):
        raise ValueError("numbered T1-T4 identities are historical aliases only")
    expected_path = (ROOT / CONTRACT_PATHS[str(semantic_id)]).resolve()
    if path.resolve() != expected_path:
        raise ValueError(f"semantic task {semantic_id!r} must live at {expected_path}")
    if value["schema_version"] != "transfer-subtask-contract-v1":
        raise ValueError("unsupported subtask contract schema")
    if not re.fullmatch(r"[a-z0-9-]+-v[1-9][0-9]*", str(value["contract_version"])):
        raise ValueError("contract_version must be a semantic, versioned identifier")
    if value["implementation_status"] not in _ALLOWED_STATUSES:
        raise ValueError("unknown implementation_status")

    semantics = _require_mapping(value["semantics"], "semantics")
    if not isinstance(semantics.get("goal"), str) or not semantics["goal"].strip():
        raise ValueError("semantics.goal must be a non-empty string")
    _require_nonempty_strings(semantics.get("excludes"), "semantics.excludes")

    events = value["event_order"]
    if not isinstance(events, list) or len(events) < 2:
        raise ValueError("event_order must contain at least two ordered events")
    event_ids = []
    for index, event in enumerate(events):
        event = _require_mapping(event, f"event_order[{index}]")
        if set(event) != {"id", "evidence"}:
            raise ValueError("each event requires exactly id and evidence")
        event_ids.append(event["id"])
        _require_nonempty_strings(event["evidence"], f"event_order[{index}].evidence")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event ids must be unique within a contract")

    interface = _require_mapping(value["interface"], "interface")
    observations = _require_mapping(interface.get("observations"), "interface.observations")
    _require_nonempty_strings(observations.get("public"), "interface.observations.public")
    _require_nonempty_strings(observations.get("hidden"), "interface.observations.hidden")
    authority = _require_mapping(interface.get("action_authority"), "interface.action_authority")
    for owner in ("method", "public_execution", "evaluator_only"):
        _require_nonempty_strings(authority.get(owner), f"interface.action_authority.{owner}")

    evaluation = _require_mapping(value["evaluation"], "evaluation")
    for field in ("success", "violations", "anti_bypass"):
        _require_nonempty_strings(evaluation.get(field), f"evaluation.{field}")

    axes = value["mechanism_axes"]
    if not isinstance(axes, list) or not axes:
        raise ValueError("mechanism_axes must be a non-empty list")
    axis_names = []
    for index, axis in enumerate(axes):
        axis = _require_mapping(axis, f"mechanism_axes[{index}]")
        if set(axis) != {"name", "public", "train", "development", "heldout"}:
            raise ValueError(
                "each mechanism axis requires name/public/train/development/heldout"
            )
        axis_names.append(axis["name"])
        for split in ("train", "development", "heldout"):
            if not isinstance(axis[split], list) or not axis[split]:
                raise ValueError(f"mechanism axis {axis['name']}.{split} is empty")
    if len(set(axis_names)) != len(axis_names):
        raise ValueError("mechanism axis names must be unique")

    source = _require_mapping(value["source_qualification"], "source_qualification")
    if set(source) != {"status", "requirements", "evidence_resolution"}:
        raise ValueError("source_qualification fields differ from the frozen schema")
    _require_nonempty_strings(source["requirements"], "source_qualification.requirements")
    if not isinstance(source["evidence_resolution"], str) or not source["evidence_resolution"]:
        raise ValueError("source_qualification.evidence_resolution is required")

    evaluator = _require_mapping(value["evaluator_evidence"], "evaluator_evidence")
    required_evaluator = {"status", "resolution", "implementation_path", "required_signals"}
    if set(evaluator) != required_evaluator:
        raise ValueError("evaluator_evidence fields differ from the frozen schema")
    _require_nonempty_strings(evaluator["required_signals"], "evaluator_evidence.required_signals")

    for key, item, location in _walk(value):
        normalized = str(key).lower().replace("-", "_")
        if "budget" in normalized and "target" in normalized:
            raise ValueError(f"target budgets do not belong in task contracts: {location}")
        if normalized.endswith("_path"):
            if not isinstance(item, str) or not item:
                raise ValueError(f"{location} must be a repository-relative path")
            resolved = (ROOT / item).resolve()
            if not resolved.is_relative_to(ROOT.resolve()) or not resolved.exists():
                raise ValueError(f"contract path does not resolve: {item}")

    pending = value["implementation_status"] == PENDING_STATUS
    evaluator_pending = evaluator["status"] == PENDING_STATUS
    if pending != evaluator_pending:
        raise ValueError("pending task and evaluator status must agree")


def load_subtask_contract(
    semantic_id: str, *, require_runnable: bool = False,
) -> SubtaskContract:
    """Load and validate one contract, optionally rejecting pending suites."""
    try:
        relative = CONTRACT_PATHS[semantic_id]
    except KeyError as exc:
        raise KeyError(f"unknown semantic subtask: {semantic_id}") from exc
    path = ROOT / relative
    payload = json.loads(path.read_text())
    validate_subtask_contract(payload, path=path)
    contract = SubtaskContract(path=path, payload=payload)
    if require_runnable and not contract.runnable:
        raise RuntimeError(
            f"subtask {semantic_id!r} is {contract.implementation_status}; "
            "physics, source skill, and evaluator must be implemented and "
            "qualified before execution"
        )
    return contract


def load_family_suite(
    family: str, *, require_runnable: bool = False,
) -> tuple[SubtaskContract, SubtaskContract]:
    """Load the two frozen semantic suites that make up one task family."""
    try:
        semantic_ids = FAMILY_SUITES[family]
    except KeyError as exc:
        raise KeyError(f"unknown transfer task family: {family}") from exc
    return tuple(
        load_subtask_contract(name, require_runnable=require_runnable)
        for name in semantic_ids
    )


def load_all_subtask_contracts() -> tuple[SubtaskContract, ...]:
    return tuple(load_subtask_contract(name) for name in CONTRACT_PATHS)

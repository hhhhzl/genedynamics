"""Matched treatment definitions for the paper's three primary ablations.

The treatment changes live here instead of being inferred from historical
checkpoint directory names.  Training and evaluation code must bind the same
``shared`` contract for every row and persist :func:`treatment_metadata` in the
checkpoint/report.  Historical ``band_only`` and ``ungated_full`` artifacts
are semantic predecessors only; they were not produced under this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


LEGACY_CONTRACT_VERSION = "keyed-insertion-clean-ablation-v1"
CONTRACT_VERSION = "cross-embodiment-clean-ablation-v2"
SHARED_LOCK_KEYS = (
    "warm_start",
    "data",
    "optimizer",
    "action_authority",
    "recovery",
    "runtime_selector",
    "checkpoint_selection",
)


@dataclass(frozen=True)
class AblationTreatment:
    """The only fields allowed to differ from the full ICDT treatment."""

    calibration: str
    response_weight: float
    task_band: bool
    anchoring_gate: str

    def __post_init__(self) -> None:
        if self.calibration not in {"target_calibrated", "nominal_uncalibrated"}:
            raise ValueError("unknown response calibration role")
        if not math.isfinite(self.response_weight) or self.response_weight < 0:
            raise ValueError("finite nonnegative response weight required")
        if not self.task_band:
            raise ValueError("the clean paper ablations retain the task band")
        if self.anchoring_gate not in {"selective", "unconditional"}:
            raise ValueError("unknown anchoring-gate treatment")


TREATMENTS = {
    "full": AblationTreatment(
        calibration="target_calibrated",
        response_weight=1.0,
        task_band=True,
        anchoring_gate="selective",
    ),
    "no_calibration": AblationTreatment(
        calibration="nominal_uncalibrated",
        response_weight=1.0,
        task_band=True,
        anchoring_gate="selective",
    ),
    "no_response_anchoring": AblationTreatment(
        calibration="target_calibrated",
        response_weight=0.0,
        task_band=True,
        anchoring_gate="selective",
    ),
    "unconditional_response_anchoring": AblationTreatment(
        calibration="target_calibrated",
        response_weight=1.0,
        task_band=True,
        anchoring_gate="unconditional",
    ),
}

EXPECTED_TREATMENT_DIFFERENCES = {
    "no_calibration": ("calibration",),
    "no_response_anchoring": ("response_weight",),
    "unconditional_response_anchoring": ("anchoring_gate",),
}


def treatment(name: str) -> AblationTreatment:
    """Return one declared treatment and reject legacy/ambiguous labels."""
    try:
        return TREATMENTS[name]
    except KeyError as error:
        raise ValueError(f"unknown clean ablation treatment: {name}") from error


def treatment_metadata(
    name: str, *, contract_version: str = CONTRACT_VERSION,
) -> dict[str, Any]:
    """Metadata that every trained checkpoint and evaluation must persist."""
    return {
        "contract_version": contract_version,
        "treatment": name,
        **asdict(treatment(name)),
        "treatment_only": True,
    }


def treatment_trainer_class(name: str):
    """Resolve the objective implementation without importing torch at metadata time."""
    treatment(name)
    from .gated_transport import PublicRecoveryGatedTransportTrainer

    # One implementation preserves every task/progress term. The explicit
    # objective flag below changes only the response mask.
    return PublicRecoveryGatedTransportTrainer


def treatment_objective_kwargs(name: str) -> dict[str, float | bool]:
    """Return the treatment-specific objective arguments.

    The task term and hinge band remain inside both trainer implementations.
    A zero response weight therefore removes only response anchoring.
    """
    value = treatment(name)
    return {
        "response_weight": value.response_weight,
        "apply_anchoring_gate": value.anchoring_gate == "selective",
    }


def _different_fields(left: AblationTreatment, right: AblationTreatment) -> tuple[str, ...]:
    return tuple(
        key for key in asdict(left)
        if getattr(left, key) != getattr(right, key)
    )


def validate_contract(document: Mapping[str, Any]) -> None:
    """Validate treatment isolation and the shared execution/training lock."""
    version = document.get("schema_version")
    if version not in {CONTRACT_VERSION, LEGACY_CONTRACT_VERSION}:
        raise ValueError("clean ablation contract version differs")
    task = document.get("task")
    if (version == LEGACY_CONTRACT_VERSION
            and task != "keyed_insertion_recovery"):
        raise ValueError("legacy clean ablation contract is for keyed insertion")
    if version == CONTRACT_VERSION and (
            not isinstance(task, str) or not task
            or document.get("objective_family") not in {
                "response_conditioned_transport", "recovery_transport",
            }):
        raise ValueError("semantic task and objective family are required")
    shared = document.get("shared")
    if not isinstance(shared, Mapping) or tuple(shared) != SHARED_LOCK_KEYS:
        raise ValueError("shared ablation lock keys or order differ")
    if any(not isinstance(shared[key], Mapping) or not shared[key] for key in SHARED_LOCK_KEYS):
        raise ValueError("every shared ablation lock must be a nonempty mapping")
    rows = document.get("treatments")
    if not isinstance(rows, Mapping) or set(rows) != set(TREATMENTS):
        raise ValueError("clean ablation treatment roster differs")
    for name, expected in TREATMENTS.items():
        actual = rows[name]
        if not isinstance(actual, Mapping):
            raise ValueError(f"{name} treatment metadata is not a mapping")
        declared = {
            key: actual.get(key)
            for key in ("calibration", "response_weight", "task_band", "anchoring_gate")
        }
        if declared != asdict(expected):
            raise ValueError(f"{name} treatment semantics differ")
        if tuple(actual.get("only_difference_from_full", ())) != (
            () if name == "full" else EXPECTED_TREATMENT_DIFFERENCES[name]
        ):
            raise ValueError(f"{name} allowed treatment difference differs")
        if name != "full" and _different_fields(TREATMENTS["full"], expected) != (
            EXPECTED_TREATMENT_DIFFERENCES[name]
        ):
            raise ValueError(f"{name} changes more than one treatment field")
    compatibility = document.get("legacy_compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("explicit legacy compatibility mapping required")
    if any(row.get("evidence_reusable") is not False for row in compatibility.values()):
        raise ValueError("legacy ablation evidence must not enter the clean comparison")


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON treatment contract."""
    document = json.loads(Path(path).read_text())
    validate_contract(document)
    return document


__all__ = [
    "AblationTreatment",
    "CONTRACT_VERSION",
    "LEGACY_CONTRACT_VERSION",
    "EXPECTED_TREATMENT_DIFFERENCES",
    "SHARED_LOCK_KEYS",
    "TREATMENTS",
    "load_contract",
    "treatment",
    "treatment_metadata",
    "treatment_objective_kwargs",
    "treatment_trainer_class",
    "validate_contract",
]

"""Read-only force/event diagnostics for keyed insertion episode evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ForceViolation:
    control_index: int
    total_force: bool
    lateral_force: bool
    force_peak_N: float
    lateral_force_peak_N: float

    @property
    def channel(self) -> str:
        labels = []
        if self.total_force:
            labels.append("total")
        if self.lateral_force:
            labels.append("lateral")
        return "+".join(labels)


def first_force_violation(
    traces: Sequence[Mapping[str, Any]],
    *,
    force_limit_N: float,
    lateral_force_limit_N: float,
) -> ForceViolation | None:
    """Reproduce the evaluator's strict force-limit predicate."""
    limits = (float(force_limit_N), float(lateral_force_limit_N))
    if any(not math.isfinite(value) or value <= 0 for value in limits):
        raise ValueError("finite positive force limits required")
    for index, trace in enumerate(traces):
        force = float(trace["force_peak"])
        lateral = float(trace["lateral_force_peak"])
        total_exceeded = force > limits[0]
        lateral_exceeded = lateral > limits[1]
        if total_exceeded or lateral_exceeded:
            return ForceViolation(
                control_index=index,
                total_force=total_exceeded,
                lateral_force=lateral_exceeded,
                force_peak_N=force,
                lateral_force_peak_N=lateral,
            )
    return None


def _scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[0] if value else default
    return default if value is None else value


def diagnose_episode(
    result: Mapping[str, Any], evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Join one result with its audited trace and expose the failure event."""
    records = evidence.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("one physical record per independent episode required")
    record = records[0]
    traces = record.get("traces")
    diagnostics = record.get("control_diagnostics")
    if not isinstance(traces, list) or not traces:
        raise ValueError("nonempty physical traces required")
    if not isinstance(diagnostics, list) or len(diagnostics) != len(traces):
        raise ValueError("one controller diagnostic per physical control required")
    task = evidence.get("task_spec", {})
    if task.get("name") != "keyed_insertion_recovery":
        raise ValueError("keyed insertion evidence required")
    config = task.get("config", {})
    violation = first_force_violation(
        traces,
        force_limit_N=config["force_limit"],
        lateral_force_limit_N=config["lateral_force_limit"],
    )
    outcome = str(result["outcome"])
    if (outcome == "overforce") != (violation is not None):
        raise ValueError("stored outcome and trace force predicate disagree")
    index = len(traces) - 1 if violation is None else violation.control_index
    diagnostic = diagnostics[index]
    previous = diagnostics[index - 1] if index else {}
    admission_count = int(diagnostic.get("transport_admission_controls", 0))
    previous_admission_count = int(previous.get("transport_admission_controls", 0))
    memory = (
        diagnostic.get("base", {}).get("observer", {}).get("memory", {})
    )
    trace = traces[index]
    request = result.get("request", {})
    if result.get("record_id") != request.get("request_id"):
        raise ValueError("result record and request identities differ")
    return {
        "record_id": result["record_id"],
        "instance_id": result.get("instance_id"),
        "outcome": outcome,
        "requested_mode": int(request["requested_mode"]),
        "mechanism_group": request["mechanism_group"],
        "force_violation": None if violation is None else asdict(violation),
        "force_violation_channel": None if violation is None else violation.channel,
        "event": {
            "control_index": index,
            "phase": int(_scalar(memory.get("phase"), -1)),
            "retreating": bool(_scalar(diagnostic.get("base", {}).get("retreating"), False)),
            "axial_support": bool(trace["axial_support"]),
            "loading": bool(trace["loading"]),
        },
        "transport": {
            "admissions_before_or_at_event": admission_count,
            "admitted_on_event_control": admission_count > previous_admission_count,
            "selected_blend": float(diagnostic.get("last_task_dominance_blend", 0.0)),
        },
        "command": {
            "applied_xy_speed_mps": math.hypot(
                float(trace["applied_action"][0]),
                float(trace["applied_action"][1]),
            ),
            "applied_z_speed_mps": float(trace["applied_action"][2]),
            "reference_lead_xy_m": math.hypot(
                float(trace["public_reference_lead"][0]),
                float(trace["public_reference_lead"][1]),
            ),
        },
    }


def _counts(rows: Iterable[Mapping[str, Any]], field) -> dict[str, int]:
    values = Counter(field(row) for row in rows)
    return {str(key): values[key] for key in sorted(values, key=lambda item: str(item))}


def summarize_paired_diagnoses(
    ours: Sequence[Mapping[str, Any]],
    task_only: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize identical-instance/request ICDT and task-only episodes."""
    def identity(row):
        return row.get("instance_id"), row["record_id"]

    ours_by_id = {identity(row): row for row in ours}
    task_by_id = {identity(row): row for row in task_only}
    if len(ours_by_id) != len(ours) or len(task_by_id) != len(task_only):
        raise ValueError("duplicate paired episode identity")
    if set(ours_by_id) != set(task_by_id):
        raise ValueError("ICDT and task-only paired identities differ")
    pairs = [(ours_by_id[key], task_by_id[key]) for key in sorted(ours_by_id)]
    failures = [row for row, _ in pairs if row["outcome"] == "overforce"]
    regressions = [
        row for row, control in pairs
        if row["outcome"] == "overforce" and control["outcome"] == "success"
    ]
    gains = [
        row for row, control in pairs
        if row["outcome"] == "success" and control["outcome"] == "overforce"
    ]
    common_failures = [
        row for row, control in pairs
        if row["outcome"] == control["outcome"] == "overforce"
    ]
    paired_outcomes = _counts(
        pairs, lambda pair: f"{pair[0]['outcome']}|{pair[1]['outcome']}",
    )
    return {
        "paired_episodes": len(pairs),
        "ours_outcomes": _counts((row for row, _ in pairs), lambda row: row["outcome"]),
        "paired_outcomes_ours_task_only": paired_outcomes,
        "overforce": {
            "episodes": len(failures),
            "common_with_task_only": len(common_failures),
            "ours_only_regressions": len(regressions),
            "ours_only_gains": len(gains),
            "channels": _counts(failures, lambda row: row["force_violation_channel"]),
            "phases": _counts(failures, lambda row: row["event"]["phase"]),
            "retreating": _counts(failures, lambda row: row["event"]["retreating"]),
            "regression_modes": _counts(regressions, lambda row: row["requested_mode"]),
            "regression_mechanisms": _counts(regressions, lambda row: row["mechanism_group"]),
            "regressions_with_prior_transport": sum(
                row["transport"]["admissions_before_or_at_event"] > 0
                for row in regressions
            ),
            "regressions_admitted_on_event_control": sum(
                row["transport"]["admitted_on_event_control"]
                for row in regressions
            ),
        },
    }


__all__ = [
    "ForceViolation",
    "diagnose_episode",
    "first_force_violation",
    "summarize_paired_diagnoses",
]

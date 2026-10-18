"""Independent strict-contact predicates for the T2 guide benchmark.

The policy never receives ``axial_support``.  It is physical evaluator truth
already stored in every contact trace.  The interval is defined from the first
loaded axial contact to the first mode-specific public outlet crossing, so the
predicate does not depend on a controller phase or method implementation.
"""
from __future__ import annotations

import numpy as np

from .evaluation import evaluate_episode


STRICT_GUIDE_VERSION = "t2-strict-guide-contact-v2"


def _maximum_false_run(values: np.ndarray) -> int:
    maximum = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def evaluate_strict_guide_episode(
    record,
    spec,
    requested_mode: int,
    *,
    minimum_contact_ratio: float = 0.5,
    maximum_contact_loss_seconds: float = 0.15,
    outlet_x_m: float = 0.455,
    outlet_y_m: float = -0.181,
):
    """Evaluate legacy task success plus contact-preserving guide traversal.

    Contact chatter is handled by a coverage requirement and a maximum gap.
    A free-space route that crosses the outlet before establishing guide
    support is rejected regardless of its later terminal force hold.
    """
    if spec.alias != "T2" or requested_mode not in (0, 1):
        raise ValueError("strict guide evaluation requires T2 and mode 0/1")
    if not 0.0 < minimum_contact_ratio <= 1.0:
        raise ValueError("minimum_contact_ratio must be in (0,1]")
    if maximum_contact_loss_seconds < 0.0:
        raise ValueError("maximum_contact_loss_seconds must be nonnegative")

    legacy = evaluate_episode(record, spec)
    rows = [row for row in record.traces if row["valid"]]
    positions = np.asarray([row["tcp_position"] for row in rows], dtype=float)
    contact = np.asarray(
        [bool(row["axial_support"]) and float(row["normal_force"]) > spec.config.contact_threshold for row in rows],
        dtype=bool,
    )
    outlet = positions[:, 0] > outlet_x_m if requested_mode == 0 else positions[:, 1] > outlet_y_m
    contact_indices = np.flatnonzero(contact)
    outlet_indices = np.flatnonzero(outlet)
    first_contact = int(contact_indices[0]) if len(contact_indices) else None
    first_outlet = int(outlet_indices[0]) if len(outlet_indices) else None
    contact_before_outlet = first_contact is not None and first_outlet is not None and first_contact <= first_outlet

    if contact_before_outlet:
        segment = contact[first_contact : first_outlet + 1]
        ratio = float(segment.mean())
        maximum_loss_steps = _maximum_false_run(segment)
        traversal_steps = int(len(segment))
        contact_steps = int(segment.sum())
    else:
        ratio = 0.0
        maximum_loss_steps = None
        traversal_steps = 0
        contact_steps = 0

    maximum_allowed_loss_steps = int(np.floor(maximum_contact_loss_seconds / spec.config.control_dt + 1e-12))
    coverage_passed = contact_before_outlet and ratio >= minimum_contact_ratio
    gap_passed = contact_before_outlet and maximum_loss_steps <= maximum_allowed_loss_steps
    strict_success = bool(legacy["success"] and coverage_passed and gap_passed)
    if not legacy["success"]:
        outcome = legacy["outcome"]
    elif not contact_before_outlet:
        outcome = "guide_contact_not_established_before_outlet"
    elif not coverage_passed:
        outcome = "guide_contact_coverage"
    elif not gap_passed:
        outcome = "guide_contact_loss"
    else:
        outcome = "success"

    return {
        **legacy,
        "evaluation_version": STRICT_GUIDE_VERSION,
        "legacy_success": legacy["success"],
        "success": strict_success,
        "task_cost": 0.0 if strict_success else 1.0,
        "outcome": outcome,
        "requested_mode": int(requested_mode),
        "first_contact_step": first_contact,
        "first_outlet_step": first_outlet,
        "contact_before_outlet": bool(contact_before_outlet),
        "traversal_steps": traversal_steps,
        "contact_steps": contact_steps,
        "continuous_guide_contact_ratio": ratio,
        "maximum_contact_loss_steps": maximum_loss_steps,
        "maximum_contact_loss_seconds": None if maximum_loss_steps is None else maximum_loss_steps * spec.config.control_dt,
        "minimum_contact_ratio_required": float(minimum_contact_ratio),
        "maximum_contact_loss_seconds_allowed": float(maximum_contact_loss_seconds),
        "outlet_threshold": {"mode_0_x_m": float(outlet_x_m), "mode_1_y_m": float(outlet_y_m)},
    }


def evaluate_strict_guide_records(records, spec, requested_modes, **kwargs):
    modes = tuple(int(mode) for mode in requested_modes)
    if len(records) != len(modes):
        raise ValueError("one requested mode is required per episode")
    rows = [evaluate_strict_guide_episode(record, spec, mode, **kwargs) for record, mode in zip(records, modes)]
    return {
        "scope": "independent strict-contact T2 evaluator; every requested episode remains in the denominator",
        "requested_episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None,
        "constraint_violations": sum(row["outcome"] not in {"success", "timeout", "incomplete"} for row in rows),
        "outcomes": {key: sum(row["outcome"] == key for row in rows) for key in sorted({row["outcome"] for row in rows})},
        "episodes": rows,
    }

"""Independent strict-contact predicates for the T2 guide benchmark.

The policy never receives ``axial_support``.  It is physical evaluator truth
already stored in every contact trace.  The interval is defined from the first
loaded axial contact to the first mode-specific public outlet crossing, so the
predicate does not depend on a controller phase or method implementation.
"""
from __future__ import annotations

import numpy as np

from .evaluation import aggregate_episode_rows, evaluate_episode


STRICT_GUIDE_VERSION = "t2-strict-guide-contact-mode-correct-v4"
CORNER_GUIDE_VERSION = "corner-guided-seating-substep-v1"


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
    """Evaluate mode-correct physical success and guide contact preservation.

    Contact chatter is handled by a coverage requirement and a maximum gap.
    A free-space route that crosses the outlet before establishing guide
    support is rejected regardless of its later terminal force hold.
    """
    if spec.alias != "T2" or requested_mode not in (0, 1):
        raise ValueError("strict guide evaluation requires T2 and mode 0/1")
    if requested_mode != record.requested_mode:
        raise ValueError("requested mode must match the episode record")
    if not 0.0 < minimum_contact_ratio <= 1.0:
        raise ValueError("minimum_contact_ratio must be in (0,1]")
    if maximum_contact_loss_seconds < 0.0:
        raise ValueError("maximum_contact_loss_seconds must be nonnegative")

    legacy = evaluate_episode(record, spec)
    rows = [row for row in record.traces if row["valid"]]
    positions = np.asarray(
        [row["tcp_position"] for row in rows], dtype=float
    ).reshape(-1, 3)
    endpoint_contact = np.asarray(
        [bool(row["axial_support"]) and float(row["normal_force"]) > spec.config.contact_threshold for row in rows],
        dtype=bool,
    )
    outlet = positions[:, 0] > outlet_x_m if requested_mode == 0 else positions[:, 1] > outlet_y_m
    exact_physics_evidence = bool(rows) and all(
        all(name in row for name in (
            "axial_support_substeps", "normal_force_substeps",
            "tcp_position_substeps",
        ))
        for row in rows
    )
    if exact_physics_evidence:
        widths = {len(np.asarray(row["axial_support_substeps"])) for row in rows}
        if widths != {spec.config.substeps}:
            raise ValueError("T2 physics-rate contact evidence has the wrong substep count")
        axial = np.concatenate([
            np.asarray(row["axial_support_substeps"], dtype=bool) for row in rows
        ])
        normal = np.concatenate([
            np.asarray(row["normal_force_substeps"], dtype=float) for row in rows
        ])
        physics_positions = np.concatenate([
            np.asarray(row["tcp_position_substeps"], dtype=float).reshape(-1, 3)
            for row in rows
        ])
        if axial.shape != normal.shape or physics_positions.shape != (len(axial), 3):
            raise ValueError("T2 physics-rate contact evidence shapes differ")
        contact = axial & (normal > spec.config.contact_threshold)
        physics_outlet = (
            physics_positions[:, 0] > outlet_x_m
            if requested_mode == 0
            else physics_positions[:, 1] > outlet_y_m
        )
        contact_indices = np.flatnonzero(contact)
        outlet_indices_physics = np.flatnonzero(physics_outlet)
        first_contact_physics = int(contact_indices[0]) if len(contact_indices) else None
        first_outlet_physics = int(outlet_indices_physics[0]) if len(outlet_indices_physics) else None
        first_contact = (
            first_contact_physics // spec.config.substeps
            if first_contact_physics is not None else None
        )
        first_outlet = (
            first_outlet_physics // spec.config.substeps
            if first_outlet_physics is not None else None
        )
        contact_before_outlet = (
            first_contact_physics is not None
            and first_outlet_physics is not None
            and first_contact_physics <= first_outlet_physics
        )
        if contact_before_outlet:
            segment = contact[first_contact_physics:first_outlet_physics + 1]
        resolution_seconds = spec.config.physics_dt
    else:
        contact = endpoint_contact
        contact_indices = np.flatnonzero(contact)
        outlet_indices_physics = np.flatnonzero(outlet)
        first_contact = int(contact_indices[0]) if len(contact_indices) else None
        first_outlet = int(outlet_indices_physics[0]) if len(outlet_indices_physics) else None
        first_contact_physics = first_contact
        first_outlet_physics = first_outlet
        contact_before_outlet = first_contact is not None and first_outlet is not None and first_contact <= first_outlet
        if contact_before_outlet:
            segment = contact[first_contact:first_outlet + 1]
        resolution_seconds = spec.config.control_dt
    if contact_before_outlet:
        ratio = float(segment.mean())
        maximum_loss_steps = _maximum_false_run(segment)
        traversal_physics_steps = int(len(segment))
        contact_physics_steps = int(segment.sum())
        traversal_steps = int(first_outlet - first_contact + 1)
        contact_steps = int(endpoint_contact[first_contact:first_outlet + 1].sum())
    else:
        ratio = 0.0
        maximum_loss_steps = None
        traversal_steps = 0
        contact_steps = 0
        traversal_physics_steps = 0
        contact_physics_steps = 0

    maximum_allowed_loss_steps = int(np.floor(maximum_contact_loss_seconds / resolution_seconds + 1e-12))
    coverage_passed = contact_before_outlet and ratio >= minimum_contact_ratio
    gap_passed = contact_before_outlet and maximum_loss_steps <= maximum_allowed_loss_steps
    strict_contact_success = bool(coverage_passed and gap_passed)
    strict_success = bool(legacy["success"] and strict_contact_success)
    if not legacy["physical_success"]:
        outcome = legacy["outcome"]
    elif not contact_before_outlet:
        outcome = "guide_contact_not_established_before_outlet"
    elif not coverage_passed:
        outcome = "guide_contact_coverage"
    elif not gap_passed:
        outcome = "guide_contact_loss"
    elif not legacy["mode_match"]:
        outcome = "mode_mismatch"
    else:
        outcome = "success"

    return {
        **legacy,
        "evaluation_version": STRICT_GUIDE_VERSION,
        "legacy_success": legacy["physical_success"],
        "strict_contact_success": strict_contact_success,
        "success": strict_success,
        "constrained_success": strict_success,
        "task_cost": 0.0 if strict_success else 1.0,
        "capped_normalized_completion_time": (
            legacy["capped_normalized_completion_time"] if strict_success else 1.0
        ),
        "outcome": outcome,
        "constraint_violation": bool(
            legacy["constraint_violation"]
            or outcome in {
                "guide_contact_not_established_before_outlet",
                "guide_contact_coverage",
                "guide_contact_loss",
            }
        ),
        "requested_mode": int(requested_mode),
        "first_contact_step": first_contact,
        "first_outlet_step": first_outlet,
        "first_contact_physics_step": first_contact_physics,
        "first_outlet_physics_step": first_outlet_physics,
        "contact_before_outlet": bool(contact_before_outlet),
        "traversal_steps": traversal_steps,
        "contact_steps": contact_steps,
        "traversal_physics_steps": traversal_physics_steps,
        "contact_physics_steps": contact_physics_steps,
        "contact_evidence_resolution_seconds": float(resolution_seconds),
        "exact_physics_contact_evidence": exact_physics_evidence,
        "continuous_guide_contact_ratio": ratio,
        "maximum_contact_loss_steps": maximum_loss_steps,
        "maximum_contact_loss_seconds": None if maximum_loss_steps is None else maximum_loss_steps * resolution_seconds,
        "minimum_contact_ratio_required": float(minimum_contact_ratio),
        "maximum_contact_loss_seconds_allowed": float(maximum_contact_loss_seconds),
        "outlet_threshold": {"mode_0_x_m": float(outlet_x_m), "mode_1_y_m": float(outlet_y_m)},
    }


def evaluate_strict_guide_records(records, spec, requested_modes, **kwargs):
    modes = tuple(int(mode) for mode in requested_modes)
    if len(records) != len(modes):
        raise ValueError("one requested mode is required per episode")
    rows = [evaluate_strict_guide_episode(record, spec, mode, **kwargs) for record, mode in zip(records, modes)]
    report = aggregate_episode_rows(
        rows,
        scope=(
            "independent strict-contact T2 evaluator; every requested episode "
            "remains in the denominator"
        ),
    )
    report["evaluation_version"] = STRICT_GUIDE_VERSION
    return report


def evaluate_corner_guide_episode(record,spec):
    """Require ordered L-corridor traversal and physical end-stop seating."""
    if spec.name!="corner_guided_seating" or spec.config.guide_profile!="planar_corner_v1":
        raise ValueError("corner evaluator requires the semantic planar-corner task")
    legacy=evaluate_episode(record,spec)
    rows=[row for row in record.traces if row["valid"]]
    required=("axial_support_substeps","normal_force_substeps","tcp_position_substeps",
              "joint_speed_substeps","force_peak_substeps","lateral_force_substeps",
              "contact_region_substeps")
    exact=bool(rows) and all(all(name in row for name in required) for row in rows)
    if not exact:
        return {**legacy,"evaluation_version":CORNER_GUIDE_VERSION,"success":False,
                "constrained_success":False,"physical_success":False,
                "task_cost":1.,"outcome":"missing_physics_substep_evidence",
                "constraint_violation":True,"group":"corner_route",
                "exact_physics_contact_evidence":False}
    widths={len(np.asarray(row["axial_support_substeps"])) for row in rows}
    if widths!={spec.config.substeps}:
        raise ValueError("corner-guide evidence has the wrong substep count")
    axial=np.concatenate([np.asarray(row["axial_support_substeps"],bool) for row in rows])
    normal=np.concatenate([np.asarray(row["normal_force_substeps"],float) for row in rows])
    positions=np.concatenate([np.asarray(row["tcp_position_substeps"],float).reshape(-1,3) for row in rows])
    joint_speed=np.concatenate([np.asarray(row["joint_speed_substeps"],float) for row in rows])
    force_peak=np.concatenate([np.asarray(row["force_peak_substeps"],float) for row in rows])
    lateral=np.concatenate([np.asarray(row["lateral_force_substeps"],float) for row in rows])
    regions=np.concatenate([np.asarray(row["contact_region_substeps"],np.uint8) for row in rows])
    if (not np.isfinite(np.r_[normal,positions.ravel(),joint_speed,force_peak,lateral]).all()
            or positions.shape!=(len(axial),3) or any(len(value)!=len(axial) for value in (normal,joint_speed,force_peak,lateral,regions))):
        raise ValueError("corner-guide physics evidence is nonfinite or shape-inconsistent")
    contact=axial&(normal>spec.config.contact_threshold)&((regions&1)!=0)
    def first(mask,after=-1):
        found=np.flatnonzero(mask&(np.arange(len(mask))>after))
        return int(found[0]) if len(found) else None
    first_contact=first(contact)
    first_segment=first((positions[:,0]>=.435)&(positions[:,1]>=-.211)&(positions[:,1]<=-.189),
                        -1 if first_contact is None else first_contact-1)
    corner_exit=first((positions[:,1]>=-.185)&(positions[:,0]>=.429)&(positions[:,0]<=.451),
                      -1 if first_segment is None else first_segment)
    second_segment=first((positions[:,1]>=-.142)&(positions[:,0]>=.429)&(positions[:,0]<=.451),
                         -1 if corner_exit is None else corner_exit)
    endstop=first((regions&4)!=0,-1 if second_segment is None else second_segment-1)
    ordered=(first_contact is not None and first_segment is not None and corner_exit is not None
             and second_segment is not None and endstop is not None
             and first_contact<=first_segment<corner_exit<second_segment<=endstop)
    def coverage(start,stop):
        if start is None or stop is None or stop<start:return (0.,None)
        segment=contact[start:stop+1]
        return float(segment.mean()),_maximum_false_run(segment)
    first_ratio,first_gap=coverage(first_contact,first_segment)
    second_ratio,second_gap=coverage(corner_exit,second_segment)
    gap_limit=int(np.floor(.15/spec.config.physics_dt+1e-12))
    coverage_ok=(first_ratio>=.5 and second_ratio>=.5 and first_gap is not None and second_gap is not None
                 and first_gap<=gap_limit and second_gap<=gap_limit)
    bypass=False
    if ordered:
        before=positions[first_contact:first_segment+1]
        after=positions[first_segment:second_segment+1]
        bypass=(np.any((before[:,1]<-.212)|(before[:,1]>-.188))
                or np.any((after[:,0]<.428)|(after[:,0]>.452))
                or np.any(positions[first_contact:endstop+1,2]>.594))
    dwell=max(1,round(.3/spec.config.physics_dt))
    terminal_start=max(0,len(positions)-dwell)
    velocity=np.linalg.norm(np.diff(positions,axis=0,prepend=positions[:1]),axis=1)/spec.config.physics_dt
    goal=np.asarray(spec.config.goal)
    terminal=(len(positions)>=dwell and np.all(np.linalg.norm(positions[terminal_start:]-goal,axis=1)<=spec.config.position_tolerance+1e-4)
              and np.all(np.abs(normal[terminal_start:]-spec.config.desired_force)<2.5)
              and np.all(force_peak[terminal_start:]<=spec.config.force_limit)
              and np.all(lateral[terminal_start:]<=spec.config.lateral_force_limit)
              and np.all(velocity[terminal_start:]<.02)
              and np.any((regions[terminal_start:]&4)!=0))
    success=bool(legacy["physical_success"] and ordered and coverage_ok and not bypass and terminal)
    if not legacy["physical_success"]: outcome=legacy["outcome"]
    elif not ordered: outcome="corner_event_order"
    elif not coverage_ok: outcome="corner_contact_coverage"
    elif bypass: outcome="corner_bypass"
    elif not terminal: outcome="endstop_not_stable"
    else: outcome="success"
    return {**legacy,"evaluation_version":CORNER_GUIDE_VERSION,"success":success,
            "constrained_success":success,"task_cost":0. if success else 1.,
            "capped_normalized_completion_time":legacy["capped_normalized_completion_time"] if success else 1.,
            "outcome":outcome,"constraint_violation":bool(legacy["constraint_violation"] or not success),
            "group":"corner_route","exact_physics_contact_evidence":True,
            "event_physics_steps":{"loaded_contact":first_contact,"first_segment":first_segment,
                                     "corner_exit":corner_exit,"second_segment":second_segment,"endstop":endstop},
            "first_segment_contact_ratio":first_ratio,"second_segment_contact_ratio":second_ratio,
            "first_segment_maximum_gap_steps":first_gap,"second_segment_maximum_gap_steps":second_gap,
            "anti_bypass_passed":bool(ordered and not bypass),"terminal_stability_passed":bool(terminal)}


def evaluate_corner_guide_records(records,spec):
    rows=[evaluate_corner_guide_episode(record,spec) for record in records]
    report=aggregate_episode_rows(rows,scope="independent semantic corner-guide evaluator; every requested episode remains in the denominator")
    report["evaluation_version"]=CORNER_GUIDE_VERSION
    return report

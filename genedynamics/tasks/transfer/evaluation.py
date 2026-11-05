"""Independent trajectory predicates; every requested reset remains a row.

Backend status is compared with the recomputed physical predicate but never
used as a success label. Partial, rejected and unexecuted attempts cost one.
This module imports no physics backend or termination implementation.
"""
from __future__ import annotations

import numpy as np

from .contact_reference import classify_contact_modes
from .routes import observed_route_modes


EVALUATION_VERSION = "cross-embodiment-mode-correct-v2"


def observed_episode_mode(record, spec):
    """Classify the executed trajectory without using its requested mode.

    Both classifiers use public geometric events already defined for the
    source datasets. Missing evidence is unknown (-1); conflicting crossing
    evidence is ambiguous (2). Neither label satisfies a requested mode.
    """
    rows = [row for row in record.traces if row["valid"]]
    if not rows:
        return -1
    positions = np.asarray([row["tcp_position"] for row in rows], dtype=float)
    valid = np.isfinite(positions).all(axis=-1)
    if spec.family == "obstacle":
        return int(observed_route_modes(positions[None, None], valid[None, None])[0])
    forces = np.asarray([row["normal_force"] for row in rows], dtype=float)
    valid &= np.isfinite(forces)
    return int(classify_contact_modes(
        positions[None], forces[None], valid[None], insertion=spec.config.insertion
    )[0])


def _angle(rotation, target):
    relative = target @ np.asarray(rotation).T
    vector = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                       relative[1, 0] - relative[0, 1]]) * .5
    return float(np.arctan2(np.linalg.norm(vector), np.clip((np.trace(relative) - 1) / 2, -1., 1.)))


def evaluate_episode(record, spec):
    cfg = spec.config
    requested_mode = int(record.requested_mode)
    if requested_mode not in (0, 1) or requested_mode != record.requested_mode:
        raise ValueError("requested mode must be 0 or 1")
    hold, loss, previous_seen, terminal = 0, 0, False, 0
    rows = record.traces
    reasons = {0: "incomplete", 1: "success", 2: "collision", 3: "joint_limit", 4: "speed_limit",
               5: "timeout", 6: "nonfinite", 7: "source_support", 8: "axial_support_loss",
               9: "overforce", 10: "body_instability", 11: "wbc_failure", 12: "support_loss"}
    peak, lateral_peak, status_mismatches, valid_steps = 0., 0., 0, 0
    target_position = np.asarray(cfg.goal_position if spec.family == "obstacle" else cfg.goal)
    yaw = cfg.goal_yaw if spec.family == "obstacle" else 0.
    c, s = np.cos(yaw), np.sin(yaw)
    target_rotation = np.array([[c, s, 0.], [s, -c, 0.], [0., 0., -1.]])
    final_error, final_angle = None, None
    for index, row in enumerate(rows):
        if not row["valid"]:
            continue
        if terminal:
            raise ValueError("episode contains physical transitions after termination")
        valid_steps += 1
        final_error = float(np.linalg.norm(np.asarray(row["tcp_position"]) - target_position))
        final_angle = _angle(row["tcp_rotation"], target_rotation)
        if spec.family == "obstacle":
            reached = final_error <= cfg.position_tolerance and final_angle <= cfg.orientation_tolerance
            hold = hold + 1 if reached else 0
            terminal = 5 if valid_steps >= cfg.timeout_steps else 0
            if hold >= cfg.hold_steps:
                terminal = 1
            if row["joint_speed_peak"] > cfg.joint_speed_limit:
                terminal = 4
            if row["joint_limit"]:
                terminal = 3
            if row["clearance"] < -cfg.collision_tolerance:
                terminal = 2
            if not row["finite"]:
                terminal = 6
        else:
            force = float(row["normal_force"])
            peak = max(peak, float(row["force_peak"]))
            lateral_peak = max(lateral_peak, float(row["lateral_force_peak"]))
            loading = bool(row["loading"])
            seen = loading and bool(row["axial_support"]) and force > cfg.contact_threshold
            history = (hold > 0 or loss > 0) if cfg.robot == "go2_z1" else (previous_seen or loss > 0)
            near_seat = row["tcp_position"][2] < cfg.goal[2] + cfg.position_tolerance if cfg.insertion else True
            lost = history and loading and not seen and near_seat
            loss = loss + 1 if lost else 0
            previous_seen = seen
            reached = (final_error < cfg.position_tolerance and final_angle < cfg.orientation_tolerance
                       and abs(force - cfg.desired_force) < 2.5)
            hold = hold + 1 if reached else 0
            terminal = 5 if valid_steps >= cfg.timeout_steps else 0
            if hold >= round(.3 / cfg.control_dt):
                terminal = 1
            if loss >= round(cfg.contact_loss_seconds / cfg.control_dt):
                terminal = 8
            if row["force_peak"] > cfg.force_limit or row["lateral_force_peak"] > cfg.lateral_force_limit:
                terminal = 9
            if row["forbidden_collision"]:
                terminal = 2
            if not row["finite"]:
                terminal = 6
            if cfg.robot == "go2_z1":
                if row["body_height"] < .15 or row["base_tilt"] > .5:
                    terminal = 10
                if row["wbc_residual"] > .01:
                    terminal = 11
                if index > 50 and row["support_count"] < 2:
                    terminal = 12
            if row["joint_limit"]:
                terminal = 3
            if row["joint_speed_peak"] > 3.:
                terminal = 4
        status_mismatches += int(int(row["backend_status"]) != terminal)
    physical_success = terminal == 1 and record.state != "rejected"
    observed_mode = observed_episode_mode(record, spec)
    mode_match = observed_mode in (0, 1) and observed_mode == requested_mode
    success = physical_success and mode_match
    reason = (
        record.rejection_reason or "rejected"
        if record.state == "rejected" else reasons[terminal]
    )
    if physical_success and not mode_match:
        reason = "mode_mismatch"
    constraint_statuses = {2, 3, 4, 6, 8, 9, 10, 11, 12}
    normalized_time = (
        min(valid_steps, cfg.timeout_steps) / cfg.timeout_steps
        if success else 1.0
    )
    return {
        "evaluation_version": EVALUATION_VERSION,
        "sample_id": record.sample_id, "seed": record.seed, "generation": record.generation,
        "success": bool(success), "status": terminal, "outcome": reason,
        "physical_success": bool(physical_success),
        "observed_mode": observed_mode, "mode_match": bool(mode_match),
        "constrained_success": bool(success),
        "rejection_reason": record.rejection_reason,
        "physical_control_steps": valid_steps,
        "physical_seconds": valid_steps * cfg.control_dt,
        "preparation_control_steps": record.preparation_control_steps,
        "preparation_seconds": record.preparation_control_steps * cfg.control_dt,
        "task_cost": 0. if success else 1.,
        "final_position_error_m": final_error, "final_orientation_error_rad": final_angle,
        "force_peak_N": peak, "lateral_force_peak_N": lateral_peak,
        "status_mismatches": status_mismatches,
        "requested_mode": requested_mode,
        "group": f"mode_{requested_mode}",
        "constraint_violation": bool(terminal in constraint_statuses),
        "capped_normalized_completion_time": float(normalized_time),
        "predeployment_rejection": bool(record.state == "rejected"),
        "controller": record.controller,
    }


def evaluate_aperture_episode(record, spec):
    """Recompute whole-envelope crossing and braking from physics-rate traces."""
    from .aperture import object_pose_and_twist, object_vertices

    cfg = spec.config
    legacy = evaluate_episode(record, spec)
    positions, rotations, twists, frame_contacts, frame_masks = [], [], [], [], []
    for row in record.traces:
        if not row["valid"]:
            continue
        required = (
            "tcp_position_substeps", "tcp_rotation_substeps",
            "tcp_twist_substeps", "frame_contact_substeps",
            "frame_contact_mask_substeps",
        )
        if any(name not in row for name in required):
            raise ValueError("aperture episode lacks physics-rate evaluator evidence")
        positions.extend(np.asarray(row["tcp_position_substeps"], dtype=float))
        rotations.extend(np.asarray(row["tcp_rotation_substeps"], dtype=float))
        twists.extend(np.asarray(row["tcp_twist_substeps"], dtype=float))
        frame_contacts.extend(np.asarray(row["frame_contact_substeps"], dtype=bool))
        frame_masks.extend(np.asarray(row["frame_contact_mask_substeps"], dtype=np.uint8))
    if not positions:
        return {
            **legacy, "evaluation_version": "semantic-aperture-brake-v1",
            "success": False, "constrained_success": False,
            "task_cost": 1., "capped_normalized_completion_time": 1.,
            "outcome": record.rejection_reason or "no_physical_transition",
            "constraint_violation": True, "exact_physics_aperture_evidence": False,
        }
    positions = np.asarray(positions)
    rotations = np.asarray(rotations)
    twists = np.asarray(twists)
    frame_contacts = np.asarray(frame_contacts)
    frame_masks = np.asarray(frame_masks)
    envelopes = np.asarray([
        object_vertices(position, rotation)
        for position, rotation in zip(positions, rotations)
    ])
    object_states = [
        object_pose_and_twist(position, rotation, twist)
        for position, rotation, twist in zip(positions, rotations, twists)
    ]
    object_positions = np.asarray([value[0] for value in object_states])
    object_twists = np.asarray([value[2] for value in object_states])
    front = envelopes[:, :, 0].max(axis=1)
    rear = envelopes[:, :, 0].min(axis=1)
    plane = cfg.aperture_plane_x

    def first(mask, *, start=0):
        indices = np.flatnonzero(np.asarray(mask) & (np.arange(len(front)) >= start))
        return int(indices[0]) if len(indices) else None

    approach = first((front >= plane - .05) & (rear < plane))
    target_yaw = cfg.goal_yaw
    c, s = np.cos(target_yaw), np.sin(target_yaw)
    target_rotation = np.asarray([[c, s, 0.], [s, -c, 0.], [0., 0., -1.]])
    angles = np.asarray([_angle(rotation, target_rotation) for rotation in rotations])
    inside_opening = (
        (envelopes[:, :, 1] >= cfg.aperture_center_y - cfg.aperture_half_width).all(axis=1)
        & (envelopes[:, :, 1] <= cfg.aperture_center_y + cfg.aperture_half_width).all(axis=1)
        & (envelopes[:, :, 2] >= cfg.aperture_center_z - cfg.aperture_half_height).all(axis=1)
        & (envelopes[:, :, 2] <= cfg.aperture_center_z + cfg.aperture_half_height).all(axis=1)
    )
    aligned = first(
        (front < plane - cfg.aperture_frame_thickness)
        & (angles <= cfg.aperture_alignment_tolerance)
        & inside_opening,
        start=0 if approach is None else approach,
    )
    whole = first(
        rear > plane + cfg.aperture_frame_thickness,
        start=0 if aligned is None else aligned,
    )
    crossing = (
        (front >= plane - cfg.aperture_frame_thickness)
        & (rear <= plane + cfg.aperture_frame_thickness)
    )
    crossing_inside = bool(crossing.any() and inside_opening[crossing].all())
    frame_clear = bool(not frame_contacts.any())
    speeds = np.linalg.norm(object_twists[:, :3], axis=1)
    in_exit = (
        (object_positions[:, 0] >= cfg.aperture_exit_zone[0])
        & (object_positions[:, 0] <= cfg.aperture_exit_zone[1])
    )
    brake = first(
        in_exit & (speeds <= cfg.aperture_brake_speed),
        start=0 if whole is None else whole,
    )
    overshoot = bool(
        brake is None and np.any(object_positions[:, 0] > cfg.aperture_exit_zone[1])
    )
    requested_mode = int(record.requested_mode)
    mode_match = False
    if approach is not None:
        offset = object_positions[approach, 1] - cfg.aperture_center_y
        mode_match = bool(offset <= 0. if requested_mode == 0 else offset >= 0.)
    ordered = bool(
        approach is not None and aligned is not None and whole is not None
        and brake is not None and approach <= aligned < whole <= brake
    )
    terminal = bool(legacy["physical_success"])
    success = bool(
        ordered and crossing_inside and frame_clear and not overshoot
        and terminal and mode_match
    )
    if not frame_clear:
        outcome = "aperture_frame_contact"
    elif not crossing_inside:
        outcome = "object_envelope_bypass"
    elif approach is None:
        outcome = "approach_missing"
    elif aligned is None:
        outcome = "orientation_not_aligned"
    elif whole is None:
        outcome = "whole_object_not_crossed"
    elif overshoot:
        outcome = "exit_zone_overshoot"
    elif brake is None:
        outcome = "exit_braking_missing"
    elif not terminal:
        outcome = legacy["outcome"]
    elif not mode_match:
        outcome = "mode_mismatch"
    else:
        outcome = "success"
    return {
        **legacy,
        "evaluation_version": "semantic-aperture-brake-v1",
        "success": success, "constrained_success": success,
        "physical_success": terminal, "mode_match": mode_match,
        "observed_mode": requested_mode if mode_match else -1,
        "task_cost": 0. if success else 1.,
        "capped_normalized_completion_time": (
            legacy["physical_control_steps"] / cfg.timeout_steps if success else 1.
        ),
        "outcome": outcome,
        "constraint_violation": bool(not success),
        "exact_physics_aperture_evidence": True,
        "frame_contact": bool(frame_contacts.any()),
        "frame_contact_mask": int(np.bitwise_or.reduce(frame_masks, initial=0)),
        "crossing_envelope_inside_opening": crossing_inside,
        "exit_zone_overshoot": overshoot,
        "event_physics_steps": {
            "approach_entered": approach,
            "orientation_aligned": aligned,
            "whole_object_crossed_aperture": whole,
            "exit_braking_completed": brake,
            "terminal_hold_completed": len(positions) - 1 if terminal else None,
        },
        "minimum_exit_speed_mps": float(speeds[in_exit].min()) if in_exit.any() else None,
        "maximum_object_x_m": float(object_positions[:, 0].max()),
    }


def evaluate_aperture_records(records, spec):
    rows = [evaluate_aperture_episode(record, spec) for record in records]
    report = aggregate_episode_rows(
        rows,
        scope=(
            "independent semantic whole-envelope aperture evaluator; every "
            "requested episode remains in the denominator"
        ),
    )
    report["evaluation_version"] = "semantic-aperture-brake-v1"
    return report


def aggregate_episode_rows(rows, *, scope):
    count = len(rows)
    successes = sum(row["success"] for row in rows)
    violations = sum(row["constraint_violation"] for row in rows)
    rejections = sum(row["predeployment_rejection"] for row in rows)
    groups = sorted({row["group"] for row in rows})
    group_success = {
        group: sum(row["success"] for row in rows if row["group"] == group)
        / sum(row["group"] == group for row in rows)
        for group in groups
    }
    return {
        "evaluation_version": EVALUATION_VERSION,
        "scope": scope,
        "requested_episodes": count, "successes": successes,
        "success_rate": successes / count if count else None,
        "constrained_success_rate": successes / count if count else None,
        "constraint_violations": violations,
        "constraint_violation_rate": violations / count if count else None,
        "capped_normalized_completion_time": (
            sum(row["capped_normalized_completion_time"] for row in rows) / count
            if count else None
        ),
        "worst_group_constrained_success": min(group_success.values()) if group_success else None,
        "group_constrained_success": group_success,
        "predeployment_rejections": rejections,
        "predeployment_rejection_rate": rejections / count if count else None,
        "mean_task_cost": sum(row["task_cost"] for row in rows) / count if count else None,
        "status_mismatches": sum(row["status_mismatches"] for row in rows),
        "outcomes": {key: sum(row["outcome"] == key for row in rows) for key in sorted({r["outcome"] for r in rows})},
        "episodes": rows,
    }


def evaluate_records(records, spec):
    from .compliant_socket import ENVIRONMENT_NAME, evaluate_compliant_socket_records
    if spec.name in {ENVIRONMENT_NAME,"compliant_fixture_insertion"}:
        return evaluate_compliant_socket_records(records, spec)
    if spec.name == "corner_guided_seating":
        from .guide_evaluation import evaluate_corner_guide_records
        return evaluate_corner_guide_records(records,spec)
    if spec.name == "aperture_brake":
        return evaluate_aperture_records(records, spec)
    if spec.alias == "T2":
        from .guide_evaluation import evaluate_strict_guide_records

        return evaluate_strict_guide_records(
            records, spec, [record.requested_mode for record in records]
        )
    rows = [evaluate_episode(record, spec) for record in records]
    return aggregate_episode_rows(
        rows,
        scope=(
            "independent CPU evaluator; every requested episode remains in "
            "the denominator"
        ),
    )

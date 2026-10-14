"""Independent trajectory predicates; every requested reset remains a row.

Backend status is compared with the recomputed physical predicate but never
used as a success label. Partial, rejected and unexecuted attempts cost one.
This module imports no physics backend or termination implementation.
"""
from __future__ import annotations

import math
import numpy as np


def _angle(rotation, target):
    relative = target @ np.asarray(rotation).T
    vector = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                       relative[1, 0] - relative[0, 1]]) * .5
    return float(np.arctan2(np.linalg.norm(vector), np.clip((np.trace(relative) - 1) / 2, -1., 1.)))


def evaluate_episode(record, spec):
    cfg = spec.config
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
    success = terminal == 1 and record.state != "rejected"
    reason = "rejected" if record.state == "rejected" else reasons[terminal]
    return {
        "sample_id": record.sample_id, "seed": record.seed, "generation": record.generation,
        "success": bool(success), "status": terminal, "outcome": reason,
        "rejection_reason": record.rejection_reason,
        "physical_control_steps": valid_steps,
        "physical_seconds": valid_steps * cfg.control_dt,
        "preparation_control_steps": record.preparation_control_steps,
        "preparation_seconds": record.preparation_control_steps * cfg.control_dt,
        "task_cost": 0. if success else 1.,
        "final_position_error_m": final_error, "final_orientation_error_rad": final_angle,
        "force_peak_N": peak, "lateral_force_peak_N": lateral_peak,
        "status_mismatches": status_mismatches,
        "controller": record.controller,
    }


def evaluate_records(records, spec):
    rows = [evaluate_episode(record, spec) for record in records]
    count = len(rows)
    successes = sum(row["success"] for row in rows)
    return {
        "scope": "independent CPU development evaluator; no native-paper or final-benchmark qualification",
        "requested_episodes": count, "successes": successes,
        "success_rate": successes / count if count else None,
        "mean_task_cost": sum(row["task_cost"] for row in rows) / count if count else None,
        "status_mismatches": sum(row["status_mismatches"] for row in rows),
        "outcomes": {key: sum(row["outcome"] == key for row in rows) for key in sorted({r["outcome"] for r in rows})},
        "episodes": rows,
    }

#!/usr/bin/env python3
"""Release the three-subtask obstacle-family formal comparison after evidence gates."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/arm/obstacle_transfer/protocol/three_subtask_formal_comparison_v1.json"
OUTPUT = ROOT / "results/arm/obstacle_transfer/validation/three_subtask_formal_release_v1"
METHODS = ("cross_ebdm", "task_only", "adaptsim", "asid", "response_mpc")
DUAL_REPORTS = {
    "cross_ebdm": "results/arm/obstacle_transfer/main/cross_ebdm/confirmation/independent_main_v2/instance_*/report.json",
    "task_only": "results/arm/obstacle_transfer/baseline/task_only/confirmation/independent_main_v2/instance_*/report.json",
    "adaptsim": "results/arm/obstacle_transfer/baseline/adaptsim/confirmation/independent_main_v2/independent_*/report.json",
    "asid": "results/arm/obstacle_transfer/baseline/asid/confirmation/independent_main_v2/independent_*/report.json",
    "response_mpc": "results/arm/obstacle_transfer/baseline/response_mpc/confirmation/independent_main_v2/instance_*/report.json",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(value, label):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} requires an exact pointer")
    path = ROOT / value["path"]
    if not path.is_file() or digest(path) != value["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def pointer(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def validate_pilot(path, task):
    report = json.loads(path.read_text())
    if (report.get("status") != "complete_development_only"
            or report.get("task") != task
            or tuple(report.get("methods", ())) != METHODS
            or report.get("formal_performance_evidence") is not False
            or report.get("all_evidence_reaudited") is not True
            or report.get("pairing", {}).get("passed") is not True):
        raise ValueError(f"{task} development pilot failed its release gate")
    summary = report["summary"]
    if (summary["cross_ebdm"]["episodes"] != 2
            or summary["cross_ebdm"]["constrained_success_rate"] != 1.
            or any(summary[name]["episodes"] != 2 for name in METHODS)):
        raise ValueError(f"{task} lacks paired full-method development evidence")
    return summary


def validate_formal_plan(path, task):
    plan = json.loads(path.read_text())
    if (plan.get("status") != "frozen_unexecuted"
            or plan.get("task") != task
            or tuple(plan.get("methods", ())) != METHODS
            or plan.get("policy", {}).get("formal_performance_evidence") is not True
            or plan.get("policy", {}).get("failure_replacement") is not False
            or len(plan.get("requests", ())) != 10
            or sum(row["mode"] == 0 for row in plan["requests"]) != 5
            or len({row["seed"] for row in plan["requests"]}) != 10):
        raise ValueError(f"{task} formal plan is not frozen and balanced")
    source = checked(plan["source_qualification"], f"{task} source qualification")
    if json.loads(source.read_text()).get("passed") is not True:
        raise ValueError(f"{task} source qualification failed")


def dual_route_summary():
    summary = {}
    artifacts = {}
    for method, pattern in DUAL_REPORTS.items():
        paths = sorted(ROOT.glob(pattern))
        if len(paths) != 8:
            raise ValueError(f"{method} dual-route formal instance count differs")
        rows = []
        for path in paths:
            report = json.loads(path.read_text())
            if (report.get("status") != "complete"
                    or report.get("scope") != "independent formal main table"
                    or report.get("method") != method
                    or report.get("evidence_audits_passed") is not True
                    or report.get("all_failures_retained") is not True
                    or report.get("replacement_requests_allowed") is not False
                    or report.get("counts", {}).get("requested") != 40
                    or report.get("counts", {}).get("execution_errors") != 0):
                raise ValueError(f"{method} dual-route report failed: {path}")
            rows.extend(report["results"])
        summary[method] = {
            "episodes": len(rows),
            "constrained_success_rate": sum(row["success"] for row in rows) / len(rows),
            "constraint_violation_rate": sum(row["constraint_violation"] for row in rows) / len(rows),
            "failure_penalized_normalized_time": statistics.mean(
                row["capped_normalized_completion_time"] for row in rows),
        }
        artifacts[method] = [pointer(path) for path in paths]
    return summary, artifacts


def main():
    config = json.loads(CONFIG.read_text())
    if (config.get("version") != "obstacle-family-three-subtask-formal-comparison-v1"
            or config.get("status") != "frozen_pending_release_gate"
            or tuple(config.get("methods", ())) != METHODS
            or len(config.get("metrics", ())) != 4):
        raise ValueError("obstacle-family release protocol differs")
    learning_path = checked(
        config["learning_evidence"]["aperture_response_training"],
        "response training report",
    )
    learning = json.loads(learning_path.read_text())
    if (learning.get("status") != "complete_development_only"
            or learning.get("all_evidence_reaudited") is not True
            or learning.get("response_model_used_for_selection") is not True
            or learning.get("fixed_gain_posthoc_selection") is not False
            or learning["budget_curve"]["60"]["task_success"] != 1.
            or learning["budget_curve"]["120"]["task_success"] != 1.):
        raise ValueError("learned response transport did not pass release gates")

    task_summaries = {}
    for task in ("aperture_brake", "sequential_aperture_reorientation"):
        cell = config["task_protocols"][task]
        pilot = checked(cell["development_pilot"], f"{task} pilot")
        formal_plan = checked(cell["formal_plan"], f"{task} formal plan")
        task_summaries[task] = validate_pilot(pilot, task)
        validate_formal_plan(formal_plan, task)

    dual_summary, dual_artifacts = dual_route_summary()
    task_summaries["dual_route_carry"] = dual_summary
    OUTPUT.mkdir(parents=True, exist_ok=False)
    report = {
        "version": "obstacle-family-three-subtask-formal-release-v1",
        "status": "released_for_formal_execution",
        "protocol": pointer(CONFIG),
        "release_requirements": config["release_requirements"],
        "development_and_existing_formal_summary": task_summaries,
        "dual_route_formal_artifacts": dual_artifacts,
        "new_formal_plans": {
            task: config["task_protocols"][task]["formal_plan"]
            for task in ("aperture_brake", "sequential_aperture_reorientation")
        },
        "new_formal_resets_opened": 0,
        "new_formal_episodes_released": 100,
        "execution_order": config["execution"]["formal_task_order"],
        "output": config["output"],
        "limitations": {
            "aperture_clean_ablations_selected_same_residual": True,
            "sequential_adaptsim_faster_in_two-seed_pilot": True,
            "release_claim": "protocol admission, not completed new formal performance"
        }
    }
    destination = OUTPUT / "report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "new_formal_episodes_released": 100,
        "development_summary": task_summaries,
        "report": str(destination.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()

"""Task-owned resolver for historical independent-table actor configs.

The configs remain immutable legacy artifacts.  Callers may dispatch with a
semantic task name; conversion to T1--T4 happens only while reading them.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .main_table_evaluation import ROOT, TASK_ROOTS
from .main_table_statistics import FIXED_METHODS, TRUE_TRAINING_INSTANCES
from .task_identity import legacy_task_id


def config_paths(task: str, method: str, revision: str = "v1") -> list[Path]:
    if revision not in {"v1", "v2"}:
        raise ValueError("unknown independent main-table revision")
    legacy = legacy_task_id(task)
    base = ROOT / TASK_ROOTS[legacy] / "confirmation" / f"independent_main_{revision}"
    if method in FIXED_METHODS:
        return [base / f"{method}_fixed.json"]
    if method in {"asid", "adaptsim"}:
        return [base / f"{method}_cohort.json"]
    return [
        base / f"{method}_instance_{index:02d}.json"
        for index in range(1, TRUE_TRAINING_INSTANCES + 1)
    ]


def instance_config(cell: dict, instance_id: str) -> dict:
    result = deepcopy(cell)
    if "instance_ids" in result:
        if instance_id not in result["instance_ids"]:
            raise ValueError("formal actor instance is absent from the cell cohort")
        result["instance_id"] = instance_id
        for key in ("output", "plan_output"):
            result[key] = result[key].replace("{instance_id}", instance_id)
    elif result.get("instance_id") != instance_id:
        raise ValueError("fixed actor identity differs")
    return result


def configs(task: str, method: str, revision: str = "v1") -> list[tuple[Path, dict]]:
    paths = config_paths(task, method, revision)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("formal cell config roster pending")
    if method in {"asid", "adaptsim"}:
        cell = json.loads(paths[0].read_text())
        ids = cell.get("instance_ids", ())
        if len(ids) != TRUE_TRAINING_INSTANCES or len(set(ids)) != len(ids):
            raise ValueError("direct baseline instance roster differs")
        return [(paths[0], instance_config(cell, instance)) for instance in ids]
    entries = [(path, json.loads(path.read_text())) for path in paths]
    expected = (
        ["fixed"] if method in FIXED_METHODS else
        [f"instance_{index:02d}" for index in range(1, TRUE_TRAINING_INSTANCES + 1)]
    )
    if [config.get("instance_id") for _, config in entries] != expected:
        raise ValueError("formal actor instance roster differs")
    return entries


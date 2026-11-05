"""Collect immutable, hash-bound ACP demonstrations from public teachers."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any

import numpy as np
import yaml

from genedynamics.core.artifacts import atomic_json
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.learning.datasets.transfer.target import (
    FOUR_TASK_VERSION,
    PublicTargetTrajectory,
)
from genedynamics.tasks.transfer.demonstrators.acp_collection import (
    ACPTeacherExecution,
    audit_evidence_bytes,
    evidence_bytes,
    execute_public_acp_teacher,
)


VERSION = "acp-public-teacher-collection-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(
    public: dict[str, np.ndarray], requests19: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    arrays = {
        **{f"public.{name}": value for name, value in public.items()},
        "requests": requests19,
    }
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".acp-teacher-", suffix=".npz",
    )
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _execution_arrays(row: ACPTeacherExecution) -> dict[str, np.ndarray]:
    return {
        **{f"public.{name}": value for name, value in row.public.items()},
        "teacher_requests19": row.requests19,
        "adapted_actions13": row.actions13,
    }


def _exact_replay(
    first: ACPTeacherExecution, second: ACPTeacherExecution,
) -> dict[str, Any]:
    first_arrays, second_arrays = _execution_arrays(first), _execution_arrays(second)
    names_equal = set(first_arrays) == set(second_arrays)
    array_equal = names_equal and all(
        np.array_equal(first_arrays[name], second_arrays[name])
        for name in first_arrays
    )
    evaluation_equal = _json_ready(first.evaluation) == _json_ready(second.evaluation)
    evidence_equal = evidence_bytes(first.evidence) == evidence_bytes(second.evidence)
    return {
        "passed": bool(array_equal and evaluation_equal and evidence_equal),
        "array_names_equal": bool(names_equal),
        "arrays_exact": bool(array_equal),
        "evaluation_exact": bool(evaluation_equal),
        "evidence_exact": bool(evidence_equal),
        "accounting": _json_ready(second.accounting),
        "wall_seconds": float(second.wall_seconds),
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "version", "task", "teacher_config", "n_steps", "scalar_stiffness",
        "require_exact_replay", "episodes", "output",
    }
    if set(config) != required or config.get("version") != VERSION:
        raise ValueError("explicit ACP public-teacher collection v1 config required")
    if config["task"] not in {"T1", "T2", "T3", "T4"}:
        raise ValueError("unknown ACP teacher collection task")
    episodes = config["episodes"]
    if len(episodes) != 8:
        raise ValueError("ACP collection requires exactly eight independent episodes")
    expected = {(mode, purpose) for mode in (0, 1) for purpose in ("fit", "validation")}
    observed = {(int(row["mode"]), str(row["purpose"])) for row in episodes}
    if observed != expected:
        raise ValueError("ACP collection must contain both splits in both modes")
    for mode in (0, 1):
        fit = sum(int(row["mode"]) == mode and row["purpose"] == "fit" for row in episodes)
        validation = sum(
            int(row["mode"]) == mode and row["purpose"] == "validation"
            for row in episodes
        )
        if (fit, validation) != (3, 1):
            raise ValueError("ACP collection requires 3 fit + 1 validation per mode")
    seeds = [int(row["seed"]) for row in episodes]
    samples = [int(row["sample_id"]) for row in episodes]
    if len(set(seeds)) != len(seeds) or len(set(samples)) != len(samples):
        raise ValueError("ACP teacher episodes require independent seeds and sample ids")


def collection_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Seal every indirect file that defines teacher execution and labels."""
    teacher_config = Path(config["teacher_config"])
    raw = yaml.safe_load(teacher_config.read_text())
    base = (teacher_config.parent / raw["base"]).resolve()
    resolved = ExperimentConfig.from_yaml(teacher_config)
    pointer = resolved.method_params["actor_binding"]["manifest"]
    actor_manifest = Path(pointer["path"])
    actual_manifest_sha = _sha256(actor_manifest)
    if actual_manifest_sha != pointer["sha256"]:
        raise ValueError("ACP teacher actor manifest differs from its declared SHA")
    root = Path(__file__).resolve().parents[5]
    sources = {
        "teacher_config": teacher_config.resolve(),
        "task_base_config": base,
        "actor_manifest": actor_manifest.resolve(),
        "acp_native19_adapter": root / "genedynamics/tasks/transfer/controllers/acp.py",
        "task_owned_execution": root / "genedynamics/tasks/transfer/demonstrators/acp_collection.py",
        "acp_collection": Path(__file__).resolve(),
    }
    return {
        "kind": "public_geometry_via_native19_adapter",
        "scalar_stiffness": float(config["scalar_stiffness"]),
        "files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "actor_manifest_declared_sha256": pointer["sha256"],
    }


def _episode_directory(output: Path, row: dict[str, Any]) -> Path:
    return (
        output / "episodes" / str(row["purpose"])
        / f"mode_{int(row['mode'])}" / f"seed_{int(row['seed'])}"
    )


def _save_execution(path: Path, row: ACPTeacherExecution, *, prefix: str = "") -> dict[str, str]:
    episode_path = path / f"{prefix}episode.npz"
    evidence_path = path / f"{prefix}evidence.json.gz"
    requests_path = path / f"{prefix}teacher_requests19.npy"
    _atomic_npz(episode_path, _execution_arrays(row))
    np.save(requests_path, row.requests19, allow_pickle=False)
    evidence_path.write_bytes(evidence_bytes(row.evidence))
    return {
        "episode": _sha256(episode_path),
        "evidence": _sha256(evidence_path),
        "teacher_requests19": _sha256(requests_path),
    }


def _collect_one(
    config: dict[str, Any], output: Path, request: dict[str, Any],
) -> dict[str, Any]:
    directory = _episode_directory(output, request)
    if directory.exists():
        raise FileExistsError(f"ACP teacher episode already exists: {directory}")
    directory.mkdir(parents=True)
    report_path = directory / "report.json"
    try:
        kwargs = dict(
            task_alias=config["task"],
            teacher_config_path=config["teacher_config"],
            episode=request,
            n_steps=int(config["n_steps"]),
            scalar_stiffness=float(config["scalar_stiffness"]),
        )
        execution = execute_public_acp_teacher(**kwargs)
        audit = audit_evidence_bytes(evidence_bytes(execution.evidence), directory)
        replay = execute_public_acp_teacher(**kwargs) if config["require_exact_replay"] else None
        replay_receipt = _exact_replay(execution, replay) if replay is not None else {
            "passed": True, "required": False,
        }
        files = _save_execution(directory, execution)
        if replay is not None:
            files.update({
                f"replay_{name}": digest
                for name, digest in _save_execution(directory, replay, prefix="replay_").items()
            })
        episode = execution.evaluation["episodes"][-1]
        exact_t2 = bool(episode.get("exact_physics_contact_evidence", True))
        support_ok = True
        if config["task"] == "T4":
            records = execution.evidence["records"]
            valid = [trace for record in records for trace in record["traces"] if trace.get("valid")]
            support_ok = bool(valid) and min(int(row.get("support_count", 0)) for row in valid) >= 2
        success = bool(
            episode["success"]
            and not episode["constraint_violation"]
            and episode["mode_match"]
            and audit["passed"]
            and exact_t2
            and support_ok
            and replay_receipt["passed"]
        )
        trajectory = PublicTargetTrajectory(
            {
                "version": FOUR_TASK_VERSION,
                "permission": "C_supervised",
                "robot": execution.evidence["task_spec"]["config"]["robot"],
                "mode": int(request["mode"]),
                "seed": int(request["seed"]),
                "purpose": request["purpose"],
                "chart": "acp_world19",
                "controls": len(execution.requests19),
                "recording_end": (
                    "terminal" if episode["status"] else "budget_stop"
                ),
                "task": config["task"],
            },
            execution.public,
            execution.requests19,
        )
        trajectory_path = directory / "public"
        trajectory.save(trajectory_path)
        files.update({child.name: _sha256(child) for child in trajectory_path.iterdir()})
        report = {
            "success": success,
            "task": config["task"],
            "mode": int(request["mode"]),
            "purpose": request["purpose"],
            "seed": int(request["seed"]),
            "sample_id": int(request["sample_id"]),
            "controls": len(execution.requests19),
            "outcome": episode["outcome"],
            "strict_task_success": bool(episode["success"]),
            "constraint_violation": bool(episode["constraint_violation"]),
            "mode_match": bool(episode["mode_match"]),
            "observed_mode": int(episode["observed_mode"]),
            "exact_physics_contact_evidence": exact_t2,
            "support_gate_passed": support_ok,
            "evidence_audit": audit,
            "exact_replay": replay_receipt,
            "evaluation": _json_ready(execution.evaluation),
            "accounting": _json_ready(execution.accounting),
            "wall_seconds": float(execution.wall_seconds),
            "canonical_model_data_sha256": _canonical_digest(
                execution.public, execution.requests19,
            ),
            "episode_sha256": files["episode"],
            "files": files,
        }
    except BaseException as error:
        report = {
            "success": False,
            "task": config["task"],
            "mode": int(request["mode"]),
            "purpose": request["purpose"],
            "seed": int(request["seed"]),
            "sample_id": int(request["sample_id"]),
            "outcome": "reset_or_execution_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    atomic_json(report_path, report)
    return report


def collect_demonstrations(config: dict[str, Any]) -> dict[str, Any]:
    """Run the fixed eight-episode schedule, retaining all failures."""
    _validate_config(config)
    output = Path(config["output"])
    manifest_path = output / "data_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"ACP collection manifest already exists: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    canonical_owners: dict[str, dict[str, Any]] = {}
    duplicates = []
    for request in config["episodes"]:
        report = _collect_one(config, output, deepcopy(request))
        directory = _episode_directory(output, request)
        canonical = report.get("canonical_model_data_sha256")
        duplicate = canonical in canonical_owners if canonical is not None else False
        if duplicate:
            duplicates.append({
                "path": str(directory / "public"),
                "canonical_model_data_sha256": canonical,
                "first": canonical_owners[canonical],
            })
        elif canonical is not None:
            canonical_owners[canonical] = {
                "seed": int(request["seed"]), "mode": int(request["mode"]),
                "purpose": request["purpose"],
            }
        public_path = directory / "public"
        rows.append({
            "path": str(public_path),
            "sha256": (public_path / "public.sha256").read_text().strip()
            if (public_path / "public.sha256").is_file() else None,
            "task": config["task"],
            "robot": (
                json.loads((public_path / "public.json").read_text())["robot"]
                if (public_path / "public.json").is_file() else None
            ),
            "mode": int(request["mode"]),
            "seed": int(request["seed"]),
            "purpose": request["purpose"],
            "controls": int(report.get("controls", 0)),
            "qualified": bool(report.get("success", False)),
            "canonical_duplicate": duplicate,
            "canonical_model_data_sha256": canonical,
            "success_filter": False,
            "replacement": False,
            "report": str(directory / "report.json"),
        })
    coverage = {
        str(mode): {
            purpose: sum(
                row["qualified"] and not row["canonical_duplicate"]
                and row["mode"] == mode and row["purpose"] == purpose
                for row in rows
            )
            for purpose in ("fit", "validation")
        }
        for mode in (0, 1)
    }
    sufficient = all(
        coverage[str(mode)]["fit"] >= 3
        and coverage[str(mode)]["validation"] >= 1
        for mode in (0, 1)
    )
    manifest = {
        "version": VERSION,
        "task": config["task"],
        "teacher": collection_provenance(config),
        "entries": rows,
        "coverage_by_mode": coverage,
        "training_data_sufficient": sufficient,
        "canonical_duplicates": duplicates,
        "all_requested_retained": len(rows) == len(config["episodes"]),
        "no_replacements": True,
        "task_predicates_unchanged": True,
        "exact_replay_required": bool(config["require_exact_replay"]),
    }
    atomic_json(manifest_path, manifest)
    return manifest


__all__ = ["VERSION", "collect_demonstrations", "collection_provenance"]

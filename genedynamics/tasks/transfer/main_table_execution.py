"""Task-owned orchestration for one sealed independent-table execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from genedynamics.core.artifacts import atomic_json
from genedynamics.tasks.transfer.main_table_evaluation import (
    GPU0_UUID, ROOT, _gpu0, formal_plan,
)
from genedynamics.tasks.transfer.main_table_configs import configs
from genedynamics.tasks.transfer.main_table_statistics import METHODS
from genedynamics.tasks.transfer.task_identity import (
    CANONICAL_TASKS,
    legacy_task_id,
)


EXECUTION_VERSION = "transfer-independent-main-table-execution-semantic-v1"


@dataclass(frozen=True)
class ExecutionSettings:
    """Machine and artifact locations supplied by the paper entrypoint."""

    output: Path
    table_output: Path
    site_packages: Path | None
    temporary_directory: Path | None
    cuda_visible_devices: str
    physical_gpu_uuid: str
    maximum_job_seconds: int = 43_200
    config_revision: str = "v2"

    @property
    def receipt(self) -> Path:
        return ROOT / self.output / "launch_receipt.json"

    @property
    def state(self) -> Path:
        return ROOT / self.output / "state.json"


def validate_settings(settings: ExecutionSettings) -> None:
    for value, label in (
        (settings.output, "execution output"),
        (settings.table_output, "table output"),
    ):
        absolute = (ROOT / value).resolve()
        if value.is_absolute() or not absolute.is_relative_to(
            (ROOT / "results/_analysis/four_task_transfer").resolve()
        ):
            raise ValueError(f"{label} must be a repository-relative analysis path")
    if settings.output == settings.table_output:
        raise ValueError("execution state and assembled table require distinct outputs")
    if settings.config_revision != "v2":
        raise ValueError("only the frozen v2 config reader is currently qualified")
    if settings.cuda_visible_devices != "0" or settings.physical_gpu_uuid != GPU0_UUID:
        raise ValueError("the frozen v2 actor configs are qualified only for physical GPU0")
    if settings.maximum_job_seconds <= 0:
        raise ValueError("maximum job seconds must be positive")
    if settings.site_packages is not None and not settings.site_packages.is_dir():
        raise FileNotFoundError(settings.site_packages)
    if (settings.temporary_directory is not None
            and not settings.temporary_directory.is_dir()):
        raise FileNotFoundError(settings.temporary_directory)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_roster() -> dict[str, str]:
    """Pin every shared implementation and task config read by formal physics."""
    roots = [
        "genedynamics/core", "genedynamics/experiments", "genedynamics/learning",
        "genedynamics/tasks/transfer", "genedynamics/envs/adapters",
        "genedynamics/envs/domains", "genedynamics/envs/obstacles",
        "genedynamics/solvers/common", "genedynamics/solvers/single/cross_ebdm",
        "genedynamics/solvers/single/fada", "genedynamics/solvers/single/dial",
        "genedynamics/solvers/single/asid", "genedynamics/solvers/single/adaptsim",
        "genedynamics/solvers/single/response_mpc",
        "configs/arm/obstacle_transfer", "configs/arm/guided_contact_alignment",
        "configs/arm/keyed_insertion_recovery",
        "configs/quadruped/mobile_keyed_insertion_recovery",
    ]
    single = [
        ".runtime/cross_ebdm/native/libcross_ebdm.so",
        "requirements/baselines/sources.lock.json",
    ]
    paths = {ROOT / name for name in single}
    for name in roots:
        for path in (ROOT / name).rglob("*"):
            if path.is_file() and path.suffix in {
                ".py", ".json", ".yaml", ".yml", ".xml", ".mjcf",
            }:
                paths.add(path)
    return {str(path.relative_to(ROOT)): _sha(path) for path in sorted(paths)}


def _source_difference(frozen: dict[str, str]) -> list[str]:
    actual = source_roster()
    return sorted(set(actual) ^ set(frozen) |
                  {name for name in set(actual) & set(frozen)
                   if actual[name] != frozen[name]})


def _status(settings: ExecutionSettings, stage: str, **details: object) -> None:
    state = {"version": EXECUTION_VERSION,
             "stage": stage, **details}
    atomic_json(settings.state, state)
    print(stage, details, flush=True)


def _child(arguments: list[str], *, settings: ExecutionSettings, gpu: bool,
           timeout: int) -> subprocess.CompletedProcess[str]:
    bootstrap = (
        "import numpy, scipy, sys; "
        + (f"sys.path.insert(0,{str(settings.site_packages)!r}); "
           if settings.site_packages is not None else "")
        + "from genedynamics.tasks.transfer.main_table_evaluation_cli import main; "
        + f"sys.argv={['main-table-evaluation', *arguments]!r}; main()"
    )
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=".", PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES=settings.cuda_visible_devices if gpu else "",
        JAX_PLATFORMS="gpu" if gpu else "cpu",
    )
    if settings.temporary_directory is not None:
        environment["TMPDIR"] = str(settings.temporary_directory)
    return subprocess.run([sys.executable, "-c", bootstrap], cwd=ROOT,
                          env=environment, capture_output=True, text=True,
                          timeout=timeout)


def plan(admission_report: dict, settings: ExecutionSettings) -> dict:
    """Build the complete GPU0 job roster without opening formal physics."""
    validate_settings(settings)

    invalidation = ROOT / (
        "results/_analysis/four_task_transfer/"
        "independent_main_table_execution_v1/invalidation.json"
    )
    if json.loads(invalidation.read_text()).get("status") != (
        "invalid_incomplete_ambiguous_physical_attempt"
    ):
        raise ValueError("the invalid GPU1 cohort is not retained")
    if admission_report.get("independent_table_eligible_cells") != 36:
        raise ValueError("the 36 actor identities no longer qualify")
    if settings.receipt.exists() or (ROOT / settings.table_output).exists():
        raise FileExistsError("GPU0 formal cohort was already launched or assembled")
    jobs = []
    paired = {}
    for task in CANONICAL_TASKS:
        legacy = legacy_task_id(task)
        for method in METHODS:
            for path, config in configs(task, method, settings.config_revision):
                checked = formal_plan(config)
                if (checked["task"] != legacy or checked["method"] != method
                        or checked["version"] !=
                        "transfer-independent-main-table-evaluation-v2"):
                    raise ValueError("GPU0 task/method/version differs")
                requests = [row["request_id"] for row in checked["requests"]]
                if task in paired and requests != paired[task]:
                    raise ValueError(f"{task} frozen requests differ across actors")
                paired[task] = requests
                if (ROOT / checked["output"]).exists():
                    raise FileExistsError(f"GPU0 formal output exists: {checked['output']}")
                jobs.append({
                    "task": task, "legacy_artifact_task": legacy,
                    "method": method,
                    "instance_id": checked["instance_id"],
                    "config": {"path": str(path.relative_to(ROOT)),
                               "sha256": _sha(path)},
                    "output": checked["output"],
                    "expected_requests": len(requests),
                })
    if len(jobs) != 232 or sum(job["expected_requests"] for job in jobs) != 11_600:
        raise ValueError("GPU0 actor or physical outcome denominator differs")
    return {
        "version": EXECUTION_VERSION,
        "status": "frozen_unexecuted",
        "canonical_task_order": list(CANONICAL_TASKS),
        "legacy_artifact_task_mapping": {
            task: legacy_task_id(task) for task in CANONICAL_TASKS
        },
        "physical_gpu_index": int(settings.cuda_visible_devices),
        "physical_gpu_uuid": settings.physical_gpu_uuid,
        "eligible_cells": 36,
        "jobs": jobs, "shared_source_and_task_config_sha256": source_roster(),
        "expected_physical_outcomes": 11_600,
        "maximum_seconds_per_actor_job": settings.maximum_job_seconds,
        "failure_replacement": False,
        "formal_policy_episodes_accessed_before_launch": 0,
        "previous_invalid_cohort": {
            "path": str(invalidation.relative_to(ROOT)),
            "sha256": _sha(invalidation),
        },
    }


def run(admission_report: dict, settings: ExecutionSettings,
        assemble: Callable[[Path, str], dict]) -> None:
    """Execute each fresh actor once; retain and stop on the first ambiguity."""
    validate_settings(settings)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != settings.cuda_visible_devices:
        raise RuntimeError("the full-table parent must expose physical GPU0")
    _gpu0()
    frozen = plan(admission_report, settings)
    atomic_json(settings.receipt, frozen)
    _status(settings, "FORMAL_LAUNCH_FROZEN", jobs=len(frozen["jobs"]),
            receipt_sha256=_sha(settings.receipt))
    sources = frozen["shared_source_and_task_config_sha256"]
    for index, job in enumerate(frozen["jobs"], start=1):
        changed = _source_difference(sources)
        if changed:
            _status(settings, "FORMAL_SHARED_SOURCE_CHANGED_RETAINED", job=index,
                    changed=changed[:12])
            raise RuntimeError("frozen GPU0 source or task config changed")
        arguments = ["--config", job["config"]["path"]]
        if job["method"] in {"asid", "adaptsim"}:
            arguments.extend(("--instance", job["instance_id"]))
        _status(settings, "FORMAL_JOB_STARTED", job=index, total=len(frozen["jobs"]),
                task=job["task"], method=job["method"],
                instance=job["instance_id"])
        try:
            executed = _child(arguments + ["--run"], settings=settings, gpu=True,
                              timeout=settings.maximum_job_seconds)
        except subprocess.TimeoutExpired as error:
            _status(settings, "FORMAL_JOB_TIMEOUT_RETAINED", job=index, error=str(error))
            raise
        print(executed.stdout[-2000:], executed.stderr[-2000:], flush=True)
        if executed.returncode:
            _status(settings, "FORMAL_RUN_FAILURE_RETAINED", job=index,
                    exit_code=executed.returncode)
            raise RuntimeError("GPU0 formal run failed; partial evidence retained")
        try:
            audited = _child(arguments + ["--audit"], settings=settings,
                             gpu=False, timeout=3600)
        except subprocess.TimeoutExpired as error:
            _status(settings, "FORMAL_AUDIT_TIMEOUT_RETAINED", job=index, error=str(error))
            raise
        print(audited.stdout[-2000:], audited.stderr[-2000:], flush=True)
        if audited.returncode:
            _status(settings, "FORMAL_AUDIT_FAILURE_RETAINED", job=index,
                    exit_code=audited.returncode)
            raise RuntimeError("GPU0 formal evidence audit failed")
        audit_path = ROOT / job["output"] / "independent_audit.json"
        audit = json.loads(audit_path.read_text())
        if (audit.get("status") != "passed"
                or audit.get("formal_performance_evidence") is not True
                or audit.get("physical_evidence_verified") != job["expected_requests"]):
            _status(settings, "FORMAL_AUDIT_STATUS_RETAINED", job=index,
                    audit_status=audit.get("status"))
            raise RuntimeError("GPU0 formal actor audit status differs")
        changed = _source_difference(sources)
        if changed:
            _status(settings, "FORMAL_SHARED_SOURCE_CHANGED_RETAINED", job=index,
                    changed=changed[:12])
            raise RuntimeError("frozen GPU0 source changed after formal actor")
        _status(settings, "FORMAL_JOB_AUDITED", job=index, total=len(frozen["jobs"]),
                audit_sha256=_sha(audit_path))
    _status(settings, "ALL_FORMAL_JOBS_AUDITED", jobs=len(frozen["jobs"]))
    try:
        assembled = assemble(ROOT / settings.table_output, settings.config_revision)
    except Exception as error:
        _status(settings, "FORMAL_TABLE_ASSEMBLY_FAILURE_RETAINED", error=str(error))
        raise
    if assembled.get("status") != "complete_formal_physical_evidence":
        _status(settings, "FORMAL_TABLE_ASSEMBLY_FAILURE_RETAINED",
                assembly_status=assembled.get("status"))
        raise RuntimeError("GPU0 formal table assembly failed")
    _status(settings, "FORMAL_TABLE_ASSEMBLED", output=str(settings.table_output))

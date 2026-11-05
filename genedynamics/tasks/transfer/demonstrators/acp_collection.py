"""Task-owned execution of one supervised ACP public-teacher episode."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from genedynamics.envs.adapters.transfer_task import FLAT_FIELDS, instantiate_task
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.acp import Native19Demonstrator
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence,
    controller_diagnostics,
    episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.specs import mechanism_spec


@dataclass(frozen=True)
class ACPTeacherExecution:
    public: dict[str, np.ndarray]
    requests19: np.ndarray
    actions13: np.ndarray
    evaluation: dict[str, Any]
    evidence: dict[str, Any]
    accounting: dict[str, Any]
    wall_seconds: float


def _public_snapshot(public: dict[str, Any]) -> dict[str, np.ndarray]:
    if set(public) != set(FLAT_FIELDS):
        raise ValueError("ACP teacher received a non-public observation schema")
    result = {}
    for name in FLAT_FIELDS:
        value = np.asarray(public[name])
        if value.shape[0] != 1:
            raise ValueError("ACP supervised collection requires batch one")
        result[name] = np.array(value[0], copy=True)
    return result


def _task_from_config(
    config: ExperimentConfig, episode: dict[str, Any], *, task_alias: str,
):
    values = deepcopy(config.env_params)
    robot = str(values.pop("robot"))
    history_len = int(values.pop("history_len", 4))
    values.pop("mode", None)
    shared_execution = values.pop("shared_execution", None)
    mechanism_values = {
        **dict(values.pop("mechanism", {}) or {}),
        **dict(episode.get("mechanism_parameters", {})),
    }
    task = instantiate_task(
        config.env_name,
        robot=robot,
        mechanism_spec=mechanism_spec(
            task_alias, mechanism_values,
            body_support_parameters=episode.get("body_support_parameters", {}),
        ),
        seed=int(episode["seed"]),
        history_len=history_len,
        config_overrides=values or None,
        shared_execution=shared_execution,
    )
    task._experiment_mode = int(episode["mode"])
    return task


def execute_public_acp_teacher(
    *, task_alias: str, teacher_config_path: str | Path,
    episode: dict[str, Any], n_steps: int, scalar_stiffness: float,
) -> ACPTeacherExecution:
    """Execute the frozen public-geometry teacher through ACP's native adapter.

    The task factory, public body intent, terminal state, and success predicate
    stay task-owned.  The ACP wrapper records the native 19D command that maps
    back to the exact shared action sent to the task.
    """
    if task_alias not in {"T1", "T2", "T3", "T4"}:
        raise ValueError("unknown four-task alias")
    if int(episode["mode"]) not in (0, 1):
        raise ValueError("ACP teacher episode mode must be 0 or 1")
    started = time.perf_counter()
    config = ExperimentConfig.from_yaml(teacher_config_path)
    if config.method != "public_geometry" or config.device != "cpu":
        raise ValueError("ACP teacher collection requires the CPU public_geometry config")
    task = _task_from_config(config, episode, task_alias=task_alias)
    initial = "jammed" if task_alias in {"T3", "T4"} else "clear"
    public = task.reset(
        (int(episode["sample_id"]),), seed=int(episode["seed"]),
        initial=initial, modes=[int(episode["mode"])],
    )
    parameters = deepcopy(config.method_params)
    parameters.update(mode=int(episode["mode"]), device="cpu")
    selection = deepcopy(parameters["actor_binding"])
    selection.update(family="main", key="public_geometry")
    parameters["actor_binding"] = selection
    teacher = create_transfer_actor(task, "public_geometry", parameters)
    actor = Native19Demonstrator(
        task, teacher, scalar_stiffness=float(scalar_stiffness),
    )
    public_rows = [_public_snapshot(public)]
    actions13: list[np.ndarray] = []
    requests19: list[np.ndarray] = []
    for _ in range(int(n_steps)):
        if bool(np.asarray(task.state.status)[0]):
            break
        action = np.asarray(actor.act(public), dtype=np.float32)
        diagnostic = controller_diagnostics(teacher)
        body_intent = task.public_body_intent(public)
        step = (
            task.step(action)
            if body_intent is None
            else task.step_contact(action, body_twist=body_intent)
        )
        public = step.observation
        task.record_control_diagnostics(diagnostic)
        actions13.append(np.asarray(action).reshape(-1, 13)[0].copy())
        requests19.append(np.asarray(actor.requests[-1], dtype=np.float32).copy())
        public_rows.append(_public_snapshot(public))
    if not actions13:
        raise ValueError("ACP public teacher executed no physical transition")
    evaluation = evaluate_records(task.records, task.spec)
    evaluation["accounting"] = dict(task.accounting)
    evaluation["lifetime_accounting"] = dict(task.lifetime_accounting)
    evidence = episode_evidence(task.records, task.spec, evaluation)
    return ACPTeacherExecution(
        public={
            name: np.stack([row[name] for row in public_rows])
            for name in FLAT_FIELDS
        },
        requests19=np.stack(requests19).astype(np.float32, copy=False),
        actions13=np.stack(actions13).astype(np.float32, copy=False),
        evaluation=evaluation,
        evidence=evidence,
        accounting=dict(task.accounting),
        wall_seconds=time.perf_counter() - started,
    )


def evidence_bytes(evidence: dict[str, Any]) -> bytes:
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return gzip.compress(raw, compresslevel=6, mtime=0)


def audit_evidence_bytes(value: bytes, directory: str | Path) -> dict[str, Any]:
    """Use the independent task audit on exactly the bytes to be persisted."""
    path = Path(directory) / ".acp-teacher-evidence-audit.json.gz"
    path.write_bytes(value)
    try:
        return audit_episode_evidence(path)
    finally:
        path.unlink(missing_ok=True)


__all__ = [
    "ACPTeacherExecution", "audit_evidence_bytes", "evidence_bytes",
    "execute_public_acp_teacher",
]

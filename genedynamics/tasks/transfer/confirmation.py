"""One-shot execution of a SHA-bound four-task transfer confirmation.

The paper script supplies paths only.  Task construction, actor binding,
post-reset receipts, failure retention, and independent evaluation live here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import multiprocessing
import subprocess
import time
import traceback
from typing import Any, Iterable

import numpy as np
import yaml

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
    RiskSetEmptyError,
)
from genedynamics.tasks.transfer.controllers.bindings import (
    create_transfer_actor,
    resolve_transfer_binding,
    validate_transfer_actor_manifest,
)
from genedynamics.tasks.transfer.evaluation import EVALUATION_VERSION, evaluate_records
from genedynamics.tasks.transfer.guide_evaluation import STRICT_GUIDE_VERSION
from genedynamics.tasks.transfer.specs import mechanism_spec


CONFIRMATION_REVISION = 14
CONFIRMATION_VERSION = f"four-task-transfer-confirmation-v{CONFIRMATION_REVISION}"
CONFIRMATION_PACKAGE_VERSION = (
    f"four-task-transfer-confirmation-package-v{CONFIRMATION_REVISION}"
)
FORMAL_DEVICE = "cuda:0"
PHYSICAL_GPU_INDEX = 1
PHYSICAL_GPU_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
NATIVE_LIBRARY = Path(".runtime/cross_ebdm/native/libcross_ebdm.so")
EXPECTED_REQUESTS = {"T1": 40, "T2": 40, "T3": 40, "T4": 80}
EXPECTED_SCHEDULE_ROWS = 21_200

TASK_ROOTS = {
    "T1": Path("configs/arm/obstacle_transfer"),
    "T2": Path("configs/arm/guided_contact_alignment"),
    "T3": Path("configs/arm/keyed_insertion_recovery"),
    "T4": Path("configs/quadruped/mobile_keyed_insertion_recovery"),
}
TASK_NAMES = {
    "T1": "obstacle_transfer",
    "T2": "guided_contact_alignment",
    "T3": "keyed_insertion_recovery",
    "T4": "mobile_keyed_insertion_recovery",
}
METHOD_ALIASES = {
    "ours_frozen_v1": "cross_ebdm",
    "ours_gated_v2": "cross_ebdm",
    "cross_ebdm": "cross_ebdm",
    "task_only": "task_only",
    "fada_task_adapted": "fada",
    "fada": "fada",
    "acp_task_adapted": "acp",
    "acp": "acp",
    "h2s2r_source_trajectory": "human2sim2robot",
    "human2sim2robot": "human2sim2robot",
    "dial_mpc": "dial_transfer",
    "dial_mpc_task_adapted": "dial_transfer",
    "dial_transfer": "dial_transfer",
    "ordinary_response_mpc": "response_mpc",
    "response_mpc": "response_mpc",
    "public_geometry": "public_geometry",
    "public_source_prior": "public_source_prior",
    "frozen_warm": "frozen_warm",
    "matching_no_band": "matching_no_band",
    "band_only": "band_only",
    "ungated_full": "ungated_full",
    "gated_full": "gated_full",
}

_WORKER: TransferConfirmation | None = None
_WORKER_BASES: dict[str, dict[str, Any]] = {}


def _initialize_confirmation_worker(
    root: str, output: str, devices: tuple[str, ...], package_report: str,
) -> None:
    global _WORKER, _WORKER_BASES
    identity = multiprocessing.current_process()._identity
    worker_index = (identity[-1] - 1) if identity else 0
    device = devices[worker_index % len(devices)]
    import torch

    torch.set_num_threads(1)
    _WORKER = TransferConfirmation(
        Path(root), output=Path(output), device=device,
        package_report=Path(package_report), continue_claimed=True,
    )
    _, _WORKER_BASES = _WORKER._admission()
    _WORKER.receipt_path = (
        _WORKER.output / f"state_receipts.worker_{os.getpid()}.jsonl"
    )


def _execute_confirmation_worker(
    payload: tuple[int, dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    if _WORKER is None:
        raise RuntimeError("confirmation worker was not initialized")
    ordinal, schedule_row, request = payload
    task = str(schedule_row.get("task", request["task"]))
    return _WORKER._execute(
        schedule_row=schedule_row, request=request,
        base=_WORKER_BASES[task], ordinal=ordinal,
    )


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_input_passes(label: str, evidence: dict[str, Any]) -> bool:
    """Apply the evidence contract for one sealed package input.

    The invalid predecessor is bound to prove zero reuse, so its required
    state is deliberately invalid rather than passed.  Every other input is a
    positive admission artifact.
    """
    if label == "invalid_predecessor":
        return bool(
            evidence.get("status") in {
                "invalid_incomplete_run", "invalid_incomplete_qualification",
            }
            and evidence.get("performance_evidence_admitted") is False
            and evidence.get("reusable_records") == 0
            and evidence.get("failure_replacement") is False
        )
    return evidence.get("passed", evidence.get("status") == "passed") is True


def _checked(root: Path, pointer: dict[str, Any], label: str) -> Path:
    path = Path(str(pointer.get("path", ""))).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = _sha(path)
    if actual != pointer.get("sha256"):
        raise ValueError(f"{label} SHA256 changed: {actual}")
    return path


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _hash_tree(value: Any, digest: hashlib._Hash | None = None) -> str:
    digest = hashlib.sha256() if digest is None else digest
    if isinstance(value, dict):
        digest.update(b"{")
        for key in sorted(value):
            digest.update(str(key).encode())
            _hash_tree(value[key], digest)
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for child in value:
            _hash_tree(child, digest)
        digest.update(b"]")
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, np.generic):
        _hash_tree(np.asarray(value), digest)
    else:
        digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _method_key(family: str, method: str) -> str:
    token = str(method).strip()
    key = METHOD_ALIASES.get(token.lower())
    if key is None:
        raise ValueError(f"unrecognized frozen method identity: {method!r}")
    if family == "ablation" and key not in {
        "frozen_warm", "task_only", "matching_no_band", "band_only",
        "ungated_full", "gated_full",
    }:
        raise ValueError(f"invalid ablation method identity: {method!r}")
    return key


def _instance(row: dict[str, Any], binding: dict[str, Any]) -> str:
    if binding["status"] == "deterministic_single_identity":
        return "fixed"
    if (
        binding["status"] == "predeployment_rejection"
        and binding.get("instance_role") == "deterministic_single_identity"
    ):
        return "fixed"
    value = (
        row.get("training_or_calibration_instance")
        or row.get("instance")
        or row.get("instance_id")
    )
    if value is None:
        record_id = str(row.get("record_id", ""))
        value = next(
            (part for part in record_id.split("/") if part.startswith("instance_")),
            None,
        )
    if value not in {f"instance_{index:02d}" for index in range(1, 9)}:
        raise ValueError(f"schedule row has invalid learned instance {value!r}")
    return str(value)


def _request_id(row: dict[str, Any]) -> str:
    value = row.get("base_request_id") or row.get("request_id")
    if not value:
        raise ValueError("schedule row has no base request identity")
    return str(value)


def _seed(request: dict[str, Any]) -> int:
    value = request.get("confirmation_seed", request.get("reset_seed"))
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("confirmation request has no integer reset seed")
    return value


def _sample_id(request: dict[str, Any]) -> int:
    value = request.get("physical_initial_slot", request.get("sample_id", 0))
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("confirmation request has invalid physical initial slot")
    return value


def _resource_counts(actor: Any, task: Any) -> dict[str, Any]:
    names = (
        "irreversible_model_calls", "irreversible_network_calls",
        "irreversible_candidate_rollouts", "irreversible_action_updates",
        "irreversible_model_forwards", "irreversible_forward_calls",
    )
    return {
        **{name: int(getattr(actor, name, 0)) for name in names},
        "task_accounting": dict(task.accounting),
        "task_lifetime_accounting": dict(task.lifetime_accounting),
    }


def _failure_outcome(
    *, task: str, request: dict[str, Any], category: str, error: BaseException,
    reset_succeeded: bool,
) -> dict[str, Any]:
    return {
        "evaluation_version": (
            STRICT_GUIDE_VERSION if task == "T2" else EVALUATION_VERSION
        ),
        "success": False,
        "constrained_success": 0.0,
        "constraint_violation": True,
        "capped_normalized_completion_time": 1.0,
        "predeployment_rejection": 0.0,
        "failure_category": category,
        "outcome": category,
        "physical_control_steps": 0,
        "preparation_control_steps": 0,
        "requested_mode": int(request["requested_mode"]),
        "reset_succeeded": reset_succeeded,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    count = len(values)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        by_group[str(row["mechanism_group"])].append(row)
    group_success = {
        group: sum(float(row["constrained_success"]) for row in group_rows)
        / len(group_rows)
        for group, group_rows in by_group.items()
    }
    return {
        "requested_episodes": count,
        "constrained_success_rate": sum(
            float(row["constrained_success"]) for row in values
        ) / count,
        "constraint_violation_rate": sum(
            float(row["constraint_violation"]) for row in values
        ) / count,
        "capped_normalized_completion_time": sum(
            float(row["capped_normalized_completion_time"]) for row in values
        ) / count,
        "worst_group_constrained_success": min(group_success.values()),
        "group_constrained_success": group_success,
        "predeployment_rejection_rate": sum(
            float(row["predeployment_rejection"]) for row in values
        ) / count,
        "target_physical_execution_count": sum(
            int(row["target_physical_execution"]) for row in values
        ),
        "outcomes": dict(Counter(str(row["failure_category"]) for row in values)),
    }


class TransferConfirmation:
    """Run exactly one already-frozen confirmation schedule."""

    def __init__(
        self, root: Path, *, output: Path, device: str = "cuda",
        package_report: Path | None = None, continue_claimed: bool = False,
        workers: int = 1, devices: tuple[str, ...] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.output = output if output.is_absolute() else self.root / output
        self.device = device
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError("workers must be positive")
        self.devices = tuple(devices or (device,))
        if not self.devices:
            raise ValueError("at least one execution device is required")
        if self.device != FORMAL_DEVICE or any(
            value != FORMAL_DEVICE for value in self.devices
        ):
            raise ValueError(
                "four-task transfer confirmation is frozen to cuda:0 only"
            )
        self.continue_claimed = bool(continue_claimed)
        self.manifests: dict[str, dict[str, Any]] = {}
        package_path = package_report or Path(
            "results/_protocol/four_task_transfer/confirmation_v14/package/report.json"
        )
        self.package_report = (
            package_path if package_path.is_absolute() else self.root / package_path
        )
        self.launch_path = self.output / "launch.json"
        self.receipt_path = self.output / "state_receipts.jsonl"
        self.outcome_path = self.output / "outcomes.jsonl"

    def _base_documents(self) -> dict[str, dict[str, Any]]:
        values = {}
        for task, relative in TASK_ROOTS.items():
            values[task] = yaml.safe_load((self.root / relative / "_base.yaml").read_text())
        return values

    def _admission(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        import mujoco  # noqa: F401 - admission requires the execution physics ABI
        if self.device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("formal CUDA actor execution is unavailable")
            if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
                raise RuntimeError("formal confirmation requires physical GPU1 only")
            uuid = subprocess.check_output([
                "nvidia-smi", "-i", str(PHYSICAL_GPU_INDEX),
                "--query-gpu=uuid", "--format=csv,noheader",
            ], text=True).strip()
            if uuid != PHYSICAL_GPU_UUID:
                raise RuntimeError(f"physical GPU1 UUID changed: {uuid}")
        expected_native = (self.root / NATIVE_LIBRARY).resolve()
        configured_native = os.environ.get("CROSS_EBDM_NATIVE_LIBRARY")
        if (
            not expected_native.is_file()
            or configured_native is None
            or Path(configured_native).expanduser().resolve() != expected_native
        ):
            raise RuntimeError(
                "formal confirmation requires CROSS_EBDM_NATIVE_LIBRARY="
                f"{expected_native}"
            )
        bases = self._base_documents()
        reports = []
        manifests = {}
        for task, base in bases.items():
            metadata = base["metadata"]
            if metadata.get("protocol_status") != "frozen_after_architecture_equivalence_v3":
                raise ValueError(f"{task} protocol is not frozen after equivalence")
            artifacts = metadata["frozen_design"]["artifacts"]
            for label in ("architecture_equivalence", "zero_physics_preflight"):
                path = _checked(self.root, artifacts[label], f"{task}.{label}")
                report = json.loads(path.read_text())
                if report.get("passed", report.get("status") == "passed") is not True:
                    raise ValueError(f"{task}.{label} did not pass")
                reports.append((label, artifacts[label]))
            manifest_pointer = base["method_params"]["actor_binding"]["manifest"]
            manifests[task] = validate_transfer_actor_manifest(manifest_pointer)
        self.manifests = manifests
        if len({json.dumps(pointer, sort_keys=True) for label, pointer in reports if label == "architecture_equivalence"}) != 1:
            raise ValueError("task bases disagree on architecture equivalence")
        if len({json.dumps(pointer, sort_keys=True) for label, pointer in reports if label == "zero_physics_preflight"}) != 1:
            raise ValueError("task bases disagree on zero-physics preflight")
        package = json.loads(self.package_report.read_text())
        if package.get("version") != CONFIRMATION_PACKAGE_VERSION:
            raise ValueError(
                f"formal confirmation requires the fresh v{CONFIRMATION_REVISION} package"
            )
        if (
            package.get("physical_gpu_index") != PHYSICAL_GPU_INDEX
            or package.get("physical_gpu_uuid") != PHYSICAL_GPU_UUID
            or package.get("logical_cuda_device") != FORMAL_DEVICE
        ):
            raise ValueError("confirmation package GPU1 binding differs")
        recovery = package.get("recovery_authorization") or {}
        fresh_package = (
            package.get("confirmation_accessed") is False
            and package.get("confirmation_states_opened") == 0
        )
        recovery_package = (
            package.get("p6_accessed") is True
            and recovery.get("status") == "full_relaunch_authorized"
            and recovery.get("execution_devices") == [FORMAL_DEVICE]
            and recovery.get("performance_evidence_admitted") is False
            and recovery.get("reusable_records") == 0
            and recovery.get("selective_replay") is False
            and isinstance(recovery.get("zero_physics_preflight"), dict)
        )
        if package.get("passed") is not True or not (
            fresh_package or recovery_package
        ):
            raise ValueError("frozen confirmation package is not admissible")
        abandoned_path = _checked(
            self.root, package["prior_abandoned_package"],
            "prior_abandoned_package",
        )
        abandoned = json.loads(abandoned_path.read_text())
        if not (
            abandoned.get("status") == "abandoned_before_confirmation_access"
            and abandoned.get("performance_evidence_admitted") is False
            and abandoned.get("reusable_records") == 0
            and abandoned.get("confirmation_states_opened") == 0
            and abandoned.get("run_directory_created") is False
            and abandoned.get("physical_steps") == 0
        ):
            raise ValueError("prior abandoned package is not a zero-access artifact")
        if recovery_package:
            _checked(
                self.root, recovery["prior_invalid_run"], "prior_invalid_run"
            )
            _checked(
                self.root, recovery["base_package"], "base_confirmation_package"
            )
            preflight_path = _checked(
                self.root,
                recovery["zero_physics_preflight"],
                "recovery_zero_physics_preflight",
            )
            if json.loads(preflight_path.read_text()).get("passed") is not True:
                raise ValueError("recovery zero-physics preflight did not pass")
        if package.get("counts", {}).get("scheduled_method_requests") != EXPECTED_SCHEDULE_ROWS:
            raise ValueError("frozen confirmation schedule cardinality changed")
        for label, pointer in package.get("audit_inputs", {}).items():
            evidence_path = _checked(self.root, pointer, f"package.{label}")
            evidence = json.loads(evidence_path.read_text())
            if label == "invalid_predecessor" and (
                package.get("prior_invalid_run") != pointer
                or package.get("old_invalid_run_reused") is not False
            ):
                raise ValueError("invalid predecessor binding or zero-reuse declaration changed")
            if not _audit_input_passes(label, evidence):
                raise ValueError(f"package audit input did not pass: {label}")
            if label == "gpu_package_calibration":
                native_path = _checked(
                    self.root, evidence["native_library"],
                    "gpu_package_calibration.native_library",
                )
                if native_path.resolve() != expected_native:
                    raise ValueError("GPU calibration used a different native library")
        return package, bases

    def _claim_launch(self, package: dict[str, Any]) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIRMATION_VERSION,
            "status": "claimed_before_sealed_schedule_access",
            "started_at": _utc(),
            "device": self.device,
            "devices": list(self.devices),
            "physical_gpu_index": PHYSICAL_GPU_INDEX,
            "physical_gpu_uuid": PHYSICAL_GPU_UUID,
            "workers": self.workers,
            "failure_replacement": False,
            "replacement_or_replay_count": 0,
            "expected_retained_failure_ordinals": [],
            "launches_allowed": 1,
            "confirmation_accessed": True,
            "package_report": {
                "path": str(self.package_report.relative_to(self.root)),
                "sha256": _sha(self.package_report),
            },
            "sealed_artifacts": {
                key: package["artifacts"][key]
                for key in ("request_roster", "evaluation_schedule")
            },
        }
        if package.get("recovery_authorization"):
            payload["recovery_authorization"] = package["recovery_authorization"]
        descriptor = os.open(
            self.launch_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _load_schedule(
        self, package: dict[str, Any]
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        roster_path = _checked(
            self.root, package["artifacts"]["request_roster"], "request_roster"
        )
        schedule_path = _checked(
            self.root, package["artifacts"]["evaluation_schedule"],
            "evaluation_schedule",
        )
        roster_rows = _read_jsonl(roster_path)
        schedule = _read_jsonl(schedule_path)
        roster = {_request_id(row): row for row in roster_rows}
        if len(roster) != len(roster_rows) or len(roster_rows) != 200:
            raise ValueError("request roster identities are not unique")
        if Counter(row["task"] for row in roster_rows) != Counter(EXPECTED_REQUESTS):
            raise ValueError("request roster task counts changed")
        if len(schedule) != EXPECTED_SCHEDULE_ROWS:
            raise ValueError("evaluation schedule does not contain 21,200 rows")
        expected_physics = int(package["counts"]["dispatch_physics"])
        actual_physics = sum(
            row.get("dispatch") == "execute_once_after_admission" for row in schedule
        )
        if actual_physics != expected_physics:
            raise ValueError("evaluation schedule physical-dispatch count changed")
        if any(_request_id(row) not in roster for row in schedule):
            raise ValueError("schedule references an unknown request")
        return roster, schedule

    @staticmethod
    def _schedule_record_id(row: dict[str, Any]) -> str:
        value = row.get("schedule_id") or row.get("record_id")
        if value:
            return str(value)
        instance = row.get("instance") or row.get("instance_id") or "fixed"
        return (
            f"{row.get('family', 'main')}/{row['method']}/{instance}/"
            f"{_request_id(row)}"
        )

    def _existing_outcomes(self) -> list[dict[str, Any]]:
        if not self.outcome_path.exists():
            return []
        rows = _read_jsonl(self.outcome_path)
        ids = [str(row["record_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("active outcome journal contains duplicate records")
        return rows

    def _merge_receipts(self) -> None:
        parts = [self.receipt_path] if self.receipt_path.exists() else []
        parts.extend(sorted(self.output.glob("state_receipts.worker_*.jsonl")))
        rows = [row for path in parts for row in _read_jsonl(path)]
        ids = [str(row["record_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("state receipt journals contain duplicate records")
        rows.sort(key=lambda row: int(row["ordinal"]))
        temporary = self.receipt_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.receipt_path)
        parts_root = self.output / "receipt_journal_parts"
        parts_root.mkdir(exist_ok=True)
        for path in parts:
            if path == self.receipt_path or not path.exists():
                continue
            os.replace(path, parts_root / path.name)

    def _selection(
        self, base: dict[str, Any], row: dict[str, Any], request: dict[str, Any]
    ) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
        task = str(row.get("task", request.get("task")))
        if task not in TASK_ROOTS or task != request.get("task"):
            raise ValueError("schedule/request task mismatch")
        family = str(row.get("family", "main")).lower()
        key = _method_key(family, str(row.get("method")))
        selection = deepcopy(base["method_params"]["actor_binding"])
        selection.update(family=family, key=key)
        document = self.manifests[task]
        manifest_row = document[family][key]
        provisional = {
            "status": manifest_row["status"],
            "instance_role": manifest_row.get("instance_role"),
        }
        instance = _instance(row, provisional)
        if instance != "fixed":
            selection["instance"] = instance
        binding = resolve_transfer_binding(
            {
                "actor_binding": selection,
                "mode": int(request["requested_mode"]),
            },
            task_alias=task,
        )
        qualification_status = str(row.get("qualification_status", ""))
        expected_dispatch = (
            "execute_once_after_admission"
            if qualification_status == "behavior_qualified"
            else "score_predeployment_rejection_without_physical_execution"
        )
        if row.get("dispatch") != expected_dispatch:
            raise ValueError(
                f"schedule dispatch differs from {task}/{family}/{key} qualification"
            )
        if expected_dispatch == "execute_once_after_admission" and binding["status"] == "predeployment_rejection":
            raise ValueError("qualification cannot execute an unavailable actor binding")
        expected_checkpoint = row.get("checkpoint_sha256")
        bound_hashes = {
            str(pointer.get("sha256"))
            for name, pointer in binding.items()
            if name in {"checkpoint", "policy_checkpoint", "response_checkpoint"}
            and isinstance(pointer, dict)
        }
        if expected_checkpoint is None:
            if binding["status"] == "instances_ready":
                raise ValueError("executable learned schedule row has no checkpoint SHA")
        elif str(expected_checkpoint) not in bound_hashes:
            raise ValueError(
                f"schedule checkpoint differs from {task}/{family}/{key}/{instance}"
            )
        return task, family, key, instance, binding

    def _execute(
        self, *, schedule_row: dict[str, Any], request: dict[str, Any],
        base: dict[str, Any], ordinal: int,
    ) -> dict[str, Any]:
        task_alias, family, key, instance, binding = self._selection(
            base, schedule_row, request
        )
        record_id = str(schedule_row.get("record_id") or (
            f"{family}/{schedule_row['method']}/{instance}/{_request_id(request)}"
        ))
        if schedule_row.get("schedule_id") not in {None, record_id}:
            raise ValueError("constructed record identity differs from frozen schedule")
        common = {
            "record_id": record_id,
            "ordinal": ordinal,
            "base_request_id": _request_id(request),
            "task": task_alias,
            "family": family,
            "method": str(schedule_row["method"]),
            "binding_key": key,
            "training_or_calibration_instance": instance,
            "instance_role": binding.get("instance_role", (
                "deterministic_single_identity" if instance == "fixed"
                else "independent_training_or_calibration_instance"
            )),
            "mechanism_group": str(request["mechanism_group"]),
            "body_support_cell": str(request["body_support_cell"]),
            "physical_initial_slot": int(request["physical_initial_slot"]),
            "requested_mode": int(request["requested_mode"]),
            "confirmation_seed": _seed(request),
            "confirmation_accessed": True,
            "performance_evidence": True,
        }
        if schedule_row["dispatch"] == "score_predeployment_rejection_without_physical_execution":
            score = schedule_row["rejection_scoring"]
            return {
                **common,
                "evaluation_version": (
                    STRICT_GUIDE_VERSION if task_alias == "T2" else EVALUATION_VERSION
                ),
                "dispatch_status": "predeployment_rejection",
                "success": False,
                "constrained_success": float(score["constrained_success"]),
                "constraint_violation": float(score["constraint_violation"]),
                "capped_normalized_completion_time": float(
                    score["capped_normalized_completion_time"]
                ),
                "predeployment_rejection": float(score["predeployment_rejection"]),
                "target_physical_execution": 0,
                "failure_category": "rejected_or_nonfinite",
                "outcome": "rejected",
                "rejection_reason": schedule_row["qualification_reason"],
                "physical_control_steps": 0,
                "preparation_control_steps": 0,
                "resource_counts": {},
            }

        seed = _seed(request)
        mode = int(request["requested_mode"])
        task = None
        actor = None
        reset_succeeded = False
        started = time.perf_counter()
        action_digest = hashlib.sha256()
        try:
            env_values = deepcopy(base.get("env_params") or {})
            support = {
                **dict(env_values.pop("body_support", {}) or {}),
                **dict(request.get("body_support_parameters") or {}),
            }
            configured_robot = env_values.pop("robot", None)
            robot = str(
                support.get("robot") or configured_robot
                or ("go2_z1" if task_alias == "T4" else "xarm7")
            )
            history_len = int(env_values.pop("history_len", 4))
            env_values.pop("mode", None)
            shared_execution = env_values.pop("shared_execution", None)
            mechanism_values = {
                **dict(env_values.pop("mechanism", {}) or {}),
                **dict(request.get("mechanism_parameters") or {}),
            }
            mechanism = mechanism_spec(
                task_alias,
                mechanism_values,
                body_support_parameters=support,
            )
            task = instantiate_task(
                TASK_NAMES[task_alias], robot=robot,
                mechanism_spec=mechanism, seed=seed, history_len=history_len,
                config_overrides=env_values or None,
                shared_execution=shared_execution,
            )
            initial = "jammed" if task_alias in {"T3", "T4"} else "clear"
            public = task.reset(
                (_sample_id(request),), seed=seed, initial=initial, modes=[mode],
            )
            reset_succeeded = True
            checkpoint = task.get_checkpoint()
            receipt = {
                **common,
                "written_before_first_actuation": True,
                "post_reset_physical_state_sha256": _hash_tree({
                    "integration": checkpoint["integration"],
                }),
                "post_reset_controller_state_sha256": _hash_tree({
                    "state": checkpoint["state"],
                    "buffers": checkpoint["buffers"],
                    "acp": checkpoint["acp"],
                }),
                "post_reset_episode_record_sha256": _hash_tree({
                    "episodes": checkpoint["episodes"],
                    "current": checkpoint["current"],
                }),
                "post_reset_state_sha256": _hash_tree({
                    "integration": checkpoint["integration"],
                    "state": checkpoint["state"], "buffers": checkpoint["buffers"],
                    "acp": checkpoint["acp"], "episodes": checkpoint["episodes"],
                    "current": checkpoint["current"],
                }),
                "task_contract_sha256": checkpoint["contract_sha256"],
                "shared_execution": deepcopy(task.shared_execution),
                "reset_accounting": dict(task.accounting),
                "created_at": _utc(),
            }
            _append_jsonl(self.receipt_path, receipt)

            actor_config = {
                **deepcopy(base.get("method_params") or {}),
                "mode": mode,
                "device": self.device,
                "actor_binding": {
                    **deepcopy(base["method_params"]["actor_binding"]),
                    "family": family,
                    "key": key,
                },
            }
            if instance != "fixed":
                actor_config["actor_binding"]["instance"] = instance
            actor = create_transfer_actor(
                task, str(binding.get("actor", key)), actor_config
            )
            maximum_steps = int(base["n_steps"])
            for _ in range(maximum_steps):
                if bool(np.asarray(task.state.status)[0]):
                    break
                try:
                    action = np.asarray(actor.act(public), dtype=np.float32)
                except RiskSetEmptyError:
                    task.records[-1].state = "rejected"
                    task.records[-1].rejection_reason = "risk_set_empty"
                    break
                action_digest.update(np.ascontiguousarray(action).tobytes())
                body_intent = task.public_body_intent(public)
                public = (
                    task.step(action).observation
                    if body_intent is None
                    else task.step_contact(action, body_twist=body_intent).observation
                )
            record = task.records[-1]
            evaluated = evaluate_records([record], task.spec)["episodes"][-1]
            outcome = {
                **evaluated,
                "constrained_success": float(evaluated["success"]),
                "failure_category": str(evaluated["outcome"]),
            }
            return {
                **common,
                **outcome,
                "dispatch_status": "completed",
                "constraint_violation": float(outcome["constraint_violation"]),
                "predeployment_rejection": float(outcome["predeployment_rejection"]),
                "target_physical_execution": 1,
                "action_stream_sha256": action_digest.hexdigest(),
                "resource_counts": _resource_counts(actor, task),
                "wall_seconds": time.perf_counter() - started,
            }
        except BaseException as error:
            category = (
                "reset_or_preparation_failure" if not reset_succeeded
                else "rejected_or_nonfinite"
            )
            failed = _failure_outcome(
                task=task_alias, request=request, category=category,
                error=error, reset_succeeded=reset_succeeded,
            )
            return {
                **common,
                **failed,
                "dispatch_status": "retained_failure",
                "target_physical_execution": int(reset_succeeded),
                "action_stream_sha256": action_digest.hexdigest(),
                "resource_counts": (
                    {} if task is None else _resource_counts(actor, task)
                ),
                "wall_seconds": time.perf_counter() - started,
                "traceback": traceback.format_exc(),
            }

    def run(self) -> dict[str, Any]:
        package, bases = self._admission()
        if self.launch_path.exists():
            existing = json.loads(self.launch_path.read_text())
            if not self.continue_claimed:
                raise FileExistsError(
                    "confirmation launch is already claimed; an audited continuation "
                    "must be explicitly requested"
                )
            if existing.get("status") not in {
                "interrupted_before_any_physics",
                "interrupted_for_parallel_continuation",
            }:
                raise ValueError(
                    "claimed launch is not in an audited continuation state"
                )
            launch = existing
            launch["continuation"] = {
                "continued_at": _utc(),
                "same_launch": True,
                "reason": str(existing.get("continuation_reason", (
                    "execution interpreter lacked MuJoCo before every reset"
                ))),
            }
            _write_json(self.launch_path, launch)
        else:
            if self.continue_claimed:
                raise FileNotFoundError("no claimed confirmation launch exists")
            self._claim_launch(package)
        roster, schedule = self._load_schedule(package)
        launch = json.loads(self.launch_path.read_text())
        launch.update(
            status="running",
            sealed_schedule_opened_at=_utc(),
            request_count=len(roster),
            scheduled_record_count=len(schedule),
        )
        _write_json(self.launch_path, launch)

        outcomes = self._existing_outcomes()
        seen = {str(row["record_id"]) for row in outcomes}
        scheduled_ids = [self._schedule_record_id(row) for row in schedule]
        if len(scheduled_ids) != len(set(scheduled_ids)):
            raise ValueError("frozen schedule contains duplicate records")
        if not seen <= set(scheduled_ids):
            raise ValueError("outcome journal contains a record outside the schedule")
        remaining = [
            (ordinal, row, roster[_request_id(row)])
            for ordinal, row in enumerate(schedule, start=1)
            if self._schedule_record_id(row) not in seen
        ]
        physical_executions = sum(
            int(row["target_physical_execution"]) for row in outcomes
        )
        retained_failures = sum(
            row["dispatch_status"] == "retained_failure" for row in outcomes
        )
        if self.workers == 1:
            result_stream = (
                self._execute(
                    schedule_row=row, request=request,
                    base=bases[str(row.get("task", request["task"]))],
                    ordinal=ordinal,
                )
                for ordinal, row, request in remaining
            )
        else:
            pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_confirmation_worker,
                initargs=(
                    str(self.root), str(self.output), self.devices,
                    str(self.package_report),
                ),
            )
            result_stream = pool.map(
                _execute_confirmation_worker, remaining, chunksize=1
            )
        try:
            for outcome in result_stream:
                ordinal = int(outcome["ordinal"])
                if outcome["record_id"] in seen:
                    raise ValueError(f"duplicate formal record: {outcome['record_id']}")
                seen.add(outcome["record_id"])
                _append_jsonl(self.outcome_path, outcome)
                outcomes.append(outcome)
                physical_executions += int(outcome["target_physical_execution"])
                retained_failures += outcome["dispatch_status"] == "retained_failure"
                completed = len(outcomes)
                if completed % 10 == 0 or completed == len(schedule):
                    launch.update(
                        completed_records=completed,
                        last_completed_ordinal=ordinal,
                        physical_executions=physical_executions,
                        retained_failures=retained_failures,
                        workers=self.workers,
                        devices=list(self.devices),
                        updated_at=_utc(),
                    )
                    _write_json(self.launch_path, launch)
        finally:
            if self.workers > 1:
                pool.shutdown(wait=True, cancel_futures=False)

        self._merge_receipts()
        receipt_rows = _read_jsonl(self.receipt_path)
        if len(receipt_rows) != physical_executions:
            raise ValueError(
                "post-reset receipt count differs from physical execution count"
            )

        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in outcomes:
            grouped[(row["task"], row["family"], row["method"])].append(row)
        aggregates = [
            {
                "task": task, "family": family, "method": method,
                **_aggregate(rows),
            }
            for (task, family, method), rows in sorted(grouped.items())
        ]
        aggregate_path = self.output / "aggregates.json"
        _write_json(aggregate_path, aggregates)
        launch.update(
            status="complete",
            completed_at=_utc(),
            completed_records=len(outcomes),
            aggregate_cells=len(aggregates),
            outcomes={
                "path": str(self.outcome_path.relative_to(self.root)),
                "sha256": _sha(self.outcome_path),
            },
            state_receipts={
                "path": str(self.receipt_path.relative_to(self.root)),
                "sha256": _sha(self.receipt_path),
            },
            aggregates={
                "path": str(aggregate_path.relative_to(self.root)),
                "sha256": _sha(aggregate_path),
            },
        )
        _write_json(self.launch_path, launch)
        return launch


def audit_confirmation(root: Path, output: Path) -> dict[str, Any]:
    """Verify a completed confirmation against its frozen schedule and journals."""
    root = root.resolve()
    output = output if output.is_absolute() else root / output
    launch_path = output / "launch.json"
    outcomes_path = output / "outcomes.jsonl"
    receipts_path = output / "state_receipts.jsonl"
    aggregates_path = output / "aggregates.json"
    launch = json.loads(launch_path.read_text())
    schedule_path = _checked(
        root, launch["sealed_artifacts"]["evaluation_schedule"],
        "evaluation_schedule",
    )
    roster_path = _checked(
        root, launch["sealed_artifacts"]["request_roster"], "request_roster"
    )
    schedule = _read_jsonl(schedule_path)
    roster = _read_jsonl(roster_path)
    outcomes = _read_jsonl(outcomes_path)
    receipts = _read_jsonl(receipts_path)
    aggregates = json.loads(aggregates_path.read_text())

    schedule_ids = [TransferConfirmation._schedule_record_id(row) for row in schedule]
    outcome_ids = [str(row["record_id"]) for row in outcomes]
    receipt_ids = [str(row["record_id"]) for row in receipts]
    scheduled_physical_ids = {
        TransferConfirmation._schedule_record_id(row)
        for row in schedule
        if row["dispatch"] == "execute_once_after_admission"
    }
    physical_outcome_ids = {
        str(row["record_id"])
        for row in outcomes if int(row["target_physical_execution"]) == 1
    }
    state_hashes: dict[str, set[str]] = defaultdict(set)
    for receipt in receipts:
        state_hashes[str(receipt["base_request_id"])].add(
            str(receipt["post_reset_state_sha256"])
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[(row["task"], row["family"], row["method"])].append(row)
    recomputed = {
        key: _aggregate(rows) for key, rows in grouped.items()
    }
    published = {
        (row["task"], row["family"], row["method"]): row
        for row in aggregates
    }
    metric_names = (
        "requested_episodes", "constrained_success_rate",
        "constraint_violation_rate", "capped_normalized_completion_time",
        "worst_group_constrained_success", "predeployment_rejection_rate",
        "target_physical_execution_count",
    )
    aggregate_exact = set(recomputed) == set(published)
    if aggregate_exact:
        for key, actual in recomputed.items():
            declared = published[key]
            for name in metric_names:
                left, right = actual[name], declared[name]
                if isinstance(left, float) or isinstance(right, float):
                    aggregate_exact &= bool(np.isclose(left, right, atol=1e-12, rtol=0))
                else:
                    aggregate_exact &= left == right
            aggregate_exact &= actual["outcomes"] == declared["outcomes"]

    retained = [
        row for row in outcomes if row["dispatch_status"] == "retained_failure"
    ]
    expected_retained = {
        int(value)
        for value in launch.get("expected_retained_failure_ordinals", [])
    }
    checks = {
        "launch_complete": launch.get("status") == "complete",
        "gpu1_only": (
            launch.get("device") == FORMAL_DEVICE
            and launch.get("devices") == [FORMAL_DEVICE]
            and launch.get("physical_gpu_index") == PHYSICAL_GPU_INDEX
            and launch.get("physical_gpu_uuid") == PHYSICAL_GPU_UUID
        ),
        "frozen_schedule_21200_unique": (
            len(schedule_ids) == len(set(schedule_ids)) == EXPECTED_SCHEDULE_ROWS
        ),
        "roster_200_unique_40_40_40_80": (
            len(roster) == len({_request_id(row) for row in roster}) == 200
            and Counter(row["task"] for row in roster) == Counter(EXPECTED_REQUESTS)
        ),
        "outcomes_exactly_cover_schedule": (
            len(outcome_ids) == len(set(outcome_ids)) == EXPECTED_SCHEDULE_ROWS
            and set(outcome_ids) == set(schedule_ids)
        ),
        "main_11600_ablation_9600": (
            sum(row["family"] == "main" for row in outcomes) == 11_600
            and sum(row["family"] == "ablation" for row in outcomes) == 9_600
        ),
        "physical_schedule_exact": physical_outcome_ids == scheduled_physical_ids,
        "one_preaction_receipt_per_physical_request": (
            len(receipt_ids) == len(set(receipt_ids)) == len(scheduled_physical_ids)
            and set(receipt_ids) == scheduled_physical_ids
            and all(row.get("written_before_first_actuation") is True for row in receipts)
        ),
        "same_request_has_identical_post_reset_state": all(
            len(values) == 1 for values in state_hashes.values()
        ),
        "only_declared_runner_failures_retained": (
            {int(row["ordinal"]) for row in retained} == expected_retained
            and all(
                row["failure_category"] == "runner_interruption_after_reset"
                for row in retained
            )
        ),
        "failure_replacement_zero": (
            launch.get("failure_replacement") is False
            and launch.get("replacement_or_replay_count", 0) == 0
            and (launch.get("supervisor_interruption") or {}).get(
                "replacement_or_replay_count", 0
            ) == 0
        ),
        "sixty_aggregate_cells_exact": (
            len(aggregates) == len(published) == len(recomputed) == 60
            and aggregate_exact
        ),
        "artifact_hashes_match_launch": all(
            _sha(root / launch[label]["path"]) == launch[label]["sha256"]
            for label in ("outcomes", "state_receipts", "aggregates")
        ),
    }
    report = {
        "version": "four-task-transfer-confirmation-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "schedule": len(schedule),
            "outcomes": len(outcomes),
            "receipts": len(receipts),
            "physical_executions": len(physical_outcome_ids),
            "retained_failures": len(retained),
            "aggregate_cells": len(aggregates),
        },
        "artifacts": {
            "launch": {"path": str(launch_path.relative_to(root)), "sha256": _sha(launch_path)},
            "outcomes": {"path": str(outcomes_path.relative_to(root)), "sha256": _sha(outcomes_path)},
            "state_receipts": {"path": str(receipts_path.relative_to(root)), "sha256": _sha(receipts_path)},
            "aggregates": {"path": str(aggregates_path.relative_to(root)), "sha256": _sha(aggregates_path)},
        },
    }
    report_path = output / "audit.json"
    _write_json(report_path, report)
    return report


__all__ = [
    "CONFIRMATION_PACKAGE_VERSION", "CONFIRMATION_REVISION",
    "CONFIRMATION_VERSION", "TransferConfirmation", "audit_confirmation",
]

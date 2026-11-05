"""Build hash-bound ACP training examples from qualified public teachers.

The input episodes remain task-owned ``PublicTargetTrajectory`` artifacts.  This
module only performs ACP's label checks and converts them to the native history
and 19D diffusion charts used by the solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

from genedynamics.core.artifacts import atomic_json
from genedynamics.learning.datasets.transfer.target import PublicTargetTrajectory
from .target_task_training import PublicACPTaskDataset, public_target_windows


VERSION = "acp-qualified-public-demonstrations-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _trajectory_model_digest(row: PublicTargetTrajectory) -> str:
    """Digest only arrays seen by ACP; metadata cannot hide duplicate rollouts."""
    return _array_digest({
        **{f"public.{name}": value for name, value in row.public.items()},
        "requests": row.requests,
    })


@dataclass(frozen=True)
class QualifiedTeacherEpisode:
    trajectory: PublicTargetTrajectory
    path: Path
    source_role: str
    files: dict[str, str]

    @property
    def receipt(self) -> dict[str, Any]:
        metadata = dict(self.trajectory.metadata)
        return {
            "path": str(self.path),
            "trajectory_sha256": self.trajectory.digest,
            "canonical_model_data_sha256": _trajectory_model_digest(self.trajectory),
            "task": self.trajectory.task,
            "robot": metadata["robot"],
            "mode": int(metadata["mode"]),
            "seed": int(metadata["seed"]),
            "purpose": metadata["purpose"],
            "controls": int(metadata["controls"]),
            "source_role": self.source_role,
            "files": self.files,
        }


def _episode_files(path: Path, report: dict[str, Any]) -> dict[str, str]:
    parent = path.parent
    required = {
        "public_json": path / "public.json",
        "public_npz": path / "public.npz",
        "public_sha256": path / "public.sha256",
        "report": parent / "report.json",
        "teacher_requests19": parent / "teacher_requests19.npy",
        "executed_episode": parent / "episode.npz",
    }
    missing = [name for name, value in required.items() if not value.is_file()]
    if missing:
        raise ValueError(f"qualified ACP episode is missing files: {missing}")
    if _sha256(required["executed_episode"]) != report.get("episode_sha256"):
        raise ValueError("executed episode SHA differs from its qualification report")
    return {name: _sha256(value) for name, value in required.items()}


def load_qualified_teacher_episodes(
    manifests: Iterable[str | Path], *, task: str, source_role: str
) -> tuple[list[QualifiedTeacherEpisode], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load qualified rows while retaining ordinary qualification failures.

    An entry marked qualified is an integrity claim, so a hash or label mismatch
    is fatal.  Unqualified entries are retained in ``rejected`` for sufficiency
    accounting and are never exposed to the learner.
    """
    accepted: list[QualifiedTeacherEpisode] = []
    rejected: list[dict[str, Any]] = []
    manifest_receipts: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for manifest_raw in manifests:
        manifest_path = Path(manifest_raw).resolve()
        manifest = json.loads(manifest_path.read_text())
        manifest_receipts.append({
            "path": str(manifest_path), "sha256": _sha256(manifest_path),
            "entries": len(manifest.get("entries", [])),
        })
        if not manifest.get("all_requested_retained") or not manifest.get("no_replacements"):
            raise ValueError("ACP input manifest must retain every requested teacher episode")
        for entry in manifest.get("entries", []):
            if entry.get("task") != task:
                continue
            path = Path(entry["path"])
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if path in seen_paths:
                raise ValueError("duplicate ACP teacher trajectory path")
            seen_paths.add(path)
            common = {
                "path": str(path), "task": task, "mode": int(entry["mode"]),
                "purpose": entry["purpose"], "seed": int(entry["seed"]),
            }
            if not entry.get("qualified", False):
                rejected.append({**common, "reason": "teacher_episode_not_qualified"})
                continue
            trajectory = PublicTargetTrajectory.load(path)
            metadata = dict(trajectory.metadata)
            expected = {
                "task": task, "robot": entry["robot"], "mode": int(entry["mode"]),
                "seed": int(entry["seed"]), "purpose": entry["purpose"],
                "controls": int(entry["controls"]),
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError("qualified ACP manifest metadata differs from trajectory")
            if metadata["permission"] != "C_supervised" or metadata["chart"] != "acp_world19":
                raise ValueError("ACP learner requires C-supervised native19 teacher commands")
            if trajectory.digest != entry["sha256"]:
                raise ValueError("qualified ACP trajectory digest differs from manifest")
            report_path = path.parent / "report.json"
            report = json.loads(report_path.read_text())
            if not report.get("success", False):
                raise ValueError("manifest qualified an unsuccessful ACP teacher episode")
            requests = np.load(path.parent / "teacher_requests19.npy", allow_pickle=False)
            if not np.array_equal(requests, trajectory.requests):
                raise ValueError("sealed public commands differ from executed ACP teacher labels")
            accepted.append(QualifiedTeacherEpisode(
                trajectory=trajectory,
                path=path,
                source_role=source_role,
                files=_episode_files(path, report),
            ))
    return accepted, rejected, manifest_receipts


def _split_arrays(
    episodes: list[QualifiedTeacherEpisode], *, history_len: int, horizon: int,
    action_down_sample_steps: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows = [episode.trajectory for episode in episodes]
    dataset = PublicACPTaskDataset(
        rows, purpose=rows[0].metadata["purpose"], history_len=history_len,
        horizon=horizon, action_chart="native19",
        action_down_sample_steps=action_down_sample_steps,
    )
    arrays = {
        **{f"observation.{name}": value.numpy() for name, value in dataset.observations.items()},
        "label.native19_relative": dataset.native_actions.numpy(),
        "label.reference_pose9_relative": dataset.native_actions[..., :9].numpy(),
        "label.virtual_target_pose9_relative": dataset.native_actions[..., 9:18].numpy(),
        "label.scalar_stiffness": dataset.native_actions[..., 18:19].numpy(),
        "observation.wrench_history_tool": dataset.observations["wrench_tool"].numpy(),
    }
    episode_indices = []
    control_indices = []
    episode_receipts = []
    for index, episode in enumerate(episodes):
        _, actions = public_target_windows(
            episode.trajectory, history_len=history_len, horizon=horizon,
            action_down_sample_steps=action_down_sample_steps,
        )
        episode_indices.extend([index] * len(actions))
        control_indices.extend(range(len(actions)))
        episode_receipts.append({
            **episode.receipt, "windows": len(actions),
            "window_sha256": _array_digest({"native19_relative": actions}),
        })
    arrays["provenance.episode_index"] = np.asarray(episode_indices, dtype=np.int32)
    arrays["provenance.control_index"] = np.asarray(control_indices, dtype=np.int32)
    return arrays, episode_receipts


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".acp-data-", suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_demonstration_data(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Prepare immutable native ACP fit/validation examples and a sufficiency report."""
    if config.get("version") != VERSION or config.get("task") not in {"T1", "T2", "T3", "T4"}:
        raise ValueError("explicit ACP demonstration-data v1 task config required")
    output = Path(output)
    history_len, horizon = int(config["history_len"]), int(config["horizon"])
    action_down_sample_steps = int(config.get("action_down_sample_steps", 1))
    if history_len < 1 or horizon < 1 or action_down_sample_steps < 1:
        raise ValueError("positive ACP history and prediction horizons required")
    accepted, rejected, manifests = load_qualified_teacher_episodes(
        config.get("source_manifests", []), task=config["task"],
        source_role=config["source_role"],
    )
    qualified_by_purpose = {
        purpose: sorted(
            [row for row in accepted if row.trajectory.metadata["purpose"] == purpose],
            key=lambda row: (int(row.trajectory.metadata["mode"]), int(row.trajectory.metadata["seed"])),
        ) for purpose in ("fit", "validation")
    }
    fit_digests = {row.trajectory.digest for row in qualified_by_purpose["fit"]}
    validation_digests = {row.trajectory.digest for row in qualified_by_purpose["validation"]}
    if fit_digests & validation_digests:
        raise ValueError("ACP train/validation trajectories overlap by canonical digest")
    deduplication = []
    by_purpose: dict[str, list[QualifiedTeacherEpisode]] = {}
    seen_model_data: set[str] = set()
    # Fit owns a duplicated numerical rollout.  A validation copy of training
    # bytes is excluded rather than silently turning train data into validation.
    for purpose in ("fit", "validation"):
        by_purpose[purpose] = []
        for row in qualified_by_purpose[purpose]:
            digest = _trajectory_model_digest(row.trajectory)
            if digest in seen_model_data:
                deduplication.append({
                    **row.receipt,
                    "reason": "duplicate_canonical_model_data_with_fit_or_earlier_episode",
                })
                continue
            seen_model_data.add(digest)
            by_purpose[purpose].append(row)
    split_reports: dict[str, Any] = {}
    for purpose, episodes in by_purpose.items():
        if not episodes:
            split_reports[purpose] = {"episodes": 0, "windows": 0, "artifact": None, "receipts": []}
            continue
        arrays, receipts = _split_arrays(
            episodes, history_len=history_len, horizon=horizon,
            action_down_sample_steps=action_down_sample_steps,
        )
        artifact = output / f"{purpose}.npz"
        _atomic_npz(artifact, arrays)
        split_reports[purpose] = {
            "episodes": len(episodes), "windows": len(arrays["provenance.episode_index"]),
            "artifact": str(artifact.resolve()), "artifact_sha256": _sha256(artifact),
            "canonical_arrays_sha256": _array_digest(arrays), "receipts": receipts,
            "array_schema": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        }
    minimum = config["minimum_episodes_per_mode"]
    coverage = {}
    for mode in (0, 1):
        coverage[str(mode)] = {
            purpose: sum(int(row.trajectory.metadata["mode"]) == mode for row in rows)
            for purpose, rows in by_purpose.items()
        }
    sufficient = all(
        coverage[str(mode)][purpose] >= int(minimum[purpose])
        for mode in (0, 1) for purpose in ("fit", "validation")
    )
    report = {
        "version": VERSION, "task": config["task"], "source_role": config["source_role"],
        "history_len": history_len, "horizon": horizon,
        "action_down_sample_steps": action_down_sample_steps,
        "source_manifests": manifests, "accepted_episodes": len(accepted),
        "usable_unique_episodes": sum(map(len, by_purpose.values())),
        "rejected_episodes": rejected, "splits": split_reports,
        "deduplicated_episodes": deduplication,
        "coverage_by_mode": coverage,
        "minimum_episodes_per_mode": minimum,
        "train_validation_disjoint": True,
        "training_data_sufficient": sufficient,
        "gpu_training_admitted": sufficient,
        "interpretation": (
            "Qualified native19 teacher data cover both modes and both fixed splits."
            if sufficient else
            "Do not launch a full ACP task pilot: at least one mode/split lacks qualified teacher episodes."
        ),
    }
    atomic_json(output / "report.json", report)
    return report


__all__ = [
    "VERSION", "QualifiedTeacherEpisode", "load_qualified_teacher_episodes",
    "prepare_demonstration_data",
]

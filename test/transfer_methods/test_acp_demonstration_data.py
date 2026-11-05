"""Qualified ACP teacher trajectories become split, hash-bound native examples."""
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from genedynamics.learning.datasets.transfer.target import (
    FOUR_TASK_VERSION,
    PublicTargetTrajectory,
)
from genedynamics.solvers.single.acp.backends import spatial
from genedynamics.solvers.single.acp.learning.demonstration_data import (
    VERSION,
    prepare_demonstration_data,
)
from genedynamics.solvers.single.acp.learning.demonstration_collection import (
    VERSION as COLLECTION_VERSION,
    _validate_config as validate_collection_config,
)
from genedynamics.solvers.single.acp.learning.task_training import pose
from test.transfer_methods.test_public_target_data import trajectory


def _episode(root, *, mode, purpose, seed):
    source = trajectory()
    world = spatial.SE3_to_pose9(pose(
        source.public["tcp_position"][:-1], source.public["tcp_rotation"][:-1],
    ))
    requests = np.concatenate((world, world, np.full((20, 1), 800. + seed)), axis=-1).astype(np.float32)
    metadata = dict(
        source.metadata, version=FOUR_TASK_VERSION, permission="C_supervised",
        chart="acp_world19", task="T3", mode=mode, purpose=purpose, seed=seed,
    )
    row = PublicTargetTrajectory(metadata, source.public, requests)
    episode = root / f"episode-{purpose}-{mode}"
    public = episode / "public"
    row.save(public)
    np.save(episode / "teacher_requests19.npy", requests)
    np.savez(episode / "episode.npz", controls=requests)
    episode_sha = hashlib.sha256((episode / "episode.npz").read_bytes()).hexdigest()
    (episode / "report.json").write_text(json.dumps({"success": True, "episode_sha256": episode_sha}))
    return {
        "path": str(public), "sha256": row.digest, "task": "T3", "robot": "xarm7",
        "mode": mode, "seed": seed, "purpose": purpose, "controls": 20,
        "qualified": True, "success_filter": False, "replacement": False,
    }


def test_preparation_emits_native_labels_wrench_history_provenance_and_split(tmp_path):
    entries = [
        _episode(tmp_path, mode=mode, purpose=purpose, seed=100 + 10 * mode + index)
        for mode in (0, 1) for index, purpose in enumerate(("fit", "validation"))
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "entries": entries, "all_requested_retained": True, "no_replacements": True,
    }))
    config = {
        "version": VERSION, "task": "T3", "source_role": "target_supervised_public_teacher",
        "source_manifests": [str(manifest)], "history_len": 8, "horizon": 16,
        "minimum_episodes_per_mode": {"fit": 1, "validation": 1},
    }
    report = prepare_demonstration_data(config, tmp_path / "prepared")
    assert report["training_data_sufficient"] and report["train_validation_disjoint"]
    assert report["coverage_by_mode"] == {
        "0": {"fit": 1, "validation": 1}, "1": {"fit": 1, "validation": 1},
    }
    with np.load(tmp_path / "prepared/fit.npz", allow_pickle=False) as arrays:
        assert arrays["label.native19_relative"].shape == (10, 16, 19)
        assert arrays["label.reference_pose9_relative"].shape[-1] == 9
        assert arrays["label.virtual_target_pose9_relative"].shape[-1] == 9
        assert arrays["label.scalar_stiffness"].shape[-1] == 1
        assert arrays["observation.wrench_history_tool"].shape[1:] == (8, 6)
        assert set(np.unique(arrays["provenance.episode_index"])) == {0, 1}


def test_qualified_episode_rejects_teacher_label_mismatch(tmp_path):
    entry = _episode(tmp_path, mode=0, purpose="fit", seed=100)
    requests = np.load(tmp_path / "episode-fit-0/teacher_requests19.npy")
    requests[0, 18] += 1
    np.save(tmp_path / "episode-fit-0/teacher_requests19.npy", requests)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "entries": [entry], "all_requested_retained": True, "no_replacements": True,
    }))
    config = {
        "version": VERSION, "task": "T3", "source_role": "target_supervised_public_teacher",
        "source_manifests": [str(manifest)], "history_len": 8, "horizon": 16,
        "minimum_episodes_per_mode": {"fit": 1, "validation": 1},
    }
    with pytest.raises(ValueError, match="teacher labels"):
        prepare_demonstration_data(config, tmp_path / "prepared")


def test_collection_schedule_requires_independent_three_fit_one_validation_per_mode():
    episodes = [
        {"mode": mode, "purpose": purpose, "seed": 100 + mode * 10 + index,
         "sample_id": 200 + mode * 10 + index}
        for mode in (0, 1)
        for index, purpose in enumerate(("fit", "fit", "fit", "validation"))
    ]
    config = {
        "version": COLLECTION_VERSION, "task": "T3",
        "teacher_config": "teacher.yaml", "n_steps": 750,
        "scalar_stiffness": 800., "require_exact_replay": True,
        "episodes": episodes, "output": "result",
    }
    validate_collection_config(config)
    duplicate = {**config, "episodes": [*episodes[:-1], {**episodes[-1], "seed": 100}]}
    with pytest.raises(ValueError, match="independent seeds"):
        validate_collection_config(duplicate)
    wrong_split = {**config, "episodes": [
        *episodes[:-1], {**episodes[-1], "purpose": "fit"},
    ]}
    with pytest.raises(ValueError, match="both splits"):
        validate_collection_config(wrong_split)

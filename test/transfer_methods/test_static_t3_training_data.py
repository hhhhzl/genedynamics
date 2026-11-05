"""Static T3 training cannot silently discard a moving fixture's observations."""
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from genedynamics.learning.datasets.transfer.target import (
    FIXTURE_SCHEMA, FIXTURE_VERSION, FOUR_TASK_VERSION, VERSION,
    PublicTargetTrajectory,
)
from test.transfer_methods.test_public_observed_response import record


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    ("train_t3_completion_value", "load_dataset", "public_features"),
    ("train_t3_mode_complete_response", "load_dataset", "response_windows"),
    ("train_t3_controller_context_response", "load_dataset", "controller_trace"),
    ("calibrate_t3_completion_response", "load_validation_rows", "response_windows"),
    ("train_t4_completion", "prepare_states", "prepare_episode"),
)


class ReachedStaticProcessing(Exception):
    """Stop before fitting or physical execution, after the input boundary."""


@pytest.fixture(scope="module")
def runners():
    return {
        name: runpy.run_path(str(ROOT / "scripts/paper/cross_ebdm" / f"{name}.py"))
        for name, _, _ in ENTRYPOINTS
    }


def trajectory(version):
    public, requests = record(12)
    public["joint_position"] = np.zeros((13, 7), np.float32)
    public["joint_velocity"] = np.zeros((13, 7), np.float32)
    public["previous_request"][1:] = requests
    metadata = dict(
        version=version, permission="B_log", robot="xarm7", mode=0,
        seed=717, purpose="fit", chart="shared13", controls=12,
        recording_end="budget_stop",
    )
    if version != VERSION:
        metadata["task"] = "T3"
    if version == FIXTURE_VERSION:
        metadata.update(environment="compliant_keyed_insertion_recovery",
                        observation_schema=FIXTURE_SCHEMA)
        public.update(
            fixture_position=np.tile([.44, -.2, 0.], (13, 1)),
            fixture_rotation=np.tile(np.eye(3), (13, 1, 1)),
            fixture_twist=np.zeros((13, 6)),
        )
    return PublicTargetTrajectory(metadata, public, requests)


def write_json(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def loader_call(tmp_path, runner, entrypoint, row, monkeypatch):
    name, function, downstream = entrypoint
    loader = runner[function]
    namespace = loader.__globals__
    row.save(tmp_path / "episode")
    entry = dict(path=str(tmp_path / "episode"), parent_sha256=row.digest,
                 prefix_sha256=row.digest, controls=len(row.requests))
    manifest = write_json(tmp_path / "manifest.json", dict(
        permission="B_log", reward=False, success_filter=False, entries=[entry],
    ))
    data = dict(
        public_log_manifests=[manifest], maximum_remaining_controls=20,
        temporal_horizon_controls=1, state_stride=1,
        episode_split=dict(modulus=3, calibration_residue=2,
                           selection_residue=1, fit_residues=[0],
                           validation_residue=0),
    )
    config = dict(version="t3-public-completion-value-training-v1",
                  model=dict(maximum_remaining_controls=20), data=data, task="T3")

    def reached(*args, **kwargs):
        raise ReachedStaticProcessing

    monkeypatch.setitem(namespace, downstream, reached)
    if name == "train_t4_completion":
        data.update(
            frozen_public_inputs=write_json(tmp_path / "frozen.json", dict(
                training_validation_episodes={}, sha256={},
            )),
            source_prior=dict(path=".", sha256={}), weak_log_manifest=manifest,
        )
        monkeypatch.setitem(namespace, "PublicLocalResidualTransport", lambda _: None)
        return lambda: loader(config, None, None, None)
    if name == "train_t3_completion_value":
        return lambda: loader(config)
    return lambda: loader(config, None)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda item: item[0])
@pytest.mark.parametrize("version", (VERSION, FOUR_TASK_VERSION, FIXTURE_VERSION))
def test_static_loaders_accept_legacy_and_reject_fixture_before_processing(
    tmp_path, runners, entrypoint, version, monkeypatch,
):
    call = loader_call(tmp_path, runners[entrypoint[0]], entrypoint,
                       trajectory(version), monkeypatch)
    if version == FIXTURE_VERSION:
        with pytest.raises(ValueError, match="static T3 public trajectory required"):
            call()
    else:
        with pytest.raises(ReachedStaticProcessing):
            call()


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda item: item[0])
def test_static_loaders_also_check_explicit_environment(
    tmp_path, runners, entrypoint, monkeypatch,
):
    call = loader_call(tmp_path, runners[entrypoint[0]], entrypoint,
                       trajectory(FOUR_TASK_VERSION), monkeypatch)
    # Exercise the entrypoint's environment guard independently of today's
    # stricter trajectory parser, so a future schema extension cannot bypass it.
    monkeypatch.setattr(PublicTargetTrajectory, "load", lambda _: SimpleNamespace(
        metadata=dict(version=FOUR_TASK_VERSION,
                      environment="compliant_keyed_insertion_recovery"),
    ))
    with pytest.raises(ValueError, match="static T3 public trajectory required"):
        call()


def test_completion_raw_public_paths_reject_fixture_fields(runners):
    dynamic = trajectory(FIXTURE_VERSION)
    value = runners["train_t3_completion_value"]["public_features"]
    with pytest.raises(ValueError, match="static T3 completion features"):
        value(dynamic.public, np.asarray([0]), 0)
    prepare = runners["train_t4_completion"]["prepare_episode"]
    with pytest.raises(ValueError, match="static T3 completion features"):
        prepare(dynamic.public, task="T3", body="xarm7", mode=0, seed=717,
                dummy=None, bank=None, feedback=None, prior=None, stride=1)


def test_static_completion_features_keep_existing_values(runners):
    import torch
    from genedynamics.solvers.single.cross_ebdm.learning.completion_value import (
        completion_value_features,
    )

    row = trajectory(FOUR_TASK_VERSION)
    ticks = np.asarray([0, 4, 12])
    actual = runners["train_t3_completion_value"]["public_features"](row.public, ticks, 0)
    arguments = {
        name: torch.from_numpy(np.asarray(row.public[field])[ticks])
        for name, field in (
            ("position", "tcp_position"), ("rotation", "tcp_rotation"),
            ("twist", "tcp_twist"), ("wrench", "measured_wrench"),
            ("goal_position", "goal_position"), ("goal_rotation", "goal_rotation"),
            ("controller_feedback", "controller_feedback"),
        )
    }
    expected = completion_value_features(
        **arguments, mode=torch.zeros(len(ticks), dtype=torch.long),
    ).numpy()
    np.testing.assert_array_equal(actual, expected)

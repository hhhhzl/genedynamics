"""Cheap data protocol tests; actual physics is qualified in execution tests."""
from dataclasses import replace

import numpy as np
import pytest

from genedynamics.tasks.transfer import ExecutionSpec, MechanismSpec, ObstacleTransferConfig
from genedynamics.tasks.transfer.runtime import ChunkResult, ControllerState, ExecutionState, StepTrace, Status
from genedynamics.solvers.single.cross_ebdm.learning.data import (
    POLICY_FEATURES, collect_dataset, load_dataset, make_episode_plan, policy_features, validate_dataset,
)
from genedynamics.tasks.transfer.routes import observed_route_modes
from genedynamics.envs.domains.manipulation.transfer_workcell.queries.source import TaskSemantics


class FixtureRunner:
    """Contract fake with one success and one failed episode in every batch."""
    def __init__(self, *, execution, task, skill, mechanism):
        self.execution, self.task, self.mechanism = execution, task, mechanism

    def reset_batch(self, request):
        b = len(request.sample_ids)
        ctrl = ControllerState(np.zeros((b,3)), np.broadcast_to(np.eye(3), (b,3,3)).copy(),
                               np.zeros((b,5,13)), np.zeros((b,13)), np.zeros((b,6)))
        return ExecutionState(None, ctrl, np.array(request.sample_ids, dtype=np.uint32),
                              np.zeros(b, dtype=np.int32), np.zeros(b, dtype=np.int32), np.zeros(b, dtype=np.int32))

    def synchronize(self, value):
        pass

    def describe(self):
        return {"task": "obstacle_transfer", "model_sha256": f"fixture-{self.task.robot}",
                "mechanism_batch": [m.__dict__ for m in self.mechanism]}

    def collect_chunk(self, state, commands):
        b, length, _ = commands.shape
        # Terminate at the first step: later steps must have valid=False.
        terminal_status = np.where(np.arange(b)%2 == 0, Status.SUCCESS, Status.NONFINITE).astype(np.int32)
        terminal_status = np.where(np.isfinite(commands).all(axis=(1,2)), terminal_status, Status.NONFINITE)
        terminal = np.zeros((b,length), dtype=bool)
        terminal[:,0] = state.status == Status.ACTIVE
        valid = terminal.copy()
        status = np.broadcast_to(np.where(state.status == 0, terminal_status, state.status)[:,None], (b,length)).copy()
        position = np.broadcast_to(np.array([.5,0.,.72]), (b,length,3)).copy()
        trace = StepTrace(position, np.broadcast_to(np.eye(3), (b,length,3,3)).copy(),
                          np.zeros((b,length)), np.zeros((b,length)), np.ones((b,length))*.005,
                          np.zeros((b,length)), np.zeros((b,length)), np.nan_to_num(commands),
                          np.zeros((b,length,7)), status, valid, terminal)
        state = state._replace(status=status[:,-1], steps=state.steps+valid.sum(axis=1))
        return ChunkResult(state, trace)


class FixtureObserver:
    def __init__(self, **kwargs):
        pass

    def observe(self, runner, state):
        b = len(state.sample_id)
        return TaskSemantics(np.zeros((b,3)), np.broadcast_to(np.eye(3), (b,3,3)).copy(),
                             np.zeros((b,6)), np.zeros(b, dtype=np.int32))


def collect(plan, **kwargs):
    def fixture_body(state):
        b = len(state.sample_id)
        return {"qpos": np.zeros((b,7)), "qvel": np.zeros((b,7))}
    return collect_dataset(plan, execution=ExecutionSpec(backend="numpy", physics_backend="mujoco", microbatch_size=2),
                           task=ObstacleTransferConfig(timeout_seconds=1), chunk_length=2, steps=4,
                           runner_factory=FixtureRunner, observer_factory=FixtureObserver,
                           state_record_factory=fixture_body, **kwargs)


def test_episode_plan_reproducible_grouped_without_chunk_or_variant_leakage():
    plan = make_episode_plan(seed=42, episodes_per_route=4, robots=("panda","xarm7"),
                             mechanisms=(MechanismSpec(), MechanismSpec(.8,2,.8)))
    assert plan == make_episode_plan(seed=42, episodes_per_route=4, robots=("panda","xarm7"),
                                    mechanisms=(MechanismSpec(), MechanismSpec(.8,2,.8)))
    assert len({e.sample_id for e in plan}) == len(plan)
    for group in {e.group_id for e in plan}:
        assert len({e.split for e in plan if e.group_id == group}) == 1
    for route in ("upper","side"):
        assert {e.split for e in plan if e.route == route} == {"train","validation"}
    with pytest.raises(ValueError, match="at least two"):
        make_episode_plan(episodes_per_route=1)


def test_source_data_masks_only_successful_train_steps_and_retains_failures(tmp_path):
    plan = make_episode_plan()
    tensors, metadata = collect(plan, role="source_demonstrations", output=tmp_path)
    assert metadata["episodes_attempted"] == 4
    assert metadata["episodes_successful"] == 2 and metadata["episodes_failed"] == 2
    assert metadata["terminal_events"] == 4 and metadata["valid_control_steps"] == 4
    assert tensors["command.raw"].shape == (4,2,2,13)
    assert tensors["prior.training_mask"].sum() == 1
    assert tensors["prior.validation_mask"].sum() == 1
    assert tensors["chunk.body.qpos"].shape == (4,2,7)
    assert tensors["episode.initial_qvel"].shape == (4,7)
    assert not tensors["prior.training_mask"][tensors["episode.split"] == 1].any()
    restored, document = load_dataset(tmp_path)
    np.testing.assert_array_equal(restored["episode.sample_id"], tensors["episode.sample_id"])
    assert document["provenance"][0]["scene"]["model_sha256"] == "fixture-panda"
    assert document["code_fingerprint"]["aggregate"] == metadata["code_fingerprint"]["aggregate"]
    features = policy_features(restored)
    assert tuple(features) == POLICY_FEATURES
    assert not any("mechanism" in name or "status" in name or "split" in name for name in features)


def test_response_data_preserves_world_order_and_has_no_prior_training_mask():
    plan = make_episode_plan(robots=("panda","xarm7"), mechanisms=(MechanismSpec(), MechanismSpec(.8,2,.8)))
    # Interleave topologies; collector grouping must not alter episode identity.
    plan = tuple(sorted(plan, key=lambda e:e.sample_id))
    tensors, metadata = collect(plan, role="response_development")
    np.testing.assert_array_equal(tensors["episode.sample_id"], [e.sample_id for e in plan])
    np.testing.assert_allclose(tensors["diagnostic.mechanism_truth"],
                               [[e.mechanism.servo_gain,e.mechanism.delay_steps,e.mechanism.payload_mass] for e in plan])
    assert not tensors["prior.training_mask"].any()
    assert metadata["execution"]["microbatch_size"] == 2
    assert len(metadata["provenance"]) == 2
    with pytest.raises(ValueError, match="nominal Panda"):
        collect(plan, role="source_demonstrations")


def test_nonfinite_raw_commands_remain_finite_with_explicit_diagnostic_mask():
    def commands(task, skill, route, steps):
        array = np.zeros((steps,13), dtype=np.float32)
        array[0,0] = np.nan
        return array
    tensors, metadata = collect(make_episode_plan(), role="source_demonstrations", command_factory=commands)
    assert all(np.isfinite(array).all() for array in tensors.values())
    assert not tensors["command.raw_finite"][:,0,0].any()
    assert metadata["nonfinite_values_sanitized"]["command.raw"] == 4
    assert metadata["episodes_failed"] == 4
    assert not tensors["prior.training_mask"].any()


def test_split_leakage_and_privileged_feature_whitelist_rejected():
    plan = make_episode_plan()
    leaked = list(plan)
    leaked[1] = replace(leaked[1], split="train" if leaked[0].split == "validation" else "validation")
    with pytest.raises(ValueError, match="cannot cross"):
        collect(leaked, role="source_demonstrations")
    tensors, metadata = collect(plan, role="source_demonstrations")
    with pytest.raises(ValueError, match="whitelist"):
        validate_dataset(tensors, dict(metadata, policy_feature_whitelist=[*POLICY_FEATURES,"diagnostic.mechanism_truth"]))
    for field in ("prior.training_mask", "trace.terminal", "episode.observed_mode"):
        changed = dict(tensors)
        changed[field] = tensors[field].copy()
        if field == "prior.training_mask":
            changed[field] = tensors["prior.validation_mask"].copy()
        else:
            changed[field].flat[0] = not bool(changed[field].flat[0])
        with pytest.raises(ValueError):
            validate_dataset(changed, metadata)
    with pytest.raises(ValueError, match="denominator"):
        validate_dataset(tensors, dict(metadata, episodes_failed=100))


def test_observed_modes_use_only_valid_actual_crossing_evidence():
    position = np.array([[[[.5,0.,.7],[.5,0.,.7]]],
                         [[[.5,-.2,.55],[.5,-.2,.55]]],
                         [[[.4,0.,.7],[.4,-.2,.55]]],
                         [[[.5,0.,.7],[.5,-.2,.55]]]])
    valid = np.ones((4,1,2), dtype=bool)
    np.testing.assert_array_equal(observed_route_modes(position, valid), [0,1,-1,2])
    np.testing.assert_array_equal(observed_route_modes(position, ~valid), [-1,-1,-1,-1])

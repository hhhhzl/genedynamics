"""Physical source support, semantic isolation and compiled batch conformance."""
import numpy as np
import pytest

from genedynamics.envs.domains.manipulation.transfer_workcell.runtime import TransferWorkcell
from genedynamics.tasks.transfer import ExecutionSpec, ObstacleTransferConfig, RandomRequest
from genedynamics.envs.domains.manipulation.transfer_workcell.queries.source import SourceQueryAdapter, SourceQuerySpec, SupportReason, TaskSemantics
from genedynamics.tasks.transfer.runtime import Status


def execution(backend):
    return ExecutionSpec(backend=backend, physics_backend="mjx" if backend == "jax" else "mujoco")


def semantics(positions, twists=None, phases=None):
    from genedynamics.envs.domains.manipulation.transfer_workcell.workcell import rotation
    batch = len(positions)
    return TaskSemantics(np.asarray(positions, np.float32), np.broadcast_to(rotation().astype(np.float32), (batch, 3, 3)).copy(),
                         np.zeros((batch, 6), np.float32) if twists is None else np.asarray(twists, np.float32),
                         np.zeros(batch, np.int32) if phases is None else np.asarray(phases, np.int32))


@pytest.fixture(scope="module")
def adapters():
    return [SourceQueryAdapter(execution=execution(b)) for b in ("numpy", "jax")]


def test_query_metadata_import_and_rejections():
    import subprocess
    import sys
    code = "import sys; from genedynamics.envs.domains.manipulation.transfer_workcell.queries.source import SourceQueryAdapter; assert not any(x in sys.modules for x in ('numpy','jax','torch','mujoco'))"
    subprocess.run([sys.executable, "-S", "-c", code], check=True)
    with pytest.raises(ValueError):
        SourceQuerySpec(iterations=0)
    with pytest.raises(ValueError, match="Panda"):
        SourceQueryAdapter(task=ObstacleTransferConfig(robot="xarm7"))


def test_target_observation_projects_to_valid_panda_with_velocity(adapters):
    outputs = []
    for adapter in adapters:
        backend = adapter.execution.backend
        target = TransferWorkcell(execution=execution(backend), task=ObstacleTransferConfig(robot="xarm7"))
        state = target.reset_batch(RandomRequest(4, (10, 20)), noise_batch=np.zeros((2, 7), np.float32))
        commands = np.zeros((2, 2, 13), np.float32)
        commands[0, :, 0], commands[1, :, 1] = .06, -.04
        state = target.collect_chunk(state, commands).state
        semantic = adapter.observe(target, state)
        result = adapter.build(semantic, RandomRequest(4, (10, 20)))
        adapter.synchronize(result)
        np.testing.assert_array_equal(result.support, [True, True])
        assert np.max(result.position_error) < .001
        assert np.max(result.orientation_error) < .01
        assert np.max(result.twist_error) < .005
        # Source controller owns its own fresh history. Target history is never copied.
        np.testing.assert_array_equal(result.virtual_state.controller.command_buffer, 0)
        np.testing.assert_array_equal(result.virtual_state.controller.integral_error, 0)
        np.testing.assert_array_equal(result.virtual_state.sample_id, [10, 20])
        source = result.virtual_state
        source_qpos = (np.stack([d.qpos for d in source.physics]) if backend == "numpy" else np.asarray(source.physics.qpos))
        target_qpos = (np.stack([d.qpos for d in state.physics]) if backend == "numpy" else np.asarray(state.physics.qpos))
        assert not np.allclose(source_qpos, target_qpos)
        response = adapter._impl.runner.collect_chunk(source, np.zeros((2, 2, 13), np.float32))
        adapter.synchronize(response)
        np.testing.assert_array_equal(response.trace.valid, True)
        assert np.all(np.isfinite(response.trace.tcp_position))
        outputs.append(result)
    np.testing.assert_allclose(outputs[0].position_error, outputs[1].position_error, atol=3e-5)
    np.testing.assert_allclose(outputs[0].twist_error, outputs[1].twist_error, atol=1e-5)


def test_failure_masks_retain_all_requests(adapters):
    shared = semantics([(.38, -.12, .55), (2., 0., 2.), (.5, 0., .52), (.38, -.12, .55), (.38, -.12, .55), (.38, -.12, .55)])
    shared.tcp_rotation[3] = np.eye(3, dtype=np.float32)*2
    shared.tcp_position[3, 0] = np.nan
    shared.tcp_twist[4, 0] = 100.
    shared.phase[5] = 1
    for adapter in adapters:
        result = adapter.build(shared, RandomRequest(7, tuple(range(6))))
        adapter.synchronize(result)
        np.testing.assert_array_equal(result.support, [True, False, False, False, False, False])
        np.testing.assert_array_equal(result.reason, [SupportReason.SUPPORTED, SupportReason.POSE_UNREACHABLE,
                                                     SupportReason.COLLISION, SupportReason.INVALID_SEMANTICS,
                                                     SupportReason.VELOCITY_UNREACHABLE, SupportReason.UNSUPPORTED_PHASE])
        assert np.all(np.isfinite(result.position_error))
        assert np.all(np.isfinite(result.orientation_error))
        assert np.all(np.isfinite(result.twist_error))
        np.testing.assert_array_equal(result.virtual_state.status, [0] + [Status.SOURCE_SUPPORT_FAILURE]*5)


def test_jax_query_identity_warmstart_and_device_residency(adapters):
    import jax
    from dataclasses import replace
    adapter = adapters[1]
    semantic = semantics([(.38, -.12, .55), (.39, -.12, .56)])
    request = RandomRequest(1, (70, 80))
    first = adapter.build(semantic, request)
    warm = adapter.build(semantic, request, previous=first.virtual_state)
    permuted = adapter.build(TaskSemantics(*(x[::-1].copy() for x in semantic)), RandomRequest(1, (80, 70)))
    adapter.synchronize(permuted)
    np.testing.assert_array_equal(first.support, [True, True])
    np.testing.assert_allclose(first.virtual_state.physics.qpos, permuted.virtual_state.physics.qpos[::-1], atol=1e-6)
    np.testing.assert_allclose(warm.virtual_state.physics.qpos, first.virtual_state.physics.qpos, atol=1e-6)
    mismatched = adapter.build(semantic, RandomRequest(1, (80, 70)), previous=first.virtual_state)
    np.testing.assert_allclose(mismatched.virtual_state.physics.qpos, first.virtual_state.physics.qpos, atol=1e-6)
    corrupted = first.virtual_state._replace(physics=first.virtual_state.physics.replace(qpos=first.virtual_state.physics.qpos.at[0, 0].set(np.nan)))
    repaired = adapter.build(semantic, request, previous=corrupted)
    np.testing.assert_allclose(repaired.virtual_state.physics.qpos, first.virtual_state.physics.qpos, atol=1e-6)
    device_semantic = TaskSemantics(*(jax.device_put(x) for x in semantic))
    with jax.transfer_guard("disallow"):
        guarded = adapter.build(device_semantic, request, previous=first.virtual_state)
        adapter.synchronize(guarded)
    np.testing.assert_array_equal(guarded.support, [True, True])
    micro = SourceQueryAdapter(execution=replace(adapter.execution, microbatch_size=1))
    small = micro.build(device_semantic, request)
    micro.synchronize(small)
    np.testing.assert_allclose(small.virtual_state.physics.qpos, first.virtual_state.physics.qpos, atol=2e-6)
    with jax.transfer_guard("disallow"):
        replay = micro.build(device_semantic, request, previous=small.virtual_state)
        micro.synchronize(replay)
    np.testing.assert_array_equal(replay.support, [True, True])
    with pytest.raises(ValueError, match="shape"):
        adapter.build(semantic, RandomRequest(1, (70,)))

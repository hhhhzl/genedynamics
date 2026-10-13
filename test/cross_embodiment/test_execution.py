"""Actual MuJoCo/MJX CPU execution; no fake physics in backend conformance."""
import numpy as np
import pytest

from genedynamics.learning.cross_embodiment import CrossEmbodiment, ObstacleTransferConfig, ExecutionSpec, MechanismSpec, RandomRequest
from genedynamics.learning.cross_embodiment.contracts import Status


def make(backend="numpy", robot="panda", **kwargs):
    return CrossEmbodiment(execution=ExecutionSpec(backend=backend, physics_backend="mjx" if backend == "jax" else "mujoco"),
                           task=ObstacleTransferConfig(robot=robot, **kwargs))


@pytest.fixture(scope="module", params=["panda", "xarm7"])
def pair(request):
    return make("numpy", request.param), make("jax", request.param)


def test_real_scene_and_nominal_gravity(pair):
    reference, _ = pair
    impl = reference._impl
    model = impl.model
    assert model.nq == model.nv == model.nu == 7
    assert all(model.geom(f"collision_link{i}").contype for i in range(1, 8))
    assert model.geom("payload_bar").contype
    state = reference.reset_batch(RandomRequest(1, (0,)), noise_batch=np.zeros((1,7)))
    np.testing.assert_allclose(impl._gravity(state.physics[0]), state.physics[0].qfrc_bias, atol=1e-8)


def test_jax_reference_chunk_parity_and_continuation(pair):
    reference, compiled = pair
    noise = np.array([[0.]*7, [.2,-.3,.1,.2,.1,-.1,.1]], dtype=np.float32)
    request = RandomRequest(7, (11, 22))
    s0 = reference.reset_batch(request, noise_batch=noise)
    j0 = compiled.reset_batch(request, noise_batch=noise)
    commands = np.zeros((2, 4, 13), dtype=np.float32)
    commands[0,:,0] = .08
    commands[1,:,1] = -.04
    commands[:,:,5] = .1
    commands[:,:,6:12] = .15
    commands[:,:,12] = 100.  # ObstacleTransfer must mask the unused force channel.
    result = reference.collect_chunk(s0, commands)
    actual = compiled.collect_chunk(j0, commands)
    compiled.synchronize(actual)
    np.testing.assert_allclose(actual.trace.tcp_position, result.trace.tcp_position, atol=3e-5, rtol=1e-4)
    np.testing.assert_allclose(actual.trace.actuator_command, result.trace.actuator_command, atol=.02, rtol=.002)
    np.testing.assert_array_equal(actual.trace.status, result.trace.status)
    assert np.all(np.asarray(actual.trace.command)[...,12] == 0)
    np.testing.assert_allclose(np.asarray(actual.trace.command)[:,0,0], [.012,0], atol=1e-8)
    np.testing.assert_array_equal(actual.state.steps, [4,4])
    first = compiled.collect_chunk(j0, commands[:,:2])
    second = compiled.collect_chunk(first.state, commands[:,2:])
    np.testing.assert_allclose(actual.state.physics.qpos, second.state.physics.qpos, atol=2e-6)
    # Numerical operations and old state are immutable; differing worlds stay independent.
    np.testing.assert_array_equal(j0.steps, [0,0])
    assert not np.allclose(actual.trace.tcp_position[0], actual.trace.tcp_position[1])
    import jax
    device_commands = jax.device_put(commands)
    with jax.transfer_guard("disallow"):
        guarded = compiled.collect_chunk(j0, device_commands)
        compiled.synchronize(guarded)


def test_payload_truth_is_not_used_by_nominal_compensation():
    reference = make()
    perturbed = CrossEmbodiment(execution=ExecutionSpec(backend="numpy", physics_backend="mujoco"),
                               mechanism=MechanismSpec(payload_mass=.9))
    a = reference.reset_batch(RandomRequest(0,(0,)), noise_batch=np.zeros((1,7))).physics[0]
    b = perturbed.reset_batch(RandomRequest(0,(0,)), noise_batch=np.zeros((1,7))).physics[0]
    np.testing.assert_allclose(reference._impl._gravity(a), perturbed._impl._gravity(b), atol=1e-9)
    assert np.linalg.norm(a.qfrc_bias-b.qfrc_bias) > .1


def test_fp32_execution_preserves_other_solvers_x64_setting():
    import jax
    with jax.experimental.enable_x64():
        runner = make("jax")
        state = runner.reset_batch(RandomRequest(0,(0,1)), noise_batch=np.zeros((2,7)))
        result = runner.collect_chunk(state, np.zeros((2,4,13), np.float32))
        runner.synchronize(result)
        assert jax.config.x64_enabled
        assert result.trace.tcp_position.dtype == np.float32
        assert result.state.physics.qpos.dtype == np.float32


def test_identity_seed_and_partial_reset(pair):
    _, runner = pair
    request = RandomRequest(31, (4, 17))
    original = runner.reset_batch(request)
    permuted = runner.reset_batch(RandomRequest(31, (17,4)))
    np.testing.assert_array_equal(original.physics.qpos[0], permuted.physics.qpos[1])
    advanced = runner.collect_chunk(original, np.zeros((2,2,13),np.float32)).state
    reset = runner.reset_batch(RandomRequest(32, (40,170)), previous=advanced, mask=np.array([True,False]))
    np.testing.assert_array_equal(reset.steps, [0,2])
    np.testing.assert_array_equal(reset.sample_id, [40,17])
    np.testing.assert_array_equal(reset.physics.qpos[1], advanced.physics.qpos[1])
    np.testing.assert_array_equal(reset.controller.command_buffer[0], 0.)


def test_device_and_capability_rejected_early():
    with pytest.raises(NotImplementedError, match="collect/evaluate"):
        CrossEmbodiment(execution=ExecutionSpec(execution_mode="train"))
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        CrossEmbodiment(execution=ExecutionSpec(device="gpu:999"))
    with pytest.raises(ValueError):
        make().collect_chunk(make().reset_batch(RandomRequest(0,(0,))), np.zeros((1,2,7)))


def test_delay_buffer_continues_across_chunks():
    runner = CrossEmbodiment(execution=ExecutionSpec(backend="numpy", physics_backend="mujoco"),
                             mechanism=MechanismSpec(delay_steps=2))
    state = runner.reset_batch(RandomRequest(0,(0,)))
    commands = np.zeros((1,4,13), np.float32)
    commands[:,:,0] = .1
    first = runner.collect_chunk(state, commands[:,:2])
    second = runner.collect_chunk(first.state, commands[:,2:])
    np.testing.assert_array_equal(first.trace.command, 0.)
    np.testing.assert_allclose(second.trace.command[0,:,0], [.012,.024])


def test_terminal_hold_orientation_timeout_and_invalid_input():
    for backend in ("numpy", "jax"):
        runner = make(backend, start_position=(.64,.08,.55), goal_yaw=0., reset_joint_noise=0., hold_seconds=.04)
        state = runner.reset_batch(RandomRequest(0,(0,1)),noise_batch=np.zeros((2,7)))
        # Second world has an invalid command; this is counted and terminates.
        commands = np.zeros((2,4,13),np.float32)
        commands[1,0,0] = np.nan
        result = runner.collect_chunk(state, commands)
        np.testing.assert_array_equal(result.state.status, [Status.SUCCESS,Status.NONFINITE])
        np.testing.assert_array_equal(result.trace.terminal.sum(axis=1), [1,1])
        np.testing.assert_array_equal(result.trace.valid.sum(axis=1), [2,1])
        np.testing.assert_array_equal(result.state.steps, [2,1])
        angle = make(backend, start_position=(.64,.08,.55), goal_yaw=.15, reset_joint_noise=0.,
                     hold_seconds=.02, timeout_seconds=.04)
        result = angle.collect_chunk(angle.reset_batch(RandomRequest(0,(0,)),noise_batch=np.zeros((1,7))), np.zeros((1,4,13),np.float32))
        assert int(result.state.status[0]) == Status.TIMEOUT


def test_collision_is_a_failure_in_both_backends(pair):
    reference, compiled = pair
    for runner in (reference, compiled):
        # Place the arm inside the obstacle using a diagnostic-only reset pose.
        from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import solve_ik, rotation
        impl = runner._impl
        q = solve_ik(impl.scene.model, impl.sid, impl.scene.start_qpos, np.array([.50,0.,.52]), rotation())
        noise = ((q-impl.scene.start_qpos)/runner.task.reset_joint_noise)[None].astype(np.float32)
        state = runner.reset_batch(RandomRequest(0,(0,)), noise_batch=noise)
        result = runner.collect_chunk(state, np.zeros((1,2,13),np.float32))
        assert int(result.state.status[0]) == Status.COLLISION
        assert int(np.asarray(result.trace.terminal).sum()) == 1

"""Same-topology mechanism batches and bounded whole-chunk dispatch."""
import numpy as np
import pytest

from genedynamics.learning.cross_embodiment import (
    CrossEmbodiment, ExecutionSpec, MechanismSpec, RandomRequest,
)


MECHANISMS = (MechanismSpec(servo_gain=.8, delay_steps=0, payload_mass=.35),
              MechanismSpec(servo_gain=1.15, delay_steps=2, payload_mass=.85))


def make(backend, mechanism, microbatch_size=None):
    return CrossEmbodiment(
        execution=ExecutionSpec(backend=backend, physics_backend="mjx" if backend == "jax" else "mujoco",
                                microbatch_size=microbatch_size),
        mechanism=mechanism,
    )


@pytest.fixture(scope="module", params=["numpy", "jax"])
def runners(request):
    backend = request.param
    return (backend, make(backend, MECHANISMS), make(backend, MECHANISMS, 1),
            tuple(make(backend, value) for value in MECHANISMS))


def commands():
    result = np.zeros((2, 4, 13), np.float32)
    result[:, :, 0] = .08
    result[:, :, 5] = .1
    result[:, :, 6:12] = .15
    return result


def qpos(state, backend):
    return np.stack([value.qpos for value in state.physics]) if backend == "numpy" else np.asarray(state.physics.qpos)


def test_mixed_mechanisms_match_independent_complete_models(runners):
    backend, mixed, _, singles = runners
    noise = np.zeros((2, 7), np.float32)
    initial = mixed.reset_batch(RandomRequest(3, (19, 23)), noise_batch=noise)
    actual = mixed.collect_chunk(initial, commands())
    mixed.synchronize(actual)
    for world, single in enumerate(singles):
        state = single.reset_batch(RandomRequest(3, (19 if world == 0 else 23,)), noise_batch=noise[:1])
        expected = single.collect_chunk(state, commands()[world:world+1])
        single.synchronize(expected)
        np.testing.assert_allclose(qpos(actual.state, backend)[world], qpos(expected.state, backend)[0], atol=3e-6)
        np.testing.assert_allclose(actual.trace.actuator_command[world], expected.trace.actuator_command[0], atol=.003)
        np.testing.assert_array_equal(actual.trace.status[world], expected.trace.status[0])
        if backend == "jax":
            # Ensure the complete MuJoCo-derived mass quantities travel through
            # the batched numeric model, rather than body_mass alone.
            for field in ("body_mass", "body_inertia", "body_subtreemass", "dof_M0", "body_invweight0"):
                if hasattr(mixed._impl.models, field):
                    np.testing.assert_array_equal(getattr(mixed._impl.models, field)[world],
                                                  getattr(single._impl.model, field))
            reference = make("numpy", MECHANISMS[world])
            native_initial = reference.reset_batch(RandomRequest(3, (19 if world == 0 else 23,)), noise_batch=noise[:1])
            native = reference.collect_chunk(native_initial, commands()[world:world+1])
            np.testing.assert_allclose(actual.trace.tcp_position[world], native.trace.tcp_position[0], atol=3e-5)
            np.testing.assert_allclose(actual.trace.actuator_command[world], native.trace.actuator_command[0], atol=.02, rtol=.002)
    np.testing.assert_array_equal(actual.trace.command[1, :2], 0.)
    np.testing.assert_allclose(actual.trace.command[1, 2:, 0], [.012, .024], atol=1e-8)
    assert not np.allclose(qpos(actual.state, backend)[0], qpos(actual.state, backend)[1])
    # Development mechanism truth is manifest metadata, not an observation.
    assert "mechanism" not in actual.trace._fields


def test_microbatch_preserves_identity_continuation_and_terminal_masks(runners):
    backend, mixed, micro, _ = runners
    request = RandomRequest(13, (7, 31))
    initial = mixed.reset_batch(request)
    micro_initial = micro.reset_batch(request)
    np.testing.assert_array_equal(qpos(initial, backend), qpos(micro_initial, backend))
    batch = commands()
    batch[0, 0, 0] = np.nan  # one terminal world must freeze across subbatches
    whole = mixed.collect_chunk(initial, batch)
    first = micro.collect_chunk(micro_initial, batch[:, :2])
    continued = micro.collect_chunk(first.state, batch[:, 2:])
    micro.synchronize(continued)
    np.testing.assert_allclose(qpos(whole.state, backend), qpos(continued.state, backend), atol=3e-6)
    np.testing.assert_array_equal(whole.state.steps, continued.state.steps)
    np.testing.assert_array_equal(whole.state.status, continued.state.status)
    np.testing.assert_array_equal(whole.trace.terminal,
                                  np.concatenate((first.trace.terminal, continued.trace.terminal), axis=1))
    np.testing.assert_array_equal(whole.trace.valid,
                                  np.concatenate((first.trace.valid, continued.trace.valid), axis=1))
    reset = micro.reset_batch(RandomRequest(14, (70, 310)), previous=continued.state,
                             mask=np.array([True, False]))
    np.testing.assert_array_equal(reset.sample_id, [70, 31])
    np.testing.assert_array_equal(reset.steps, [0, 4])
    np.testing.assert_array_equal(qpos(reset, backend)[1], qpos(continued.state, backend)[1])
    np.testing.assert_array_equal(reset.controller.command_buffer[0], 0.)
    if backend == "jax":
        import jax
        device_batch = jax.device_put(commands())
        # Warmed microbatch calls may slice and concatenate device tensors but
        # must not convert a tensor to NumPy or transfer it to the host.
        with jax.transfer_guard("disallow"):
            guarded = micro.collect_chunk(micro_initial, device_batch)
            micro.synchronize(guarded)


def test_mechanism_binding_permutation_and_size_validation(runners):
    backend, mixed, _, _ = runners
    permuted = make(backend, MECHANISMS[::-1], 1)
    original = mixed.reset_batch(RandomRequest(27, (4, 17)))
    reverse = permuted.reset_batch(RandomRequest(27, (17, 4)))
    np.testing.assert_array_equal(qpos(original, backend), qpos(reverse, backend)[::-1])
    actual = mixed.collect_chunk(original, commands())
    reversed_result = permuted.collect_chunk(reverse, commands()[::-1])
    np.testing.assert_allclose(qpos(actual.state, backend), qpos(reversed_result.state, backend)[::-1], atol=3e-6)
    with pytest.raises(ValueError, match="mechanism batch"):
        mixed.reset_batch(RandomRequest(0, (0,)))
    assert mixed.describe()["mechanism_batch_size"] == 2
    assert permuted.describe()["microbatch_size"] == 1


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_microbatch_size(size):
    with pytest.raises(ValueError, match="microbatch_size"):
        ExecutionSpec(microbatch_size=size)


def test_empty_mechanism_tuple_is_not_silently_defaulted():
    with pytest.raises(TypeError, match="mechanism"):
        CrossEmbodiment(mechanism=())


def test_delay_capacity_is_validated_for_every_mechanism():
    with pytest.raises(ValueError, match="delay"):
        CrossEmbodiment(mechanism=(MechanismSpec(), MechanismSpec(delay_steps=5)))

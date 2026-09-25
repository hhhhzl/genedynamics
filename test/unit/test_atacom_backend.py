"""Geometry and wrapper contracts for the task-adapted ATACOM baseline."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from genedynamics.solvers.single.atacom.backends.atacom_jax import (
    _damped_qr_pinv_null,
    _pinv_null,
    atacom_constraint_dims,
    atacom_null_dim,
    init_slack,
    make_atacom_transform,
)
from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper
from genedynamics.solvers.single.atacom.prior import AtacomHorizonPrior
from genedynamics.solvers.single.dial.spline import NodeSpline


@struct.dataclass
class _State:
    obs: jax.Array
    info: dict


class _ToyConstraintEnv:
    action_size = 4
    observation_size = 2
    manifold_constraint_size = 1
    inequality_constraint_size = 1
    backend = "generalized"
    dt = 1.0

    def reset(self, _key):
        return _State(
            obs=jnp.zeros((2,), jnp.float32),
            info={"goal": jnp.asarray(0.25, jnp.float32)},
        )

    def manifold_residual(self, state, actions):
        return actions[..., 0] - state.info["goal"]

    def constraint_residual(self, _state, action):
        return jnp.zeros((0,), action.dtype), jnp.asarray([action[1] - 0.5])

    def step(self, state, action):
        return state.replace(obs=state.obs.at[0].set(action[0]))


class _ToyPredictiveConstraintEnv(_ToyConstraintEnv):
    """State safety is static, while the known dynamics expose control authority."""

    def constraint_residual(self, state, action):
        del action
        return jnp.zeros((0,), state.obs.dtype), jnp.asarray(
            [state.obs[1] - 0.5], state.obs.dtype
        )

    def atacom_constraint_residual(self, state, action):
        # Discrete counterpart of g(q) + K_g J_g(q) q_dot: the candidate
        # action controls the predicted next safety coordinate.
        return jnp.zeros((0,), action.dtype), jnp.asarray(
            [state.obs[1] + action[1] - 0.5], action.dtype
        )


def _augmented_constraint(env, state, action, slack):
    f = env.manifold_residual(state, action[None]).reshape(-1)
    g = env.constraint_residual(state, action)[1].reshape(-1)
    return jnp.concatenate([f, g + 0.5 * slack ** 2])


def test_environment_owned_dimensions_determine_policy_null_size():
    env = _ToyConstraintEnv()
    assert atacom_constraint_dims(env) == (1, 1)
    assert atacom_null_dim(env) == 3


def test_svd_null_basis_annihilates_augmented_jacobian():
    A = jnp.asarray([
        [1.0, 2.0, -1.0, 0.0, 0.5],
        [0.0, 1.0, 1.0, -2.0, 0.3],
    ], jnp.float32)
    pinv, null = _pinv_null(A, n_c=2)
    np.testing.assert_allclose(A @ null, 0.0, atol=2e-6)
    np.testing.assert_allclose(A @ pinv, jnp.eye(2), atol=2e-6)


def test_damped_qr_projection_is_finite_for_rank_deficient_rows():
    A = jnp.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], jnp.float32)
    pinv, null = _damped_qr_pinv_null(A, n_c=3)
    assert pinv.shape == (4, 3)
    assert null.shape == (4, 1)
    assert bool(jnp.all(jnp.isfinite(pinv)))
    assert bool(jnp.all(jnp.isfinite(null)))
    np.testing.assert_allclose(A @ null, 0.0, atol=2e-6)


def test_error_correction_reduces_augmented_constraint_norm():
    env = _ToyConstraintEnv()
    state = env.reset(jax.random.PRNGKey(0))
    slack = init_slack(env, state)
    zero = jnp.zeros((env.action_size,), jnp.float32)
    before = _augmented_constraint(env, state, zero, slack)
    transform = make_atacom_transform(env, Kc=1.0, time_step=1.0)
    action, slack_next = transform(
        state, jnp.zeros((atacom_null_dim(env),), jnp.float32), slack
    )
    after = _augmented_constraint(env, state, action, slack_next)
    assert float(jnp.linalg.norm(after)) < float(jnp.linalg.norm(before))
    assert action.shape == (4,)
    assert slack_next.shape == (1,)


def test_task_owned_predictive_constraint_supplies_action_safety_jacobian():
    env = _ToyPredictiveConstraintEnv()
    state = env.reset(jax.random.PRNGKey(0)).replace(
        obs=jnp.asarray([0.0, 0.8], jnp.float32)
    )
    zero = jnp.zeros((env.action_size,), jnp.float32)
    static_g = env.constraint_residual(state, zero)[1]
    static_jacobian = jax.jacobian(
        lambda u: env.constraint_residual(state, u)[1]
    )(zero)
    predictive_jacobian = jax.jacobian(
        lambda u: env.atacom_constraint_residual(state, u)[1]
    )(zero)
    np.testing.assert_allclose(static_jacobian, 0.0)
    assert float(jnp.linalg.norm(predictive_jacobian)) > 0.9

    slack = init_slack(env, state)
    transform = make_atacom_transform(env, Kc=1.0, time_step=1.0)
    action, _ = transform(
        state, jnp.zeros((atacom_null_dim(env),), jnp.float32), slack
    )
    corrected_g = env.atacom_constraint_residual(state, action)[1]
    assert float(corrected_g[0]) < float(static_g[0])


def test_wrapper_exposes_3d_policy_action_and_executes_4d_control():
    env = _ToyConstraintEnv()
    wrapped = AtacomEnvWrapper(env, Kc=1.0)
    state = wrapped.reset(jax.random.PRNGKey(0))
    assert wrapped.action_size == 3
    next_state = wrapped.step(state, jnp.zeros((3,), jnp.float32))
    assert next_state.info["atacom_u"].shape == (4,)
    assert next_state.info["atacom_s"].shape == (1,)
    assert bool(jnp.all(jnp.isfinite(next_state.info["atacom_u"])))


class _ToyTangentPrior:
    output_dim = 3

    def act(self, obs, *, key=None, deterministic=True):
        if deterministic:
            return jnp.zeros((3,), jnp.float32)
        return 0.05 * jax.random.normal(key, (3,))

    def logp_of_sequence(self, obs_seq, act_seq):
        del obs_seq
        return -0.5 * jnp.sum(act_seq * act_seq)


def test_horizon_prior_maps_tangent_samples_to_full_control_nodes():
    env = _ToyConstraintEnv()
    state = env.reset(jax.random.PRNGKey(0))
    prior = AtacomHorizonPrior(
        env, _ToyTangentPrior(), Hsample=16, Hnode=4, ctrl_dt=0.02
    )
    proposals = prior.sample_horizons(
        state, key=jax.random.PRNGKey(5), n_samples=3
    )
    assert proposals.trajectories.shape == (3, 5, 4)
    assert proposals.log_prob.shape == (3,)
    assert np.all(np.asarray(proposals.expert_id) == 2)
    assert bool(jnp.all(jnp.isfinite(proposals.trajectories)))
    dense = NodeSpline.build(4, 16, 0.02).node2u_batch(
        proposals.trajectories
    )
    # The task equality is enforced by the model-based ATACOM transform, not
    # by the tangent network.
    np.testing.assert_allclose(dense[:, :, 0], 0.25, atol=2e-5)


def test_horizon_prior_carries_slack_only_after_commit():
    env = _ToyConstraintEnv()
    state = env.reset(jax.random.PRNGKey(0))
    prior = AtacomHorizonPrior(
        env, _ToyTangentPrior(), Hsample=4, Hnode=2, ctrl_dt=0.02
    )
    prior.warm_start(state)
    pending = np.asarray(prior._pending_slack)
    assert prior._slack is None
    prior.commit()
    np.testing.assert_allclose(prior._slack, pending)
    assert prior._pending_slack is None
    prior.reset()
    assert prior._slack is None

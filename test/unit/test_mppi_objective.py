import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from genedynamics.core.energy import EnergyTerm, LegacyEnergyFunctional
from genedynamics.solvers.single.mppi.backends.mppi_jax import MPPIBackendJax


class _ScalarEnv:
    act_dim = 1

    def __init__(self, target: float = 1.0):
        self.target = np.asarray([target], dtype=np.float32)

    def jax_transition(self, state, action):
        return state + action

    def terminal_distance_jax(self, state):
        return jnp.abs(state[0] - self.target[0])


def _backend(energy, **overrides):
    params = {
        "env_adapter": _ScalarEnv(),
        "legacy_energy": energy,
        "horizon": 3,
        "dt": 1.0,
        "num_samples": 8,
        "num_iterations": 2,
        "noise_sigma": 0.1,
        "lambda_": 1.0,
        "action_limit": 1.0,
        "terminal_energy_weight": 0.0,
        "guide_weight": 0.0,
        "position_dim": 1,
        "seed": 0,
    }
    params.update(overrides)
    return MPPIBackendJax(**params)


def test_rollout_uses_time_indices_and_terminal_distance():
    energy = LegacyEnergyFunctional(
        {"time": EnergyTerm(lambda _x, _u, ctx: jnp.asarray(ctx["t"]), 1.0)}
    )
    backend = _backend(energy, terminal_energy_weight=10.0)

    rewards = np.asarray(
        backend._rollout_rewards_fn(
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.zeros((3, 1), dtype=jnp.float32),
        )
    )

    # Stage costs are t={0,1,2}; terminal distance is 1 with weight 10.
    np.testing.assert_allclose(rewards, np.asarray([0.0, -1.0, -12.0]))


def test_zero_mean_remains_candidate_when_noise_is_worse():
    energy = LegacyEnergyFunctional(
        {"control": EnergyTerm(lambda _x, u, _ctx: jnp.sum(u**2), 1.0)}
    )
    backend = _backend(
        energy,
        env_adapter=_ScalarEnv(target=0.0),
        horizon=4,
        noise_sigma=0.5,
    )

    result = backend.plan(
        np.asarray([0.0], dtype=np.float32), jax.random.PRNGKey(0)
    )

    np.testing.assert_allclose(result["actions"], np.zeros((4, 1)), atol=1e-7)
    assert result["total_reward"] == pytest.approx(0.0)
    assert result["reward_history"].shape == (backend.num_iterations,)
    assert result["diffusion_actions_traj"].shape == (
        backend.num_iterations,
        backend.horizon,
        backend.act_dim,
    )
    assert result["diffusion_sampled_actions"].shape == (
        backend.num_iterations,
        backend.num_samples,
        backend.horizon,
        backend.act_dim,
    )
    assert result["optimization_cost_history"].shape == (
        backend.num_iterations,
    )
    # Shared cost-viz history is clean-to-noisy, so index 0 is the final
    # incumbent and must match the action sequence returned by MPPI.
    np.testing.assert_allclose(
        result["diffusion_actions_traj"][0], result["actions"], atol=1e-7
    )

"""Fedguide gates: shared priors package + the additive prior= seam.

Covers the brax-free parts (protocols, registry, elite buffer, jax diffusion
prior, and the prior= warm-start seam's additive safety on a mock rollout). The
brax RL prior (jax backend = brax.training) is exercised in the docker test."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genedynamics.learning.priors import (
    DiffusionPrior, EliteBuffer, Prior, list_priors, make_prior,
)
from genedynamics.solvers.single.mdac.backends.mdac_jax import MdacBackendJax


# --- registry / elite buffer ---

def test_registry_names():
    assert set(list_priors()) == {"rl", "diffusion"}


def test_elite_buffer_topk_and_capacity():
    eb = EliteBuffer(capacity=3, seed=0)
    for r in [0.1, 0.9, 0.5, 0.2]:
        eb.add(np.zeros((2, 3)), np.full((2, 3), r), r)
    assert len(eb) == 3                                  # lowest (0.1) evicted
    _, _, rw = eb.topk(2)
    assert np.allclose(rw, [0.9, 0.5])
    o, a, r = eb.sample(4, elite_frac=0.5)
    assert o.shape[0] == 4 and a.shape[0] == 4


# --- jax diffusion prior (pure jax, no brax) ---

def test_diffusion_prior_protocol_and_ops():
    dp = make_prior("diffusion", dim=12, nu=2, n_warm_nodes=6, n_steps=10)
    assert isinstance(dp, DiffusionPrior) and isinstance(dp, Prior)
    s = dp.score(jnp.zeros(12), 3)
    assert s.shape == (12,) and jnp.all(jnp.isfinite(s))
    dp.train(np.random.default_rng(0).standard_normal((24, 12)).astype("float32"),
             steps=5, batch=8)
    assert dp.warm_start(None).shape == (6, 2)


# --- prior= seam additive safety (mock rollout, no brax) ---

class _FakePrior:
    """Minimal Prior: constant warm-start, for additive-safety testing."""
    output_dim = 2

    def __init__(self, hn1, nu, val=0.3):
        self._u = jnp.full((hn1, nu), val, jnp.float32)

    def act(self, obs, *, key=None, deterministic=True):
        return self._u[0]

    def logp_of_sequence(self, obs_seq, act_seq):
        return jnp.array(0.0)

    def warm_start(self, state):
        return self._u


COMMON = dict(nu=2, Hnode=4, Hsample=16, Nsample=32, Ndiffuse_init=4, Ndiffuse=2,
              temp_sample=0.06, action_limit=1.0, ctrl_dt=0.02, seed=0)
TARGET = jnp.array([0.4, -0.2])


def _mock_rollout(state, us, t0):
    return -jnp.sum((us - TARGET) ** 2, axis=-1)


def _replan(backend, rng):
    return backend.replan(jnp.zeros((1,)), backend.init_plan_var(),
                          backend.make_schedule(backend.Ndiffuse_init), rng)


def test_fake_prior_conforms():
    assert isinstance(_FakePrior(5, 2), Prior)


def test_prior_seam_additive_safety():
    rng = jax.random.PRNGKey(5)
    base = _replan(MdacBackendJax(rollout_fn=_mock_rollout, **COMMON), rng)

    # lambda_shift=1.0 => U_init = 1*U_shift + 0*U_rl = U_shift => byte-identical
    b1 = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    b1.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    b1.use_rl_prior, b1.prior_lambda_shift = True, 1.0
    # Isolate the legacy warm-start seam. The full Phase-4 path intentionally
    # keeps U_rl as a separate candidate even when lambda_shift=1.
    b1.prior_include_incumbent = False
    b1.prior_trust_radius = 0.0
    b1.prior_acceptance = False
    y1 = _replan(b1, rng)
    assert float(jnp.max(jnp.abs(y1 - base))) < 1e-6      # prior mix is identity at lam=1

    # lambda_shift=0.0 => U_init = U_rl (nonzero) => changes the plan
    b0 = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    b0.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    b0.use_rl_prior, b0.prior_lambda_shift = True, 0.0
    y0 = _replan(b0, rng)
    assert float(jnp.max(jnp.abs(y0 - base))) > 1e-4      # warm-start replaced => differs
    assert jnp.all(jnp.isfinite(y0))


def test_prior_none_is_unchanged():
    rng = jax.random.PRNGKey(9)
    base = _replan(MdacBackendJax(rollout_fn=_mock_rollout, **COMMON), rng)
    b = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)   # prior stays None
    assert b.prior is None
    assert float(jnp.max(jnp.abs(_replan(b, rng) - base))) == 0.0


def test_prior_acceptance_and_force_veto():
    backend = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    U_rl = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.broadcast_to(TARGET, U_rl.shape)

    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)
    selected, info = backend._accept_refinement(
        X0 := jnp.zeros((1,)), U_rl, refined, jnp.float32(0.0)
    )
    np.testing.assert_allclose(selected, refined)
    assert float(info["prior_accepted"]) == 1.0
    assert float(info["prior_predicted_improvement"]) > 0.0

    def force_risk(state, us):
        violation = jnp.mean((us[:, 0] > 0.35).astype(jnp.float32))
        return jnp.asarray([violation, 0.0, 0.0, 0.0])

    backend.risk_fn = force_risk
    selected, info = backend._accept_refinement(
        X0, U_rl, refined, jnp.float32(0.0)
    )
    np.testing.assert_allclose(selected, U_rl)
    assert float(info["prior_accepted"]) == 0.0
    assert float(info["prior_force_veto"]) == 1.0


def test_prior_trust_region_bounds_refinement():
    backend = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    U_rl = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.2)
    backend.prior = _FakePrior(
        COMMON["Hnode"] + 1, COMMON["nu"], val=0.2
    )
    backend.use_rl_prior = True
    backend.prior_trust_radius = 0.1
    projected = backend._project_to_prior(
        jnp.full_like(U_rl, 1.0), U_rl
    )
    rms = jnp.sqrt(jnp.mean((projected - U_rl) ** 2))
    assert float(rms) <= 0.100001

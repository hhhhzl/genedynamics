"""Fedguide gates: shared priors package + the additive prior= seam.

Covers the brax-free parts (protocols, registry, elite buffer, jax diffusion
prior, and the prior= warm-start seam's additive safety on a mock rollout). The
brax RL prior (jax backend = brax.training) is exercised in the docker test."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace
from typing import NamedTuple

from genedynamics.learning.priors import (
    DiffusionPrior, EliteBuffer, Prior, ProposalBatch, StructuredPrior,
    list_priors, make_prior,
)
from genedynamics.learning.reliability import (
    FEATURE_NAMES, LinearReliabilityModel, RISK_NAMES,
)
from genedynamics.solvers.single.mdac.backends.mdac_jax import MdacBackendJax
from genedynamics.solvers.common.rl_policy_controller import RLPolicyController
from genedynamics.solvers.single.mdac.run_experiment import (
    _discover_algorithm_configs,
    _evaluation_configs,
    _prior_diagnostics,
    _solver_cfg,
)


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


class _FakeStructuredPrior(_FakePrior):
    def sample_horizons(self, state, *, key, n_samples):
        return ProposalBatch(
            trajectories=jnp.broadcast_to(
                self._u, (int(n_samples),) + self._u.shape
            ),
            log_prob=jnp.zeros((int(n_samples),), jnp.float32),
            expert_id=jnp.ones((int(n_samples),), jnp.int32),
        )


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


def test_structured_prior_batch_is_jax_pytree_and_backward_compatible():
    prior = _FakeStructuredPrior(5, 2)
    assert isinstance(prior, Prior)
    assert isinstance(prior, StructuredPrior)
    batch = prior.sample_horizons(
        None, key=jax.random.PRNGKey(0), n_samples=3
    )
    assert batch.trajectories.shape == (3, 5, 2)
    assert batch.log_prob.shape == (3,)
    assert batch.expert_id.shape == (3,)
    assert len(jax.tree_util.tree_leaves(batch)) == 3


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


def test_structured_proposals_use_fixed_gaussian_budget_and_are_opt_in():
    rng = jax.random.PRNGKey(23)
    legacy = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    legacy.prior = _FakeStructuredPrior(
        COMMON["Hnode"] + 1, COMMON["nu"], val=0.3
    )
    legacy.use_rl_prior = True
    legacy.prior_stochastic_samples = 0
    y_legacy = _replan(legacy, rng)

    structured = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    structured.prior = legacy.prior
    structured.use_rl_prior = True
    structured.prior_stochastic_samples = 4
    y_structured, info = structured.replan_with_info(
        jnp.zeros((1,)), structured.init_plan_var(),
        structured.make_schedule(structured.Ndiffuse_init), rng,
    )
    assert y_structured.shape == y_legacy.shape
    assert jnp.all(jnp.isfinite(y_structured))
    assert float(info["proposal_stochastic_count"]) == 4.0
    # Four structured candidates replace four Gaussian candidates; the reverse
    # kernel's declared stochastic budget itself never changes.
    assert structured.Nsample == legacy.Nsample == COMMON["Nsample"]


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


def test_learned_reliability_is_additional_acceptance_veto():
    class Reliability:
        def predict_upper(self, features):
            return jnp.asarray([features[0], 0.0, 0.0, 0.0])

        def support_score(self, features):
            return jnp.asarray(0.0)

    backend = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)
    backend.reliability_model = Reliability()
    backend.reliability_feature_fn = lambda state, action: action[:1]
    backend.reliability_force_limit = 0.35
    backend.reliability_deformation_limit = float("inf")
    backend.reliability_risk_tolerance = jnp.zeros((4,), jnp.float32)
    incumbent = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.broadcast_to(TARGET, incumbent.shape)

    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, refined, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, incumbent)
    assert float(info["prior_accepted"]) == 0.0
    assert float(info["prior_force_veto"]) == 1.0
    assert "reliability_risk_refined" in info


def test_linear_reliability_fit_calibration_and_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(160, len(FEATURE_NAMES))).astype(np.float32)
    weights = rng.normal(scale=0.1, size=(len(FEATURE_NAMES), len(RISK_NAMES)))
    train_y = np.maximum(train_x @ weights + 0.2, 0.0).astype(np.float32)
    cal_x = rng.normal(size=(80, len(FEATURE_NAMES))).astype(np.float32)
    cal_y = np.maximum(cal_x @ weights + 0.25, 0.0).astype(np.float32)
    model = LinearReliabilityModel.fit(
        train_x, train_y, cal_x, cal_y, quantile=0.9
    )
    upper = np.asarray(model.predict_upper(cal_x))
    assert upper.shape == cal_y.shape
    assert np.all(np.mean(cal_y <= upper + 1e-6, axis=0) >= 0.9)
    assert np.mean(np.asarray(model.in_support(cal_x))) >= 0.9
    assert not bool(model.in_support(np.full(len(FEATURE_NAMES), 1.0e6)))

    path = tmp_path / "reliability.json"
    model.save(path, metadata={"split": "unit"})
    loaded = LinearReliabilityModel.load(path)
    np.testing.assert_allclose(loaded.predict_upper(cal_x), upper, atol=1e-6)


def test_no_acceptance_skips_counterfactual_rollouts():
    backend = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior_acceptance = False
    backend.risk_fn = lambda state, us: (_ for _ in ()).throw(
        AssertionError("unchecked ablation must not evaluate risk")
    )
    U_rl = jnp.full((COMMON["Hnode"] + 1, COMMON["nu"]), 0.3)
    refined = jnp.broadcast_to(TARGET, U_rl.shape)
    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), U_rl, refined, jnp.float32(0.0)
    )
    np.testing.assert_allclose(selected, refined)
    assert float(info["prior_accepted"]) == 1.0
    assert bool(jnp.isnan(info["prior_predicted_improvement"]))


def test_receding_incumbent_fallback_never_returns_raw_policy():
    backend = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    backend.prior_fallback_mode = "receding_incumbent"
    incumbent = jnp.zeros((COMMON["Hnode"] + 1, COMMON["nu"]), jnp.float32)
    worse = jnp.full_like(incumbent, -0.8)
    backend.risk_fn = lambda state, us: jnp.zeros((4,), jnp.float32)

    selected, info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, worse, jnp.float32(1.0)
    )
    np.testing.assert_allclose(selected, incumbent)
    assert float(info["prior_accepted"]) == 0.0
    assert "prior_risk_incumbent" in info

    cold, cold_info = backend._accept_refinement(
        jnp.zeros((1,)), incumbent, worse, jnp.float32(0.0)
    )
    np.testing.assert_allclose(cold, worse)
    assert float(cold_info["prior_accepted"]) == 1.0


def test_receding_incumbent_uses_terminal_hold_shift():
    backend = MdacBackendJax(rollout_fn=_mock_rollout, **COMMON)
    backend.prior = _FakePrior(COMMON["Hnode"] + 1, COMMON["nu"])
    backend.use_rl_prior = True
    nodes = jnp.linspace(
        -0.4, 0.8, (COMMON["Hnode"] + 1) * COMMON["nu"]
    ).reshape(COMMON["Hnode"] + 1, COMMON["nu"])

    backend.prior_fallback_mode = "rl"
    np.testing.assert_allclose(
        backend.shift(nodes), backend.spline.shift_nodes(nodes)
    )
    backend.prior_fallback_mode = "receding_incumbent"
    np.testing.assert_allclose(
        backend.shift(nodes), backend.spline.shift_nodes_terminal_hold(nodes)
    )


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


def test_raw_rl_controller_matches_receding_horizon_contract():
    class State(NamedTuple):
        obs: jax.Array

    class Env:
        def step(self, state, action):
            return State(obs=state.obs + action)

    controller = RLPolicyController(
        Env(), lambda obs, key: jnp.ones_like(obs) * 0.25
    )
    x0 = State(obs=jnp.zeros((2,), jnp.float32))
    result = controller.run_receding(
        x0,
        3,
        jax.random.PRNGKey(0),
        collect_states=False,
    )
    assert result.horizon == 3
    assert len(result.states) == 2
    np.testing.assert_allclose(result.states[-1].obs, [0.75, 0.75])


def test_algorithm_level_cpu_comparison_matrix_and_policy_routing():
    root = "configs/arm/impedence/cpu_controllability_geometry/comparison"
    configs = _discover_algorithm_configs(root)
    assert len(configs) == 3 * 8
    assert {c["algorithm"] for c in configs} == {
        "dial", "mppi", "pegasusflow", "issa", "atacom",
        "standalone_rl", "model_based_only", "full_mdac",
    }
    assert {c["suite"] for c in configs} == {
        "rigid_cylinder", "soft_cylinder", "soft_unseen",
    }
    assert all(c["seeds"] == [10, 11] for c in configs)
    for cfg in configs:
        factory, sampling = _solver_cfg(cfg)
        uses_policy = cfg["algorithm"] in {
            "standalone_rl", "issa", "atacom", "full_mdac",
        }
        assert ("policy_ckpt" in factory) == uses_policy
        assert "policy_ckpt" not in sampling
        uses_reliability = cfg["algorithm"] in {
            "model_based_only", "full_mdac",
        }
        assert ("reliability_ckpt" in factory) == uses_reliability
        assert "reliability_ckpt" not in sampling
    full = next(c for c in configs if c["algorithm"] == "full_mdac")
    assert full["method_params"]["prior_acceptance"] is True
    assert full["method_params"]["prior_fallback_mode"] == "receding_incumbent"
    assert full["method_params"]["reliability_deformation_limit"] == 0.5
    safe_unseen = next(
        c for c in configs
        if c["algorithm"] == "full_mdac" and c["suite"] == "soft_unseen"
    )
    assert safe_unseen["env_params"]["f_target"] == 15.0
    assert safe_unseen["env_params"]["grav_comp"] == 0.45
    standalone_path = f"{root}/standalone_rl.yaml"
    seeded = _evaluation_configs(
        standalone_path, "/tmp/shared_ppo_seed1.pkl", policy_seed=1
    )
    assert all(c["policy_training_seed"] == 1 for c in seeded)
    assert all(c["policy_training_steps"] == 2_000_000 for c in seeded)
    assert all(
        c["method_params"]["policy_ckpt"] == "/tmp/shared_ppo_seed1.pkl"
        for c in seeded
    )


def test_prior_diagnostics_aggregate_acceptance_and_risk():
    result = SimpleNamespace(infos=[
        {
            "prior_accepted": jnp.asarray(1.0),
            "prior_predicted_improvement": jnp.asarray(0.2),
            "prior_risk_ok": jnp.asarray(1.0),
            "prior_force_veto": jnp.asarray(0.0),
            "prior_risk_rl": jnp.asarray([0.0, 0.1, 0.2, 0.3]),
            "prior_risk_refined": jnp.asarray([0.0, 0.0, 0.1, 0.2]),
        },
        {
            "prior_accepted": jnp.asarray(0.0),
            "prior_predicted_improvement": jnp.asarray(-0.1),
            "prior_risk_ok": jnp.asarray(0.0),
            "prior_force_veto": jnp.asarray(1.0),
            "prior_risk_rl": jnp.asarray([0.0, 0.2, 0.4, 0.6]),
            "prior_risk_refined": jnp.asarray([0.1, 0.3, 0.5, 0.7]),
        },
    ])
    diagnostics = _prior_diagnostics(result)
    assert diagnostics["prior_acceptance_rate"] == pytest.approx(0.5)
    assert diagnostics["prior_fallback_rate"] == pytest.approx(0.5)
    assert diagnostics["prior_force_veto_rate"] == pytest.approx(0.5)
    assert diagnostics["prior_risk_rl_contact_loss"] == pytest.approx(0.15)


def test_best_eval_selector_pairs_params_with_later_metrics():
    from scripts.tasks.robot.arm.train_rl_baseline import _BestEvalSelector

    selector = _BestEvalSelector("eval/episode_realized_progress")
    selector.observe(0, {"eval/episode_realized_progress": 0.0})
    selector.capture(0, None, jnp.asarray([0.0]))
    selector.capture(100, None, jnp.asarray([1.0]))
    selector.observe(100, {
        "eval/episode_reward": -2.0,
        "eval/episode_realized_progress": 0.8,
    })
    selector.capture(200, None, jnp.asarray([2.0]))
    selector.observe(200, {
        "eval/episode_reward": -1.0,
        "eval/episode_realized_progress": 0.5,
    })

    assert selector.step == 100
    assert selector.score == pytest.approx(0.8)
    assert selector.metrics["eval/episode_reward"] == pytest.approx(-2.0)
    np.testing.assert_allclose(selector.params, [1.0])


def test_panda_residual_action_wrapper_adds_and_clips_bias():
    from genedynamics.envs.domains.manipulation.panda_brax import (
        PandaResidualActionEnv,
    )

    class State(NamedTuple):
        obs: jax.Array

    class Env:
        action_size = 2
        observation_size = 1
        backend = "mjx"
        dt = 0.02

        def step(self, state, action):
            return State(obs=action)

    wrapped = PandaResidualActionEnv(Env(), [0.25, -0.5], [0.5, 0.25])
    result = wrapped.step(
        State(obs=jnp.zeros((1,), jnp.float32)),
        jnp.asarray([0.9, 0.25]),
    )
    np.testing.assert_allclose(result.obs, [0.7, -0.4375])

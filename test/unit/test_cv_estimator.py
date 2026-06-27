"""Stage 3 (contribution 2) — multi-fidelity control-variate score estimator.

Synthetic-reward unit tests (CPU, NO simulator) substantiating the central ML
claim: the control-variate estimator is (a) unbiased toward the high-fidelity
objective, (b) lower-variance than spending the same #high-fidelity evals on a
pure high-fidelity estimate, and (c) closer to the true high-fidelity score
than a single-fidelity estimator at a fixed compute budget.

Synthetic model (one diffusion step, M proposals):
    μ_m  ~ N(0,1)                     true per-proposal value
    R_hi = μ + N(0, σ_hi²)            unbiased, expensive
    R_lo = μ + bias + N(0, σ_lo²)     systematically biased, cheap
The differences R_hi−R_lo cancel μ, so Var(diff) ≪ Var(R_hi): exactly the
regime where the control variate pays off.
"""

from __future__ import annotations

import numpy as np
import pytest

from genedynamics.solvers.single.mrmfmbd_ablation.estimator_system import (
    corrected_rewards,
    cv_score_weighted_mean,
    softmax_weights,
    weighted_mean,
    estimator_diagnostics,
    optimal_subset_size,
    single_fidelity_pool,
    realized_cost,
    BudgetDual,
)


def _make_step(M, rng, sigma_hi=0.05, sigma_lo=0.1, bias=0.5):
    mu = rng.standard_normal(M)
    R_hi = mu + sigma_hi * rng.standard_normal(M)
    R_lo = mu + bias + sigma_lo * rng.standard_normal(M)
    return mu, R_lo, R_hi


def test_corrected_rewards_endpoints():
    rng = np.random.default_rng(0)
    M = 32
    _, R_lo, R_hi = _make_step(M, rng)
    # K = M (all hi) → exactly R_hi.
    allidx = np.arange(M)
    Rt_all = corrected_rewards(R_lo, R_hi[allidx], allidx, M)
    np.testing.assert_allclose(Rt_all, R_hi, rtol=0, atol=1e-9)
    # K = 0 (no hi) → exactly R_lo.
    Rt_none = corrected_rewards(R_lo, np.array([]), np.array([], dtype=int), M)
    np.testing.assert_allclose(Rt_none, R_lo, rtol=0, atol=1e-9)


def test_corrected_mean_is_unbiased_for_high_fidelity():
    rng = np.random.default_rng(1)
    M, K, trials = 64, 12, 3000
    _, R_lo, R_hi = _make_step(M, rng)
    target = float(R_hi.mean())
    ests = np.empty(trials)
    for t in range(trials):
        S = rng.choice(M, K, replace=False)
        ests[t] = corrected_rewards(R_lo, R_hi[S], S, M).mean()
    # E[mean(R̃)] = mean(R_hi). MC stderr ~ std/sqrt(trials).
    stderr = ests.std() / np.sqrt(trials)
    assert abs(ests.mean() - target) < 5 * stderr + 1e-3


def test_control_variate_reduces_variance_at_equal_hifi_count():
    """Var of the CV mean estimate < Var of a pure-hi mean over the SAME K evals."""
    rng = np.random.default_rng(2)
    M, K, trials = 64, 12, 4000
    _, R_lo, R_hi = _make_step(M, rng)
    cv = np.empty(trials)
    hi = np.empty(trials)
    for t in range(trials):
        S = rng.choice(M, K, replace=False)
        cv[t] = corrected_rewards(R_lo, R_hi[S], S, M).mean()  # all-M low-fi + K corrections
        hi[t] = R_hi[S].mean()                                  # K hi-fi only
    # μ_m cancels in the differences → CV variance is much smaller.
    assert cv.var() < 0.25 * hi.var()


def test_cv_beats_single_fidelity_score_at_fixed_budget():
    """At a fixed per-step budget, the CV weighted-mean (the MBD score) is
    closer to the true high-fidelity weighted-mean than a single-fidelity
    estimator that can only afford a smaller proposal pool."""
    rng = np.random.default_rng(3)
    M_big = 128
    c_lo, c_hi, budget, T = 1.0, 10.0, 300.0, 0.5
    K = optimal_subset_size(budget, M_big, c_lo, c_hi)
    n_single = single_fidelity_pool(budget, c_hi)
    assert K >= 1 and n_single >= 1 and n_single < M_big

    trials = 200
    mse_cv = np.empty(trials)
    mse_sf = np.empty(trials)
    for t in range(trials):
        mu, R_lo, R_hi = _make_step(M_big, rng)
        theta = mu  # each proposal's parameter is its own value
        truth = weighted_mean(theta, softmax_weights(mu, T))  # noiseless hi-fi over all M_big

        S = rng.choice(M_big, K, replace=False)
        tb_cv, _, _ = cv_score_weighted_mean(theta, R_lo, R_hi[S], S, T)

        # single-fidelity: afford n_single hi-fi evals → weight only those.
        Ssf = rng.choice(M_big, n_single, replace=False)
        tb_sf = weighted_mean(theta[Ssf], softmax_weights(R_hi[Ssf], T))

        mse_cv[t] = (tb_cv - truth) ** 2
        mse_sf[t] = (tb_sf - truth) ** 2
    assert mse_cv.mean() < mse_sf.mean()


def test_budget_allocation_respects_budget():
    c_lo, c_hi = 1.0, 7.0
    M = 64
    for B in (80.0, 200.0, 500.0):
        K = optimal_subset_size(B, M, c_lo, c_hi)
        assert 0 <= K <= M
        assert realized_cost(M, K, c_lo, c_hi) <= B + 1e-9
        # one more hi eval would exceed budget (unless K == M).
        if K < M:
            assert realized_cost(M, K + 1, c_lo, c_hi) > B
    # If even the low-fi sweep blows the budget, K = 0.
    assert optimal_subset_size(10.0, M, c_lo, c_hi) == 0


def test_budget_dual_tracks_target():
    dual = BudgetDual(nu=0.0, eta=0.05, target=100.0)
    # Persistent overspend pushes ν up.
    for _ in range(50):
        dual.update(150.0)
    assert dual.nu > 0.0
    hi = dual.nu
    # Persistent underspend pulls ν back to the 0 floor.
    for _ in range(200):
        dual.update(10.0)
    assert dual.nu == 0.0 and hi > 0.0


def test_diagnostics_report_variance_reduction():
    rng = np.random.default_rng(5)
    M = 64
    _, R_lo, R_hi = _make_step(M, rng)
    S = rng.choice(M, 16, replace=False)
    d = estimator_diagnostics(R_lo, R_hi[S], S)
    assert d["corr"] > 0.9          # lo/hi strongly correlated
    assert d["var_reduction"] > 0.5  # differences much tighter than R_hi spread

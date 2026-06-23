"""Stage-2 unit tests: risk-sensitive high-fidelity certification rho_H.

Covers the host aggregator `risk_sensitive_marginalize_np` that the method
backend's `_fine_revalidate` and `codesign.py`'s final eval use to certify a
design by its worst-case-robust score (new_version.txt eq:high_fidelity_risk),
instead of the legacy cross-regime mean.
"""
from __future__ import annotations

import numpy as np
import pytest

from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (
    risk_sensitive_marginalize_np,
)


def test_tau_to_zero_is_worst_mode():
    R = np.array([2.0, 10.0, 8.0, 6.0])
    lp = np.zeros(4)
    rho = risk_sensitive_marginalize_np(R, lp, 1e-3)
    assert abs(rho - R.min()) < 1e-2, f"tau->0 should be worst-mode min={R.min()}, got {rho}"


def test_tau_to_inf_is_prior_mean():
    # Requires the normalized prior fix (sum_m p(m)=1); otherwise blows up by -tau*logC.
    R = np.array([2.0, 10.0, 8.0, 6.0])
    lp = np.zeros(4)
    rho = risk_sensitive_marginalize_np(R, lp, 1e6)
    assert abs(rho - R.mean()) < 1e-2, f"tau->inf should be prior-mean {R.mean()}, got {rho}"


def test_soft_min_below_mean():
    R = np.array([2.0, 10.0, 8.0, 6.0])
    lp = np.zeros(4)
    for tau in (0.5, 1.0, 3.0):
        rho = risk_sensitive_marginalize_np(R, lp, tau)
        assert rho <= R.mean() + 1e-9, f"rho_H must be <= mean (soft-min), tau={tau}"
        assert rho >= R.min() - 1e-9, f"rho_H must be >= worst-mode, tau={tau}"


def test_robustness_preference_at_small_tau():
    # A robust design (high worst-mode) should certify HIGHER than a brittle
    # design with a better mean but a catastrophic regime, at small tau.
    robust = np.array([7.0, 7.5, 7.0, 7.5])      # mean 7.25, worst 7.0
    brittle = np.array([0.5, 12.0, 12.0, 12.0])  # mean 9.125, worst 0.5
    lp = np.zeros(4)
    tau = 0.5
    assert risk_sensitive_marginalize_np(robust, lp, tau) > \
        risk_sensitive_marginalize_np(brittle, lp, tau)
    # ...but at large tau (≈mean) the brittle one's higher mean wins.
    assert risk_sensitive_marginalize_np(brittle, lp, 1e6) > \
        risk_sensitive_marginalize_np(robust, lp, 1e6)


def test_batched_shape_and_value():
    R = np.array([2.0, 10.0, 8.0, 6.0])
    lp = np.zeros(4)
    batch = np.stack([R, R + 1.0, R - 2.0])  # (3, 4)
    out = risk_sensitive_marginalize_np(batch, lp, 1.0)
    assert out.shape == (3,)
    # row-wise consistency with the per-row scalar call
    for i in range(3):
        assert abs(out[i] - risk_sensitive_marginalize_np(batch[i], lp, 1.0)) < 1e-9


def test_nonuniform_prior_normalized():
    # An unnormalized log-prior gives the SAME result as its normalized form
    # (the helper normalizes internally).
    R = np.array([3.0, 9.0, 5.0])
    lp_raw = np.array([0.0, 0.0, 0.0])
    lp_shift = lp_raw + 5.0  # same distribution, shifted in log space
    a = risk_sensitive_marginalize_np(R, lp_raw, 1.0)
    b = risk_sensitive_marginalize_np(R, lp_shift, 1.0)
    assert abs(a - b) < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

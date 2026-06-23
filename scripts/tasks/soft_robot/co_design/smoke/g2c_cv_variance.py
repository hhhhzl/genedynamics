#!/usr/bin/env python3
"""G2c — control-variate score variance < single-fidelity at matched budget (GPU).

Replicates test_cv_estimator.test_cv_beats_single_fidelity_score_at_fixed_budget
on the REAL JAX-MPM crawling scene (the unit test uses synthetic rewards). One
denoising step = M proposals around a center. Each is evaluated at:
  * LOW  fidelity (30 env-steps) — all M  (cheap, biased)
  * HIGH fidelity (200 env-steps) — used to form the truth + the CV/SF estimates
Matched budget B: CV affords M low + K high; single-fidelity affords n_single high.
The MBD score is the importance-weighted clean mean θ̄. We compare MSE-to-truth.

Gate: mean MSE(CV) < mean MSE(single-fidelity), and lo/hi correlation > 0,
var_reduction > 0 (the control variate genuinely helps on MPM rewards).

Run: python scripts/tasks/soft_robot/co_design/smoke/g2c_cv_variance.py
"""
from __future__ import annotations

import dataclasses
import sys
import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
from genedynamics.envs.external.jax_mpm.scene import rollout_return_batch
from genedynamics.solvers.single.mrmfmbd_ablation.estimator_system.control_variate import (
    cv_score_weighted_mean, weighted_mean, softmax_weights, estimator_diagnostics,
)
from genedynamics.solvers.single.mrmfmbd_ablation.estimator_system.budget import (
    optimal_subset_size, single_fidelity_pool, realized_cost,
)

# Fidelity = PHYSICS RESOLUTION (substeps per env-step), the paper's intended
# "multi-fidelity by substeps" axis, which preserves reward correlation far
# better than episode-length truncation (which anti-correlates here). Low =
# coarse dt (fewer substeps, cheaper); High = fine dt. Same 200 env-steps.
ENV_STEPS = 200
DT_LOW = 1.0e-3              # coarse: 8 substeps/step
T = 0.5
SIGMA = 0.20                 # proposal spread around the center
TRIALS = 30
M = 24


def main() -> int:
    print("backend:", jax.default_backend())
    prov = JaxMpmTaskDomainProvider()
    ev = prov.create_evaluator(
        ".", voxel_dims=[4, 3, 4], n_grid=64, reward_shaping_weight=100.0,
        act_strength_base=24.0, scale=50.0, task="crawling_ground",
    )
    scene, cfg_hi = ev._scene, ev._mpm_cfg
    cfg_lo = dataclasses.replace(cfg_hi, dt=DT_LOW)        # coarse physics (cheaper)
    fr = jnp.asarray(float(ev._mode_friction[0]), jnp.float32)

    vx, vy, vz = cfg_hi.voxel_dims
    x_dim = int(vx * vy * vz)
    phi_dim = int(cfg_hi.n_actuators * cfg_hi.n_sin_waves + 4 * cfg_hi.n_actuators)
    x_lo, x_hi, phi_lo, phi_hi = 0.2, 1.0, -0.5, 0.5
    scale = np.concatenate([np.full(x_dim, (x_hi - x_lo) / 2), np.full(phi_dim, (phi_hi - phi_lo) / 2)]).astype(np.float32)
    center = np.concatenate([np.full(x_dim, 0.6), np.zeros(phi_dim)]).astype(np.float32)
    lo = np.concatenate([np.full(x_dim, x_lo), np.full(phi_dim, phi_lo)]).astype(np.float32)
    hi = np.concatenate([np.full(x_dim, x_hi), np.full(phi_dim, phi_hi)]).astype(np.float32)

    # Cost ∝ substeps (per env-step) since env-steps are equal. Budget pinned so
    # CV gets K∈[1,M) hi-fi and single-fidelity gets n_single<M hi-fi.
    C_LO = float(cfg_lo.substeps_per_env_step)
    C_HI = float(cfg_hi.substeps_per_env_step)
    budget = M * C_LO + 6 * C_HI
    K = optimal_subset_size(budget, M, C_LO, C_HI)
    n_single = single_fidelity_pool(budget, C_HI)
    print(f"low={cfg_lo.substeps_per_env_step}substeps high={cfg_hi.substeps_per_env_step}substeps "
          f"(cost ratio {C_LO/C_HI:.2f})  M={M} budget={budget:.0f}  K(cv hi-fi)={K}  "
          f"n_single(sf hi-fi)={n_single}")
    assert 1 <= K < M and 1 <= n_single < M

    def rollout(theta_batch, scene_cfg, steps):
        x = jnp.asarray(theta_batch[:, :x_dim]); phi = jnp.asarray(theta_batch[:, x_dim:])
        frb = jnp.full((theta_batch.shape[0],), fr, jnp.float32)
        rs, _ = rollout_return_batch(x, phi, frb, scene, scene_cfg, steps)
        return np.asarray(rs)

    rng = np.random.default_rng(0)
    mse_cv = np.empty(TRIALS); mse_sf = np.empty(TRIALS)
    corrs = []; varreds = []
    for t in range(TRIALS):
        eps = rng.standard_normal((M, center.shape[0])).astype(np.float32)
        theta = np.clip(center[None] + SIGMA * scale[None] * eps, lo[None], hi[None])
        R_lo = rollout(theta, cfg_lo, ENV_STEPS)   # (M,) coarse physics, cheap/biased
        R_hi = rollout(theta, cfg_hi, ENV_STEPS)   # (M,) fine physics (truth + subsets)
        truth = weighted_mean(theta, softmax_weights(R_hi, T))           # all-M hi-fi score

        S = np.argsort(R_lo)[-K:]            # CV subset: top-K by low-fi
        tb_cv, _, _ = cv_score_weighted_mean(theta, R_lo, R_hi[S], S, T)  # M low + K high
        Ssf = rng.choice(M, n_single, replace=False)                     # SF: random n_single hi-fi
        tb_sf = weighted_mean(theta[Ssf], softmax_weights(R_hi[Ssf], T))

        mse_cv[t] = float(np.sum((tb_cv - truth) ** 2))
        mse_sf[t] = float(np.sum((tb_sf - truth) ** 2))
        d = estimator_diagnostics(R_lo, R_hi[S], S)
        corrs.append(d["corr"]); varreds.append(d["var_reduction"])

    cv_m, sf_m = float(mse_cv.mean()), float(mse_sf.mean())
    print(f"\nmean MSE-to-truth:  CV={cv_m:.4e}   single-fidelity={sf_m:.4e}   "
          f"ratio CV/SF={cv_m/max(sf_m,1e-12):.3f}")
    print(f"lo/hi corr (mean over trials): {np.nanmean(corrs):+.3f}   "
          f"var_reduction (mean): {np.nanmean(varreds):+.3f}")
    ok = cv_m < sf_m
    print(f"\nCV score variance < single-fidelity at matched budget -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Proof: a GENERAL registered solver (CEM) optimizes soft-robot co-design through
the standard Solver(dynamics, energy).solve() contract — no baseline layer.

Builds the crawling_ground co-design problem via codesign_env.build_codesign_problem
(action a -> design theta -> MPM reward, reparameterized into the solver's native
zero-centered action space), constructs the real `CEMSolver`, and runs solve().
Pass criterion: it runs, returns a finite best reward, and CEM improves the reward
over its iterations.

Run: python scripts/tasks/soft_robot/co_design/smoke/codesign_solver_bridge.py
"""
from __future__ import annotations

import sys
import numpy as np
import jax.numpy as jnp

from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
from genedynamics.envs.external.jax_mpm.codesign_env import build_codesign_problem
from genedynamics.solvers.single.cem.cem import CEMSolver
from genedynamics.core.backends.runtime import RuntimeBackendManager


def main() -> int:
    vx, vy, vz = 4, 3, 4
    x_opt_dim = vx * vy * (vz // 2)   # z-symmetry: optimizer sees half the voxels (24)
    phi_dim = 80
    print(f"co-design action dim D = {x_opt_dim} + {phi_dim} = {x_opt_dim + phi_dim}")

    prov = JaxMpmTaskDomainProvider()
    ev = prov.create_evaluator(".", voxel_dims=[vx, vy, vz], n_grid=64, reward_shaping_weight=100.0,
                               act_strength_base=24.0, scale=50.0, task="crawling_ground")
    scene, cfg = ev._scene, ev._mpm_cfg

    dynamics, energy, x0 = build_codesign_problem(
        scene, cfg, x_opt_dim=x_opt_dim, phi_dim=phi_dim,
        x_lo=0.2, x_hi=1.0, x_mean=0.6, phi_lo=-0.5, phi_hi=0.5, phi_mean=0.0,
        friction=0.5, num_env_steps=100, z_sym=True, voxel_dims=(vx, vy, vz),
    )

    # The GENERAL CEM solver — unchanged — consumes the co-design (dynamics, energy).
    RuntimeBackendManager.set_backend("jax")
    backend = RuntimeBackendManager.get_backend()
    print("backend:", backend.name)
    solver = CEMSolver(
        dynamics, energy, backend,
        horizon=1, dt=1.0, num_samples=16, num_iterations=5,
        elite_frac=0.25, init_std=0.8, min_std=0.1, action_limit=3.0, seed=0,
    )

    # reward of the prior-mean design (action a = 0) as a baseline-to-beat
    r0 = float(dynamics.jax_transition(jnp.zeros(1), jnp.zeros(dynamics.act_dim))[0])
    print(f"prior-mean (a=0) reward = {r0:+.4f}")

    traj = solver.solve(x0, horizon=1)
    best_a = np.asarray(traj.actions[0], dtype=np.float32)
    theta = np.asarray(dynamics.action_to_theta(jnp.asarray(best_a)))
    r_best = float(dynamics.jax_transition(jnp.zeros(1), jnp.asarray(best_a))[0])

    print(f"CEM best reward = {r_best:+.4f}  (theta dim {theta.shape[0]}, "
          f"x in [{theta[:x_opt_dim].min():.2f},{theta[:x_opt_dim].max():.2f}])")
    ok = np.isfinite(r_best) and r_best >= r0
    print(f"\nGENERAL CEM solver did co-design via Solver(dynamics,energy).solve()  "
          f"-> {'PASS' if ok else 'FAIL'}  (improved over a=0: {r_best - r0:+.4f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

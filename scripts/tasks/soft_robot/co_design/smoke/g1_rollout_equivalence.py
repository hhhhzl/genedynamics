#!/usr/bin/env python3
"""G1.1 — end-to-end one-hot rollout == legacy rollout equivalence (GPU).

Kernel-level equivalence (`_stress_and_J_weighted` one-hot == `_stress_and_J`)
is already unit-tested (test_actuator_weight_stress.py, Δ≈2e-11). This is the
END-TO-END confirmation over a full 200-step crawling_ground rollout: feeding the
scene a one-hot `actuator_weight` (derived from `actuator_id`, passive→zero row)
must reproduce the legacy hard-index path's reward.

Gate: max |Δreward| < 1e-4 over several random (x, φ). Unlocks the weighted
actuation path used by A2 (G2a) and DiffuseBot.

Run: python scripts/tasks/soft_robot/co_design/smoke/g1_rollout_equivalence.py
"""
from __future__ import annotations

import sys
import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
from genedynamics.envs.external.jax_mpm.scene import rollout_return

TOL = 1e-4
N_TRIALS = 5
NUM_ENV_STEPS = 200


def main() -> int:
    print("backend:", jax.default_backend())
    prov = JaxMpmTaskDomainProvider()
    # Match crawling_ground.yaml evaluator_runtime.
    ev = prov.create_evaluator(
        ".", voxel_dims=[4, 3, 4], n_grid=64, reward_shaping_weight=100.0,
        act_strength_base=24.0, scale=50.0, task="crawling_ground",
    )
    scene = ev._scene                  # legacy: actuator_weight is None
    cfg = ev._mpm_cfg
    friction = jnp.asarray(float(ev._mode_friction[0]), jnp.float32)

    K = int(cfg.n_actuators)
    aid = np.asarray(scene.actuator_id)          # (N,) int, -1 = passive
    N = int(aid.shape[0])
    # one-hot per particle; passive (aid < 0) -> all-zero row.
    W = np.zeros((N, K), dtype=np.float32)
    pos = aid >= 0
    W[np.arange(N)[pos], aid[pos]] = 1.0
    scene_oh = scene._replace(actuator_weight=jnp.asarray(W))
    print(f"n_particles={N}  n_actuators={K}  passive={int((~pos).sum())}  "
          f"legacy actuator_weight is None: {scene.actuator_weight is None}")

    vx, vy, vz = cfg.voxel_dims
    x_dim = int(vx * vy * vz)
    phi_dim = int(cfg.n_actuators * cfg.n_sin_waves + 4 * cfg.n_actuators)
    print(f"x_dim(n_voxels)={x_dim}  phi_dim={phi_dim}  num_env_steps={NUM_ENV_STEPS}")

    rng = np.random.default_rng(0)
    max_abs = 0.0
    for t in range(N_TRIALS):
        x = jnp.asarray(rng.uniform(0.2, 1.0, x_dim).astype(np.float32))
        phi = jnp.asarray(rng.uniform(-0.5, 0.5, phi_dim).astype(np.float32))
        r_legacy, _, _ = rollout_return(x, phi, friction, scene, cfg, NUM_ENV_STEPS)
        r_weight, _, _ = rollout_return(x, phi, friction, scene_oh, cfg, NUM_ENV_STEPS)
        d = abs(float(r_legacy) - float(r_weight))
        max_abs = max(max_abs, d)
        print(f"  trial {t}: legacy={float(r_legacy):+.6f}  one-hot={float(r_weight):+.6f}  |Δ|={d:.3e}")

    ok = max_abs < TOL
    print(f"\nmax |Δreward| = {max_abs:.3e}  (tol {TOL:.0e})  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

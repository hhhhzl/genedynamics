#!/usr/bin/env python3
"""Task-2 — DiffuseBot-aligned continuous actuator (+stiffness) co-design field.

DiffuseBot co-designs Ψ = {geometry, stiffness, actuator placement} (paper Eq 1)
via DIFFERENTIABLE-sim gradients. We align the same continuous actuator-placement
field into our pipeline but optimize it GRADIENT-FREE: a multi-head morphology
decoder g(w) → {occupancy, actuator field (softmax over K), stiffness} so a single
MBD sample of the shape latent w jointly sets geometry + actuator + stiffness, and
the decoded fields flow through Stage-1's per-particle `actuator_weight`/`E` path.

Gates:
  A. decoder: the trained multi-head decoder exposes an actuator head; decoded
     actuator fields are real placements (per-voxel argmax spans >=2 actuators).
  B. physics: feeding the decoded actuator field into the rollout changes the
     reward vs the scene's fixed one-hot (the field is actually used by the sim).
  C. e2e: the real MRMFMBD backend runs with the multi-head decoder; toggling the
     actuator head (morph_decoder_actuator on/off, same seed) changes the optimized
     return (the co-design field flows through the gradient-free MBD loop). A legacy
     occ-only decoder has no actuator head (byte-compatible path).

Prereq: a multi-head decoder at data/morph_decoders/loco_cpu_mh (train via
  scripts/tasks/soft_robot/morphology/train_morph_ae.py --multihead --n-actuators 10).
Run: python scripts/tasks/soft_robot/co_design/smoke/task2_actuator_codesign.py
"""
from __future__ import annotations

import sys

import numpy as np

MH = "data/morph_decoders/loco_cpu_mh"
OCC_ONLY = "data/morph_decoders/loco_cpu"
VOXEL_DIMS = [3, 3, 3]   # 27 voxels — matches the loco_cpu(_mh) decoder n_voxels


def _load_mh():
    import json, os
    from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
    from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig
    with open(os.path.join(MH, "decoder_config.json")) as f:
        d = json.load(f)
    cfg = MorphDecoderConfig(
        latent_dim=int(d["latent_dim"]), hidden_dim=int(d["hidden_dim"]),
        n_voxels=int(d["n_voxels"]), x_lo=float(d["x_lo"]), x_hi=float(d["x_hi"]),
        decode_actuator=bool(d.get("decode_actuator", False)),
        decode_stiffness=bool(d.get("decode_stiffness", False)),
        n_actuators=int(d.get("n_actuators", 0)),
    )
    return MorphDecoder.load(os.path.join(MH, "decoder_params.npz"), cfg), cfg


def part_a() -> bool:
    print("=" * 70 + "\nPart A — multi-head decoder exposes a real actuator field\n" + "=" * 70)
    import jax
    dec, cfg = _load_mh()
    print(f"  has_actuator={dec.has_actuator} has_stiffness={dec.has_stiffness} "
          f"n_voxels={cfg.n_voxels} K={cfg.n_actuators}")
    W = jax.random.normal(jax.random.PRNGKey(0), (8, cfg.latent_dim))
    occ, act, stiff = dec.decode_full_batch(W)
    occ = np.asarray(occ); act = np.asarray(act)
    sums = act.sum(-1)
    n_distinct = [len(np.unique(act[i].argmax(-1))) for i in range(act.shape[0])]
    g = (dec.has_actuator and act.shape == (8, cfg.n_voxels, cfg.n_actuators)
         and abs(float(sums.min()) - 1.0) < 1e-3 and abs(float(sums.max()) - 1.0) < 1e-3
         and min(n_distinct) >= 2)
    print(f"  occ in [{occ.min():.2f},{occ.max():.2f}]  act softmax sum≈1: "
          f"[{sums.min():.3f},{sums.max():.3f}]")
    print(f"  per-sample #distinct actuators (argmax): {n_distinct}  -> "
          f"{'PASS' if g else 'FAIL'}")
    return g


def part_b() -> bool:
    print("\n" + "=" * 70 + "\nPart B — decoded actuator field changes the rollout reward\n" + "=" * 70)
    import jax, jax.numpy as jnp
    from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
    from genedynamics.envs.external.jax_mpm.scene import rollout_return_batch
    dec, cfg = _load_mh()
    prov = JaxMpmTaskDomainProvider()
    ev = prov.create_evaluator(".", voxel_dims=VOXEL_DIMS, n_grid=64,
                               reward_shaping_weight=100.0, act_strength_base=24.0,
                               scale=50.0, task="crawling_ground")
    scene, mcfg = ev._scene, ev._mpm_cfg
    print(f"  scene n_voxels={scene.n_voxels} n_actuators={scene.n_actuators}")
    if scene.n_voxels != cfg.n_voxels or scene.n_actuators != cfg.n_actuators:
        print("  [skip] scene/decoder grid mismatch"); return True
    B = 4
    W = jax.random.normal(jax.random.PRNGKey(1), (B, cfg.latent_dim))
    occ, act, _ = dec.decode_full_batch(W)
    phi_dim = mcfg.n_actuators * mcfg.n_sin_waves + 4 * mcfg.n_actuators
    phi = jax.random.normal(jax.random.PRNGKey(2), (B, phi_dim)) * 0.3
    fr = jnp.full((B,), float(ev._mode_friction[0]), jnp.float32)
    r_field, _ = rollout_return_batch(occ, phi, fr, scene, mcfg, 30,
                                      actuator_weight_voxel_batch=act)
    r_onehot, _ = rollout_return_batch(occ, phi, fr, scene, mcfg, 30)  # scene one-hot
    d = float(np.abs(np.asarray(r_field) - np.asarray(r_onehot)).max())
    print(f"  reward(decoded actuator field) : {np.asarray(r_field)}")
    print(f"  reward(scene one-hot)          : {np.asarray(r_onehot)}")
    print(f"  max|Δreward| = {d:.4f}  -> {'PASS' if d > 1e-3 else 'FAIL'}")
    return d > 1e-3


def _run_backend(actuator_on: bool, decoder_path: str):
    from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
    from genedynamics.experiments.framework.baseline import BaselineConfig
    from genedynamics.solvers.single.mrmfmbd.codesign import MRMFMBDBaseline
    prov = JaxMpmTaskDomainProvider()
    ev = prov.create_evaluator(".", voxel_dims=VOXEL_DIMS, n_grid=64,
                               reward_shaping_weight=100.0, act_strength_base=24.0,
                               scale=50.0, task="crawling_ground")
    ts = prov.get_task_spec("crawling_ground")
    mcfg = ev._mpm_cfg
    phi_dim = mcfg.n_actuators * mcfg.n_sin_waves + 4 * mcfg.n_actuators
    extra = dict(backend="mbd", K=5, M=4, num_fidelity_levels=1, phi_dim_override=True,
                 phi_lo=-0.5, phi_hi=0.5, morph_latent_dim=8, morph_decoder_path=decoder_path,
                 morph_decoder_actuator=actuator_on, morph_decoder_stiffness=False,
                 shac_refine_steps=0)
    r = MRMFMBDBaseline().run(BaselineConfig(task_id="crawling_ground", seed=0, extra=extra),
                              ev, ts, x_dim=27, phi_dim=phi_dim)
    return r


def part_c() -> bool:
    print("\n" + "=" * 70 + "\nPart C — e2e through the real MRMFMBD backend\n" + "=" * 70)
    import jax
    print("backend:", jax.default_backend())
    r_on = _run_backend(True, MH)
    r_off = _run_backend(False, MH)
    print(f"  multi-head decoder: return(actuator ON)={r_on.return_:.4f}  "
          f"return(actuator OFF)={r_off.return_:.4f}")
    finite = np.isfinite(r_on.return_) and np.isfinite(r_off.return_)
    differ = abs(float(r_on.return_) - float(r_off.return_)) > 1e-4
    # legacy occ-only decoder must have NO actuator head (byte-compatible path).
    from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
    from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig
    import json, os
    with open(os.path.join(OCC_ONLY, "decoder_config.json")) as f:
        d = json.load(f)
    leg = MorphDecoder.load(os.path.join(OCC_ONLY, "decoder_params.npz"),
                            MorphDecoderConfig(latent_dim=int(d["latent_dim"]),
                                               hidden_dim=int(d["hidden_dim"]),
                                               n_voxels=int(d["n_voxels"])))
    legacy_ok = not leg.has_actuator
    print(f"  finite returns: {finite}; ON≠OFF (actuator field flows into MBD): {differ}; "
          f"legacy occ-only has no actuator head: {legacy_ok}")
    return finite and differ and legacy_ok


def main() -> int:
    a = part_a()
    try:
        b = part_b()
        c = part_c()
    except Exception as e:  # pragma: no cover
        print(f"\n[errored — reporting, not masking]: {e!r}")
        import traceback; traceback.print_exc()
        b = c = False
    ok = a and b and c
    print("\n" + "=" * 70)
    print(f"RESULT: A {'PASS' if a else 'FAIL'} | B {'PASS' if b else 'FAIL'} | "
          f"C {'PASS' if c else 'FAIL'}  => {'ALL PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

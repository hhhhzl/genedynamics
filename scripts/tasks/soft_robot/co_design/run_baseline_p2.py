"""Run a P2 baseline (CEM / CMA-ES) from a config, on CPU, and rho_H-certify.

Forces the JAX CPU backend (so it runs alongside GPU policy training without
contending for the GPU). Reads the FAIR P2 config (same A2 latent morphology +
risk_codesign multi-regime objective + sinusoid controller), runs the solver per
seed, then certifies the returned (w, phi) by re-rolling over the regimes and
scoring the SAME risk-sensitive rho_H.

  JAX_PLATFORMS=cpu python scripts/tasks/soft_robot/co_design/run_baseline_p2.py \
    --config configs/soft_robot/co_design/baselines/cem_crawling_p2.yaml
"""
import os
# Device select BEFORE importing jax. P2_DEVICE=cpu (default) forces CPU so the
# baseline can run alongside GPU training; P2_DEVICE=gpu inherits the launch env
# (pair with XLA_PYTHON_CLIENT_PREALLOCATE=false + MEM_FRACTION to coexist on one GPU).
if os.environ.get("P2_DEVICE", "cpu").lower() == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import time

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="", help="override config seeds, e.g. 0,1,2")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    er = cfg["evaluator_runtime"]
    mp = dict(cfg["method_params"])
    name = str(cfg["baseline_name"]).lower()
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else cfg.get("seeds", [0]))

    import jax.numpy as jnp
    import jax
    from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
    from genedynamics.experiments.framework.codesign_runner import codesign_problem_from_evaluator
    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.solvers.single.cem.cem import CEMSolver
    from genedynamics.solvers.single.cmaes.cmaes import CMAESSolver
    from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
    from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig
    from genedynamics.envs.external.jax_mpm.scene import rollout_return
    from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (
        risk_sensitive_marginalize_np)

    print(f"[{name}] backend={jax.default_backend()} | config={os.path.basename(args.config)} "
          f"seeds={seeds}", flush=True)

    prov = JaxMpmTaskDomainProvider()
    # Open-loop trajectory controller passthrough (matched with run_learned_codesign).
    _ckkw = {}
    _ck = str(er.get("controller_kind", mp.get("controller_kind", "sinusoid")))
    if _ck == "open_loop":
        _ckkw = {"controller_kind": "open_loop",
                 "n_control_nodes": int(er.get("n_control_nodes", mp.get("n_control_nodes", 50))),
                 "n_actuators": int(er.get("n_actuators", mp.get("n_actuators", 10))),
                 "env_horizon": int(er.get("env_horizon", mp.get("env_horizon", 200)))}
        mp["phi_lo"] = -3.0; mp["phi_hi"] = 3.0
    ev = prov.create_evaluator(".", voxel_dims=er["voxel_dims"], n_grid=int(er["n_grid"]),
                               reward_shaping_weight=float(er["reward_shaping_weight"]),
                               act_strength_base=float(er["act_strength_base"]),
                               scale=float(er["scale"]), task="crawling_ground",
                               mode_friction=er["mode_friction"], **_ckkw,
                               **{k: er[k] for k in ("mode_slope_deg","mode_mass_scale","mode_init_vel") if k in er})
    mcfg = ev._mpm_cfg
    if _ck == "open_loop":
        phi_dim = int(mcfg.n_control_nodes if mcfg.n_control_nodes > 0 else mcfg.env_horizon) * mcfg.n_actuators
    else:
        phi_dim = mcfg.n_actuators * mcfg.n_sin_waves + 4 * mcfg.n_actuators
    x_dim = int(mp.get("morph_latent_dim", 0)) or int(np.prod([int(v) for v in er["voxel_dims"]]))
    dyn, energy, x0, x_opt_dim = codesign_problem_from_evaluator(
        ev, x_dim=x_dim, phi_dim=phi_dim, method_params=mp)
    RuntimeBackendManager.set_backend("jax")
    backend = RuntimeBackendManager.get_backend()

    dpath = str(mp.get("morph_decoder_path", "") or "")
    dec = None
    if dpath:
        _dc = json.load(open(os.path.join(dpath, "decoder_config.json")))
        _mc = MorphDecoderConfig(latent_dim=int(_dc["latent_dim"]), hidden_dim=int(_dc["hidden_dim"]),
                                 n_voxels=int(_dc["n_voxels"]), x_lo=float(_dc["x_lo"]), x_hi=float(_dc["x_hi"]),
                                 decode_actuator=bool(_dc.get("decode_actuator", False)),
                                 n_actuators=int(_dc.get("n_actuators", 0)))
        dec = MorphDecoder.load(os.path.join(dpath, "decoder_params.npz"), _mc)
    nmodes = int(mp.get("num_modes", 4))
    fr_table = np.asarray(ev._mode_friction, np.float32)[:nmodes]
    steps = int(mp.get("num_env_steps", 200))
    tau = float(mp.get("risk_temperature", 1.0))

    Solver = CEMSolver if name == "cem" else CMAESSolver
    kw = dict(num_samples=int(mp["num_samples"]), num_iterations=int(mp["num_iterations"]),
              elite_frac=float(mp["elite_frac"]), init_std=float(mp.get("init_std", 0.8)),
              min_std=float(mp.get("min_std", 0.1)), action_limit=3.0)

    per_seed = []
    for sd in seeds:
        t0 = time.time()
        solver = Solver(dyn, energy, backend, horizon=1, dt=1.0, seed=sd, **kw)
        traj = solver.solve(x0, horizon=1)
        best_a = np.asarray(traj.actions[0], np.float32)
        theta = np.asarray(dyn.action_to_theta(jnp.asarray(best_a)))
        w, phi = theta[:x_opt_dim], theta[x_opt_dim:]
        if dec is not None:
            occ, act, _ = dec.decode_full_batch(jnp.asarray(w[None]))
            occ = occ[0]; aw = act[0] if act is not None else None
        else:
            occ = jnp.asarray(w); aw = None              # raw-voxel occupancy (no decoder)
        rets = [float(rollout_return(occ, jnp.asarray(phi), jnp.asarray(float(fr)), ev._scene, mcfg,
                                     steps, actuator_weight_voxel=aw)[0]) for fr in fr_table]
        rho = float(risk_sensitive_marginalize_np(np.asarray(rets), np.zeros(len(rets)), tau))
        wall = (time.time() - t0) / 60.0
        per_seed.append({"seed": sd, "rho_H": rho, "per_mode": rets,
                         "worst_mode": float(np.min(rets)), "wall_min": wall,
                         "x_latent": w.tolist(), "phi": phi.tolist()})  # for COM diagnostics
        print(f"[{name}] seed={sd} rho_H={rho:.4f} per_mode={[round(r,3) for r in rets]} "
              f"worst={min(rets):.3f} wall={wall:.1f}min", flush=True)

    rhos = [r["rho_H"] for r in per_seed]
    out = {"method": name, "config": os.path.basename(args.config), "n_grid": int(er["n_grid"]),
           "design_evals": kw["num_samples"] * kw["num_iterations"], "num_modes": nmodes,
           "rho_H_mean": float(np.mean(rhos)), "rho_H_std": float(np.std(rhos)),
           "worst_mode_mean": float(np.mean([r["worst_mode"] for r in per_seed])),
           "per_seed": per_seed}
    od = cfg["output_dir"]; os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "results.json"), "w"), indent=2)
    print(f"[{name}] ALL DONE rho_H={np.mean(rhos):.4f}+/-{np.std(rhos):.4f} "
          f"-> {od}/results.json", flush=True)


if __name__ == "__main__":
    main()

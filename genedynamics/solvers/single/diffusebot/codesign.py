"""DiffuseBot co-design registration adapter (Table-1 "DiffuseBot").

Thin BaselineProtocol shell — mirrors codesign_optimizers/shac.py. The ACTUAL
algorithm (physics-augmented generative diffusion + SinWaveOpenLoop controller
gradient, faithful to /workspace/DiffuseBot) lives in the independent solver
`solvers/single/diffusebot/`. This adapter just pulls the differentiable
JAX-MPM scene/cfg (+ optional A2 decoder) off the evaluator and drives it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genedynamics.experiments.framework.baseline import BaselineConfig, BaselineResult, BaselineProtocol


class DiffuseBotBaseline(BaselineProtocol):
    """Registration adapter for the DiffuseBot solver (solvers/single/diffusebot/)."""

    @property
    def name(self) -> str:
        return "diffusebot"

    def run(
        self,
        config: BaselineConfig,
        evaluator: Any,
        task_spec: Any,
        *,
        x_dim: int,
        phi_dim: int,
        **kwargs: Any,
    ) -> BaselineResult:
        import os
        import json
        from genedynamics.solvers.single.diffusebot import DiffuseBotConfig, DiffuseBotSolver

        if not hasattr(evaluator, "_scene") or not hasattr(evaluator, "_mpm_cfg"):
            raise RuntimeError(
                "DiffuseBotBaseline requires a JAX-MPM evaluator with `_scene` and "
                "`_mpm_cfg` (physics-augmented guidance differentiates the rollout)."
            )
        extra = dict(config.extra)
        scene, mpm_cfg = evaluator._scene, evaluator._mpm_cfg

        # Single mode (DiffuseBot is mode-blind): first regime's friction.
        friction = float(extra.get("friction", 0.5))
        if getattr(evaluator, "_regime_bank", None):
            friction = float(evaluator._regime_bank[0].friction)
        elif getattr(evaluator, "_mode_friction", None):
            friction = float(evaluator._mode_friction[0])

        # Optional A2 decoder (latent-w morphology, matching Ours) — same path as mrmfmbd.
        morph_decoder = None
        latent_dim = int(extra.get("morph_latent_dim", 0))
        dpath = str(extra.get("morph_decoder_path", "") or "")
        if latent_dim > 0 and dpath:
            from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
            from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig
            with open(os.path.join(dpath, "decoder_config.json")) as f:
                dc = json.load(f)
            mcfg = MorphDecoderConfig(
                latent_dim=int(dc["latent_dim"]), hidden_dim=int(dc["hidden_dim"]),
                n_voxels=int(dc["n_voxels"]), x_lo=float(dc["x_lo"]), x_hi=float(dc["x_hi"]),
                beta_kl=float(dc.get("beta_kl", 1e-3)),
                decode_actuator=bool(dc.get("decode_actuator", False)),
                decode_stiffness=bool(dc.get("decode_stiffness", False)),
                n_actuators=int(dc.get("n_actuators", 0)),
                e_lo=float(dc.get("e_lo", 0.5)), e_hi=float(dc.get("e_hi", 3.0)),
            )
            morph_decoder = MorphDecoder.load(os.path.join(dpath, "decoder_params.npz"), mcfg)

        cfg = DiffuseBotConfig(
            n_diffuse=int(extra.get("n_diffuse", extra.get("K", 50))),
            beta0=float(extra.get("beta0", 1e-4)), betaT=float(extra.get("betaT", 2e-2)),
            guidance_scale=float(extra.get("guidance_scale", 1.0)),
            latent_dim=latent_dim,
            controller_iters=int(extra.get("controller_iters", 20)),
            controller_lr=float(extra.get("controller_lr", 5e-3)),
            n_iters=int(extra.get("n_iters", 200)), lr=float(extra.get("lr", 1e-2)),
            x_lo=float(extra.get("x_lo", 0.2)), x_hi=float(extra.get("x_hi", 1.0)),
            phi_lo=float(extra.get("phi_lo", -0.5)), phi_hi=float(extra.get("phi_hi", 0.5)),
            x_mean=float(extra.get("x_mean", 0.6)), seed=int(config.seed),
            num_env_steps=int(extra.get("num_env_steps", getattr(mpm_cfg, "env_horizon", 200))),
            mode=str(extra.get("diffusebot_mode", "diffusion")),
        )

        morph0 = extra.get("morphology", None)
        _minit = str(extra.get("morph_init_path", "") or "")
        if _minit:
            morph0 = np.load(_minit).astype(np.float32).ravel()   # prior-seed latent w0 (TripoSG)
        res = DiffuseBotSolver(
            scene, mpm_cfg, cfg, morphology=morph0, morph_decoder=morph_decoder,
            friction=friction,
        ).solve()

        x_star = np.asarray(res["x"], np.float32)
        phi_star = np.asarray(res["phi"], np.float32)
        return BaselineResult(
            theta=np.concatenate([x_star, phi_star]),
            x=x_star, phi=phi_star,
            return_=float(res["return_"]), success=bool(res["success"]),
            num_evaluations=int(res.get("num_evaluations", 0)),
            wall_time=float(res.get("wall_time", 0.0)),
            metadata={"method": "diffusebot", "mode": cfg.mode, "history": res.get("history", [])},
        )

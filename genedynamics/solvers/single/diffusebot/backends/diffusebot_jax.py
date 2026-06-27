"""JAX-MPM backend for DiffuseBot.

Reproduces DiffuseBot's two pillars in the differentiable jax_mpm:

  (1) Physics-augmented diffusion over morphology. We sample the design from the
      posterior p(w) ∝ prior(w)·exp(reward) by ANNEALED LANGEVIN — the faithful
      gradient analog of DiffuseBot's classifier-guided reverse diffusion
      (conditionals/base.py: guidance = ∇reward through the projection). For the
      A2 latent the prior is N(0,I) (score = −w); g(w) → occupancy. The physics
      reward gradient ∇_w reward is obtained by autodiff through the decoder and
      the differentiable MPM rollout.

  (2) SinWaveOpenLoop controller (the env's `compute_actuation` φ) optimized by
      Adam through truncated-BPTT in the same differentiable rollout.

`mode="first_order"` drops the diffusion noise/prior and runs pure Adam gradient
ascent on (w, φ) — DiffuseBot's first-order ablation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.envs.external.jax_mpm.scene import rollout_return


class DiffuseBotBackendJax:
    def __init__(self, *, scene, mpm_cfg, cfg, morphology=None, morph_decoder=None,
                 friction: float = 0.5, terrain_height=None):
        self.scene = scene
        self.mpm_cfg = mpm_cfg
        self.cfg = cfg
        self.decoder = morph_decoder
        self.friction = float(friction)
        self.terrain_height = terrain_height
        self.num_env_steps = int(cfg.num_env_steps)

        # design dim: latent w (if a decoder is given) else occupancy x
        self.use_latent = bool(cfg.latent_dim > 0 and morph_decoder is not None)
        if self.use_latent:
            self.design_dim = int(cfg.latent_dim)
        else:
            self.design_dim = int(getattr(scene, "n_voxels", None) or np.asarray(scene.voxel_id).max() + 1)
        # phi dim from the env controller layout (n_act*K + 4*n_act)
        nf = int(getattr(mpm_cfg, "n_sin_waves", 4))
        na = int(getattr(mpm_cfg, "n_actuators", 10))
        self.phi_dim = na * nf + 4 * na
        self._morph0 = morphology

    # ------------------------------------------------------------------ #
    def _build(self):
        cfg, dec = self.cfg, self.decoder
        x_lo, x_hi = float(cfg.x_lo), float(cfg.x_hi)
        fr = jnp.asarray(self.friction, jnp.float32)
        th = self.terrain_height

        def occ_from_design(d):
            if self.use_latent:
                return dec.decode(d)               # g(w) → occupancy in [x_lo,x_hi]
            return jnp.clip(d, x_lo, x_hi)

        # Truncated-BPTT horizon: differentiating through the FULL 200-step MPM
        # rollout stores the whole trajectory × all particles -> OOM (hundreds of
        # GiB). DiffuseBot/SHAC avoid this with SHORT-horizon gradients. Use
        # cfg.controller_h (>0) else a safe default; cap to num_env_steps.
        # BPTT through MPM at n_grid=128 stores the 128^3 grid per substep, so the
        # differentiable-sim gradient is memory-bound (~2.3 GiB per env-step here).
        # That ceiling is itself a contribution-1 datapoint: the gradient baseline
        # can only afford a very short horizon where gradient-free (ours/cmaes) is
        # unconstrained. Default to h=8 to fit a 24 GiB GPU.
        gh = int(cfg.controller_h) if int(cfg.controller_h) > 0 else 8
        self._grad_h = max(4, min(gh, self.num_env_steps))

        def reward(design, phi):
            occ = occ_from_design(design)
            r, _, _ = rollout_return(occ, phi, fr, self.scene, self.mpm_cfg,
                                     self._grad_h, terrain_height=th)
            return r

        self._reward = reward
        # jax.checkpoint (remat) = DiffuseBot's gradient checkpointing: the backward
        # pass RECOMPUTES the MPM forward instead of storing the full 128^3-grid
        # trajectory tape, so BPTT through the differentiable sim fits in GPU memory.
        reward_ckpt = jax.checkpoint(reward)
        self._rg_design = jax.jit(jax.value_and_grad(reward_ckpt, argnums=0))
        self._rg_phi = jax.jit(jax.value_and_grad(reward_ckpt, argnums=1))
        self._occ = jax.jit(occ_from_design)

    # ------------------------------------------------------------------ #
    def solve(self) -> Dict[str, Any]:
        self._build()
        cfg = self.cfg
        rng = np.random.RandomState(cfg.seed)

        # --- init design + controller ---
        if self.use_latent and self._morph0 is not None and np.asarray(self._morph0).ravel().shape[0] == self.design_dim:
            design = jnp.asarray(np.asarray(self._morph0, np.float32).ravel(), jnp.float32)  # prior-seed latent w0
        elif self.use_latent:
            design = jnp.asarray(rng.randn(self.design_dim) * 0.1, jnp.float32)   # near prior mean
        elif self._morph0 is not None:
            design = jnp.asarray(np.clip(np.asarray(self._morph0, np.float32).reshape(-1),
                                         cfg.x_lo, cfg.x_hi), jnp.float32)
        else:
            design = jnp.full((self.design_dim,), float(cfg.x_mean), jnp.float32)
        phi = jnp.zeros((self.phi_dim,), jnp.float32)

        # annealed Langevin step sizes (beta schedule) for the diffusion mode
        betas = np.linspace(cfg.beta0, cfg.betaT, cfg.n_diffuse).astype(np.float32)
        history, best = [], {"r": -np.inf, "design": design, "phi": phi}
        # Adam state for the controller
        mp = jnp.zeros_like(phi); vp = jnp.zeros_like(phi)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t0 = time.perf_counter()

        def adam_phi(phi, g, t, m, v):
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
            phi = jnp.clip(phi + cfg.controller_lr * mh / (jnp.sqrt(vh) + eps), cfg.phi_lo, cfg.phi_hi)
            return phi, m, v

        first_order = (cfg.mode == "first_order")
        n_outer = cfg.n_iters if first_order else cfg.n_diffuse
        pstep = 0
        # design Adam state (first-order mode)
        md = jnp.zeros_like(design); vd = jnp.zeros_like(design)

        for k in range(n_outer):
            # (2) controller gradient optimization (SinWaveOpenLoop via BPTT)
            for _ in range(1 if first_order else cfg.controller_iters):
                pstep += 1
                r_p, g_p = self._rg_phi(design, phi)
                phi, mp, vp = adam_phi(phi, g_p, pstep, mp, vp)

            # (1) morphology update
            r_d, g_d = self._rg_design(design, phi)
            if first_order:
                md = b1 * md + (1 - b1) * g_d; vd = b2 * vd + (1 - b2) * g_d * g_d
                mh = md / (1 - b1 ** (k + 1)); vh = vd / (1 - b2 ** (k + 1))
                design = design + cfg.lr * mh / (jnp.sqrt(vh) + eps)
            else:
                # physics-augmented annealed Langevin: prior score (−design for the
                # N(0,I) latent) + classifier guidance (physics ∇reward) + noise.
                beta = float(betas[k])
                prior_score = (-design) if self.use_latent else 0.0
                noise = jnp.asarray(rng.randn(self.design_dim) * np.sqrt(beta), jnp.float32)
                design = design + 0.5 * beta * (prior_score + cfg.guidance_scale * g_d) + noise
            if not self.use_latent:
                design = jnp.clip(design, cfg.x_lo, cfg.x_hi)

            r = float(r_d)
            history.append({"iter": k, "reward": r})
            if r > best["r"]:
                best = {"r": r, "design": design, "phi": phi}

        # final occupancy + return
        occ = np.asarray(self._occ(best["design"]), np.float32)
        x_full = occ if not self.use_latent else occ  # report decoded occupancy
        return {
            "x": np.asarray(best["design"], np.float32) if self.use_latent else x_full,
            "occupancy": x_full,
            "phi": np.asarray(best["phi"], np.float32),
            "return_": float(best["r"]),
            "success": bool(best["r"] > 0.0),
            "wall_time": time.perf_counter() - t0,
            "history": history,
            "num_evaluations": (cfg.n_iters if first_order else cfg.n_diffuse * (cfg.controller_iters + 1)),
        }

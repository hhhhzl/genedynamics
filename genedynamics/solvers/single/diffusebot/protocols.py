"""Config for the DiffuseBot solver.

Faithful to Wang et al., "DiffuseBot: Breeding Soft Robots With Physics-Augmented
Generative Diffusion Models" (NeurIPS 2023). Reference code:
/workspace/DiffuseBot (softzoo + diffusebot). The two pillars reproduced here:

  1. Physics-augmented diffusion for MORPHOLOGY — reverse diffusion over the
     design, guided each step by the gradient of the differentiable-physics
     reward w.r.t. the design (classifier-style guidance; cf.
     diffusebot/conditionals/base.py `gradient()` = ∇reward through the SDF
     projection). In our framework the design is the A2 shape latent w (the
     VAE N(0,I) prior is the diffusion prior); g(w) → occupancy.

  2. Controller = SinWaveOpenLoop (cf. diffusebot/conditionals/diff_phys_utils/
     controllers.py) optimized by gradient (Adam) through truncated-BPTT in the
     differentiable MPM. Here the controller is the env's `compute_actuation` φ.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiffuseBotConfig:
    # --- diffusion over morphology (physics-augmented) ---
    n_diffuse: int = 50            # reverse-diffusion steps (DiffuseBot design iters)
    beta0: float = 1.0e-4
    betaT: float = 2.0e-2
    guidance_scale: float = 1.0    # classifier-guidance strength (physics ∇reward)
    latent_dim: int = 0            # >0 → diffuse the A2 latent w; 0 → diffuse occupancy x

    # --- controller (SinWaveOpenLoop) gradient optimization ---
    controller_iters: int = 20     # Adam steps on φ per design iter
    controller_lr: float = 5.0e-3
    controller_h: int = 0          # 0 → full horizon BPTT; >0 → truncated short horizon

    # --- shared ---
    n_iters: int = 200             # total optimization budget (first-order ablation)
    lr: float = 1.0e-2             # design Adam lr (used in pure first-order mode)
    x_lo: float = 0.2
    x_hi: float = 1.0
    phi_lo: float = -0.5
    phi_hi: float = 0.5
    x_mean: float = 0.6
    seed: int = 0
    num_env_steps: int = 200
    # "diffusion" (physics-augmented, default, faithful) vs "first_order"
    # (pure Adam gradient ascent — DiffuseBot's first-order ablation).
    mode: str = "diffusion"

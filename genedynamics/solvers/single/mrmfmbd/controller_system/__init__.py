"""Learned closed-loop controller for the method path (Stage 5; new_version.txt
§IV-A): vartheta = D_beta(c, x), a_t ~ pi_{D_beta(c,x)}(. | o_t, psi(t), e_x).

Compact, gradient-free-friendly instantiation:
  - pi is a fixed-architecture MLP with params beta captured at trace time.
  - the controller LATENT c (the z^MB diffusion block, what the sampler explores
    and SHAC refines) and the morphology embedding e_x = E_chi(x) condition pi by
    INPUT concatenation (D_beta = identity hyper-decoder). beta and E_chi are
    fixed projections; only c is sampled/refined.
The sinusoidal controller (`scene.compute_actuation`) remains the default and the
ablation; this module is used only when controller_type='learned'.
"""
from .policy import (
    init_policy, policy_apply, init_embed, embed, time_encoding, obs_dim,
    ControllerParams,
)
from .io import save_controller, load_controller
from .critic import (
    init_critic, critic_value, lambda_returns, critic_td_loss, critic_sgd_step,
    state_dim, CriticParams,
)

__all__ = [
    "save_controller", "load_controller",
    "init_policy", "policy_apply", "init_embed", "embed", "time_encoding",
    "obs_dim", "ControllerParams",
    "init_critic", "critic_value", "lambda_returns", "critic_td_loss",
    "critic_sgd_step", "state_dim", "CriticParams",
]

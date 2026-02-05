from __future__ import annotations

"""
Vendored SafeDiffuser GaussianDiffusion.

This file is adapted from the external SafeDiffuser repo to live under
`enerdynamics/solvers/single/safediffuser/patch/` (dpcc-style layout).

Edits vs upstream:
- replace `import diffuser.utils as utils` with local `diffuser_utils`
- remove hard-coded `cuda:0` in condition formatting
- keep Progress/Silent via local `diffuser_utils.progress`
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import einops

try:
    from qpth.qp import QPFunction, QPSolvers  # type: ignore
except Exception:  # pragma: no cover
    QPFunction = None
    QPSolvers = None

from enerdynamics.solvers.single.safediffuser import diffuser_utils as utils
from .helpers import cosine_beta_schedule, extract, apply_conditioning, Losses


def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=1000,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.norm_mins = 0
        self.norm_maxs = 0
        self.safe1 = 0
        self.safe2 = 0

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    # ----------------- compatibility helpers -----------------

    def _format_conditions(self, conditions, batch_size):
        # Note: SafeDiffuser Policy formats conditions; keep for completeness.
        device = self.betas.device
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, "observations")  # type: ignore[attr-defined]
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=device)
        conditions = utils.apply_dict(einops.repeat, conditions, "d -> repeat d", repeat=batch_size)
        return conditions

    def get_loss_weights(self, action_weight, discount, weights_dict):
        self.action_weight = action_weight
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        discounts = discount ** torch.arange(self.horizon, dtype=torch.float32)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights[0, : self.action_dim] = action_weight
        return loss_weights

    # ----------------- sampling core -----------------

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        x = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        # Safety hooks (kept from upstream, but can be extended for new tasks).
        # Upstream contains task-specific invariance / QP projection code here.
        return x

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False, projector=None):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion:
            diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        safe1, safe2 = [], []
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_prev = x
            xp1 = self.p_sample(x_prev, cond, timesteps)
            xp1 = apply_conditioning(xp1, cond, self.action_dim)

            if projector is not None:
                # Force upstream SafeDiffuser semantics: correct the denoising update x -> xp1.
                # (No fallbacks to action-only correction.)
                xp1 = projector.invariance(x_prev, xp1)

            # Enforce conditioning after safety hooks as well.
            x = apply_conditioning(xp1, cond, self.action_dim)
            # keep interface
            try:
                safe1.append(self.safe1.unsqueeze(0))
                safe2.append(self.safe2.unsqueeze(0))
            except Exception:
                pass
            progress.update({"t": i})
            if return_diffusion:
                diffusion.append(x)

        try:
            self.safe1 = torch.cat(safe1, dim=0)
            self.safe2 = torch.cat(safe2, dim=0)
        except Exception:
            pass

        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, return_diffusion=True, projector=None, **kwargs):
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(
            shape,
            cond,
            return_diffusion=return_diffusion,
            projector=projector,
            *args,
            **kwargs,
        )

    def forward(self, cond, *args, projector=None, **kwargs):
        return self.conditional_sample(cond=cond, projector=projector, *args, **kwargs)


from __future__ import annotations

import torch

from diffuser.models.helpers import apply_conditioning


class SafeDiffuserTorchStepper:
    """
    SafeDiffuser-style stepper that matches DPCC's denoising architecture, but applies
    SafeDiffuser invariance/QP correction on every denoising step:

      xp1 = projector.invariance(x_prev, xp1)
    """

    def __init__(self, diffusion_model):
        self.diffusion = diffusion_model

    def _noise_scale(self) -> float:
        # Match DPCC stepper defaults for consistency.
        return 0.5

    def _init_scale(self) -> float:
        # Match DPCC stepper defaults for consistency.
        return 0.5

    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):
        # Keep the same core mean/variance computation as DPCC stepper.
        _ = (projector, constraints)
        if getattr(self.diffusion, "returns_condition", False):
            epsilon_cond = self.diffusion.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.diffusion.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.diffusion.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self.diffusion.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.diffusion.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.diffusion.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            raise RuntimeError('SafeDiffuserTorchStepper expects diffusion.clip_denoised=True (matches DPCC defaults).')

        model_mean, posterior_variance, posterior_log_variance = self.diffusion.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None, projector=None, constraints=None):
        b, *_, _device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            cond=cond,
            t=t,
            returns=returns,
            projector=projector,
            constraints=constraints,
        )
        noise = self._noise_scale() * torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        returns=None,
        return_diffusion=False,
        projector=None,
        constraints=None,
        repeat_last=0,
    ):
        _ = (constraints, repeat_last)
        device = self.diffusion.betas.device
        batch_size = shape[0]

        x = self._init_scale() * torch.randn(shape, device=device)
        x = apply_conditioning(
            x,
            cond,
            self.diffusion.action_dim,
            goal_dim=getattr(self.diffusion, "goal_dim", 0),
        )

        diffusion_chain = [x] if return_diffusion else None

        for i in reversed(range(0, self.diffusion.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_prev = x
            xp1 = self.p_sample(x_prev, cond, timesteps, returns=returns)
            xp1 = apply_conditioning(
                xp1,
                cond,
                self.diffusion.action_dim,
                goal_dim=getattr(self.diffusion, "goal_dim", 0),
            )

            if projector is not None:
                xp1 = projector.invariance(x_prev, xp1)

            x = apply_conditioning(
                xp1,
                cond,
                self.diffusion.action_dim,
                goal_dim=getattr(self.diffusion, "goal_dim", 0),
            )

            if diffusion_chain is not None:
                diffusion_chain.append(x)

        infos = {"projection_costs": {}}
        if diffusion_chain is not None:
            infos["diffusion"] = torch.stack(diffusion_chain, dim=1)
        return x, infos

    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        batch_size = len(cond[0])
        horizon = horizon or self.diffusion.horizon
        shape = (batch_size, horizon, self.diffusion.transition_dim)
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)


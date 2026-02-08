from __future__ import annotations

import torch

from diffuser.models.helpers import apply_conditioning


class DPCCTorchStepper:
    """
    Single DPCC stepper that owns denoising-step logic in solver space.
    """

    def __init__(self, diffusion_model):
        self.diffusion = diffusion_model

    def _noise_scale(self) -> float:
        return 0.5

    def _init_scale(self) -> float:
        return 0.5

    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):
        if self.diffusion.returns_condition:
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
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.diffusion.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )

        if projector is not None and projector.gradient:
            if self.diffusion.goal_dim > 0:
                grad = projector.compute_gradient(x_recon[:, :, :-self.diffusion.goal_dim], constraints)
            else:
                grad = projector.compute_gradient(x_recon, constraints)
            model_mean = model_mean + grad

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
        device = self.diffusion.betas.device

        batch_size = shape[0]
        x = self._init_scale() * torch.randn(shape, device=device)
        x = apply_conditioning(
            x,
            cond,
            self.diffusion.action_dim,
            goal_dim=self.diffusion.goal_dim,
        )

        diffusion_chain = [x] if return_diffusion else None
        costs = {}

        last_timestep = -repeat_last if repeat_last > 0 and projector is not None else 0
        for i in reversed(range(last_timestep, self.diffusion.n_timesteps)):
            t = i if i >= 0 else 0
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

            use_grad = (
                projector is not None
                and projector.gradient
                and t <= projector.diffusion_timestep_threshold * self.diffusion.n_timesteps
            )
            x = self.p_sample(
                x,
                cond,
                timesteps,
                returns=returns,
                projector=projector if use_grad else None,
                constraints=constraints if use_grad else None,
            )

            x = apply_conditioning(
                x,
                cond,
                self.diffusion.action_dim,
                goal_dim=self.diffusion.goal_dim,
            )

            use_project = (
                projector is not None
                and (not projector.gradient)
                and t <= projector.diffusion_timestep_threshold * self.diffusion.n_timesteps
            )
            if use_project:
                if self.diffusion.goal_dim > 0:
                    x[:, :, :-self.diffusion.goal_dim], projection_costs = projector.project(
                        x[:, :, :-self.diffusion.goal_dim], constraints
                    )
                else:
                    x, projection_costs = projector.project(x, constraints)
                costs[i] = projection_costs

            x = apply_conditioning(
                x,
                cond,
                self.diffusion.action_dim,
                goal_dim=self.diffusion.goal_dim,
            )

            if return_diffusion:
                diffusion_chain.append(x)

        infos = {"projection_costs": costs}
        if return_diffusion:
            infos["diffusion"] = torch.stack(diffusion_chain, dim=1)

        return x, infos

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, _device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            cond=cond,
            t=t,
            returns=returns,
        )
        noise = self._noise_scale() * torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.diffusion.betas.device

        batch_size = shape[0]
        x = self._init_scale() * torch.randn(shape, device=device)
        x = apply_conditioning(
            x,
            cond,
            self.diffusion.action_dim,
            goal_dim=self.diffusion.goal_dim,
        )

        diffusion_chain = [x] if return_diffusion else None

        for i in reversed(range(0, self.diffusion.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(
                x,
                cond,
                self.diffusion.action_dim,
                goal_dim=self.diffusion.goal_dim,
            )
            if return_diffusion:
                diffusion_chain.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion_chain, dim=1)
        return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        batch_size = len(cond[0])
        horizon = horizon or self.diffusion.horizon
        shape = (batch_size, horizon, self.diffusion.transition_dim)
        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

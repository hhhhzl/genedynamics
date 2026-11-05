"""Deterministic few-step DDIM residual parameterization of task transport.

The denoising chain is optimized through the existing ICDT objective; it is
not trained with demonstration noise-prediction or claimed to estimate a score.
"""
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from genedynamics.learning.checkpoints.torch import to_cpu
from .response import PublicResponseSpec
from .residual_transport import PublicTransportSpec
from .task_response import task_amplitude
from .task_transport import PublicTaskResidualTransport, T2_TRANSPORT_COORDINATES, project_task_owned_requests


VERSION = "public-task-ddim-residual-transport-v1"


class PublicTaskDiffusionTransport(PublicTaskResidualTransport):
    """Same shared13 candidate and T2 task-owned projection as the MLP policy."""

    def __init__(self, response_spec=None, spec=None, *, denoising_steps=4,
                 latent_seed=0, t2_execution_scale=.25):
        super().__init__(response_spec, spec, t2_execution_scale=t2_execution_scale)
        if (not 2 <= denoising_steps <= 8 or isinstance(latent_seed, bool)
                or not isinstance(latent_seed, int) or latent_seed < 0):
            raise ValueError("two to eight deterministic DDIM steps and a nonnegative latent seed required")
        self.denoising_steps = denoising_steps
        self.latent_seed = latent_seed
        context_dim = 2*self.response_spec.feature_dim + 26*self.response_spec.horizon + 4 + 9
        output_dim = 13*self.response_spec.horizon
        sizes = (context_dim + output_dim + denoising_steps, *self.spec.hidden, output_dim)
        layers = []
        for index, (before, after) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(before, after))
            if index < len(sizes)-2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        latent = np.random.default_rng(latent_seed).standard_normal(output_dim).astype(np.float32)
        self.register_buffer("fixed_latent", torch.from_numpy(latent), persistent=True)
        beta = torch.linspace(.01, .08, denoising_steps)
        self.register_buffer("alpha_bar", torch.cat((torch.ones(1), torch.cumprod(1-beta, dim=0))), persistent=True)

    def forward(self, target_features, source_features, source_commands, base_proto,
                phase, own_decoder_context):
        if self.spec.residual_scale != .025:
            raise ValueError("diffusion and MLP candidates require the same 0.025 envelope")
        if (source_commands.shape != base_proto.shape
                or base_proto.shape[1:] != (self.response_spec.horizon, 13)):
            raise ValueError("shared source/base request chunks required")
        context = torch.cat((target_features, source_features,
            source_commands.flatten(start_dim=1), base_proto.flatten(start_dim=1),
            torch.nn.functional.one_hot(phase.long(), 4).to(target_features.dtype),
            own_decoder_context), dim=1)
        latent = self.fixed_latent.to(context.dtype).expand(len(context), -1)
        state = latent
        null_state = latent
        for step in range(self.denoising_steps, 0, -1):
            clock = torch.nn.functional.one_hot(
                torch.full((len(context),), step-1, device=context.device, dtype=torch.long),
                self.denoising_steps).to(context.dtype)
            epsilon = self.network(torch.cat((context, state, clock), dim=1))
            current = self.alpha_bar[step].to(context.dtype)
            previous = self.alpha_bar[step-1].to(context.dtype)
            carry = torch.sqrt(previous/current)
            correction = torch.sqrt(1-previous) - carry*torch.sqrt(1-current)
            state = carry*state + correction*epsilon
            null_state = carry*null_state
        residual = torch.tanh(state-null_state).reshape(-1, self.response_spec.horizon, 13)
        amplitude = task_amplitude(target_features)[:, None]
        mask = base_proto.new_zeros(13)
        mask[list(T2_TRANSPORT_COORDINATES)] = 1.
        obstacle = (target_features[..., -7:-6] > .5)[:, None]
        mask = torch.where(obstacle, torch.ones_like(mask), mask)
        value = base_proto + self.spec.residual_scale*amplitude*residual*mask
        value = torch.minimum(torch.maximum(value, -amplitude), amplitude)
        return project_task_owned_requests(target_features, value, base_proto,
                                           t2_execution_scale=self.t2_execution_scale)

    def checkpoint(self):
        return dict(version=VERSION, response_spec=asdict(self.response_spec),
                    transport_spec=asdict(self.spec), denoising_steps=self.denoising_steps,
                    latent_seed=self.latent_seed, t2_execution_scale=self.t2_execution_scale,
                    state_dict=to_cpu(self.state_dict()), private_state=False,
                    stochastic_noise=False, training_objective="original ICDT; no score matching",
                    sampling="fixed latent, deterministic DDIM, one chunk per replanning tick")

    @classmethod
    def from_checkpoint(cls, payload, device="cpu"):
        if (payload.get("version") != VERSION or payload.get("private_state") is not False
                or payload.get("stochastic_noise") is not False):
            raise ValueError("deterministic diffusion transport checkpoint required")
        response = dict(payload["response_spec"])
        transport = dict(payload["transport_spec"])
        response["hidden"] = tuple(response["hidden"])
        transport["hidden"] = tuple(transport["hidden"])
        model = cls(PublicResponseSpec(**response), PublicTransportSpec(**transport),
                    denoising_steps=payload["denoising_steps"], latent_seed=payload["latent_seed"],
                    t2_execution_scale=payload["t2_execution_scale"]).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model

# Derived from Yifan Hou ACP (MIT), commit 74d68eca3bfb5cb59da131c6351d9e1c9b129ff6.
# Copyright (c) 2024 Yifan Hou@Stanford. See LICENSE and SOURCE.json.
"""Native ACP sparse diffusion policy; checkpoint keys match the author's Mod1 class.

The observation encoder and scheduler are actual native components. This port does
not replace the vision/force encoder with a generic policy callback.
"""
from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .normalizer import LinearNormalizer
from .conditional_unet1d import ConditionalUnet1D
from .module_attr_mixin import ModuleAttrMixin

class BaseImagePolicy(ModuleAttrMixin):
    def reset(self):
        pass

class DiffusionUnetTimmMod1Policy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler,
        obs_encoder: nn.Module,
        num_inference_steps=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        input_pertub=0.1,
        inpaint_fixed_action_prefix=False,
        train_diffusion_n_samples=1,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
        sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
            "down_sample_steps"
        ]

        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())

        # create diffusion model
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model_sparse = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model_sparse = model_sparse
        self.noise_scheduler = noise_scheduler
        self.sparse_normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.sparse_action_horizon = sparse_action_horizon
        self.sparse_action_down_sample_steps = sparse_action_down_sample_steps
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs
        self.sparse_loss = 0

        # store intermediate results from sparse model
        self.sparse_nobs_encode = None
        self.sparse_naction_pred = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= training  ============
    def set_normalizer(
        self,
        sparse_normalizer: LinearNormalizer,
    ):
        self.sparse_normalizer.load_state_dict(sparse_normalizer.state_dict())

    def get_normalizer(self):
        return self.sparse_normalizer

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model_sparse
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self,
        obs: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        obs: include keys from shape_meta['sample']['obs'],
        """
        obs_dict_sparse = obs["sparse"]

        ##
        ## =================  Part one: Sparse =================
        ##
        nobs_sparse = self.sparse_normalizer.normalize(obs_dict_sparse)

        batch_size = next(iter(nobs_sparse.values())).shape[0]

        # condition through global feature
        sparse_nobs_encode = self.obs_encoder(nobs_sparse)

        # empty data for action
        cond_data = torch.zeros(
            size=(batch_size, self.sparse_action_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        sparse_naction_pred = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=sparse_nobs_encode,
            **self.kwargs,
        )

        # unnormalize prediction
        assert sparse_naction_pred.shape == (
            batch_size,
            self.sparse_action_horizon,
            self.action_dim,
        )
        sparse_action_pred = self.sparse_normalizer["action"].unnormalize(
            sparse_naction_pred
        )

        self.sparse_nobs_encode = sparse_nobs_encode
        self.sparse_naction_pred = sparse_naction_pred

        result = {"sparse": sparse_action_pred}
        return result

    def compute_loss(self, batch, args):
        # normalize input
        assert "valid_mask" not in batch
        nobs_sparse = self.sparse_normalizer.normalize(batch["obs"]["sparse"])
        nactions_sparse = self.sparse_normalizer["action"].normalize(
            batch["action"]["sparse"]
        )

        sparse_nobs_encode = self.obs_encoder(nobs_sparse)

        ##
        ## =================  Part one: Sparse =================
        ##
        trajectory = nactions_sparse

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(
            trajectory.shape, device=trajectory.device
        )

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (nactions_sparse.shape[0],),
            device=trajectory.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps
        )

        # Predict the noise residual
        pred_sparse = self.model_sparse(
            noisy_trajectory, timesteps, local_cond=None, global_cond=sparse_nobs_encode
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        sparse_loss = F.mse_loss(pred_sparse, target, reduction="mean")
        self.sparse_loss = sparse_loss

        loss = sparse_loss

        return loss

    def forward(self, batch, flags):
        return self.compute_loss(batch, flags)

    def get_loss_components(self):
        return self.sparse_loss

ACPDiffusionPolicy = DiffusionUnetTimmMod1Policy

# Derived from Yifan Hou ACP (MIT), commit 74d68eca3bfb5cb59da131c6351d9e1c9b129ff6.
# Copyright (c) 2024 Yifan Hou@Stanford. See LICENSE and SOURCE.json.
import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from .module_attr_mixin import ModuleAttrMixin
from .pytorch_util import replace_submodules

# from diffusion_policy.model.vision.force_spec_encoder import ForceSpecEncoder, convert_to_spec
from .causal_force import ForceEncoder

logger = logging.getLogger(__name__)


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class TimmObsEncoderWithForce(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        fuse_mode: str,
        reduce_pretrained_lr: bool,
        vision_encoder_cfg: dict,
        force_encoder_cfg: dict,
        # replace BatchNorm with GroupNorm
        # use single rgb model for all rgb inputs
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        position_encoding: str = "learnable",
    ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        wrench_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        model_list = []
        feature_dim_list = []
        for cfg in [vision_encoder_cfg, force_encoder_cfg]:
            if cfg.model_name == "causalconv":
                model = ForceEncoder(cfg.feature_dim)
            else:
                model = timm.create_model(
                    model_name=cfg.model_name,
                    pretrained=cfg.pretrained,
                    global_pool=cfg.global_pool,
                    num_classes=0,
                )
            if cfg.frozen:
                assert cfg.pretrained
                for param in model.parameters():
                    param.requires_grad = False

            if cfg.model_name.startswith("resnet"):
                # the last layer is nn.Identity() because num_classes is 0
                # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
                if cfg.downsample_ratio == 32:
                    modules = list(model.children())[:-2]
                    model = torch.nn.Sequential(*modules)
                    feature_dim = 512
                elif cfg.downsample_ratio == 16:
                    modules = list(model.children())[:-3]
                    model = torch.nn.Sequential(*modules)
                    feature_dim = 256
                else:
                    raise NotImplementedError(
                        f"Unsupported downsample_ratio: {cfg.downsample_ratio}"
                    )
            elif cfg.model_name.startswith("convnext"):
                # the last layer is nn.Identity() because num_classes is 0
                # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
                if cfg.downsample_ratio == 32:
                    modules = list(model.children())[:-2]
                    model = torch.nn.Sequential(*modules)
                    feature_dim = 1024
                else:
                    raise NotImplementedError(
                        f"Unsupported downsample_ratio: {cfg.downsample_ratio}"
                    )
            elif cfg.model_name == "causalconv":
                feature_dim = cfg.feature_dim
            else:
                feature_dim = 768
            feature_dim_list.append(feature_dim)

            if cfg.use_group_norm and not cfg.pretrained:
                model = replace_submodules(
                    root_module=model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=(
                            (x.num_features // 16)
                            if (x.num_features % 16 == 0)
                            else (x.num_features // 8)
                        ),
                        num_channels=x.num_features,
                    ),
                )
            model_list.append(model)

        vision_encoder, force_encoder = model_list
        self.v_feature_dim, self.f_feature_dim = feature_dim_list
        if force_encoder_cfg.model_name.startswith("resnet"):
            force_encoder = ForceEncoder(force_encoder, force_encoder_cfg.norm_spec)

        image_shape = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            if type == "rgb":
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if vision_encoder_cfg.transforms is not None and not isinstance(
            vision_encoder_cfg.transforms[0], torch.nn.Module
        ):
            assert vision_encoder_cfg.transforms[0].type == "RandomCrop"
            ratio = vision_encoder_cfg.transforms[0].ratio
            vision_encoder_cfg.transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True),
            ] + vision_encoder_cfg.transforms[1:]
        vision_transform = (
            nn.Identity()
            if vision_encoder_cfg.transforms is None
            else torch.nn.Sequential(*vision_encoder_cfg.transforms)
        )

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)

                vision_encoder = (
                    vision_encoder
                    if vision_encoder_cfg.share_rgb_model
                    else copy.deepcopy(vision_encoder)
                )
                key_model_map[key] = vision_encoder
                key_transform_map[key] = vision_transform
            elif type == "low_dim":
                if "wrench" in key:
                    print("key_model_map adding wrench key:", key)
                    wrench_keys.append(key)

                    force_encoder = (
                        force_encoder
                        if force_encoder_cfg.share_force_model
                        else copy.deepcopy(force_encoder)
                    )
                    key_model_map[key] = force_encoder
                else:
                    if not attr.get("ignore_by_policy", False):
                        low_dim_keys.append(key)
            elif type == "timestamp":
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print("rgb keys:         ", rgb_keys)
        print("low_dim_keys keys:", low_dim_keys)

        self.vision_encoder_cfg = vision_encoder_cfg
        self.force_encoder_cfg = force_encoder_cfg
        self.shape_meta = shape_meta
        self.fuse_mode = fuse_mode
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.wrench_keys = wrench_keys
        self.key_shape_map = key_shape_map
        self.position_encoding = position_encoding

        feature_map_shape = [
            x // vision_encoder_cfg.downsample_ratio for x in image_shape
        ]
        if vision_encoder_cfg.model_name.startswith("vit"):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if vision_encoder_cfg.feature_aggregation == "all_tokens":
                # Use all tokens from ViT
                pass
            elif vision_encoder_cfg.feature_aggregation is not None:
                logger.warn(
                    f"vit will use the CLS token. feature_aggregation ({vision_encoder_cfg.feature_aggregation}) is ignored!"
                )
                vision_encoder_cfg.feature_aggregation = None

        if vision_encoder_cfg.feature_aggregation == "soft_attention":
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False), nn.Softmax(dim=1)
            )
        elif vision_encoder_cfg.feature_aggregation == "spatial_embedding":
            self.spatial_embedding = torch.nn.Parameter(
                torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim)
            )
        elif vision_encoder_cfg.feature_aggregation == "transformer":
            if position_encoding == "learnable":
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(
                        feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim
                    )
                )
            elif position_encoding == "sinusoidal":
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, feature_dim, 2).float()
                    * (-math.log(2 * num_features) / feature_dim)
                )
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4,
            )
        elif vision_encoder_cfg.feature_aggregation == "attention_pool_2d":
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim,
            )
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

        if fuse_mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(self.v_feature_dim * 3, 1024), nn.ReLU(), nn.Linear(1024, 512)
            )

        elif fuse_mode == "modality-attention":
            self.transformer_encoder = torch.nn.TransformerEncoderLayer(
                d_model=self.v_feature_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.0,
            )
            n_features = len(rgb_keys) * shape_meta["sample"]["obs"]["sparse"]["rgb_0"][
                "horizon"
            ] + len(wrench_keys)
            self.linear_projection = nn.Linear(
                self.v_feature_dim * n_features, self.v_feature_dim
            )
            if position_encoding == "learnable":
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(n_features, self.v_feature_dim)
                )

    def aggregate_feature(self, model_name, agg_mode, feature):
        if model_name.startswith("vit"):
            assert agg_mode is None  # vit uses the CLS token
            return feature[:, 0, :]

        # resnet
        assert len(feature.shape) == 4
        if agg_mode == "attention_pool_2d":
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2)  # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2)  # B, 7*7, 512

        if agg_mode == "avg":
            return torch.mean(feature, dim=[1])
        elif agg_mode == "max":
            return torch.amax(feature, dim=[1])
        elif agg_mode == "soft_attention":
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif agg_mode == "spatial_embedding":
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif agg_mode == "transformer":
            zero_feature = torch.zeros(
                feature.shape[0], 1, feature.shape[-1], device=feature.device
            )
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = (
                torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            )
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert agg_mode is None
            return feature

    def forward(self, obs_dict):
        """Assume each image key is (B, T, C, H, W)"""
        features = list()
        modality_features = list()
        low_dim_features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(
                model_name=self.vision_encoder_cfg.model_name,
                agg_mode=self.vision_encoder_cfg.feature_aggregation,
                feature=raw_feature,
            )
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))
            modality_features.append(feature.reshape(B, T, -1))

        for key in self.wrench_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.permute(0, 2, 1)
            feature = self.key_model_map[key](data.float())[:, :, 0]
            assert len(feature.shape) == 2 and feature.shape[0] == B
            features.append(feature.reshape(B, -1))
            modality_features.append(feature.unsqueeze(1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            # directly concatenate actions
            features.append(data.reshape(B, -1))
            low_dim_features.append(data.reshape(B, -1))

        # concatenate all features
        if self.fuse_mode == "concat":
            result = torch.cat(features, dim=-1)
        elif self.fuse_mode == "mlp":
            result = self.mlp(torch.cat(modality_features, dim=-1))
            result = torch.concat([result, torch.cat(low_dim_features, dim=-1)], dim=1)
        elif self.fuse_mode == "modality-attention":
            in_embeds = torch.cat(modality_features, dim=1)  # [batch, n_features, D]
            if self.position_encoding == "learnable":
                if self.position_embedding.device != in_embeds.device:
                    self.position_embedding = self.position_embedding.to(feature.device)
                in_embeds = in_embeds + self.position_embedding
            out_embeds = self.transformer_encoder(in_embeds)  # [batch, n_features, D]
            result = torch.concat(
                [out_embeds[:, i] for i in range(out_embeds.shape[1])], dim=1
            )
            result = self.linear_projection(result)
            result = torch.concat([result, torch.cat(low_dim_features, dim=-1)], dim=1)

        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        sample_obs_shape_meta = self.shape_meta["sample"]["obs"]["sparse"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "timestamp":
                continue

            shape = tuple(attr["shape"])
            horizon = sample_obs_shape_meta[key]["horizon"]
            this_obs = torch.zeros(
                (1, horizon) + shape, dtype=self.dtype, device=self.device
            )
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1

        return example_output.shape


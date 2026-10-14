# SPDX-License-Identifier: Apache-2.0
# Derived from LeCAR-Lab/FADA commit 290b3a9c5c0ebe2e9851b61350617544be05e9bc.
# Source: src/holosoma/holosoma/fada/common/lora_utils.py
# Changes: isolate native PyTorch core; replace Holosoma imports with local imports.
from __future__ import annotations

import copy
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .norm_stats import DEFAULT_NORM_EPS, validate_norm_stats

def _extract_norm_stats(
    payload: dict[str, Any],
    *,
    key: str,
    mean_key: str,
    std_key: str,
    dim: int,
    required: bool,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    raw_stats = payload.get(key)
    if raw_stats is None:
        if required:
            raise RuntimeError(f"io_normalization=True requires checkpoint field '{key}', but it is missing.")
        return None
    if not isinstance(raw_stats, dict):
        raise RuntimeError(f"Invalid {key} payload type: {type(raw_stats)}")
    if mean_key not in raw_stats or std_key not in raw_stats:
        if required:
            raise RuntimeError(
                f"io_normalization=True requires {key}.{mean_key}/{std_key}, but checkpoint is missing them."
            )
        return None

    mean = torch.as_tensor(raw_stats[mean_key], device=device, dtype=torch.float32).flatten()
    std = torch.as_tensor(raw_stats[std_key], device=device, dtype=torch.float32).flatten()
    if mean.numel() != dim or std.numel() != dim:
        raise RuntimeError(
            f"Checkpoint {key} shape mismatch: mean={tuple(mean.shape)}, std={tuple(std.shape)}, expected=({dim},)"
        )
    eps = float(raw_stats.get("eps", DEFAULT_NORM_EPS))
    # Rejects NaN/inf in mean/std/eps as well as bad shapes.
    validate_norm_stats(mean, std, eps, key=key)
    std = torch.clamp(std, min=eps)
    return {"mean": mean, "std": std}



class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear, keeping base weights frozen."""

    def __init__(self, linear: nn.Linear, *, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.A = nn.Linear(self.in_features, self.r, bias=False)
        self.B = nn.Linear(self.r, self.out_features, bias=False)
        # Keep adapter weights on the same device/dtype as the replaced backbone layer.
        target_device = linear.weight.device
        target_dtype = linear.weight.dtype
        self.A.to(device=target_device, dtype=target_dtype)
        self.B.to(device=target_device, dtype=target_dtype)
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = self.B(self.A(self.dropout(x))) * self.scaling
        return base + delta

    def merged_weight(self) -> torch.Tensor:
        return self.weight.detach() + self.scaling * (self.B.weight.detach() @ self.A.weight.detach())



class LoRAMultiheadAttention(nn.Module):
    """LoRA wrapper for nn.MultiheadAttention packed QKV projection."""

    def __init__(self, mha: nn.MultiheadAttention, *, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        if not bool(getattr(mha, "_qkv_same_embed_dim", True)):
            raise ValueError("LoRAMultiheadAttention only supports same embed dim for q/k/v projections.")

        self.embed_dim = int(mha.embed_dim)
        self.num_heads = int(mha.num_heads)
        self.dropout = float(mha.dropout)
        self.batch_first = bool(mha.batch_first)
        self.add_zero_attn = bool(mha.add_zero_attn)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)

        self.in_proj_weight = nn.Parameter(mha.in_proj_weight.detach().clone(), requires_grad=False)
        if mha.in_proj_bias is not None:
            self.in_proj_bias = nn.Parameter(mha.in_proj_bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("in_proj_bias", None)

        if mha.bias_k is not None:
            self.bias_k = nn.Parameter(mha.bias_k.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias_k", None)
        if mha.bias_v is not None:
            self.bias_v = nn.Parameter(mha.bias_v.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias_v", None)

        self.out_proj = mha.out_proj
        # QKV LoRA is applied as a fused low-rank delta on packed in_proj_weight so it stays
        # compatible with PyTorch's optimized MultiheadAttention forward path.
        self.A = nn.Linear(self.embed_dim, self.r, bias=False)
        self.B = nn.Linear(self.r, 3 * self.embed_dim, bias=False)
        target_device = mha.in_proj_weight.device
        target_dtype = mha.in_proj_weight.dtype
        self.A.to(device=target_device, dtype=target_dtype)
        self.B.to(device=target_device, dtype=target_dtype)
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.B.weight)

    def merged_in_proj_weight(self) -> torch.Tensor:
        return self.in_proj_weight.detach() + self.scaling * (self.B.weight.detach() @ self.A.weight.detach())

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(0, 1)
                else:
                    query, key = (tensor.transpose(0, 1) for tensor in (query, key))
                    value = key
            else:
                query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))

        out_proj_weight, out_proj_bias = _resolve_linear_weight_bias(self.out_proj)
        common_kwargs = dict(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight + self.scaling * (self.B.weight @ self.A.weight),
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
        )
        try:
            attn_output, attn_weights = F.multi_head_attention_forward(
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                **common_kwargs,
            )
        except TypeError:
            try:
                attn_output, attn_weights = F.multi_head_attention_forward(
                    average_attn_weights=average_attn_weights,
                    **common_kwargs,
                )
            except TypeError:
                attn_output, attn_weights = F.multi_head_attention_forward(**common_kwargs)

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_weights



def _resolve_linear_weight_bias(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(module, LoRALinear):
        return module.merged_weight(), module.bias
    if isinstance(module, nn.Linear):
        return module.weight, module.bias
    raise TypeError(f"Unsupported projection module type: {type(module)}")



def _inject_lora_modules(
    module: nn.Module,
    *,
    should_replace,
    r: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
    replaced: list[str] | None = None,
) -> list[str]:
    if replaced is None:
        replaced = []

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and should_replace(full_name):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
            continue
        _inject_lora_modules(
            child,
            should_replace=should_replace,
            r=r,
            alpha=alpha,
            dropout=dropout,
            prefix=full_name,
            replaced=replaced,
        )
    return replaced



def _inject_lora_multihead_attention_modules(
    module: nn.Module,
    *,
    should_replace,
    r: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
    replaced: list[str] | None = None,
) -> list[str]:
    if replaced is None:
        replaced = []

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.MultiheadAttention) and should_replace(full_name):
            setattr(module, name, LoRAMultiheadAttention(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(f"{full_name}.in_proj_weight")
            continue
        _inject_lora_multihead_attention_modules(
            child,
            should_replace=should_replace,
            r=r,
            alpha=alpha,
            dropout=dropout,
            prefix=full_name,
            replaced=replaced,
        )
    return replaced



def extract_lora_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    adapters: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name.endswith(".A.weight") or name.endswith(".B.weight"):
            adapters[name] = param.detach().cpu().clone()
    return adapters



def extract_trainable_parameter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable[name] = param.detach().cpu()
    return trainable



def build_merged_state_dict(model_with_lora: nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model_with_lora.state_dict()
    merged: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".A.weight") or key.endswith(".B.weight"):
            continue
        merged[key] = value.detach().clone()

    for module_name, module in model_with_lora.named_modules():
        if module_name == "":
            continue
        if isinstance(module, LoRALinear):
            weight_key = f"{module_name}.weight"
            merged[weight_key] = module.merged_weight().detach().clone()
            if module.bias is not None:
                merged[f"{module_name}.bias"] = module.bias.detach().clone()
            continue
        if isinstance(module, LoRAMultiheadAttention):
            merged[f"{module_name}.in_proj_weight"] = module.merged_in_proj_weight().detach().clone()

    return merged



def _clone_state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


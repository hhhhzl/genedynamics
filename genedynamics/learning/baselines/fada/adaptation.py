# SPDX-License-Identifier: Apache-2.0
# Derived from LeCAR-Lab/FADA commit 290b3a9c5c0ebe2e9851b61350617544be05e9bc.
# Source: src/holosoma/holosoma/fada/planner_idm/finetune_idm_lora.py
# Changes: isolate native PyTorch core; replace Holosoma imports with local imports.
from __future__ import annotations

import torch
from torch import optim
from torch.nn import functional as F
from genedynamics.solvers.single.fada.backends.lora import (
    LoRALinear, LoRAMultiheadAttention, _inject_lora_modules,
    _inject_lora_multihead_attention_modules,
)
from .loss import weighted_horizon_mse

def _normalize_idm_if_enabled(
    history_obs: torch.Tensor,
    history_act: torch.Tensor,
    current_command: torch.Tensor,
    future_obs: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    io_norm_enabled: bool,
    obs_stats: dict[str, torch.Tensor] | None,
    action_stats: dict[str, torch.Tensor] | None,
    command_stats: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not io_norm_enabled:
        return history_obs, history_act, current_command, future_obs, target_actions

    if obs_stats is None or action_stats is None or command_stats is None:
        raise RuntimeError("Missing normalization stats for io_normalization=True")

    obs_mean = obs_stats["mean"].view(1, 1, -1)
    obs_std = obs_stats["std"].view(1, 1, -1)
    act_mean = action_stats["mean"].view(1, 1, -1)
    act_std = action_stats["std"].view(1, 1, -1)
    cmd_mean = command_stats["mean"].view(1, -1)
    cmd_std = command_stats["std"].view(1, -1)

    return (
        (history_obs - obs_mean) / obs_std,
        (history_act - act_mean) / act_std,
        (current_command - cmd_mean) / cmd_std,
        (future_obs - obs_mean) / obs_std,
        (target_actions - act_mean) / act_std,
    )



def _normalize_lora_target_scope(scope: str) -> str:
    token = str(scope).strip().lower().replace("-", "_")
    if token in {"encoder_decoder", "full", "all"}:
        return "encoder_decoder"
    if token in {"encoder_decoder_qkv", "full_qkv", "all_qkv", "lora_all"}:
        return "encoder_decoder_qkv"
    if token in {"decoder_only", "decoder", "lora_mlp"}:
        return "decoder_only"
    if token in {"decoder_only_qkv", "decoder_qkv"}:
        return "decoder_only_qkv"
    if token in {"all_linear", "mlp", "mlp_all"}:
        return "all_linear"
    raise ValueError(
        f"Unsupported IDM LoRA target scope: {scope!r}. "
        "Expected one of: encoder_decoder, decoder_only, encoder_decoder_qkv, decoder_only_qkv, all_linear, lora_mlp, lora_all."
    )



def _normalize_finetune_method(method: str) -> str:
    token = str(method).strip().lower().replace("-", "_")
    if token in {"lora", "idm_lora"}:
        return "lora"
    if token in {"full", "full_finetune", "all"}:
        return "full"
    raise ValueError(f"Unsupported finetune method: {method!r}. Expected one of: lora, full.")



def _freeze_all_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = False



def _mark_lora_trainables(model) -> None:
    for name, param in model.named_parameters():
        if name.endswith(".A.weight") or name.endswith(".B.weight"):
            param.requires_grad = True



def _is_idm_backbone_linear(name: str, *, target_scope: str) -> bool:
    normalized_scope = _normalize_lora_target_scope(target_scope)
    if name.startswith("idm.decoder."):
        return True
    if normalized_scope in {"encoder_decoder", "encoder_decoder_qkv"} and name.startswith("idm.history_encoder."):
        return True
    return False



def _is_idm_backbone_attention(name: str, *, target_scope: str) -> bool:
    normalized_scope = _normalize_lora_target_scope(target_scope)
    if normalized_scope not in {"decoder_only_qkv", "encoder_decoder_qkv"}:
        return False
    if name.startswith("idm.decoder."):
        return True
    if normalized_scope == "encoder_decoder_qkv" and name.startswith("idm.history_encoder."):
        return True
    return False



def inject_lora_into_idm_backbone(
    model,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_scope: str = "encoder_decoder",
) -> list[str]:
    normalized_scope = _normalize_lora_target_scope(target_scope)

    _freeze_all_parameters(model)

    replaced = _inject_lora_modules(
        model,
        should_replace=lambda name: _is_idm_backbone_linear(name, target_scope=normalized_scope),
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    if normalized_scope in {"decoder_only_qkv", "encoder_decoder_qkv"}:
        replaced = _inject_lora_multihead_attention_modules(
            model,
            should_replace=lambda name: _is_idm_backbone_attention(name, target_scope=normalized_scope),
            r=r,
            alpha=alpha,
            dropout=dropout,
            replaced=replaced,
        )
    if not replaced:
        raise RuntimeError(
            f"No IDM backbone linear layers matched for LoRA injection (target_scope={normalized_scope})."
        )

    _mark_lora_trainables(model)
    return replaced



def enable_full_idm_finetune(model) -> list[str]:
    _freeze_all_parameters(model)
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        if name.startswith("idm.") or name.startswith("idm_decoder.") or name.startswith("shared_history_encoder."):
            param.requires_grad = True
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("No IDM parameters found for full finetuning.")
    return ["idm"]



def configure_idm_finetune_method(
    model,
    *,
    finetune_method: str,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_target_scope: str,
) -> tuple[list[str], str, str | None]:
    resolved_method = _normalize_finetune_method(finetune_method)
    if resolved_method == "lora":
        effective_scope = lora_target_scope
        replaced = inject_lora_into_idm_backbone(
            model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_scope=effective_scope,
        )
        return replaced, resolved_method, _normalize_lora_target_scope(effective_scope)
    replaced = enable_full_idm_finetune(model)
    return replaced, resolved_method, None



def _loss_metric_key(loss_target: str) -> str:
    token = str(loss_target).strip().lower()
    if token == "idm":
        return "idm_action_loss"
    raise ValueError(f"Unsupported finetune loss target: {loss_target!r}")



def _run_epoch(
    *,
    model,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    io_norm_enabled: bool,
    obs_stats: dict[str, torch.Tensor] | None,
    action_stats: dict[str, torch.Tensor] | None,
    command_stats: dict[str, torch.Tensor] | None,
    pred_horizon_k: int,
    grad_clip: float,
    show_progress: bool,
    epoch_desc: str,
    max_batches: int | None = None,
    loss_target: str = "idm",
    anchor_lambda: float = 1.0,
    horizon_gamma: float = 1.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    lora_active = any(isinstance(m, (LoRALinear, LoRAMultiheadAttention)) for m in model.modules())
    disable_eval_fastpath = (not is_train) and lora_active and bool(torch.backends.mha.get_fastpath_enabled())
    fastpath_prev = bool(torch.backends.mha.get_fastpath_enabled())
    if disable_eval_fastpath:
        torch.backends.mha.set_fastpath_enabled(False)

    total_loss = 0.0
    total_loss_unweighted = 0.0
    total_planner_cycle_loss = 0.0
    total_action_anchor_loss = 0.0
    total_action_anchor_loss_unweighted = 0.0
    total_steps = 0
    loss_metric_key = _loss_metric_key(loss_target)

    iterator = loader
    if show_progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(loader, desc=epoch_desc, leave=False, dynamic_ncols=True)
        except Exception:
            iterator = loader

    try:
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break
            history_obs = batch["history_obs"].to(device=device, dtype=torch.float32)
            history_act = batch["history_act"].to(device=device, dtype=torch.float32)
            current_command = batch["current_command"].to(device=device, dtype=torch.float32)
            history_valid = batch["history_valid_mask"].to(device=device, dtype=torch.bool)
            future_obs = batch["future_observations"].to(device=device, dtype=torch.float32)
            target_actions = batch["target_actions"].to(device=device, dtype=torch.float32)

            history_obs, history_act, current_command, future_obs, target_actions = _normalize_idm_if_enabled(
                history_obs,
                history_act,
                current_command,
                future_obs,
                target_actions,
                io_norm_enabled=io_norm_enabled,
                obs_stats=obs_stats,
                action_stats=action_stats,
                command_stats=command_stats,
            )

            with torch.set_grad_enabled(is_train):
                idm_current_command = model._resolve_idm_current_command(
                    current_command,
                    idm_use_current_command_for_history=model.idm_use_current_command_for_history,
                )
                pred_actions = model.predict_idm_actions(
                    history_obs,
                    history_act,
                    idm_current_command,
                    future_obs,
                    history_valid_mask=history_valid,
                )
                pred_actions = pred_actions[:, :pred_horizon_k, :]
                # Adapt target when the IDM head outputs a single action step
                # (pred_horizon=1) but the batch carries a multi-step target.
                idm_target = target_actions
                if pred_actions.shape[1] == 1 and idm_target.shape[1] > 1:
                    idm_target = idm_target[:, :1, :]
                loss_uw = F.mse_loss(pred_actions, idm_target[:, :pred_horizon_k, :])
                loss = weighted_horizon_mse(pred_actions, idm_target[:, :pred_horizon_k, :], horizon_gamma)

                if is_train:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip > 0.0:
                        params = [p for p in model.parameters() if p.requires_grad]
                        if params:
                            torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_steps += 1
            total_loss_unweighted += float(loss_uw.detach().cpu().item())
    finally:
        if disable_eval_fastpath:
            torch.backends.mha.set_fastpath_enabled(fastpath_prev)

    if total_steps <= 0:
        result = {"loss": float("nan"), "steps": 0.0}
    else:
        result = {"loss": total_loss / total_steps, "steps": float(total_steps)}
    result[loss_metric_key] = result["loss"]
    if total_steps > 0:
        result["idm_action_loss_unweighted"] = total_loss_unweighted / total_steps
    return result


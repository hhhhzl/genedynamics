# SPDX-License-Identifier: Apache-2.0
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Selected unmodified methods from pinned FADA, licensed under Apache-2.0.
# See solvers/single/fada/LICENSE, NOTICE and native_training_manifest.json.
"""Pinned native FADA training pass definitions; task bindings live separately."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from .loss import weighted_horizon_mse


class NativeTrainingPasses:
    @staticmethod
    def _empty_source_counts() -> dict[str, int]:
        return {
            "optimal": 0,
            "suboptimal": 0,
            "suboptimal_expert": 0,
            "online": 0,
            "online_trajectory": 0,
        }


    @classmethod
    def _merge_source_counts(cls, *counts: dict[str, int]) -> dict[str, int]:
        merged = cls._empty_source_counts()
        for count in counts:
            for key in merged:
                merged[key] += int(count.get(key, 0))
        return merged


    @staticmethod
    def _prefix_source_counts(prefix: str, counts: dict[str, int]) -> dict[str, int]:
        return {
            f"{prefix}_optimal_batch_size": int(counts.get("optimal", 0)),
            f"{prefix}_suboptimal_batch_size": int(counts.get("suboptimal", 0)),
            f"{prefix}_suboptimal_expert_batch_size": int(counts.get("suboptimal_expert", 0)),
            f"{prefix}_online_batch_size": int(counts.get("online", 0)),
            f"{prefix}_online_trajectory_batch_size": int(counts.get("online_trajectory", 0)),
        }


    def _build_sample_stats(
        self,
        *,
        planner_counts: dict[str, int],
        idm_counts: dict[str, int],
    ) -> dict[str, int]:
        total_counts = self._merge_source_counts(planner_counts, idm_counts)
        stats = {
            "optimal_batch_size": int(total_counts["optimal"]),
            "suboptimal_batch_size": int(total_counts["suboptimal"]),
            "suboptimal_expert_batch_size": int(total_counts["suboptimal_expert"]),
            "online_batch_size": int(total_counts["online"]),
            "online_trajectory_batch_size": int(total_counts["online_trajectory"]),
        }
        stats.update(self._prefix_source_counts("planner", planner_counts))
        stats.update(self._prefix_source_counts("idm", idm_counts))
        return stats


    def _planner_batch_counts(self, *, mode: str) -> dict[str, int]:
        # Planner always uses oracle-relabeled targets for all sources, so no
        # online-trajectory / suboptimal-expert splits are applied; those keys
        # are kept at 0 for schema consistency with IDM counts.
        #
        # Warmup (mode="offline") uses a separate ratio so warmup can target a
        # balanced optimal:suboptimal split (e.g., 50/50) without affecting the
        # DAgger-phase offline:online 50/50 balance.
        if mode == "offline":
            base_suboptimal_ratio = float(self.cfg.warmup_planner_suboptimal_batch_ratio)
        else:
            base_suboptimal_ratio = float(self.cfg.planner_suboptimal_batch_ratio)
        effective_suboptimal_ratio = base_suboptimal_ratio if self.planner_suboptimal_enabled else 0.0
        suboptimal_bs = int(round(int(self.cfg.batch_size) * effective_suboptimal_ratio))
        suboptimal_bs = max(0, min(suboptimal_bs, int(self.cfg.batch_size)))
        remaining = int(self.cfg.batch_size) - suboptimal_bs
        if mode == "offline":
            return {
                "optimal": int(remaining),
                "suboptimal": int(suboptimal_bs),
                "suboptimal_expert": 0,
                "online": 0,
                "online_trajectory": 0,
            }
        if mode != "mixed":
            raise ValueError(f"Unknown planner train mode: {mode}")
        if remaining <= 0:
            return {
                "optimal": 0,
                "suboptimal": int(suboptimal_bs),
                "suboptimal_expert": 0,
                "online": 0,
                "online_trajectory": 0,
            }
        optimal_bs = int(round(remaining * float(self.cfg.mixed_batch_offline_ratio)))
        optimal_bs = max(0, min(optimal_bs, remaining))
        online_bs = remaining - optimal_bs
        return {
            "optimal": int(optimal_bs),
            "suboptimal": int(suboptimal_bs),
            "suboptimal_expert": 0,
            "online": int(online_bs),
            "online_trajectory": 0,
        }


    def _idm_batch_counts(self, *, mode: str) -> dict[str, int]:
        # Warmup (mode="offline") uses a separate ratio so warmup can target a
        # balanced optimal:(suboptimal_real + suboptimal_expert) split (e.g.,
        # 50/50) without affecting the DAgger-phase offline:online 50/50 balance.
        if mode == "offline":
            base_suboptimal_ratio = float(self.cfg.warmup_idm_suboptimal_batch_ratio)
        else:
            base_suboptimal_ratio = float(self.cfg.idm_suboptimal_batch_ratio)
        effective_suboptimal_ratio = base_suboptimal_ratio if self.suboptimal_enabled else 0.0
        total_suboptimal_bs = int(round(int(self.cfg.batch_size) * effective_suboptimal_ratio))
        total_suboptimal_bs = max(0, min(total_suboptimal_bs, int(self.cfg.batch_size)))
        # Split total suboptimal budget between real-trajectory and expert-shadow targets.
        expert_ratio = float(self.cfg.suboptimal_expert_batch_ratio)
        expert_ratio = max(0.0, min(1.0, expert_ratio))
        suboptimal_expert_bs = int(round(total_suboptimal_bs * expert_ratio))
        suboptimal_expert_bs = max(0, min(suboptimal_expert_bs, total_suboptimal_bs))
        suboptimal_bs = total_suboptimal_bs - suboptimal_expert_bs
        remaining = int(self.cfg.batch_size) - total_suboptimal_bs
        if mode == "offline":
            # No online data in offline/warmup mode.
            return {
                "optimal": int(remaining),
                "suboptimal": int(suboptimal_bs),
                "suboptimal_expert": int(suboptimal_expert_bs),
                "online": 0,
                "online_trajectory": 0,
            }
        if mode != "mixed":
            raise ValueError(f"Unknown IDM train mode: {mode}")
        if remaining <= 0:
            return {
                "optimal": 0,
                "suboptimal": int(suboptimal_bs),
                "suboptimal_expert": int(suboptimal_expert_bs),
                "online": 0,
                "online_trajectory": 0,
            }
        optimal_bs = int(round(remaining * float(self.cfg.mixed_batch_offline_ratio)))
        optimal_bs = max(0, min(optimal_bs, remaining))
        total_online_bs = remaining - optimal_bs
        # Split total online budget between oracle-shadow and real-trajectory targets.
        # When use_generalized_idm=False, skip oracle-shadow online data so IDM only
        # sees real dynamics (executed actions, not oracle-relabeled actions).
        if not bool(self.cfg.use_generalized_idm):
            online_trajectory_bs = total_online_bs
            online_bs = 0
        else:
            trajectory_ratio = float(self.cfg.online_trajectory_batch_ratio)
            trajectory_ratio = max(0.0, min(1.0, trajectory_ratio))
            online_trajectory_bs = int(round(total_online_bs * trajectory_ratio))
            online_trajectory_bs = max(0, min(online_trajectory_bs, total_online_bs))
            online_bs = total_online_bs - online_trajectory_bs
        return {
            "optimal": int(optimal_bs),
            "suboptimal": int(suboptimal_bs),
            "suboptimal_expert": int(suboptimal_expert_bs),
            "online": int(online_bs),
            "online_trajectory": int(online_trajectory_bs),
        }


    def _apply_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Add per-term uniform noise to obs tensor (B, T, D) or (B, D)."""
        noise_scales = self._build_augment_noise_scales()
        if noise_scales is None:
            return obs
        shape = obs.shape
        if obs.ndim == 3:
            noise_scales = noise_scales.view(1, 1, -1)
        elif obs.ndim == 2:
            noise_scales = noise_scales.view(1, -1)
        noise = (torch.rand(shape, device=obs.device, dtype=obs.dtype) * 2.0 - 1.0) * noise_scales
        return obs + noise


    def _apply_history_mask(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Randomly zero out history time steps and update valid mask."""
        ratio = float(self.cfg.augment_history_mask_ratio)
        if ratio <= 0.0:
            return history_obs, history_act, history_valid_mask
        B, T, _ = history_obs.shape
        # Never mask the last time step (current obs)
        mask_candidates = T - 1
        if mask_candidates <= 0:
            return history_obs, history_act, history_valid_mask
        # Generate random mask for first T-1 steps
        drop_mask = torch.rand((B, mask_candidates), device=history_obs.device) < ratio
        # Pad with False for the last time step (never drop)
        full_drop = torch.cat([drop_mask, torch.zeros((B, 1), device=history_obs.device, dtype=torch.bool)], dim=1)
        history_obs = history_obs.clone()
        history_act = history_act.clone()
        history_obs[full_drop] = 0.0
        history_act[full_drop] = 0.0
        if history_valid_mask is not None:
            history_valid_mask = history_valid_mask.clone()
            history_valid_mask[full_drop] = False
        return history_obs, history_act, history_valid_mask


    def _apply_future_mask(self, future: torch.Tensor) -> torch.Tensor:
        """Randomly zero out future time steps (works for both obs and actions)."""
        ratio = float(self.cfg.augment_future_mask_ratio)
        if ratio <= 0.0:
            return future
        B, T, _ = future.shape
        drop_mask = torch.rand((B, T), device=future.device) < ratio
        future = future.clone()
        future[drop_mask] = 0.0
        return future


    def _apply_action_noise(self, actions: torch.Tensor) -> torch.Tensor:
        """Add uniform noise to action tensor (B, T, D) or (B, D)."""
        level = float(self.cfg.augment_action_noise_level)
        if level <= 0.0:
            return actions
        noise = (torch.rand_like(actions) * 2.0 - 1.0) * level
        return actions + noise


    def _to_delta_obs(self, target_obs: torch.Tensor, ref_obs: torch.Tensor) -> torch.Tensor:
        """Convert absolute future obs target to delta relative to ref_obs.

        target_obs: (B, K, D) — absolute future observations
        ref_obs: (B, D) — reference (current) observation
        Returns: (B, K, D) — delta predictions
        """
        return target_obs - ref_obs.unsqueeze(1)


    def _from_delta_obs(self, delta_obs: torch.Tensor, ref_obs: torch.Tensor) -> torch.Tensor:
        """Reconstruct absolute future obs from delta prediction.

        delta_obs: (B, K, D) — delta predictions
        ref_obs: (B, D) — reference (current) observation
        Returns: (B, K, D) — absolute future observations
        """
        return delta_obs + ref_obs.unsqueeze(1)


    def _compute_per_term_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Compute MSE decomposed by compact obs term."""
        results = {}
        for term_name, start, end in self._obs_term_slices():
            term_loss = F.mse_loss(pred[..., start:end], target[..., start:end])
            results[term_name] = float(term_loss.detach().cpu().item())
        return results


    def _normalize_batch_inputs(
        self,
        *,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        target_actions: torch.Tensor,
        target_next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_history_obs = history_obs
        model_history_act = history_act
        model_current_command = current_command
        target_actions_loss_input = target_actions
        target_obs_loss_input = target_next_obs
        if self._use_io_normalization():
            obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
            action_mean, action_std = self._ensure_action_norm_stats_for_training()
            command_mean, command_std = self._ensure_command_norm_stats_for_training()
            obs_mean_seq = obs_mean.view(1, 1, -1)
            obs_std_seq = obs_std.view(1, 1, -1)
            action_mean_seq = action_mean.view(1, 1, -1)
            action_std_seq = action_std.view(1, 1, -1)
            command_mean_seq = command_mean.view(1, -1)
            command_std_seq = command_std.view(1, -1)
            model_history_obs = (history_obs - obs_mean_seq) / obs_std_seq
            model_history_act = (history_act - action_mean_seq) / action_std_seq
            model_current_command = (current_command - command_mean_seq) / command_std_seq
            target_actions_loss_input = (target_actions - action_mean_seq) / action_std_seq
            target_obs_loss_input = (target_next_obs - obs_mean_seq) / obs_std_seq
        return (
            model_history_obs,
            model_history_act,
            model_current_command,
            target_actions_loss_input,
            target_obs_loss_input,
        )


    def _compute_planner_batch_loss_from_tensors(
        self,
        *,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        target_next_obs: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
        # Save clean reference obs for delta conversion (before noise)
        ref_obs = history_obs[:, -1, :]  # (B, D)

        # Apply augmentation (noise + masking) to inputs only
        if augment:
            history_obs = self._apply_obs_noise(history_obs)
            history_act = self._apply_action_noise(history_act)
            history_obs, history_act, history_valid_mask = self._apply_history_mask(
                history_obs, history_act, history_valid_mask,
            )

        (
            model_history_obs,
            model_history_act,
            model_current_command,
            _target_actions_loss_input,
            target_obs_loss_input,
        ) = self._normalize_batch_inputs(
            history_obs=history_obs,
            history_act=history_act,
            current_command=current_command,
            target_actions=torch.zeros(
                (history_obs.shape[0], self.cfg.pred_horizon, self.act_dim),
                device=history_obs.device,
                dtype=history_obs.dtype,
            ),
            target_next_obs=target_next_obs,
        )

        # Delta target: predict change relative to current obs
        if bool(self.cfg.planner_predict_delta):
            # Normalize ref_obs the same way as target
            if self._use_io_normalization():
                obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
                ref_obs_norm = (ref_obs - obs_mean) / obs_std
            else:
                ref_obs_norm = ref_obs
            target_obs_loss_input = self._to_delta_obs(target_obs_loss_input, ref_obs_norm)

        pred_future_obs = self.student_policy.planner(
            model_history_obs,
            model_current_command,
            history_action=(model_history_act if self.student_policy.planner_use_action_history else None),
            history_valid_mask=history_valid_mask,
        )
        planner_loss = F.mse_loss(pred_future_obs, target_obs_loss_input)
        per_term = self._compute_per_term_mse(pred_future_obs, target_obs_loss_input)
        weighted_total = float(self.cfg.planner_obs_loss_coef) * planner_loss
        return weighted_total, planner_loss, per_term


    def _resolve_idm_future_obs(
        self,
        *,
        planner_future_obs: torch.Tensor,
        teacher_future_obs: torch.Tensor,
        ref_obs_norm: torch.Tensor | None = None,
        use_teacher_forcing: bool | None = None,
        teacher_forcing_ratio: float | None = None,
        detach_override: bool | None = None,
    ) -> torch.Tensor:
        use_tf = bool(self.cfg.idm_use_teacher_forcing) if use_teacher_forcing is None else bool(use_teacher_forcing)
        teacher_ratio = float(self.cfg.idm_teacher_forcing_ratio) if teacher_forcing_ratio is None else float(
            teacher_forcing_ratio
        )
        if not use_tf:
            teacher_ratio = 0.0
        # Reconstruct absolute from delta if planner uses delta prediction
        should_detach = bool(self.cfg.idm_detach_planner_future_obs) if detach_override is None else bool(detach_override)
        if should_detach:
            planner_source = planner_future_obs.detach()
        else:
            planner_source = planner_future_obs
        if bool(self.cfg.planner_predict_delta) and ref_obs_norm is not None:
            planner_source = self._from_delta_obs(planner_source, ref_obs_norm)
        if teacher_ratio <= 0.0:
            return planner_source
        if teacher_ratio >= 1.0:
            return teacher_future_obs
        batch_size = int(planner_future_obs.shape[0])
        mix_mask = (
            torch.rand((batch_size, 1, 1), device=planner_future_obs.device, dtype=planner_future_obs.dtype)
            < teacher_ratio
        ).to(dtype=planner_future_obs.dtype)
        return mix_mask * teacher_future_obs + (1.0 - mix_mask) * planner_source


    def _compute_idm_batch_loss_from_tensors(
        self,
        *,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        target_actions: torch.Tensor,
        target_next_obs: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
        use_teacher_forcing: bool | None = None,
        teacher_forcing_ratio: float | None = None,
        detach_override: bool | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Save clean reference obs for delta reconstruction (before noise)
        ref_obs = history_obs[:, -1, :]  # (B, D)

        # Apply augmentation to inputs
        if augment:
            history_obs = self._apply_obs_noise(history_obs)
            history_act = self._apply_action_noise(history_act)
            history_obs, history_act, history_valid_mask = self._apply_history_mask(
                history_obs, history_act, history_valid_mask,
            )

        (
            model_history_obs,
            model_history_act,
            model_current_command,
            target_actions_loss_input,
            target_obs_loss_input,
        ) = self._normalize_batch_inputs(
            history_obs=history_obs,
            history_act=history_act,
            current_command=current_command,
            target_actions=target_actions,
            target_next_obs=target_next_obs,
        )

        # Resolve effective TF ratio early to skip planner forward on pure-TF paths.
        _use_tf = bool(self.cfg.idm_use_teacher_forcing) if use_teacher_forcing is None else bool(use_teacher_forcing)
        _teacher_ratio = (
            float(self.cfg.idm_teacher_forcing_ratio) if teacher_forcing_ratio is None else float(teacher_forcing_ratio)
        )
        if not _use_tf:
            _teacher_ratio = 0.0

        if _teacher_ratio >= 1.0:
            # Pure TF: teacher obs is used directly — skip planner forward entirely.
            idm_future_obs = target_obs_loss_input
        else:
            # Compute normalized ref_obs for delta reconstruction
            ref_obs_norm: torch.Tensor | None = None
            if bool(self.cfg.planner_predict_delta):
                if self._use_io_normalization():
                    obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
                    ref_obs_norm = (ref_obs - obs_mean) / obs_std
                else:
                    ref_obs_norm = ref_obs

            planner_future_obs = self.student_policy.planner(
                model_history_obs,
                model_current_command,
                history_action=(model_history_act if self.student_policy.planner_use_action_history else None),
                history_valid_mask=history_valid_mask,
            )
            idm_future_obs = self._resolve_idm_future_obs(
                planner_future_obs=planner_future_obs,
                teacher_future_obs=target_obs_loss_input,
                ref_obs_norm=ref_obs_norm,
                use_teacher_forcing=use_teacher_forcing,
                teacher_forcing_ratio=teacher_forcing_ratio,
                detach_override=detach_override,
            )
        # Apply future obs noise + masking to IDM input (after teacher forcing resolution)
        if augment:
            idm_future_obs = self._apply_obs_noise(idm_future_obs)
            idm_future_obs = self._apply_future_mask(idm_future_obs)
        idm_current_command = self.student_policy._resolve_idm_current_command(
            model_current_command,
            idm_use_current_command_for_history=self.student_policy.idm_use_current_command_for_history,
        )
        pred_actions = self.student_policy.predict_idm_actions(
            model_history_obs,
            model_history_act,
            idm_current_command,
            idm_future_obs,
            history_valid_mask=history_valid_mask,
        )
        # Hybrid IDM outputs (B, 1, A); slice target to first step. Gamma is enforced
        # to 0.0 at config-validation time, so this is consistent and not a silent fallback.
        if pred_actions.shape[1] != target_actions_loss_input.shape[1]:
            if pred_actions.shape[1] != 1:
                raise RuntimeError(
                    f"IDM pred_actions shape {tuple(pred_actions.shape)} incompatible with "
                    f"target shape {tuple(target_actions_loss_input.shape)}"
                )
            target_actions_loss_input = target_actions_loss_input[:, :1, :]
        idm_loss_unweighted = F.mse_loss(pred_actions, target_actions_loss_input)
        gamma = float(self.cfg.idm_action_loss_horizon_gamma)
        idm_loss = weighted_horizon_mse(pred_actions, target_actions_loss_input, gamma)
        weighted_total = float(self.cfg.idm_action_loss_coef) * idm_loss
        return weighted_total, idm_loss, idm_loss_unweighted


    def _compute_planner_batch_loss_from_batch(
        self,
        batch: dict[str, np.ndarray],
        *,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
        history_obs, history_act, current_command, _target_actions, target_next_obs, history_valid_mask = (
            self._batch_to_tensors(batch)
        )
        if target_next_obs is None:
            raise RuntimeError("planner batch requires target_next_observations")
        return self._compute_planner_batch_loss_from_tensors(
            history_obs=history_obs,
            history_act=history_act,
            current_command=current_command,
            target_next_obs=target_next_obs,
            history_valid_mask=history_valid_mask,
            augment=augment,
        )


    def _compute_idm_batch_loss_from_batch(
        self,
        batch: dict[str, np.ndarray],
        *,
        use_teacher_forcing: bool | None = None,
        teacher_forcing_ratio: float | None = None,
        use_trajectory_targets: bool = False,
        detach_override: bool | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_obs, history_act, current_command, target_actions, target_next_obs, history_valid_mask = (
            self._batch_to_tensors(batch)
        )
        # For suboptimal data, use trajectory targets (real causal dynamics from
        # executed_act / obs) instead of chunk targets (oracle-relabeled).
        # For optimal/online data, keep expert chunk targets (expert causality).
        if use_trajectory_targets:
            if "trajectory_target_actions" in batch:
                target_actions = torch.from_numpy(batch["trajectory_target_actions"]).to(
                    device=self.device, dtype=torch.float32,
                )
            if "trajectory_target_next_observations" in batch:
                target_next_obs = torch.from_numpy(batch["trajectory_target_next_observations"]).to(
                    device=self.device, dtype=torch.float32,
                )
            # Drop boundary samples whose trajectory targets are zero-filled
            # (anchor too close to episode end for K consecutive steps).
            if "trajectory_target_valid" in batch:
                valid_mask = torch.from_numpy(batch["trajectory_target_valid"]).to(device=self.device)
                if not valid_mask.all():
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    if valid_idx.numel() == 0:
                        zero = torch.zeros((), device=self.device, dtype=torch.float32)
                        return zero, zero, zero
                    history_obs = history_obs[valid_idx]
                    history_act = history_act[valid_idx]
                    current_command = current_command[valid_idx]
                    target_actions = target_actions[valid_idx]
                    if target_next_obs is not None:
                        target_next_obs = target_next_obs[valid_idx]
                    history_valid_mask = history_valid_mask[valid_idx]
        if target_next_obs is None:
            raise RuntimeError("IDM batch requires target_next_observations")
        return self._compute_idm_batch_loss_from_tensors(
            history_obs=history_obs,
            history_act=history_act,
            current_command=current_command,
            target_actions=target_actions,
            target_next_obs=target_next_obs,
            history_valid_mask=history_valid_mask,
            use_teacher_forcing=use_teacher_forcing,
            teacher_forcing_ratio=teacher_forcing_ratio,
            detach_override=detach_override,
            augment=augment,
        )


    def _idm_source_uses_teacher_forcing(self, source_name: str) -> bool:
        # Trajectory sources (suboptimal, online_trajectory) always use TF=1.0:
        # their labels are real executed actions that must be paired with real obs.
        # Oracle-shadow sources (optimal, suboptimal_expert, online) follow the
        # global cfg flag and idm_teacher_forcing_ratio uniformly.
        if source_name in {"suboptimal", "online_trajectory"}:
            return True
        return bool(self.cfg.idm_use_teacher_forcing)


    def _compute_idm_batch_loss_from_source_batches(
        self,
        batches_by_source: dict[str, dict[str, np.ndarray]],
        *,
        force_teacher_forcing_ratio: float | None = None,
        detach_override: bool | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weighted_losses: list[torch.Tensor] = []
        idm_losses: list[torch.Tensor] = []
        idm_losses_unweighted: list[torch.Tensor] = []
        batch_sizes: list[int] = []
        # Real-causal sources (suboptimal, online_trajectory) route targets to the
        # executed-action / real-next-obs fields; oracle-causal sources (optimal,
        # online, suboptimal_expert) keep oracle shadow chunks.
        for source_name in ("optimal", "suboptimal", "suboptimal_expert", "online", "online_trajectory"):
            batch = batches_by_source.get(source_name)
            if batch is None:
                continue
            # Trajectory sources are always fully TF (ratio=1.0); oracle-shadow sources
            # (optimal, suboptimal_expert, online) share the same idm_teacher_forcing_ratio.
            # force_teacher_forcing_ratio overrides all per-source ratios when provided.
            if force_teacher_forcing_ratio is not None:
                tf_ratio = float(force_teacher_forcing_ratio)
                use_tf = tf_ratio > 0.0
            else:
                tf_ratio = 1.0 if source_name in {"suboptimal", "online_trajectory"} else None
                use_tf = self._idm_source_uses_teacher_forcing(source_name)
            loss_total, idm_loss, idm_loss_uw = self._compute_idm_batch_loss_from_batch(
                batch,
                use_teacher_forcing=use_tf,
                use_trajectory_targets=(source_name in {"suboptimal", "online_trajectory"}),
                teacher_forcing_ratio=tf_ratio,
                detach_override=detach_override,
                augment=augment,
            )
            weighted_losses.append(loss_total)
            idm_losses.append(idm_loss)
            idm_losses_unweighted.append(idm_loss_uw)
            batch_sizes.append(int(batch["history_obs"].shape[0]))
        if not weighted_losses or not batch_sizes:
            raise RuntimeError("No IDM source batches were provided for loss computation")
        total_batch = int(sum(batch_sizes))
        device = weighted_losses[0].device
        dtype = weighted_losses[0].dtype
        weights = [torch.as_tensor(bs / total_batch, device=device, dtype=dtype) for bs in batch_sizes]
        loss_total = sum(weight * loss for weight, loss in zip(weights, weighted_losses, strict=False))
        idm_loss = sum(weight * loss for weight, loss in zip(weights, idm_losses, strict=False))
        idm_loss_unweighted = sum(weight * loss for weight, loss in zip(weights, idm_losses_unweighted, strict=False))
        return loss_total, idm_loss, idm_loss_unweighted


    def _train_step_separate_pass(self, *, mode: str) -> tuple[float, dict[str, int], dict[str, float]]:
        assert self.idm_optimizer is not None, "idm_optimizer required for separate-pass mode"
        self.student_model.train()

        planner_batch, planner_counts = self._sample_planner_batch(mode=mode)
        idm_batches_by_source, idm_counts = self._sample_idm_batches_by_source(mode=mode)

        idm_params = list(self.student_model.idm_parameters())

        # ── IDM pass (first): all sources, forced TF=1.0, pure inverse dynamics ──
        idm_loss_total, idm_loss, idm_loss_unweighted = self._compute_idm_batch_loss_from_source_batches(
            idm_batches_by_source, force_teacher_forcing_ratio=1.0
        )
        self.idm_optimizer.zero_grad(set_to_none=True)
        idm_loss_total.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(idm_params, self.cfg.grad_clip)
        self.idm_optimizer.step()

        # ── Planner pass (second): planner obs loss + IDM action loss (oracle-shadow, TF=0) ──
        # Freeze IDM params so autograd skips their gradient computation entirely.
        # Save prior requires_grad to safely restore even if backward() raises or IDM is partially frozen.
        prev_requires_grad = [p.requires_grad for p in idm_params]
        for p in idm_params:
            p.requires_grad_(False)

        planner_loss_total, planner_loss, planner_per_term = self._compute_planner_batch_loss_from_batch(planner_batch)

        idm_planner_loss = torch.zeros((), device=self.device, dtype=planner_loss_total.dtype)
        idm_planner_loss_uw = torch.zeros((), device=self.device, dtype=planner_loss_total.dtype)
        idm_planner_loss_total_weighted = torch.zeros((), device=self.device, dtype=planner_loss_total.dtype)
        _, idm_planner_loss, idm_planner_loss_uw = self._compute_idm_batch_loss_from_batch(
            planner_batch,
            use_teacher_forcing=False,
            teacher_forcing_ratio=0.0,
            detach_override=False,
        )
        coef = float(self.cfg.idm_planner_pass_action_loss_coef)
        idm_planner_loss_total_weighted = coef * float(self.cfg.idm_action_loss_coef) * idm_planner_loss
        total_planner_loss = planner_loss_total + idm_planner_loss_total_weighted

        self.optimizer.zero_grad(set_to_none=True)
        try:
            total_planner_loss.backward()
        finally:
            for p, prev in zip(idm_params, prev_requires_grad):
                p.requires_grad_(prev)
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(self.student_model.planner_parameters()), self.cfg.grad_clip)
        self.optimizer.step()

        self.train_step_count += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.idm_lr_scheduler is not None:
            self.idm_lr_scheduler.step()

        loss_total_combined = total_planner_loss + idm_loss_total
        sample_stats = self._build_sample_stats(planner_counts=planner_counts, idm_counts=idm_counts)
        loss_metrics = {
            "loss_total": float(loss_total_combined.detach().cpu().item()),
            "planner_loss_total": float(planner_loss_total.detach().cpu().item()),
            "planner_future_obs_loss": float(planner_loss.detach().cpu().item()),
            "planner_future_obs_loss_weighted": float(float(self.cfg.planner_obs_loss_coef) * planner_loss.detach().cpu().item()),
            "idm_loss_total": float(idm_loss_total.detach().cpu().item()),
            "idm_action_loss": float(idm_loss.detach().cpu().item()),
            "idm_action_loss_unweighted": float(idm_loss_unweighted.detach().cpu().item()),
            "idm_action_loss_weighted": float(float(self.cfg.idm_action_loss_coef) * idm_loss.detach().cpu().item()),
            "idm_planner_pass_action_loss": float(idm_planner_loss.detach().cpu().item()),
            "idm_planner_pass_action_loss_unweighted": float(idm_planner_loss_uw.detach().cpu().item()),
            "idm_planner_pass_action_loss_weighted": float(idm_planner_loss_total_weighted.detach().cpu().item()),
        }
        if planner_per_term is not None:
            for term_name, term_loss in planner_per_term.items():
                loss_metrics[f"planner_term/{term_name}"] = float(term_loss)
        return float(loss_total_combined.detach().cpu().item()), sample_stats, loss_metrics


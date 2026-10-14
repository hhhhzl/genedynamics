# SPDX-License-Identifier: Apache-2.0
# Derived from LeCAR-Lab/FADA commit 290b3a9c5c0ebe2e9851b61350617544be05e9bc.
# Source: src/holosoma/holosoma/fada/planner_idm/model.py
# Changes: isolate native PyTorch core; replace Holosoma imports with local imports.
from __future__ import annotations

import torch
from torch import nn
from .positional import SinusoidalPositionalEncoding

class _PositionalEncodingMixin:
    @staticmethod
    def _build_positional_encoding(
        *,
        d_model: int,
        seq_len: int,
        use_learned: bool,
    ) -> tuple[nn.Parameter | None, SinusoidalPositionalEncoding | None]:
        if use_learned:
            embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.normal_(embedding, mean=0.0, std=0.02)
            return embedding, None
        return None, SinusoidalPositionalEncoding(d_model, max_len=seq_len)

    @staticmethod
    def _resolve_positional_encoding(
        *,
        pos_embedding: nn.Parameter | None,
        sinusoidal: SinusoidalPositionalEncoding | None,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if pos_embedding is not None:
            return pos_embedding[:, :seq_len, :].to(device=device, dtype=dtype)
        if sinusoidal is None:
            raise RuntimeError("Both learned and sinusoidal positional encodings are missing")
        return sinusoidal(seq_len, device=device, dtype=dtype)



class Planner(nn.Module, _PositionalEncodingMixin):
    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        history_len: int,
        pred_horizon: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_learned_positional_encoding: bool = False,
        use_action_history: bool = False,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or act_dim <= 0 or cmd_dim <= 0:
            raise ValueError("obs_dim, act_dim, cmd_dim must be positive")
        if history_len <= 0 or pred_horizon <= 0:
            raise ValueError("history_len and pred_horizon must be positive")
        if d_model <= 0 or nhead <= 0 or num_layers <= 0:
            raise ValueError("d_model, nhead, num_layers must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cmd_dim = int(cmd_dim)
        self.history_len = int(history_len)
        self.pred_horizon = int(pred_horizon)
        self.d_model = int(d_model)
        self.use_action_history = bool(use_action_history)

        self.obs_embed = nn.Linear(self.obs_dim, self.d_model)
        self.cmd_embed = nn.Linear(self.cmd_dim, self.d_model)
        self.act_embed = nn.Linear(self.act_dim, self.d_model) if self.use_action_history else None
        self.history_pos_embedding, self.history_sinusoidal_positional_encoding = self._build_positional_encoding(
            d_model=self.d_model,
            seq_len=self.history_len,
            use_learned=bool(use_learned_positional_encoding),
        )
        self.embed_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.obs_head = nn.Linear(self.d_model, self.pred_horizon * self.obs_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        history_obs: torch.Tensor,
        current_command: torch.Tensor,
        history_action: torch.Tensor | None,
        history_valid_mask: torch.Tensor | None,
    ) -> None:
        if history_obs.ndim != 3 or current_command.ndim != 2:
            raise ValueError("history_obs must be rank-3 and current_command must be rank-2")
        if history_obs.shape[0] != current_command.shape[0]:
            raise ValueError("Batch dimension mismatch across planner inputs")
        if history_obs.shape[1] != self.history_len:
            raise ValueError("Planner history length mismatch")
        if history_obs.shape[2] != self.obs_dim:
            raise ValueError(f"history_obs feature mismatch: got {history_obs.shape[2]}, expected {self.obs_dim}")
        if current_command.shape[1] != self.cmd_dim:
            raise ValueError(
                f"current_command feature mismatch: got {current_command.shape[1]}, expected {self.cmd_dim}"
            )
        if self.use_action_history:
            if history_action is None:
                raise ValueError("planner requires history_action when use_action_history=True")
            if history_action.ndim != 3:
                raise ValueError("history_action must be rank-3")
            if history_action.shape[:2] != history_obs.shape[:2]:
                raise ValueError("history_action shape mismatch with history_obs")
            if history_action.shape[2] != self.act_dim:
                raise ValueError(
                    f"history_action feature mismatch: got {history_action.shape[2]}, expected {self.act_dim}"
                )
        if history_valid_mask is not None:
            if history_valid_mask.ndim != 2:
                raise ValueError("history_valid_mask must be rank-2 [B, H]")
            if history_valid_mask.shape != history_obs.shape[:2]:
                raise ValueError("history_valid_mask shape mismatch")

    def forward(
        self,
        history_obs: torch.Tensor,
        current_command: torch.Tensor,
        *,
        history_action: torch.Tensor | None = None,
        history_valid_mask: torch.Tensor | None = None,
        skip_empty_history_check: bool = False,
    ) -> torch.Tensor:
        self._validate_inputs(history_obs, current_command, history_action, history_valid_mask)

        current_command_embed = self.cmd_embed(current_command).unsqueeze(1).expand(-1, self.history_len, -1)
        tokens = self.obs_embed(history_obs) + current_command_embed
        if self.use_action_history:
            assert self.act_embed is not None and history_action is not None
            tokens = tokens + self.act_embed(history_action)
        pos = self._resolve_positional_encoding(
            pos_embedding=self.history_pos_embedding,
            sinusoidal=self.history_sinusoidal_positional_encoding,
            seq_len=self.history_len,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = self.embed_dropout(tokens + pos)

        src_key_padding_mask: torch.Tensor | None = None
        if history_valid_mask is not None:
            valid_mask = history_valid_mask.to(device=tokens.device)
            if valid_mask.dtype != torch.bool:
                valid_mask = valid_mask > 0
            src_key_padding_mask = ~valid_mask
            if not skip_empty_history_check and torch.any(torch.all(src_key_padding_mask, dim=1)):
                raise ValueError("planner history_valid_mask contains an empty history sample")

        features = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        context = features[:, -1, :]
        return self.obs_head(context).view(-1, self.pred_horizon, self.obs_dim)



class IDM(nn.Module, _PositionalEncodingMixin):
    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int | None = None,
        history_len: int,
        pred_horizon: int,
        d_model: int = 256,
        nhead: int = 8,
        encoder_num_layers: int = 6,
        decoder_num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_learned_positional_encoding: bool = False,
        use_command_history: bool = False,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or act_dim <= 0:
            raise ValueError("obs_dim and act_dim must be positive")
        if bool(use_command_history) and (cmd_dim is None or int(cmd_dim) <= 0):
            raise ValueError("cmd_dim must be positive when use_command_history=True")
        if history_len <= 0 or pred_horizon <= 0:
            raise ValueError("history_len and pred_horizon must be positive")
        if d_model <= 0 or nhead <= 0 or encoder_num_layers <= 0 or decoder_num_layers <= 0:
            raise ValueError("d_model, nhead, encoder_num_layers, decoder_num_layers must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cmd_dim = (int(cmd_dim) if cmd_dim is not None else 0)
        self.history_len = int(history_len)
        self.pred_horizon = int(pred_horizon)
        self.d_model = int(d_model)
        self.use_command_history = bool(use_command_history)

        self.history_obs_embed = nn.Linear(self.obs_dim, self.d_model)
        self.history_act_embed = nn.Linear(self.act_dim, self.d_model)
        self.history_cmd_embed = nn.Linear(self.cmd_dim, self.d_model) if self.use_command_history else None
        self.future_obs_embed = nn.Linear(self.obs_dim, self.d_model)
        (
            self.history_pos_embedding,
            self.history_sinusoidal_positional_encoding,
        ) = self._build_positional_encoding(
            d_model=self.d_model,
            seq_len=self.history_len,
            use_learned=bool(use_learned_positional_encoding),
        )
        (
            self.future_pos_embedding,
            self.future_sinusoidal_positional_encoding,
        ) = self._build_positional_encoding(
            d_model=self.d_model,
            seq_len=self.pred_horizon,
            use_learned=bool(use_learned_positional_encoding),
        )
        self.history_dropout = nn.Dropout(dropout)
        self.future_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_num_layers)
        self.action_head = nn.Linear(self.d_model, self.act_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor | None,
        future_obs: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
    ) -> None:
        if history_obs.ndim != 3 or history_act.ndim != 3 or future_obs.ndim != 3:
            raise ValueError("IDM inputs must be rank-3")
        if history_obs.shape[0] != history_act.shape[0] or history_obs.shape[0] != future_obs.shape[0]:
            raise ValueError("Batch dimension mismatch across IDM inputs")
        if history_obs.shape[1] != self.history_len or history_act.shape[1] != self.history_len:
            raise ValueError("IDM history length mismatch")
        if future_obs.shape[1] != self.pred_horizon:
            raise ValueError(
                f"future_obs horizon mismatch: got {future_obs.shape[1]}, expected {self.pred_horizon}"
            )
        if history_obs.shape[2] != self.obs_dim or future_obs.shape[2] != self.obs_dim:
            raise ValueError("IDM obs_dim mismatch")
        if history_act.shape[2] != self.act_dim:
            raise ValueError("IDM act_dim mismatch")
        if self.use_command_history:
            if current_command is None:
                raise ValueError("IDM requires current_command when use_command_history=True")
            if current_command.ndim != 2 or current_command.shape[0] != history_obs.shape[0]:
                raise ValueError("current_command shape mismatch for IDM")
            if current_command.shape[1] != self.cmd_dim:
                raise ValueError("IDM cmd_dim mismatch")
        if history_valid_mask is not None:
            if history_valid_mask.ndim != 2 or history_valid_mask.shape != history_obs.shape[:2]:
                raise ValueError("history_valid_mask shape mismatch for IDM")

    def forward(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor | None,
        future_obs: torch.Tensor,
        *,
        history_valid_mask: torch.Tensor | None = None,
        skip_empty_history_check: bool = False,
    ) -> torch.Tensor:
        self._validate_inputs(history_obs, history_act, current_command, future_obs, history_valid_mask)

        history_tokens = self.history_obs_embed(history_obs) + self.history_act_embed(history_act)
        if self.use_command_history:
            assert self.history_cmd_embed is not None and current_command is not None
            current_command_embed = self.history_cmd_embed(current_command).unsqueeze(1).expand(
                -1,
                self.history_len,
                -1,
            )
            history_tokens = history_tokens + current_command_embed
        history_pos = self._resolve_positional_encoding(
            pos_embedding=self.history_pos_embedding,
            sinusoidal=self.history_sinusoidal_positional_encoding,
            seq_len=self.history_len,
            device=history_tokens.device,
            dtype=history_tokens.dtype,
        )
        history_tokens = self.history_dropout(history_tokens + history_pos)

        future_tokens = self.future_obs_embed(future_obs)
        future_pos = self._resolve_positional_encoding(
            pos_embedding=self.future_pos_embedding,
            sinusoidal=self.future_sinusoidal_positional_encoding,
            seq_len=self.pred_horizon,
            device=future_tokens.device,
            dtype=future_tokens.dtype,
        )
        future_tokens = self.future_dropout(future_tokens + future_pos)

        memory_key_padding_mask: torch.Tensor | None = None
        if history_valid_mask is not None:
            valid_mask = history_valid_mask.to(device=history_tokens.device)
            if valid_mask.dtype != torch.bool:
                valid_mask = valid_mask > 0
            memory_key_padding_mask = ~valid_mask
            if not skip_empty_history_check and torch.any(torch.all(memory_key_padding_mask, dim=1)):
                raise ValueError("IDM history_valid_mask contains an empty history sample")

        memory = self.history_encoder(history_tokens, src_key_padding_mask=memory_key_padding_mask)
        decoded = self.decoder(
            tgt=future_tokens,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.action_head(decoded)



class PlannerIDMPolicy(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        history_len: int,
        pred_horizon: int,
        planner_d_model: int = 256,
        planner_nhead: int = 8,
        planner_num_layers: int = 6,
        planner_dim_feedforward: int = 1024,
        planner_dropout: float = 0.1,
        planner_use_learned_positional_encoding: bool = False,
        planner_use_action_history: bool = False,
        idm_d_model: int = 256,
        idm_nhead: int = 8,
        idm_encoder_num_layers: int = 6,
        idm_decoder_num_layers: int = 4,
        idm_dim_feedforward: int = 1024,
        idm_dropout: float = 0.1,
        idm_use_learned_positional_encoding: bool = False,
        idm_use_current_command_for_history: bool = False,
        planner_predict_delta: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cmd_dim = int(cmd_dim)
        self.history_len = int(history_len)
        self.pred_horizon = int(pred_horizon)
        self.predict_future_obs = True
        self.planner_use_action_history = bool(planner_use_action_history)
        self.idm_use_current_command_for_history = bool(idm_use_current_command_for_history)
        self.planner_predict_delta = bool(planner_predict_delta)

        self.planner = Planner(
            obs_dim=obs_dim,
            act_dim=act_dim,
            cmd_dim=cmd_dim,
            history_len=history_len,
            pred_horizon=pred_horizon,
            d_model=planner_d_model,
            nhead=planner_nhead,
            num_layers=planner_num_layers,
            dim_feedforward=planner_dim_feedforward,
            dropout=planner_dropout,
            use_learned_positional_encoding=planner_use_learned_positional_encoding,
            use_action_history=planner_use_action_history,
        )
        self.idm = IDM(
            obs_dim=obs_dim,
            act_dim=act_dim,
            cmd_dim=cmd_dim,
            history_len=history_len,
            pred_horizon=pred_horizon,
            d_model=idm_d_model,
            nhead=idm_nhead,
            encoder_num_layers=idm_encoder_num_layers,
            decoder_num_layers=idm_decoder_num_layers,
            dim_feedforward=idm_dim_feedforward,
            dropout=idm_dropout,
            use_learned_positional_encoding=idm_use_learned_positional_encoding,
            use_command_history=idm_use_current_command_for_history,
        )

    @staticmethod
    def _resolve_idm_current_command(
        current_command: torch.Tensor,
        *,
        idm_use_current_command_for_history: bool,
    ) -> torch.Tensor | None:
        if not bool(idm_use_current_command_for_history):
            return None
        return current_command

    def forward(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        *,
        history_valid_mask: torch.Tensor | None = None,
        return_obs: bool = False,
        skip_empty_history_check: bool = False,
        future_obs_override: torch.Tensor | None = None,
        detach_planner_future_obs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        planner_raw_output = self.planner(
            history_obs,
            current_command,
            history_action=(history_act if self.planner_use_action_history else None),
            history_valid_mask=history_valid_mask,
            skip_empty_history_check=skip_empty_history_check,
        )
        # Reconstruct absolute obs from delta if planner uses delta prediction
        if self.planner_predict_delta:
            ref_obs = history_obs[:, -1, :]  # (B, D)
            planner_future_obs = planner_raw_output + ref_obs.unsqueeze(1)
        else:
            planner_future_obs = planner_raw_output
        idm_future_obs = future_obs_override if future_obs_override is not None else planner_future_obs
        if future_obs_override is None and detach_planner_future_obs:
            idm_future_obs = idm_future_obs.detach()
        idm_current_command = self._resolve_idm_current_command(
            current_command,
            idm_use_current_command_for_history=self.idm_use_current_command_for_history,
        )
        pred_actions = self.predict_idm_actions(
            history_obs,
            history_act,
            idm_current_command,
            idm_future_obs,
            history_valid_mask=history_valid_mask,
            skip_empty_history_check=skip_empty_history_check,
        )
        if not return_obs:
            return pred_actions
        return pred_actions, planner_future_obs

    def predict_idm_actions(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor | None,
        future_obs: torch.Tensor,
        *,
        history_valid_mask: torch.Tensor | None = None,
        skip_empty_history_check: bool = False,
    ) -> torch.Tensor:
        assert self.idm is not None
        return self.idm(
            history_obs,
            history_act,
            current_command,
            future_obs,
            history_valid_mask=history_valid_mask,
            skip_empty_history_check=skip_empty_history_check,
        )

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=True)

    def load_split_state_dict(
        self,
        *,
        planner_state_dict: dict[str, torch.Tensor],
        idm_state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.planner.load_state_dict(planner_state_dict, strict=True)
        assert self.idm is not None
        self.idm.load_state_dict(idm_state_dict, strict=True)

    def planner_parameters(self):
        return self.planner.parameters()

    def idm_parameters(self):
        assert self.idm is not None
        return self.idm.parameters()

    def planner_state_dict(self) -> dict[str, torch.Tensor]:
        return self.planner.state_dict()

    def idm_state_dict(self) -> dict[str, torch.Tensor]:
        assert self.idm is not None
        return self.idm.state_dict()



def build_student_policy(
    cfg,
    *,
    obs_dim: int,
    act_dim: int,
    cmd_dim: int,
) -> nn.Module:
    """Construct the DAgger student policy (:class:`PlannerIDMPolicy`) from cfg fields."""
    return PlannerIDMPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        cmd_dim=cmd_dim,
        history_len=int(cfg.history_len),
        pred_horizon=int(cfg.pred_horizon),
        planner_d_model=int(cfg.planner_d_model),
        planner_nhead=int(cfg.planner_nhead),
        planner_num_layers=int(cfg.planner_num_layers),
        planner_dim_feedforward=int(cfg.planner_dim_feedforward),
        planner_dropout=float(cfg.planner_dropout),
        planner_use_learned_positional_encoding=bool(cfg.planner_use_learned_positional_encoding),
        planner_use_action_history=bool(cfg.planner_use_action_history),
        idm_d_model=int(cfg.idm_d_model),
        idm_nhead=int(cfg.idm_nhead),
        idm_encoder_num_layers=int(cfg.idm_encoder_num_layers),
        idm_decoder_num_layers=int(cfg.idm_decoder_num_layers),
        idm_dim_feedforward=int(cfg.idm_dim_feedforward),
        idm_dropout=float(cfg.idm_dropout),
        idm_use_learned_positional_encoding=bool(cfg.idm_use_learned_positional_encoding),
        idm_use_current_command_for_history=bool(cfg.idm_use_current_command_for_history),
        planner_predict_delta=bool(cfg.planner_predict_delta),
    )


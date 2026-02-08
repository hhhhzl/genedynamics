from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from enerdynamics.solvers.single.safediffuser.diffuser_utils.normalization import PlanningNormalizer
from enerdynamics.solvers.single.safediffuser.patch.diffusion import GaussianDiffusion
from enerdynamics.solvers.single.safediffuser.patch.temporal import TemporalUnet


@dataclass(frozen=True)
class LoadedSafeDiffuser:
    diffusion: Any
    ema: Any
    normalizer: PlanningNormalizer
    epoch: int


def _latest_epoch(checkpoint_dir: Path) -> int:
    candidates = glob.glob(str(checkpoint_dir / "state_*.pt"))
    best = -1
    for p in candidates:
        name = Path(p).name
        try:
            n = int(name.replace("state_", "").replace(".pt", ""))
        except Exception:
            continue
        best = max(best, n)
    if best < 0:
        raise FileNotFoundError(f"No state_*.pt found under {checkpoint_dir}")
    return best


def load_planning_checkpoint(
    checkpoint_dir: str | Path,
    *,
    epoch: int | str = "latest",
    device: str = "cuda:0",
) -> LoadedSafeDiffuser:
    """
    Load a *converted* SafeDiffuser checkpoint stored in a repo-independent format.

    Expected files:
      - safediffuser_planning.yaml   (architecture + normalizer mins/maxs)
      - state_<epoch>.pt             (dict with keys: model, ema OR just a state_dict)
    """
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    cfg_path = checkpoint_dir / "safediffuser_planning.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    horizon = int(cfg["horizon"])
    observation_dim = int(cfg["observation_dim"])
    action_dim = int(cfg["action_dim"])
    n_timesteps = int(cfg.get("n_timesteps", 256))
    dim = int(cfg.get("dim", 32))
    dim_mults = tuple(cfg.get("dim_mults", (1, 4, 8)))

    obs_mins_v = cfg.get("obs_mins")
    obs_maxs_v = cfg.get("obs_maxs")
    act_mins_v = cfg.get("act_mins")
    act_maxs_v = cfg.get("act_maxs")

    if obs_mins_v is None or obs_maxs_v is None or act_mins_v is None or act_maxs_v is None:
        raise ValueError(
            "Missing obs/act mins/maxs in `safediffuser_planning.yaml`. "
            "This runtime loader no longer reads `data_config.pkl` / `dataset_config.pkl`. "
            "Re-run `scripts/convert_safediffuser_checkpoint.py` to regenerate the yaml."
        )

    obs_mins = np.asarray(obs_mins_v, dtype=np.float32)
    obs_maxs = np.asarray(obs_maxs_v, dtype=np.float32)
    act_mins = np.asarray(act_mins_v, dtype=np.float32)
    act_maxs = np.asarray(act_maxs_v, dtype=np.float32)

    normalizer = PlanningNormalizer(obs_mins, obs_maxs, act_mins, act_maxs)

    model = TemporalUnet(
        horizon=horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim,
        dim=dim,
        dim_mults=dim_mults,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=n_timesteps,
        loss_type=str(cfg.get("loss_type", "l2")),
        clip_denoised=bool(cfg.get("clip_denoised", True)),
        predict_epsilon=bool(cfg.get("predict_epsilon", False)),
    ).to(device)

    # attach for optional _format_conditions compatibility
    diffusion.normalizer = normalizer  # type: ignore[attr-defined]

    epoch_i: int
    epoch_str = str(epoch).strip().lower()
    if epoch_str == "latest":
        epoch_i = _latest_epoch(checkpoint_dir)
        state_path = checkpoint_dir / f"state_{epoch_i}.pt"
    elif epoch_str == "best":
        epoch_i = -1
        state_path = checkpoint_dir / "state_best.pt"
    else:
        epoch_i = int(epoch)
        state_path = checkpoint_dir / f"state_{epoch_i}.pt"
    data = torch.load(state_path, map_location="cpu")

    # Accept either a dict with "model"/"ema" or a raw state_dict.
    if isinstance(data, dict) and ("model" in data or "ema" in data):
        model_sd = data.get("model")
        ema_sd = data.get("ema", model_sd)
    else:
        model_sd = data
        ema_sd = data

    diffusion.load_state_dict(model_sd)
    diffusion.eval()

    ema = GaussianDiffusion(
        model=model,
        horizon=horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=n_timesteps,
        loss_type=str(cfg.get("loss_type", "l2")),
        clip_denoised=bool(cfg.get("clip_denoised", True)),
        predict_epsilon=bool(cfg.get("predict_epsilon", False)),
    ).to(device)
    ema.normalizer = normalizer  # type: ignore[attr-defined]
    ema.load_state_dict(ema_sd)
    ema.eval()

    return LoadedSafeDiffuser(diffusion=diffusion, ema=ema, normalizer=normalizer, epoch=epoch_i)


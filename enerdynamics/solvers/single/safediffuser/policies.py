from __future__ import annotations

from collections import namedtuple
from typing import Any, Dict, Iterable, Optional, Tuple

import einops
import numpy as np
import torch


Trajectories = namedtuple("Trajectories", "actions observations")


def _to_np(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _apply_dict(fn, d: Dict[int, Any], *args, **kwargs) -> Dict[int, Any]:
    return {k: fn(v, *args, **kwargs) for k, v in d.items()}


class SafeDiffuserPolicy:
    """
    SafeDiffuser-specific Policy wrapper.

    Goals:
    - Keep SafeDiffuser integration self-contained (no trajectory_selection logic).
    - Put `cond` and `returns` on the *model device* (fixes CPU/CUDA mismatch risks).
    - Optionally return diffusion paths for debugging / visualization.
    """

    def __init__(
        self,
        *,
        model: Any,
        normalizer: Any,
        projector: Any | None = None,
        preprocess_fns: Optional[Iterable[Any]] = None,
        test_ret: float = 0.0,
    ):
        self.model = model
        self.normalizer = normalizer
        self.projector = projector
        self.test_ret = float(test_ret)

        preprocess_fns = list(preprocess_fns or [])
        try:
            from diffuser.datasets.preprocessing import get_policy_preprocess_fn  # type: ignore

            self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        except Exception:
            # If preprocessing utilities aren't available, default to identity.
            self.preprocess_fn = lambda x: x

        self.action_dim = int(getattr(model, "action_dim", getattr(normalizer, "action_dim", 0)))
        if self.action_dim <= 0:
            raise ValueError("SafeDiffuserPolicy requires a positive action_dim on model or normalizer.")

    @property
    def device(self) -> torch.device:
        params = list(getattr(self.model, "parameters", lambda: [])())
        return params[0].device if len(params) else torch.device("cpu")

    def _format_conditions(self, conditions: Dict[int, Any], batch_size: int) -> Dict[int, torch.Tensor]:
        # Preprocess each conditioned observation (usually identity for avoiding).
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        # Normalize to model training scale.
        conditions = _apply_dict(self.normalizer.normalize, conditions, "observations")
        # Convert to torch on model device.
        conditions = _apply_dict(
            lambda v: torch.as_tensor(v, dtype=torch.float32, device=self.device),
            conditions,
        )
        # Repeat for batch sampling: [obs_dim] -> [B, obs_dim]
        conditions = _apply_dict(einops.repeat, conditions, "d -> repeat d", repeat=batch_size)
        return conditions

    def __call__(
        self,
        conditions: Dict[int, Any],
        *,
        batch_size: int = 1,
        horizon: Optional[int] = None,
        test_ret: Optional[float] = None,
        constraints: Any = None,
        return_diffusion: bool = True,
    ) -> Tuple[np.ndarray, Any, Optional[np.ndarray]]:
        """
        Returns:
        - action: (action_dim,) numpy array for selected trajectory at t=0
        - trajectories: Trajectories(actions[B,H,A], observations[B,H,O]) *unnormalized* numpy arrays
        - diffusion_paths: Optional diffusion observations (B,T,H,O) numpy array if available
        """
        B = int(batch_size)
        if B <= 0:
            raise ValueError("batch_size must be positive")

        cond_t = self._format_conditions(conditions, B)

        tr = self.test_ret if test_ret is None else float(test_ret)
        returns = torch.full((B, 1), tr, dtype=torch.float32, device=self.device)

        # Run reverse diffusion.
        samples, infos = self.model(
            cond_t,
            returns=returns,
            projector=self.projector,
            constraints=constraints,
            horizon=horizon,
            return_diffusion=return_diffusion,
        )

        samples_np = np.asarray(_to_np(samples), dtype=np.float32)  # (B,H,transition)
        actions = samples_np[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(actions, "actions")

        obs = samples_np[:, :, self.action_dim :]
        obs = self.normalizer.unnormalize(obs, "observations")

        diffusion_paths = None
        if isinstance(infos, dict) and "diffusion" in infos:
            diff = np.asarray(_to_np(infos["diffusion"]), dtype=np.float32)  # (B,T,H,transition)
            diff_obs = diff[:, :, :, self.action_dim :]
            diffusion_paths = self.normalizer.unnormalize(diff_obs, "observations")

        trajectories = Trajectories(actions, obs)
        return actions, trajectories, diffusion_paths


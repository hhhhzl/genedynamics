from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from genedynamics.solvers.single.safediffuser.tools.utils.normalization_value_extractor import (
    load_limits_from_dataset_artifacts,
)


def _install_minimal_diffuser_shims() -> Dict[str, types.ModuleType]:
    """
    Install minimal in-memory module shims so that DPCC/SafeDiffuser pickled Config objects
    can be unpickled without the upstream `diffuser` package.
    """

    inserted: Dict[str, types.ModuleType] = {}

    def ensure(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        sys.modules[name] = m
        inserted[name] = m
        return m

    ensure("diffuser").__path__ = []  # type: ignore[attr-defined]
    ensure("diffuser.utils")
    mod_config = ensure("diffuser.utils.config")

    class Config:
        """Pickle-compatible Config shim with `_dict`."""

        def __init__(self, _class=None, **kwargs: Any):
            self._class = _class
            self._dict = dict(kwargs)

        def __call__(self, *args: Any, **kwargs: Any):
            if self._class is None:
                raise TypeError("Config has no _class")
            merged = dict(self._dict)
            merged.update(kwargs)
            return self._class(*args, **merged)

    mod_config.Config = Config  # type: ignore[attr-defined]

    # Common referenced globals by module path.
    mod_models = ensure("diffuser.models")
    mod_models.__path__ = []  # type: ignore[attr-defined]

    # diffusion_config.pkl often references diffuser.models.diffusion.GaussianDiffusion
    mod_diffusion = ensure("diffuser.models.diffusion")

    class GaussianDiffusion:
        """Dummy GaussianDiffusion class for unpickling only."""

        pass

    mod_diffusion.GaussianDiffusion = GaussianDiffusion  # type: ignore[attr-defined]

    # model_config.pkl sometimes references these
    ensure("diffuser.models.temporal").TemporalUnet = type("TemporalUnet", (), {})  # type: ignore[attr-defined]
    mod_unet = ensure("diffuser.models.unet1d_temporal_cond")
    Dummy = type("UNet1DTemporalCondModel", (), {})
    mod_unet.UNet1DTemporalCondModel = Dummy  # type: ignore[attr-defined]
    mod_unet.Unet1DTemporalCond = Dummy  # type: ignore[attr-defined]

    # trainer_config.pkl references diffuser.utils.training.Trainer (not needed here, but harmless)
    mod_training = ensure("diffuser.utils.training")
    mod_training.Trainer = type("Trainer", (), {})  # type: ignore[attr-defined]

    # dataset_config.pkl references diffuser.datasets.sequence.SequenceDataset
    ensure("diffuser.datasets")
    mod_sequence = ensure("diffuser.datasets.sequence")
    mod_sequence.SequenceDataset = type("SequenceDataset", (), {})  # type: ignore[attr-defined]

    return inserted


def _load_pickle(path: Path) -> Any:
    inserted = _install_minimal_diffuser_shims()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        for name in list(inserted.keys()):
            sys.modules.pop(name, None)


def _extract_cfg(config_obj: Any) -> Dict[str, Any]:
    # SafeDiffuser stores Config with a `_dict` field.
    d = getattr(config_obj, "_dict", None)
    if not isinstance(d, dict):
        raise TypeError(f"Unexpected config type {type(config_obj)}; missing _dict")
    return dict(d)


def _extract_model_kind(config_obj: Any) -> str:
    cls = getattr(config_obj, "_class", None)
    if cls is None:
        return "unknown"
    name = getattr(cls, "__name__", str(cls))
    mod = getattr(cls, "__module__", "")
    return f"{mod}.{name}".strip(".")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert SafeDiffuser logdir to genedynamics safediffuser checkpoint")
    ap.add_argument("--input-dir", type=str, required=True, help="SafeDiffuser logdir containing *_config.pkl and state_*.pt")
    ap.add_argument("--output-dir", type=str, required=True, help="Output directory for converted checkpoint")

    ap.add_argument(
        "--dataset-data-dir",
        type=str,
        required=True,
        help=(
            "Offline dataset directory used to compute mins/maxs. "
            "For avoiding-d3il this must contain `env_*.pkl` files. "
            "This is required because DPCC/SafeDiffuser `dataset_config.pkl` typically does NOT store mins/maxs "
            "for LimitsNormalizer."
        ),
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg_pkl = input_dir / "model_config.pkl"
    diffusion_cfg_pkl = input_dir / "diffusion_config.pkl"
    if not model_cfg_pkl.exists() or not diffusion_cfg_pkl.exists():
        raise FileNotFoundError(f"Missing model_config.pkl/diffusion_config.pkl in {input_dir}")

    model_cfg = _load_pickle(model_cfg_pkl)
    diffusion_cfg = _load_pickle(diffusion_cfg_pkl)

    model_kind = _extract_model_kind(model_cfg)
    model_dict = _extract_cfg(model_cfg)
    diffusion_dict = _extract_cfg(diffusion_cfg)

    # Pull core params. (We currently only support TemporalUnet + GaussianDiffusion.)
    horizon = int(diffusion_dict.get("horizon", model_dict.get("horizon")))
    if not horizon:
        raise ValueError("Could not infer horizon from configs")

    observation_dim = int(diffusion_dict.get("observation_dim"))
    action_dim = int(diffusion_dict.get("action_dim"))
    if not observation_dim or not action_dim:
        # best-effort fallback
        transition_dim = int(model_dict.get("transition_dim", 0))
        cond_dim = int(model_dict.get("cond_dim", 0))
        if cond_dim:
            observation_dim = cond_dim
        if transition_dim and observation_dim:
            action_dim = transition_dim - observation_dim
    if not observation_dim or not action_dim:
        raise ValueError("Could not infer observation_dim/action_dim from configs")

    n_timesteps = int(diffusion_dict.get("n_timesteps", diffusion_dict.get("n_diffusion_steps", 256)))
    dim = int(model_dict.get("dim", 32))
    dim_mults = tuple(model_dict.get("dim_mults", (1, 4, 8)))

    # Compute mins/maxs from the offline dataset using dataset_config.pkl + env_*.pkl files.
    dataset_config_pkl = input_dir / "dataset_config.pkl"
    if not dataset_config_pkl.exists():
        raise FileNotFoundError(
            f"Missing `dataset_config.pkl` under input-dir: {dataset_config_pkl}. "
            "This converter does not accept hard-coded/default mins/maxs."
        )

    dataset_data_dir = Path(args.dataset_data_dir).expanduser().resolve()
    limits = load_limits_from_dataset_artifacts(
        dataset_config_pkl=dataset_config_pkl,
        dataset_data_dir=dataset_data_dir,
    )
    obs_mins, obs_maxs, act_mins, act_maxs = (
        limits.obs_mins,
        limits.obs_maxs,
        limits.act_mins,
        limits.act_maxs,
    )

    if obs_mins.size != observation_dim or obs_maxs.size != observation_dim:
        raise ValueError(f"obs mins/maxs size must match observation_dim={observation_dim}")
    if act_mins.size != action_dim or act_maxs.size != action_dim:
        raise ValueError(f"act mins/maxs size must match action_dim={action_dim}")

    # Copy ALL weights into output-dir so runtime can select any epoch via plan_config.diffusion_epoch.
    src_states = sorted(input_dir.glob("state_*.pt"))

    if not src_states:
        raise FileNotFoundError(f"No state_*.pt found under {input_dir}")

    copied_epochs: list[int] = []
    for src_state in src_states:
        if not src_state.exists():
            raise FileNotFoundError(f"Missing {src_state}")
        dst_state = output_dir / src_state.name
        shutil.copy2(src_state, dst_state)
        try:
            copied_epochs.append(int(src_state.stem.replace("state_", "")))
        except Exception:
            pass

    # Optional: also copy a `state_best.pt` if present.
    src_best = input_dir / "state_best.pt"
    has_best = False
    if src_best.exists():
        shutil.copy2(src_best, output_dir / src_best.name)
        has_best = True

    # Write repo-independent planning yaml.
    out_cfg: Dict[str, Any] = {
        "source_model_kind": model_kind,
        "horizon": horizon,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "n_timesteps": n_timesteps,
        "dim": dim,
        "dim_mults": list(dim_mults),
        "loss_type": diffusion_dict.get("loss_type", "l2"),
        "clip_denoised": bool(diffusion_dict.get("clip_denoised", True)),
        "predict_epsilon": bool(diffusion_dict.get("predict_epsilon", False)),
        "obs_mins": obs_mins.tolist(),
        "obs_maxs": obs_maxs.tolist(),
        "act_mins": act_mins.tolist(),
        "act_maxs": act_maxs.tolist(),
    }
    # Convenience metadata for humans: integers plus a special marker for `state_best.pt`.
    available_epochs: list[Any] = sorted(set(int(x) for x in copied_epochs))
    if has_best:
        available_epochs.append("best")
    if available_epochs:
        out_cfg["available_epochs"] = available_epochs
    with open(output_dir / "safediffuser_planning.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    print(f"Wrote converted checkpoint to: {output_dir}")


if __name__ == "__main__":
    main()

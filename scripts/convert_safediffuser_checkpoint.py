from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from safediffuser_utils.normalization_value_extractor import load_limits_from_dataset_artifacts


def _add_safediffuser_to_syspath(safediffuser_root: Path) -> None:
    safediffuser_root = safediffuser_root.resolve()
    if str(safediffuser_root) not in sys.path:
        sys.path.insert(0, str(safediffuser_root))
    diffuser_subdir = safediffuser_root / "diffuser"
    if diffuser_subdir.exists() and str(diffuser_subdir) not in sys.path:
        sys.path.insert(0, str(diffuser_subdir))


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


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
    ap = argparse.ArgumentParser(description="Convert SafeDiffuser logdir to enerdynamics safediffuser checkpoint")
    ap.add_argument("--safediffuser-root", type=str, default=None, help="Path to external SafeDiffuser repo (for unpickling Config)")
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

    if args.safediffuser_root:
        _add_safediffuser_to_syspath(Path(args.safediffuser_root))

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


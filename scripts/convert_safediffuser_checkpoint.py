from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml


def _parse_list_floats(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return np.asarray([float(p) for p in parts], dtype=np.float32)


def _latest_epoch(input_dir: Path) -> int:
    best = -1
    for p in input_dir.glob("state_*.pt"):
        try:
            n = int(p.stem.replace("state_", ""))
        except Exception:
            continue
        best = max(best, n)
    if best < 0:
        raise FileNotFoundError(f"No state_*.pt found under {input_dir}")
    return best


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
    ap.add_argument("--epoch", type=str, default="latest", help="Epoch number or 'latest'")

    ap.add_argument("--obs-mins", type=str, default=None, help="Comma list, e.g. '0.2,-0.3,0.2,-0.3' (x_des,y_des,x,y)")
    ap.add_argument("--obs-maxs", type=str, default=None, help="Comma list, e.g. '0.8,0.4,0.8,0.4'")
    ap.add_argument("--act-mins", type=str, default=None, help="Comma list, e.g. '-0.05,-0.05'")
    ap.add_argument("--act-maxs", type=str, default=None, help="Comma list, e.g. '0.05,0.05'")

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

    # Defaults for d3il-avoiding (4D obs, 2D act).
    if args.obs_mins is None:
        obs_mins = np.asarray([0.2, -0.3, 0.2, -0.3], dtype=np.float32)
    else:
        obs_mins = _parse_list_floats(args.obs_mins)
    if args.obs_maxs is None:
        obs_maxs = np.asarray([0.8, 0.4, 0.8, 0.4], dtype=np.float32)
    else:
        obs_maxs = _parse_list_floats(args.obs_maxs)
    if args.act_mins is None:
        act_mins = np.asarray([-0.05, -0.05], dtype=np.float32)
    else:
        act_mins = _parse_list_floats(args.act_mins)
    if args.act_maxs is None:
        act_maxs = np.asarray([0.05, 0.05], dtype=np.float32)
    else:
        act_maxs = _parse_list_floats(args.act_maxs)

    if obs_mins.size != observation_dim or obs_maxs.size != observation_dim:
        raise ValueError(f"obs mins/maxs size must match observation_dim={observation_dim}")
    if act_mins.size != action_dim or act_maxs.size != action_dim:
        raise ValueError(f"act mins/maxs size must match action_dim={action_dim}")

    # Pick epoch and copy state file.
    if args.epoch == "latest":
        epoch_i = _latest_epoch(input_dir)
    else:
        epoch_i = int(args.epoch)
    src_state = input_dir / f"state_{epoch_i}.pt"
    if not src_state.exists():
        raise FileNotFoundError(f"Missing {src_state}")
    dst_state = output_dir / src_state.name
    shutil.copy2(src_state, dst_state)

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
    with open(output_dir / "safediffuser_planning.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    print(f"Wrote converted checkpoint to: {output_dir}")


if __name__ == "__main__":
    main()


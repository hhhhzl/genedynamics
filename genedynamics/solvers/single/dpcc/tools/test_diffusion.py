#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _ensure_paths(project_root: Path) -> None:
    third_party = project_root / "third_party"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyYAML is required. Please install with `pip install pyyaml`.") from exc

    if not path.exists():
        raise FileNotFoundError(f"DPCC config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_env(env_name: str, render: bool):
    if env_name == "d3il_avoiding_9d":
        from genedynamics.experiments.plugins.environments.d3il_avoiding_9d import D3ILAvoiding9DPlugin

        return D3ILAvoiding9DPlugin().create_env({"render": render})

    if env_name == "d3il_avoiding":
        from genedynamics.experiments.plugins.environments.d3il_avoiding import D3ILAvoidingPlugin

        return D3ILAvoidingPlugin().create_env({"render": render})

    raise ValueError(f"Unsupported env_name={env_name}. Use d3il_avoiding or d3il_avoiding_9d.")


def _default_indices(env_name: str) -> Dict[str, Dict[str, int]]:
    if env_name == "d3il_avoiding_9d":
        return {
            "observations": {"x": 0, "y": 1},
            "actions": {
                "qdot1": 0,
                "qdot2": 1,
                "qdot3": 2,
                "qdot4": 3,
                "qdot5": 4,
                "qdot6": 5,
                "qdot7": 6,
            },
        }

    # d3il_avoiding (4D state + 2D action)
    return {
        "observations": {"x_des": 0, "y_des": 1, "x": 2, "y": 3},
        "actions": {"vx": 0, "vy": 1},
    }


def _build_indices_from_cfg(dpcc_cfg: Dict[str, Any], env_name: str) -> Dict[str, Dict[str, int]]:
    robot_name = "avoiding"
    obs = (dpcc_cfg.get("observation_indices", {}) or {}).get(robot_name)
    act = (dpcc_cfg.get("action_indices", {}) or {}).get(robot_name)

    if env_name == "d3il_avoiding_9d":
        # Force 9D semantics to avoid mixing with 4D indices in legacy yaml.
        return _default_indices(env_name)

    if isinstance(obs, dict) and isinstance(act, dict):
        return {"observations": obs, "actions": act}
    return _default_indices(env_name)


def _summary(info: Dict[str, Any]) -> Dict[str, float]:
    n_success = np.asarray(info.get("n_success", []), dtype=np.float32)
    n_success_safe = np.asarray(info.get("n_success_and_constraints", []), dtype=np.float32)
    n_steps = np.asarray(info.get("n_steps", []), dtype=np.float32)
    n_viol = np.asarray(info.get("n_violations", []), dtype=np.float32)
    avg_time = np.asarray(info.get("avg_time_all", []), dtype=np.float32)

    def _mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size > 0 else float("nan")

    return {
        "success_rate": _mean(n_success),
        "safe_success_rate": _mean(n_success_safe),
        "avg_steps": _mean(n_steps),
        "avg_violations": _mean(n_viol),
        "avg_plan_time_s": _mean(avg_time),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained DPCC diffuser checkpoint in D3IL env.")
    parser.add_argument("--env-name", type=str, default="d3il_avoiding_9d", choices=["d3il_avoiding", "d3il_avoiding_9d"])
    parser.add_argument("--dataset", type=str, default="avoiding-d3il-9d")
    parser.add_argument("--exp", type=str, default="avoiding-d3il-9d")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--loadbase", type=str, default="logs")
    parser.add_argument("--diffusion-loadpath", type=str, required=True,
                        help="Example: diffusion/H8_K20_Dmodels.GaussianDiffusion")
    parser.add_argument("--epoch", type=str, default="latest", help="latest | best | <int>")

    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-episode-length", type=int, default=200)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--variant", type=str, default="diffuser")
    parser.add_argument("--solver", type=str, default="scipy")
    parser.add_argument("--disable-projection", action="store_true")

    parser.add_argument(
        "--dpcc-config-path",
        type=str,
        default="genedynamics/solvers/single/dpcc/config/projection_eval.yaml",
    )
    parser.add_argument("--save-json", type=str, default=None)

    args = parser.parse_args()

    project_root = _resolve_project_root()
    _ensure_paths(project_root)

    dpcc_cfg = _load_yaml((project_root / args.dpcc_config_path).resolve())
    env = _build_env(args.env_name, render=args.render)

    try:
        import diffuser.utils as dpcc_utils
        from genedynamics.solvers.single.dpcc.backends.dpcc_plan_torch import DPCCBackendTorch

        epoch: Any = args.epoch
        if args.epoch.isdigit():
            epoch = int(args.epoch)

        diffusion_exp = dpcc_utils.load_diffusion(
            args.loadbase,
            args.dataset,
            args.diffusion_loadpath,
            str(args.seed),
            epoch=epoch,
            device=args.device,
        )

        indices = _build_indices_from_cfg(dpcc_cfg, args.env_name)

        plan_config = {
            "exp": args.exp,
            "horizon": args.horizon,
            "batch_size": args.batch_size,
            "max_episode_length": args.max_episode_length,
            "n_trials": args.n_trials,
            "variant": args.variant,
            "solver": args.solver,
            "disable_projection": args.disable_projection,
        }

        backend = DPCCBackendTorch(
            env=env,
            diffusion=diffusion_exp.diffusion,
            normalizer=diffusion_exp.dataset.normalizer,
            plan_config=plan_config,
            constraint_config=dpcc_cfg,
            indices=indices,
            device=args.device,
            seed=args.seed,
        )

        result = backend.plan()
        info = result.get("info", {})
        metrics = _summary(info)

        print("[dpcc_test_diffusion] done")
        print(json.dumps({
            "env_name": args.env_name,
            "dataset": args.dataset,
            "diffusion_loadpath": args.diffusion_loadpath,
            "epoch": str(epoch),
            "metrics": metrics,
        }, indent=2))

        if args.save_json:
            out_path = Path(args.save_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "args": vars(args),
                    "metrics": metrics,
                    "info": {
                        "n_success": np.asarray(info.get("n_success", [])).tolist(),
                        "n_success_and_constraints": np.asarray(info.get("n_success_and_constraints", [])).tolist(),
                        "n_steps": np.asarray(info.get("n_steps", [])).tolist(),
                        "n_violations": np.asarray(info.get("n_violations", [])).tolist(),
                        "total_violations": np.asarray(info.get("total_violations", [])).tolist(),
                        "avg_time_all": np.asarray(info.get("avg_time_all", [])).tolist(),
                    },
                }, f, indent=2)
            print(f"[dpcc_test_diffusion] saved: {out_path}")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

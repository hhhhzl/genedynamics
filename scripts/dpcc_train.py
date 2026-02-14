#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _load_config_module(config_file: Path):
    spec = importlib.util.spec_from_file_location("dpcc_train_cfg", str(config_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config file: {config_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _eval_fstring(template: str, args_ns: SimpleNamespace) -> str:
    val = template.replace("{", "{args.").replace("f:", "")
    return eval(f"f'{val}'", {}, {"args": args_ns})


def _prepare_args(params: dict, dataset: str, seed: int, device_override: str | None) -> SimpleNamespace:
    args = SimpleNamespace()
    args.dataset = dataset

    for k, v in params.items():
        setattr(args, k, copy.deepcopy(v))

    args.seed = seed
    if device_override is not None:
        args.device = device_override

    for key, old in list(vars(args).items()):
        if isinstance(old, str) and old.startswith("f:"):
            setattr(args, key, _eval_fstring(old, args))

    if callable(getattr(args, "exp_name", None)):
        args.exp_name = args.exp_name(args)

    args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name, str(args.seed))
    os.makedirs(args.savepath, exist_ok=True)

    with open(os.path.join(args.savepath, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, skipkeys=True)

    return args


def _attach_stepper(diffusion):
    from enerdynamics.solvers.single.dpcc.stepper import DPCCTorchStepper

    stepper = DPCCTorchStepper(diffusion)
    diffusion.p_mean_variance = stepper.p_mean_variance
    diffusion.p_sample = stepper.p_sample
    diffusion.p_sample_loop = stepper.p_sample_loop
    diffusion.grad_p_sample = stepper.grad_p_sample
    diffusion.grad_p_sample_loop = stepper.grad_p_sample_loop
    diffusion.grad_conditional_sample = stepper.grad_conditional_sample
    return stepper


def train_one_seed(args):
    import diffuser.utils as utils

    _set_seed(args.seed)

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        env=args.dataset,
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        include_returns=args.include_returns,
        # Keep DPCC behavior for avoiding: rewards are <= 1 per step.
        returns_scale=args.max_path_length,
        discount=args.discount,
    )

    dataset = dataset_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, "model_config.pkl"),
        horizon=args.horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim,
        dim_mults=args.dim_mults,
        returns_condition=args.returns_condition,
        dim=args.dim,
        condition_dropout=args.condition_dropout,
        device=args.device,
    )

    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, "diffusion_config.pkl"),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        goal_dim=dataset.goal_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_discount=args.loss_discount,
        returns_condition=args.returns_condition,
        condition_guidance_w=args.condition_guidance_w,
        device=args.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, "trainer_config.pkl"),
        train_test_split=args.train_test_split,
        ema_decay=args.ema_decay,
        n_train_steps=args.n_train_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        results_folder=args.savepath,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    _attach_stepper(diffusion)
    trainer = trainer_config(diffusion, dataset)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train DPCC diffusion from enerdynamics/scripts.")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to DPCC training config python file (default: ../dpcc/config/avoiding-d3il.py).",
    )
    parser.add_argument("--dataset", type=str, default="avoiding-d3il")
    parser.add_argument("--seeds", type=str, default="5,6,7,8,9")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    third_party = project_root / "third_party"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))

    config_file = Path(args.config_file) if args.config_file else (project_root.parent / "dpcc" / "config" / "avoiding-d3il.py")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    cfg_module = _load_config_module(config_file)
    if not hasattr(cfg_module, "base") or "diffusion" not in cfg_module.base:
        raise ValueError(f"Invalid training config format: {config_file}")

    diffusion_params = copy.deepcopy(cfg_module.base["diffusion"])
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        seed_args = _prepare_args(diffusion_params, args.dataset, seed, args.device)
        print(f"[dpcc_train] seed={seed} savepath={seed_args.savepath}")
        train_one_seed(seed_args)


if __name__ == "__main__":
    main()

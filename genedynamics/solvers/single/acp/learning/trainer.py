"""Native ACP objective/AdamW/EMA with complete deterministic continuation state."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from diffusers.optimization import get_scheduler

from genedynamics.solvers.single.acp.learning.models.ema_model import EMAModel


def deterministic_runtime():
    """Use deterministic FP32 CUDA kernels for reproducible source/resume audits."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return {"deterministic_algorithms": True, "attention_backend": "math",
            "tf32": False, "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def _rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count differs from visible devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _gradient_state(model):
    """Include native optimizer-omitted gradients, which accumulate upstream."""
    return {name:parameter.grad for name,parameter in model.named_parameters()}


def native_optimizer(model, cfg, *, mode="upstream"):
    """Preserve the author's exact groups, and expose the omitted trainable parameters."""
    options = OmegaConf.to_container(cfg.optimizer, resolve=True)
    options.pop("_target_", None)
    if mode == "all_trainable":
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **options), []
    if mode != "upstream":
        raise ValueError("optimizer mode must be upstream or explicitly adapted all_trainable")
    rgb = [parameter for key in model.obs_encoder.rgb_keys
           for parameter in model.obs_encoder.key_model_map[key].parameters() if parameter.requires_grad]
    diffusion = list(model.model_sparse.parameters())
    lr = options["lr"] * (0.1 if cfg.policy.obs_encoder.reduce_pretrained_lr else 1)
    optimizer = torch.optim.AdamW([{"params": diffusion}, {"params": rgb, "lr": lr}], **options)
    included = {id(parameter) for parameter in [*rgb, *diffusion]}
    omitted = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and id(parameter) not in included]
    return optimizer, omitted


class ACPTrainer:
    def __init__(self, model, cfg, *, device="cuda:0", total_steps, optimizer_mode="upstream"):
        if int(cfg.training.gradient_accumulate_every) != 1:
            raise ValueError("native qualification currently requires author's released accumulation=1 profile")
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        # Native policy caches inference features and its most recent loss. The
        # author's workspace constructs EMA before those graphs exist; discard
        # diagnostics when taking an already evaluated policy into training.
        self.model.sparse_nobs_encode = None
        self.model.sparse_naction_pred = None
        self.model.sparse_loss = 0
        self.ema_model = copy.deepcopy(model).to(self.device) if cfg.training.use_ema else None
        self.optimizer, self.omitted_trainable_parameters = native_optimizer(model, cfg, mode=optimizer_mode)
        self.optimizer_mode = optimizer_mode
        self.total_steps = int(total_steps)
        self.scheduler = get_scheduler(str(cfg.training.lr_scheduler), self.optimizer,
                                       num_warmup_steps=int(cfg.training.lr_warmup_steps),
                                       num_training_steps=self.total_steps)
        if self.ema_model is not None:
            options = OmegaConf.to_container(cfg.ema, resolve=True)
            options.pop("_target_", None)
            self.ema = EMAModel(model=self.ema_model, **options)
        else:
            self.ema = None
        self.global_step = 0
        self.epoch = 0
        self.data_state = {}

    def train_step(self, batch):
        self.model.train()
        loss = self.model.compute_loss(batch, None)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.step(self.model)
        self.global_step += 1
        return float(loss.detach().cpu())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "genedynamics.acp.native-training.v2",
            "cfg": OmegaConf.to_container(self.cfg, resolve=False),
            "state_dicts": {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict()},
            "global_step": self.global_step, "epoch": self.epoch,
            "rng": _rng_state(), "data_state": self.data_state,
            "optimizer_mode": self.optimizer_mode, "total_steps": self.total_steps,
            "model_gradients": _gradient_state(self.model),
        }
        if self.ema is not None:
            payload["state_dicts"]["ema_model"] = self.ema_model.state_dict()
            payload["ema_state"] = {name: getattr(self.ema, name) for name in (
                "decay", "optimization_step", "update_after_step", "inv_gamma", "power", "min_value", "max_value")}
        temporary = path.with_name(path.name+".partial")
        torch.save(payload, temporary)
        temporary.replace(path)

    def restore(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format"] not in {"genedynamics.acp.native-training.v1","genedynamics.acp.native-training.v2"}:
            raise ValueError("not a complete native ACP training continuation checkpoint")
        if "model_gradients" not in payload and self.omitted_trainable_parameters:
            raise ValueError("legacy checkpoint lacks accumulated gradients omitted by the native optimizer")
        if payload["optimizer_mode"] != self.optimizer_mode or payload["total_steps"] != self.total_steps:
            raise ValueError("resume optimizer mode or complete LR schedule differs")
        self.model.load_state_dict(payload["state_dicts"]["model"], strict=True)
        # Native DictOfTensorMixin replaces its normalizer ParameterDict while
        # loading. Move the recreated statistics back to the requested device.
        self.model.to(self.device)
        gradients=payload.get("model_gradients",{})
        if gradients and set(gradients)!=set(dict(self.model.named_parameters())):
            raise ValueError("checkpoint gradient parameter schema differs")
        for name,parameter in self.model.named_parameters():
            gradient=gradients.get(name)
            parameter.grad=None if gradient is None else gradient.to(device=parameter.device,dtype=parameter.dtype).clone()
        self.optimizer.load_state_dict(payload["state_dicts"]["optimizer"])
        self.scheduler.load_state_dict(payload["state_dicts"]["scheduler"])
        if self.ema is not None:
            self.ema_model.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
            self.ema_model.to(self.device)
            for name, value in payload["ema_state"].items():
                setattr(self.ema, name, value)
        self.global_step, self.epoch = payload["global_step"], payload["epoch"]
        self.data_state = payload["data_state"]
        _restore_rng(payload["rng"])
        return payload

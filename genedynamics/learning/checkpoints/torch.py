"""Portable PyTorch training-state and random-state helpers."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return copy.deepcopy(value)


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((
        value["algorithm"],
        value["keys"].cpu().numpy().astype(np.uint32),
        value["position"],
        value["has_gauss"],
        value["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG resume requires the same visible GPU count")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


__all__ = ["capture_rng_state", "restore_rng_state", "to_cpu"]


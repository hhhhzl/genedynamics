# SPDX-License-Identifier: Apache-2.0
# Derived from LeCAR-Lab/FADA commit 290b3a9c5c0ebe2e9851b61350617544be05e9bc.
# Source: src/holosoma/holosoma/fada/common/norm_stats.py
# Changes: isolate native PyTorch core; replace Holosoma imports with local imports.
from __future__ import annotations

import torch

DEFAULT_NORM_EPS = 1e-6

def validate_norm_stats(
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float,
    *,
    key: str,
) -> None:
    """Raise `RuntimeError` if these checkpoint normalization stats are unusable.

    `key` is the checkpoint field the stats came from (e.g. `obs_norm_stats`) and is
    named in the message.
    """
    for name, tensor in (("mean", mean), ("std", std)):
        if not bool(torch.isfinite(tensor).all()):
            bad = int((~torch.isfinite(tensor)).sum())
            first = int(torch.nonzero(~torch.isfinite(tensor)).flatten()[0])
            raise RuntimeError(
                f"Checkpoint {key}.{name} contains {bad} non-finite value(s) "
                f"(first at index {first}, value {tensor.flatten()[first].item()!r}). "
                "Normalizing with it makes every downstream observation, metric and exported "
                "weight NaN without raising. Re-export the checkpoint from a run whose "
                "normalization statistics are finite."
            )
    if bool((std < 0).any()):
        first = int(torch.nonzero(std < 0).flatten()[0])
        raise RuntimeError(
            f"Checkpoint {key}.std has a negative entry at index {first} "
            f"({std.flatten()[first].item()!r}). No variance estimator produces one; it would be "
            "clamped up to eps and silently rescale that dimension. (std == 0 is fine and is "
            "clamped to eps as before.)"
        )
    if not (float(eps) > 0.0) or float(eps) != float(eps) or float(eps) == float("inf"):
        raise RuntimeError(
            f"Checkpoint {key}.eps must be a finite value > 0, got {eps!r}. It is the floor "
            "applied to std, so a non-positive or non-finite eps either divides by zero or "
            "propagates NaN."
        )


"""Persist / load the learned controller (policy beta + embedding E_chi + critic
V_psi) so a warmup-trained policy can be saved and reused by the co-design run /
diagnostics instead of re-training from scratch."""
from __future__ import annotations

import numpy as np


def save_controller(path: str, policy, E_proj, critic) -> None:
    d = {f"pp_{k}": np.asarray(v) for k, v in policy.items()}
    d["E_proj"] = np.asarray(E_proj)
    if critic:
        d.update({f"cr_{k}": np.asarray(v) for k, v in critic.items()})
    np.savez(path, **d)


def load_controller(path):
    import jax.numpy as jnp
    z = np.load(path)
    policy = {k[3:]: jnp.asarray(z[k]) for k in z.files if k.startswith("pp_")}
    E_proj = jnp.asarray(z["E_proj"])
    critic = {k[3:]: jnp.asarray(z[k]) for k in z.files if k.startswith("cr_")}
    return policy, E_proj, critic

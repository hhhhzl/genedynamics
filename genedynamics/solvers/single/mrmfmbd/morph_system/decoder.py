"""A2 morphology latent decoder + VAE (pure JAX — no optax/flax dependency).

`decode(params, w, cfg)` maps a shape latent w → per-voxel occupancy in
[x_lo, x_hi] (the x_morph the rollout consumes). It is a small MLP, kept
pure-JAX so it can be jitted INSIDE the MBD fast-path scan when A2 is wired in.

Training (`train_vae`) is offline on a static dataset of occupancy vectors
(no simulator). Adam is hand-rolled because the target conda env has no optax.
The VAE's N(0, I) latent prior is what makes the joint MF/MB prior term
log p(w) = -½‖w‖² a clean standard-normal (see theta_prior).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from .specs import MorphDecoderConfig

Params = Dict[str, jnp.ndarray]


# ---------------------------------------------------------------------------
# MLP params + forward
# ---------------------------------------------------------------------------


def init_params(key, cfg: MorphDecoderConfig) -> Params:
    """Glorot-uniform init for the encoder (n→h→{μ,logσ²}) + decoder (d→h→n)."""
    d, h, n = int(cfg.latent_dim), int(cfg.hidden_dim), int(cfg.n_voxels)
    ks = jax.random.split(key, 5)

    def glorot(k, shape):
        lim = jnp.sqrt(6.0 / float(sum(shape)))
        return jax.random.uniform(k, shape, minval=-lim, maxval=lim)

    return {
        "enc_w1": glorot(ks[0], (n, h)), "enc_b1": jnp.zeros((h,)),
        "enc_mu_w": glorot(ks[1], (h, d)), "enc_mu_b": jnp.zeros((d,)),
        "enc_lv_w": glorot(ks[2], (h, d)), "enc_lv_b": jnp.zeros((d,)),
        "dec_w1": glorot(ks[3], (d, h)), "dec_b1": jnp.zeros((h,)),
        "dec_out_w": glorot(ks[4], (h, n)), "dec_out_b": jnp.zeros((n,)),
    }


def encode(params: Params, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """x: (..., n_voxels) occupancy in [0,1] → (mu, logvar), each (..., d)."""
    hdn = jnp.tanh(x @ params["enc_w1"] + params["enc_b1"])
    mu = hdn @ params["enc_mu_w"] + params["enc_mu_b"]
    logvar = hdn @ params["enc_lv_w"] + params["enc_lv_b"]
    return mu, logvar


def _decode_logits(params: Params, w: jnp.ndarray) -> jnp.ndarray:
    hdn = jnp.tanh(w @ params["dec_w1"] + params["dec_b1"])
    return hdn @ params["dec_out_w"] + params["dec_out_b"]


def decode_occ01(params: Params, w: jnp.ndarray) -> jnp.ndarray:
    """Latent → occupancy in [0,1] (the trained reconstruction target range)."""
    return jax.nn.sigmoid(_decode_logits(params, w))


def decode(params: Params, w: jnp.ndarray, cfg: MorphDecoderConfig) -> jnp.ndarray:
    """Latent → physical per-voxel occupancy in [x_lo, x_hi] (the rollout x_morph)."""
    occ01 = decode_occ01(params, w)
    return cfg.x_lo + (cfg.x_hi - cfg.x_lo) * occ01


# ---------------------------------------------------------------------------
# VAE loss + hand-rolled Adam training
# ---------------------------------------------------------------------------


def vae_loss(params: Params, x: jnp.ndarray, key, cfg: MorphDecoderConfig):
    """β-VAE loss: MSE recon in [0,1] + β·KL(N(μ,σ²)‖N(0,I)). Returns (loss, aux)."""
    mu, logvar = encode(params, x)
    eps = jax.random.normal(key, mu.shape)
    z = mu + jnp.exp(0.5 * logvar) * eps
    recon = decode_occ01(params, z)
    recon_loss = jnp.mean((recon - x) ** 2)
    kl = -0.5 * jnp.mean(jnp.sum(1.0 + logvar - mu ** 2 - jnp.exp(logvar), axis=-1))
    return recon_loss + cfg.beta_kl * kl, (recon_loss, kl)


def _adam_init(params: Params):
    z = {k: jnp.zeros_like(v) for k, v in params.items()}
    return z, {k: jnp.zeros_like(v) for k, v in params.items()}


def _adam_step(params, grads, m, v, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m = {k: b1 * m[k] + (1 - b1) * grads[k] for k in params}
    v = {k: b2 * v[k] + (1 - b2) * grads[k] ** 2 for k in params}
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    params = {
        k: params[k] - lr * (m[k] / bc1) / (jnp.sqrt(v[k] / bc2) + eps)
        for k in params
    }
    return params, m, v


def train_vae(
    dataset,
    cfg: MorphDecoderConfig,
    *,
    steps: int = 2000,
    lr: float = 1e-2,
    seed: int = 0,
) -> Tuple[Params, List[Tuple[int, float, float, float]]]:
    """Full-batch β-VAE training on a (M, n_voxels) occupancy dataset in [0,1].

    Returns (params, history) where history is a list of
    (step, total_loss, recon_loss, kl) sampled ~10x over the run.
    """
    X = jnp.asarray(np.asarray(dataset, dtype=np.float32))
    key = jax.random.PRNGKey(int(seed))
    key, ik = jax.random.split(key)
    params = init_params(ik, cfg)
    m, v = _adam_init(params)

    loss_and_grad = jax.jit(
        jax.value_and_grad(lambda p, k: vae_loss(p, X, k, cfg), has_aux=True)
    )

    history: List[Tuple[int, float, float, float]] = []
    every = max(1, int(steps) // 10)
    for t in range(1, int(steps) + 1):
        key, sk = jax.random.split(key)
        (loss, (rl, kl)), grads = loss_and_grad(params, sk)
        params, m, v = _adam_step(params, grads, m, v, t, lr=lr)
        if t == 1 or t % every == 0:
            history.append((t, float(loss), float(rl), float(kl)))
    return params, history


# ---------------------------------------------------------------------------
# Serialization + runtime wrapper
# ---------------------------------------------------------------------------


def save_params(path: str, params: Params) -> None:
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def load_params(path: str) -> Params:
    d = np.load(path)
    return {k: jnp.asarray(d[k]) for k in d.files}


class MorphDecoder:
    """Trained decoder: decode(w) → physical occupancy (jitted, hot-loop ready)."""

    def __init__(self, params: Params, cfg: MorphDecoderConfig):
        self.params = params
        self.cfg = cfg
        self._decode = jax.jit(lambda w: decode(self.params, w, cfg))

    def decode(self, w) -> jnp.ndarray:
        """w: (latent_dim,) → occupancy (n_voxels,) in [x_lo, x_hi]."""
        return self._decode(jnp.asarray(w, dtype=jnp.float32))

    def decode_batch(self, W) -> jnp.ndarray:
        """W: (B, latent_dim) → (B, n_voxels)."""
        return jax.vmap(self._decode)(jnp.asarray(W, dtype=jnp.float32))

    def reconstruct(self, x) -> jnp.ndarray:
        """Encode-decode round trip in [0,1] (for recon diagnostics)."""
        mu, _ = encode(self.params, jnp.asarray(x, dtype=jnp.float32))
        return decode_occ01(self.params, mu)

    @classmethod
    def load(cls, params_path: str, cfg: MorphDecoderConfig) -> "MorphDecoder":
        return cls(load_params(params_path), cfg)

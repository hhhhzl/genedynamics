"""Stage 2 (A2) — morphology latent decoder + VAE unit tests (CPU, no sim).

Verifies the offline, simulator-free A2 building block:
  1. decode() output shape (n_voxels,) and range [x_lo, x_hi]; jittable + vmap.
  2. determinism (same w → same occupancy).
  3. train_vae reduces loss and yields a usable reconstruction on a small
     structured dataset (proves the AE pipeline end to end, no bank needed).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from genedynamics.solvers.single.mrmfmbd.morph_system import (  # noqa: E402
    MorphDecoderConfig,
    MorphDecoder,
    init_params,
    decode,
    train_vae,
)


def _toy_dataset(M=24, n=27, seed=0):
    """Structured occupancy: a few latent 'body types' + noise, in [0,1]."""
    rng = np.random.default_rng(seed)
    protos = rng.random((3, n)).astype(np.float32)
    idx = rng.integers(0, 3, size=M)
    X = protos[idx] + 0.05 * rng.standard_normal((M, n)).astype(np.float32)
    return np.clip(X, 0.0, 1.0)


def test_decode_shape_and_range():
    cfg = MorphDecoderConfig(latent_dim=8, hidden_dim=16, n_voxels=27)
    params = init_params(jax.random.PRNGKey(0), cfg)
    w = jax.random.normal(jax.random.PRNGKey(1), (cfg.latent_dim,))
    occ = decode(params, w, cfg)
    assert occ.shape == (cfg.n_voxels,)
    occ_np = np.asarray(occ)
    assert occ_np.min() >= cfg.x_lo - 1e-6
    assert occ_np.max() <= cfg.x_hi + 1e-6


def test_decode_jit_and_vmap():
    cfg = MorphDecoderConfig(latent_dim=8, hidden_dim=16, n_voxels=27)
    params = init_params(jax.random.PRNGKey(2), cfg)
    dec = MorphDecoder(params, cfg)
    W = jax.random.normal(jax.random.PRNGKey(3), (5, cfg.latent_dim))
    batch = dec.decode_batch(W)
    assert batch.shape == (5, cfg.n_voxels)
    # single decode is jitted; calling twice is deterministic.
    o1 = np.asarray(dec.decode(W[0]))
    o2 = np.asarray(dec.decode(W[0]))
    np.testing.assert_array_equal(o1, o2)


def test_vae_training_reduces_loss_and_reconstructs():
    cfg = MorphDecoderConfig(latent_dim=6, hidden_dim=32, n_voxels=27, beta_kl=1e-3)
    X = _toy_dataset(M=24, n=cfg.n_voxels, seed=1)
    params, history = train_vae(X, cfg, steps=400, lr=1e-2, seed=0)

    first_loss = history[0][1]
    last_loss = history[-1][1]
    assert last_loss < first_loss, f"loss did not drop: {first_loss} -> {last_loss}"

    # Deterministic (mu-based) reconstruction should be close in [0,1].
    dec = MorphDecoder(params, cfg)
    recon = np.asarray(dec.reconstruct(X))
    recon_mse = float(np.mean((recon - X) ** 2))
    # Baseline: predicting the dataset mean has MSE = per-voxel variance.
    mean_mse = float(np.mean((X - X.mean(axis=0, keepdims=True)) ** 2))
    assert recon_mse < mean_mse, f"recon {recon_mse} not better than mean {mean_mse}"
    assert recon_mse < 0.05, f"recon MSE too high: {recon_mse}"


def test_latent_prior_is_standard_normalish():
    """Encoded posterior means should be O(1) (β-VAE pulls them toward N(0,I)),
    so the joint-MBD prior term log p(w) = -½‖w‖² is well-scaled."""
    from genedynamics.solvers.single.mrmfmbd.morph_system import encode

    cfg = MorphDecoderConfig(latent_dim=6, hidden_dim=32, n_voxels=27, beta_kl=1e-2)
    X = _toy_dataset(M=24, n=cfg.n_voxels, seed=2)
    params, _ = train_vae(X, cfg, steps=400, lr=1e-2, seed=0)
    mu, _ = encode(params, jnp.asarray(X))
    assert float(jnp.max(jnp.abs(mu))) < 8.0  # not blowing up

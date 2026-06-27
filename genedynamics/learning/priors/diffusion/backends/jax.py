"""JAX learned diffusion prior `s_theta` (DDPM score over control sequences).

A small conditioned score MLP trained (DDPM denoising) on the elite buffer's
flattened control sequences. Exposes `score(x, t)` for the transport `score_g`
seam (`S_hat = omega_mb S_mb + omega_mf s_theta`) and `warm_start`
via a few ancestral-sampling steps. Pure JAX (no brax) — testable on fedguide.

Minimal by design: MDAC's minimal config sets `omega_mf = 0` (s_theta off), so
this is the optional full-path component; it is built strictly AFTER elite
collection from a minimal RL prior.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp


def _init_mlp(key, in_dim, hidden, out_dim):
    ks = jax.random.split(key, len(hidden) + 1)
    sizes = [in_dim] + list(hidden) + [out_dim]
    params = []
    for i, k in enumerate(ks):
        w = jax.random.normal(k, (sizes[i], sizes[i + 1])) * jnp.sqrt(2.0 / sizes[i])
        b = jnp.zeros((sizes[i + 1],))
        params.append((w, b))
    return params


def _mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def _t_embed(t, dim=16):
    # sinusoidal time embedding; t in [0,1]
    freqs = jnp.exp(jnp.linspace(0.0, 4.0, dim // 2))
    ang = t[..., None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


class JaxDiffusionPrior:
    """DDPM score-MLP prior over flattened control sequences."""

    def __init__(self, *, dim: int, n_warm_nodes: int = 5, nu: int = 1,
                 hidden=(128, 128), n_steps: int = 50, seed: int = 0):
        self.dim = int(dim)                 # flattened sequence dim (= n_nodes*nu)
        self.nu = int(nu)
        self.n_warm_nodes = int(n_warm_nodes)
        self.output_dim = int(nu)
        self.K = int(n_steps)
        self._temb = 16
        key = jax.random.PRNGKey(seed)
        self.params = _init_mlp(key, self.dim + self._temb, hidden, self.dim)
        betas = jnp.linspace(1e-4, 0.02, self.K)
        self.alphas_bar = jnp.cumprod(1.0 - betas)

    def _eps_pred(self, params, x, t01):
        h = jnp.concatenate([x, _t_embed(jnp.asarray(t01), self._temb)], axis=-1)
        return _mlp(params, h)

    def score(self, x: Any, t: Any) -> Any:
        """`s_theta(x,t) = -eps_pred / sqrt(1-abar_t)` (score = -eps/sigma)."""
        x = jnp.reshape(jnp.asarray(x), (-1,))
        k = jnp.clip(jnp.asarray(t, jnp.int32), 0, self.K - 1)
        abar = self.alphas_bar[k]
        eps = self._eps_pred(self.params, x, k.astype(jnp.float32) / self.K)
        return -eps / jnp.sqrt(jnp.clip(1.0 - abar, 1e-6, None))

    def train(self, data: Any, *, steps: int = 500, lr: float = 1e-3, batch: int = 64, seed: int = 0):
        """DDPM denoising training on elite flattened sequences `data` (N, dim)."""
        data = jnp.asarray(data, jnp.float32).reshape(data.shape[0], -1)
        params = self.params
        key = jax.random.PRNGKey(seed)

        def loss_fn(params, x0, k, noise):
            abar = self.alphas_bar[k][:, None]
            xt = jnp.sqrt(abar) * x0 + jnp.sqrt(1.0 - abar) * noise
            eps = jax.vmap(lambda xi, ki: self._eps_pred(params, xi, ki))(
                xt, k.astype(jnp.float32) / self.K)
            return jnp.mean((eps - noise) ** 2)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn))
        for _ in range(int(steps)):
            key, k1, k2, k3 = jax.random.split(key, 4)
            idx = jax.random.randint(k1, (batch,), 0, data.shape[0])
            x0 = data[idx]
            kk = jax.random.randint(k2, (batch,), 0, self.K)
            noise = jax.random.normal(k3, x0.shape)
            _, g = grad_fn(params, x0, kk, noise)
            params = [(w - lr * gw, b - lr * gb) for (w, b), (gw, gb) in zip(params, g)]
        self.params = params
        return self

    def warm_start(self, state: Any) -> Any:
        """Ancestral DDPM sample -> control sequence (n_warm_nodes, nu)."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (self.dim,))
        for k in range(self.K - 1, -1, -1):
            abar = self.alphas_bar[k]
            eps = self._eps_pred(self.params, x, jnp.asarray(float(k) / self.K))
            x = (x - jnp.sqrt(1.0 - abar) * eps) / jnp.sqrt(jnp.clip(abar, 1e-6, None))
            x = jnp.clip(x, -1.0, 1.0)
        return x.reshape(self.n_warm_nodes, self.nu)

    def act(self, obs: Any, *, key: Optional[Any] = None, deterministic: bool = True) -> Any:
        return self.warm_start(obs)[0]

    def logp_of_sequence(self, obs_seq: Any, act_seq: Any) -> Any:
        raise NotImplementedError("diffusion prior log-likelihood is intractable; "
                                  "use it via score_g, not the log-weight.")


__all__ = ["JaxDiffusionPrior"]

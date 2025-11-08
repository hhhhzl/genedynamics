import jax
import jax.numpy as jnp
from jax import Array


def langevin_step(x: Array,
                  grad: Array,
                  metric_inv: Array,
                  dt: float,
                  noise_std: float = 0.0,
                  rng_key: jax.random.PRNGKey = None) -> Array:
    """
    Single Langevin diffusion update (fully JAX-compatible).

    x_{k+1} = x_k - dt * G^{-1} grad + sqrt(2*dt)*noise

    Args:
        x: current variable (n,) or (batch, n)
        grad: gradient of energy w.r.t x (same shape as x)
        metric_inv: inverse metric tensor G^{-1} (n,n)
        dt: step size
        noise_std: noise coefficient (0 for deterministic gradient descent)
        rng_key: JAX random key (required if noise_std > 0)
    """
    # Deterministic drift term
    drift = jnp.einsum('ij,...j->...i', metric_inv, grad)
    x_next = x - dt * drift

    # Stochastic noise term (if enabled)
    if noise_std > 0.0:
        assert rng_key is not None, "rng_key must be provided when noise_std > 0"
        noise = jax.random.normal(rng_key, shape=x.shape)
        x_next = x_next + jnp.sqrt(2.0 * dt) * noise_std * noise

    return x_next


def euler_step(x: Array,
               grad: Array,
               metric_inv: Array,
               dt: float) -> Array:
    """no noise version"""
    drift = metric_inv @ grad
    return x - dt * drift

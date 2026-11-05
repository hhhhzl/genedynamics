"""JAX runtime uses explicit device placement and functional derivatives."""

import jax
import jax.numpy as jnp
import numpy as np


class JaxRuntime:
    name = "jax"
    autodiff = True

    def __init__(self, device="cpu", dtype="float32"):
        platform = "gpu" if device.startswith("cuda") else device
        devices = jax.devices(platform)
        index = int(device.split(":")[1]) if ":" in device else 0
        self.device, self.dtype, self.xp = devices[index], getattr(jnp, dtype), jnp

    def array(self, value):
        return jax.device_put(jnp.asarray(value, dtype=self.dtype), self.device)

    def to_numpy(self, value):
        return np.asarray(value)

    def scalar(self, value):
        return float(value)

    def copy(self, value):
        return jnp.array(value)

    def clip(self, value, lower, upper):
        return jnp.clip(value, lower, upper)

    def eye(self, size, like):
        return jnp.eye(size, dtype=like.dtype)

    def normal(self, shape, seed):
        return jax.device_put(
            jax.random.normal(jax.random.PRNGKey(seed), shape, self.dtype), self.device
        )

    def value_and_grad(self, function, value):
        return jax.value_and_grad(function)(value)

    def jacobian(self, function, value):
        return jax.jacrev(function)(value)

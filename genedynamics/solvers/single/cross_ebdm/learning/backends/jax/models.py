"""Device-resident ensemble MLP, Adam and whole-update-chunk scan."""
import jax
import jax.numpy as jnp
from genedynamics.core.backends.runtime.device import select_device
from ...models import TrainState


def forward(parameters, inputs):
    value = jnp.broadcast_to(inputs, (parameters[0][0].shape[0], *inputs.shape))
    for i, (weight, bias) in enumerate(parameters):
        value = jnp.einsum("ebi,eio->ebo", value, weight) + bias[:, None, :]
        if i + 1 != len(parameters):
            value = jnp.tanh(value)
    return value


def regression_loss(parameters, batch):
    error = forward(parameters, batch.inputs) - batch.targets[None]
    return jnp.sum(error**2 * batch.weights[None, :, None]) / jnp.maximum(
        parameters[0][0].shape[0] * error.shape[-1] * jnp.sum(batch.weights), 1.)


def adam_update(state, gradient, learning_rate):
    step = state.step + 1
    first = jax.tree.map(lambda m, g: .9*m + .1*g, state.first, gradient)
    second = jax.tree.map(lambda v, g: .999*v + .001*g*g, state.second, gradient)
    parameters = jax.tree.map(lambda p,m,v: p-learning_rate*(m/(1-.9**step))/(
        jnp.sqrt(v/(1-.999**step))+1e-8), state.parameters, first, second)
    return TrainState(parameters, first, second, step)


class JaxModel:
    def __init__(self, spec, device):
        self.spec, self.device = spec, select_device(device)
        self.predict = jax.jit(forward, device=self.device)
        self.train_step = jax.jit(self._step, device=self.device)
        self.train_chunk = jax.jit(self._chunk, device=self.device)
        self.input_gradient = jax.jit(jax.grad(lambda p,x,c: jnp.sum(forward(p,x)*c), argnums=1),
                                      device=self.device)

    def initialize(self, seed):
        with jax.default_device(self.device):
            key = jax.random.PRNGKey(seed)
            sizes = (self.spec.input_dim, *self.spec.hidden, self.spec.output_dim)
            layers = []
            for i, (n,m) in enumerate(zip(sizes[:-1], sizes[1:])):
                weight = jax.random.normal(jax.random.fold_in(key,i), (self.spec.ensemble,n,m), dtype=jnp.float32)
                layers.append((weight*jnp.sqrt(jnp.float32(2./(n+m))),
                               jnp.zeros((self.spec.ensemble,m), dtype=jnp.float32)))
            parameters = tuple(layers)
            zero = jax.tree.map(jnp.zeros_like, parameters)
            return TrainState(parameters, zero, zero, jnp.int32(0))

    def _step(self, state, batch):
        loss, gradient = jax.value_and_grad(regression_loss)(state.parameters, batch)
        updated = adam_update(state, gradient, self.spec.learning_rate)
        # Padding-only batches must not advance Adam moments or time.
        updated = jax.tree.map(lambda new,old: jnp.where(jnp.sum(batch.weights)>0,new,old), updated,state)
        return updated, loss

    def _chunk(self, state, batches):
        return jax.lax.scan(self._step, state, batches)

    def import_parameters(self, tensors):
        sizes = (self.spec.input_dim,*self.spec.hidden,self.spec.output_dim)
        expected = {f"layer.{i}.{name}" for i in range(len(sizes)-1) for name in ("weight","bias")}
        if set(tensors) != expected:
            raise ValueError("model layer manifest mismatch")
        layers = []
        for i,(n,m) in enumerate(zip(sizes[:-1],sizes[1:])):
            w,b = tensors[f"layer.{i}.weight"],tensors[f"layer.{i}.bias"]
            if w.shape != (self.spec.ensemble,n,m) or b.shape != (self.spec.ensemble,m):
                raise ValueError("model layer shape mismatch")
            layers.append((jax.device_put(jnp.asarray(w,dtype=jnp.float32),self.device),
                           jax.device_put(jnp.asarray(b,dtype=jnp.float32),self.device)))
        return tuple(layers)

    @staticmethod
    def synchronize(value):
        jax.block_until_ready(value)

    @staticmethod
    def export_tensors(tensors):
        return jax.device_get(tensors)


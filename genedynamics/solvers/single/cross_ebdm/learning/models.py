"""Portable learned models; providers own arrays, differentiation and optimizers.

Weights are always [ensemble,in,out]. No tensor-library import is needed to
construct metadata or select a provider. Noise is an explicit input, never an
implicit global generator. This interface is independent of physics execution.
"""
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple, Protocol
from .artifact_bundle import load_bundle, save_bundle


@dataclass(frozen=True)
class ModelSpec:
    input_dim: int
    output_dim: int
    hidden: tuple = (64, 64)
    ensemble: int = 1
    learning_rate: float = .001

    def __post_init__(self):
        import math
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1
               for x in (self.input_dim, self.output_dim, self.ensemble, *self.hidden)):
            raise ValueError("model dimensions must be positive integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")


class RegressionBatch(NamedTuple):
    inputs: Any
    targets: Any
    weights: Any  # [B]; zero masks padding/failure, never removes denominator silently


class TrainState(NamedTuple):
    parameters: Any
    first: Any
    second: Any
    step: Any


class ModelBackend(Protocol):
    def initialize(self, seed: int) -> TrainState: ...
    def predict(self, parameters, inputs): ...
    def train_step(self, state, batch: RegressionBatch): ...
    def train_chunk(self, state, batches: RegressionBatch): ...
    def input_gradient(self, parameters, inputs, cotangent): ...
    def synchronize(self, value): ...


class LearnedModel:
    """Thin solver-style frontend with lazy backend selection."""
    def __init__(self, spec, *, backend="jax", device="cpu"):
        self.spec, self.backend, self.device = spec, backend, device
        if backend == "jax":
            from .backends.jax.models import JaxModel
            self._impl = JaxModel(spec, device)
        elif backend == "torch":
            from .backends.torch.models import TorchModel
            self._impl = TorchModel(spec, device)
        elif backend == "native":
            from .backends.native.models import NativeModel
            self._impl = NativeModel(spec, device)
        else:
            raise ValueError("learned backend must be jax, torch or native")

    def initialize(self, seed=0):
        return self._impl.initialize(seed)

    def predict(self, parameters, inputs):
        return self._impl.predict(parameters, inputs)

    def train_step(self, state, batch):
        return self._impl.train_step(state, batch)

    def train_chunk(self, state, batches):
        return self._impl.train_chunk(state, batches)

    def input_gradient(self, parameters, inputs, cotangent):
        return self._impl.input_gradient(parameters, inputs, cotangent)

    def synchronize(self, value):
        return self._impl.synchronize(value)

    def import_parameters(self, tensors):
        return self._impl.import_parameters(tensors)

    def save(self, path, parameters, *, metadata=None):
        tensors = {f"layer.{i}.{name}": value for i, layer in enumerate(parameters)
                   for name, value in zip(("weight", "bias"), layer)}
        # Provider conversion happens exclusively at the artifact boundary.
        tensors = self._impl.export_tensors(tensors)
        save_bundle(path, tensors, {**(metadata or {}), "artifact_kind": "learned_inference",
                                    "model_spec": asdict(self.spec), "activation": "tanh"})

    @classmethod
    def load(cls, path, *, backend="jax", device="cpu"):
        tensors, metadata = load_bundle(path)
        if metadata.get("artifact_kind") != "learned_inference" or metadata.get("activation") != "tanh":
            raise ValueError("unsupported learned model artifact")
        values = dict(metadata["model_spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(ModelSpec(**values), backend=backend, device=device)
        return model, model.import_parameters(tensors), metadata


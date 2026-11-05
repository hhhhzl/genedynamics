"""Do not import accelerator libraries until a runtime is requested."""


def make_runtime(name="numpy", device="cpu", dtype="float32"):
    if name == "numpy":
        from .implementations.numpy_runtime import NumpyRuntime

        return NumpyRuntime(device, dtype)
    if name == "torch":
        from .implementations.torch_runtime import TorchRuntime

        return TorchRuntime(device, dtype)
    if name == "jax":
        from .implementations.jax_runtime import JaxRuntime

        return JaxRuntime(device, dtype)
    raise ValueError(f"Unknown tensor runtime: {name!r}")

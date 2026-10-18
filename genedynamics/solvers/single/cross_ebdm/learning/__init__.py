"""Models, datasets, fitting, and checkpoints for Cross-EBDM."""

# Runtime and policy entrypoints live under ``envs`` and ``solvers``.  Keeping
# this initializer tensor-runtime free lets tools inspect training metadata
# without importing JAX or PyTorch.

__all__: list[str] = []

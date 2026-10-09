"""Project a generative action batch with a JAX kernel on CPU or GPU.

Run: python examples/batched_constraints.py --device gpu --samples 4096
The halfspaces are fixed; this is an array-composition example, not a robot task.
"""

import argparse
import time

import jax
import jax.numpy as jnp

from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.constraints.solvers.jaxopt_osqp_solver import solve_hard_qp_jax


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def run(device: jax.Device, samples: int, horizon: int, repeats: int) -> None:
    RuntimeBackendManager.set_backend("jax", device=device.platform)
    backend = RuntimeBackendManager.get_backend()
    probe = backend.tensor([0.0])
    assert probe.devices() == {device}, "Backend array is on the wrong device"

    proposals = backend.randn((samples, horizon, 2), jax.random.PRNGKey(0))
    A = backend.tensor(jnp.eye(2, dtype=jnp.float32))
    b = backend.tensor([0.2, -0.1], dtype=jnp.float32)

    def project_action(action):
        return solve_hard_qp_jax(action, A, b, control_limit=1.0, maxiter=20)

    project_batch = jax.jit(jax.vmap(jax.vmap(project_action)))
    proposals.block_until_ready()
    started = time.perf_counter()
    filtered, residual = project_batch(proposals)
    jax.block_until_ready((filtered, residual))
    first_ms = 1000.0 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        filtered, residual = project_batch(proposals)
        jax.block_until_ready((filtered, residual))
    warm_ms = 1000.0 * (time.perf_counter() - started) / repeats

    assert proposals.devices() == filtered.devices() == {device}
    assert filtered.shape == (samples, horizon, 2)
    assert residual.shape == (samples, horizon)
    assert bool(jnp.isfinite(filtered).all()), "Non-finite projected actions"
    assert float(jnp.abs(filtered).max()) <= 1.0, "Action bounds violated"
    max_residual = float(residual.max())
    assert max_residual < 1e-6, f"Halfspace residual too large: {max_residual}"
    print(f"Device: {device}")
    print(f"Proposals -> filtered: {proposals.shape} -> {filtered.shape}")
    print(f"Residuals: {residual.shape}; maximum: {max_residual:.3g}")
    print(f"Compile + first projection: {first_ms:.3f} ms")
    print(f"Mean warm projection ({repeats} calls): {warm_ms:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--samples", type=positive_int, default=64)
    parser.add_argument("--horizon", type=positive_int, default=20)
    parser.add_argument("--repeats", type=positive_int, default=10)
    args = parser.parse_args()
    try:
        device = jax.devices(args.device)[0]
    except (RuntimeError, IndexError) as exc:
        raise SystemExit(
            f"Requested {args.device} is unavailable. For NVIDIA GPU setup, see "
            "docs/recipes/gpu-planning.md. This example does not fall back to CPU."
        ) from exc
    with jax.default_device(device):
        run(device, args.samples, args.horizon, args.repeats)


if __name__ == "__main__":
    main()
